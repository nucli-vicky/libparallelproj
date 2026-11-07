#include "parallelproj.h"
#include "joseph3d_tof_sino_fwd_worker.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

// single-image kernel (same code as the kernel of joseph3d_tof_sino_fwd)
__global__ void joseph3d_tof_sino_fwd_batch_single_kernel(const float *lor_start,
                                                          const float *lor_end,
                                                          const float *image,
                                                          const float *image_origin,
                                                          const float *voxel_size,
                                                          float *projection_values,
                                                          std::size_t num_lors,
                                                          const int *image_dim,
                                                          float tof_bin_width,
                                                          const float *tof_sigma,
                                                          const float *tof_center_offset,
                                                          float num_sigmas,
                                                          short num_tof_bins,
                                                          unsigned char is_lor_dependent_tof_sigma,
                                                          unsigned char is_lor_dependent_tof_center_offset)
{
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_lors)
    {
        joseph3d_tof_sino_fwd_worker(i, lor_start, lor_end, image, image_origin, voxel_size, projection_values, image_dim, tof_bin_width,
                                     tof_sigma, tof_center_offset, num_sigmas, num_tof_bins,
                                     is_lor_dependent_tof_sigma, is_lor_dependent_tof_center_offset);
    }
}

//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////

void joseph3d_tof_sino_fwd_batch(const float *lor_start,
                                 const float *lor_end,
                                 const float *image,
                                 const float *image_origin,
                                 const float *voxel_size,
                                 float *projection_values,
                                 std::size_t num_lors,
                                 const int *image_dim,
                                 std::size_t num_images,
                                 float tof_bin_width,
                                 const float *tof_sigma,
                                 const float *tof_center_offset,
                                 float num_sigmas,
                                 short num_tof_bins,
                                 unsigned char is_lor_dependent_tof_sigma,
                                 unsigned char is_lor_dependent_tof_center_offset,
                                 int device_id,
                                 int threads_per_block)
{
    // Calculate nvoxels from image_dim - image_dim can be device pointer!
    std::size_t nvoxels = cuda_nvoxels_from_img_dim(image_dim);
    if (num_images == 0)
        return;

    // Set the CUDA device
    if (device_id >= 0)
    {
        cudaError_t set_err = cudaSetDevice(device_id);
        if (set_err != cudaSuccess)
        {
            cudaGetLastError(); // clear last-error state before throwing
            throw std::runtime_error(
                std::string("cudaSetDevice failed: ") + cudaGetErrorString(set_err));
        }
    }

    /////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////
    // copy arrays to device if needed
    /////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////

    // Handle lor_start (read mostly)
    CudaDevicePtr<float> d_lor_start;
    handle_cuda_input_array(lor_start, &d_lor_start.ptr, sizeof(float) * num_lors * 3, d_lor_start.owns, device_id, cudaMemAdviseSetReadMostly);

    // Handle lor_end (read mostly)
    CudaDevicePtr<float> d_lor_end;
    handle_cuda_input_array(lor_end, &d_lor_end.ptr, sizeof(float) * num_lors * 3, d_lor_end.owns, device_id, cudaMemAdviseSetReadMostly);

    // Handle image (read mostly)
    CudaDevicePtr<float> d_image;
    handle_cuda_input_array(image, &d_image.ptr, sizeof(float) * nvoxels * num_images, d_image.owns, device_id, cudaMemAdviseSetReadMostly);

    // Handle image_origin (read mostly)
    CudaDevicePtr<float> d_image_origin;
    handle_cuda_input_array(image_origin, &d_image_origin.ptr, sizeof(float) * 3, d_image_origin.owns, device_id, cudaMemAdviseSetReadMostly);

    // Handle voxel_size (read mostly)
    CudaDevicePtr<float> d_voxel_size;
    handle_cuda_input_array(voxel_size, &d_voxel_size.ptr, sizeof(float) * 3, d_voxel_size.owns, device_id, cudaMemAdviseSetReadMostly);

    // Handle projection_values (write access) - size depends on TOF bins
    std::size_t projection_values_size = sizeof(float) * num_lors * num_tof_bins * num_images;
    CudaDevicePtr<float> d_projection_values;
    handle_cuda_input_array(projection_values, &d_projection_values.ptr, projection_values_size, d_projection_values.owns, device_id, cudaMemAdviseSetAccessedBy);

    // Handle image_dim (read mostly)
    CudaDevicePtr<int> d_image_dim;
    handle_cuda_input_array(image_dim, &d_image_dim.ptr, sizeof(int) * 3, d_image_dim.owns, device_id, cudaMemAdviseSetReadMostly);

    // Handle tof_sigma (read mostly)
    CudaDevicePtr<float> d_tof_sigma;
    std::size_t tof_sigma_size = is_lor_dependent_tof_sigma ? sizeof(float) * num_lors : sizeof(float);
    handle_cuda_input_array(tof_sigma, &d_tof_sigma.ptr, tof_sigma_size, d_tof_sigma.owns, device_id, cudaMemAdviseSetReadMostly);

    // Handle tof_center_offset (read mostly)
    CudaDevicePtr<float> d_tof_center_offset;
    std::size_t tof_center_offset_size = is_lor_dependent_tof_center_offset ? sizeof(float) * num_lors : sizeof(float);
    handle_cuda_input_array(tof_center_offset, &d_tof_center_offset.ptr, tof_center_offset_size, d_tof_center_offset.owns, device_id, cudaMemAdviseSetReadMostly);

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    // launch the kernel
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////

    int num_blocks = (int)((num_lors + threads_per_block - 1) / threads_per_block);
    // Flush any stale (non-sticky) last-error state - e.g. left behind by a
    // previously failed CUDA call of the CALLER (other libraries, earlier
    // failed calls into this library, ...). CUDA only resets the last error
    // when it is read, so without this flush the launch check below would
    // misattribute such a stale error to this kernel launch.
    cudaGetLastError();
    // The images are projected one after the other with the single-image
    // kernel. Sharing the bilinear and TOF weights between images does not
    // pay off on the GPU here: the forward projection adds to the TOF bins of
    // the LOR in global memory for every plane, and for a group of images
    // these updates no longer fit in the L2 cache. (The OpenMP build uses
    // groups of images.)
    for (std::size_t g = 0; g < num_images; ++g)
    {
        joseph3d_tof_sino_fwd_batch_single_kernel<<<num_blocks, threads_per_block>>>(d_lor_start.ptr, d_lor_end.ptr, d_image.ptr + g * nvoxels, d_image_origin.ptr, d_voxel_size.ptr, d_projection_values.ptr + g * num_lors * num_tof_bins, num_lors, d_image_dim.ptr, tof_bin_width, d_tof_sigma.ptr, d_tof_center_offset.ptr, num_sigmas, num_tof_bins, is_lor_dependent_tof_sigma, is_lor_dependent_tof_center_offset);
    }

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess)
        throw std::runtime_error(
            std::string("CUDA kernel launch failed: ") + cudaGetErrorString(launch_err));
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess)
    {
        cudaGetLastError(); // clear last-error state before throwing
        throw std::runtime_error(
            std::string("CUDA kernel error: ") + cudaGetErrorString(sync_err));
    }

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    // Copy the result back to the host if (and only if) the device buffer is a
    // temporary copy we allocated ourselves (owns == true). If the caller
    // passed device/managed memory, the kernel wrote to it directly.
    //
    // No explicit cudaFree calls are needed here: every device buffer is held
    // by a CudaDevicePtr (RAII, see cuda_utils.h), whose destructor frees the
    // memory automatically when this function returns OR when an exception is
    // thrown anywhere above — so no leaks on error paths.
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////

    if (d_projection_values.owns)
        cuda_memcpy_d2h(projection_values, d_projection_values.ptr, projection_values_size);
}
