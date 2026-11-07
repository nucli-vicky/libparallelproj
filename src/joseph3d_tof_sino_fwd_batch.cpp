#include "parallelproj.h"
#include "joseph3d_tof_sino_fwd_worker.h"
#include "joseph3d_tof_sino_fwd_batch_worker.h"

template <int T>
static void joseph3d_tof_sino_fwd_batch_omp(const float *lor_start,
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
                                unsigned char is_lor_dependent_tof_center_offset)
{
#pragma omp parallel for
  for (long long i = 0; i < static_cast<long long>(num_lors); ++i)
  {
    joseph3d_tof_sino_fwd_batch_worker<T>(i, lor_start, lor_end, image, image_origin, voxel_size, projection_values, image_dim, tof_bin_width, tof_sigma, tof_center_offset, num_sigmas, num_tof_bins, is_lor_dependent_tof_sigma, is_lor_dependent_tof_center_offset, num_lors, num_images);
  }
}

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
  check_image_plane_size(image_dim[1], image_dim[2]);

  // group size T: number of images that share the bilinear and TOF weights of a plane
  if (num_images >= 8)
    joseph3d_tof_sino_fwd_batch_omp<8>(lor_start, lor_end, image, image_origin, voxel_size, projection_values, num_lors, image_dim, num_images, tof_bin_width, tof_sigma, tof_center_offset, num_sigmas, num_tof_bins, is_lor_dependent_tof_sigma, is_lor_dependent_tof_center_offset);
  else if (num_images >= 4)
    joseph3d_tof_sino_fwd_batch_omp<4>(lor_start, lor_end, image, image_origin, voxel_size, projection_values, num_lors, image_dim, num_images, tof_bin_width, tof_sigma, tof_center_offset, num_sigmas, num_tof_bins, is_lor_dependent_tof_sigma, is_lor_dependent_tof_center_offset);
  else if (num_images >= 2)
    joseph3d_tof_sino_fwd_batch_omp<2>(lor_start, lor_end, image, image_origin, voxel_size, projection_values, num_lors, image_dim, num_images, tof_bin_width, tof_sigma, tof_center_offset, num_sigmas, num_tof_bins, is_lor_dependent_tof_sigma, is_lor_dependent_tof_center_offset);
  else if (num_images == 1)
  {
    // one image: use the single-image worker
#pragma omp parallel for
    for (long long i = 0; i < static_cast<long long>(num_lors); ++i)
    {
      joseph3d_tof_sino_fwd_worker(i, lor_start, lor_end, image, image_origin, voxel_size, projection_values, image_dim, tof_bin_width, tof_sigma, tof_center_offset, num_sigmas, num_tof_bins, is_lor_dependent_tof_sigma, is_lor_dependent_tof_center_offset);
    }
  }
}
