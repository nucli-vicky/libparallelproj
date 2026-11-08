#include "parallelproj.h"
#include "joseph3d_tof_sino_fwd_worker.h"

void joseph3d_tof_sino_fwd(const float *lor_start,
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
                           unsigned char is_lor_dependent_tof_center_offset,
                           int device_id,
                           int threads_per_block)
{

  check_image_plane_size(image_dim[1], image_dim[2]);

#pragma omp parallel for
  for (long long i = 0; i < static_cast<long long>(num_lors); ++i)
  {
    joseph3d_tof_sino_fwd_worker(i, lor_start, lor_end, image, image_origin, voxel_size, projection_values, image_dim, tof_bin_width,
                                 tof_sigma, tof_center_offset, num_sigmas, num_tof_bins,
                                 is_lor_dependent_tof_sigma, is_lor_dependent_tof_center_offset);
  }
}
