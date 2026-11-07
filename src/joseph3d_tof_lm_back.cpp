#include "parallelproj.h"
#include "joseph3d_tof_lm_back_worker.h"

void joseph3d_tof_lm_back(const float *lor_start,
                          const float *lor_end,
                          float *image,
                          const float *image_origin,
                          const float *voxel_size,
                          const float *projection_values,
                          std::size_t num_events,
                          const int *image_dim,
                          float tof_bin_width,
                          const float *tof_sigma,
                          const float *tof_center_offset,
                          float num_sigmas,
                          const short *tof_bin_index,
                          short num_tof_bins,
                          unsigned char is_lor_dependent_tof_sigma,
                          unsigned char is_lor_dependent_tof_center_offset,
                          int device_id,
                          int threads_per_block)
{

  check_image_plane_size(image_dim[1], image_dim[2]);

#pragma omp parallel for
  for (long long i = 0; i < static_cast<long long>(num_events); ++i)
  {
    joseph3d_tof_lm_back_worker(i, lor_start, lor_end, image, image_origin, voxel_size, projection_values, image_dim, tof_bin_width,
                                tof_sigma, tof_center_offset, num_sigmas, tof_bin_index, num_tof_bins,
                                is_lor_dependent_tof_sigma, is_lor_dependent_tof_center_offset);
  }
}
