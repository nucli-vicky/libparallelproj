#include "parallelproj.h"
#include "joseph3d_back_worker.h"

void joseph3d_back(const float *lor_start,
                   const float *lor_end,
                   float *image,
                   const float *image_origin,
                   const float *voxel_size,
                   const float *projection_values,
                   std::size_t num_lors,
                   const int *image_dim,
                   int device_id,
                   int threads_per_block)
{

  check_image_plane_size(image_dim[1], image_dim[2]);

#pragma omp parallel for
  for (long long i = 0; i < static_cast<long long>(num_lors); ++i)
  {
    joseph3d_back_worker(i, lor_start, lor_end, image, image_origin, voxel_size, projection_values, image_dim);
  }
}
