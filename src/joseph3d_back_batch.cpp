#include "parallelproj.h"
#include "joseph3d_back_worker.h"
#include "joseph3d_back_batch_worker.h"

template <int T>
static void joseph3d_back_batch_omp(const float *lor_start,
                                    const float *lor_end,
                                    float *image,
                                    const float *image_origin,
                                    const float *voxel_size,
                                    const float *projection_values,
                                    std::size_t num_lors,
                                    const int *image_dim,
                                    std::size_t num_images)
{
#pragma omp parallel for
  for (long long i = 0; i < static_cast<long long>(num_lors); ++i)
  {
    joseph3d_back_batch_worker<T>(i, lor_start, lor_end, image, image_origin, voxel_size, projection_values, image_dim, num_lors, num_images);
  }
}

void joseph3d_back_batch(const float *lor_start,
                         const float *lor_end,
                         float *image,
                         const float *image_origin,
                         const float *voxel_size,
                         const float *projection_values,
                         std::size_t num_lors,
                         const int *image_dim,
                         std::size_t num_images,
                         int device_id,
                         int threads_per_block)
{
  check_image_plane_size(image_dim[1], image_dim[2]);

  // group size T: number of images that share the bilinear weights of a plane
  if (num_images >= 8)
    joseph3d_back_batch_omp<8>(lor_start, lor_end, image, image_origin, voxel_size, projection_values, num_lors, image_dim, num_images);
  else if (num_images >= 4)
    joseph3d_back_batch_omp<4>(lor_start, lor_end, image, image_origin, voxel_size, projection_values, num_lors, image_dim, num_images);
  else if (num_images >= 2)
    joseph3d_back_batch_omp<2>(lor_start, lor_end, image, image_origin, voxel_size, projection_values, num_lors, image_dim, num_images);
  else if (num_images == 1)
  {
    // one image: use the single-image worker
#pragma omp parallel for
    for (long long i = 0; i < static_cast<long long>(num_lors); ++i)
    {
      joseph3d_back_worker(i, lor_start, lor_end, image, image_origin, voxel_size, projection_values, image_dim);
    }
  }
}
