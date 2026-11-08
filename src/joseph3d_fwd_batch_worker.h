#pragma once
#include "cuda_compat.h"
#include "utils.h"

// Forward projection of LOR i for a batch of num_images images.
// image holds the images one after the other ([num_images, n0, n1, n2]),
// projection_values the projections one after the other
// ([num_images, num_lors]).
// The ray geometry is computed once per LOR. For every traversed plane, the
// bilinear corners and weights are computed once per group of T images and
// applied to all images in the group.
template <int T>
WORKER_QUALIFIER inline void joseph3d_fwd_batch_worker(std::size_t i,
                                                       const float *lor_start,
                                                       const float *lor_end,
                                                       const float *image,
                                                       const float *image_origin,
                                                       const float *voxel_size,
                                                       float *projection_values,
                                                       const int *image_dim,
                                                       std::size_t num_lors,
                                                       std::size_t num_images)
{
  int n0 = image_dim[0];
  int n1 = image_dim[1];
  int n2 = image_dim[2];
  std::size_t nvoxels = std::size_t(n0) * n1 * n2;

  int direction;
  float cf;

  float a0 = 0, a1 = 0, a2 = 0;
  float b0 = 0, b1 = 0, b2 = 0;

  int istart = -1;
  int iend = -1;

  float d0 = lor_end[3 * i + 0] - lor_start[3 * i + 0];
  float d1 = lor_end[3 * i + 1] - lor_start[3 * i + 1];
  float d2 = lor_end[3 * i + 2] - lor_start[3 * i + 2];

  float dr;

  // test whether the ray intersects the image cube
  // if it does not, istart and iend are set to -1
  // if it does, direction is set to the principal axis
  // and istart and iend are set to the first and last voxel planes
  // that are intersected
  // cf is the correction factor voxel_size[dir]/cos[dir]
  ray_cube_intersection_joseph(lor_start + 3 * i, lor_end + 3 * i, image_origin, voxel_size, image_dim, direction, cf, istart, iend);

  // if the ray does not intersect the image cube, all projections are 0
  if (istart == -1)
  {
    for (std::size_t b = 0; b < num_images; ++b)
      projection_values[b * num_lors + i] = 0.0f;
    return;
  }

  // the coordinates of the intersection of the ray with plane k along the
  // principal axis, in voxel units of the two other axes, are k*a + b
  if (direction == 0)
  {
    dr = d0;
    a1 = (d1 * voxel_size[direction]) / (voxel_size[1] * dr);
    b1 = (lor_start[3 * i + 1] - image_origin[1] + d1 * (image_origin[direction] - lor_start[3 * i + direction]) / dr) / voxel_size[1];
    a2 = (d2 * voxel_size[direction]) / (voxel_size[2] * dr);
    b2 = (lor_start[3 * i + 2] - image_origin[2] + d2 * (image_origin[direction] - lor_start[3 * i + direction]) / dr) / voxel_size[2];
  }
  else if (direction == 1)
  {
    dr = d1;
    a0 = (d0 * voxel_size[direction]) / (voxel_size[0] * dr);
    b0 = (lor_start[3 * i + 0] - image_origin[0] + d0 * (image_origin[direction] - lor_start[3 * i + direction]) / dr) / voxel_size[0];
    a2 = (d2 * voxel_size[direction]) / (voxel_size[2] * dr);
    b2 = (lor_start[3 * i + 2] - image_origin[2] + d2 * (image_origin[direction] - lor_start[3 * i + direction]) / dr) / voxel_size[2];
  }
  else
  {
    dr = d2;
    a0 = (d0 * voxel_size[direction]) / (voxel_size[0] * dr);
    b0 = (lor_start[3 * i + 0] - image_origin[0] + d0 * (image_origin[direction] - lor_start[3 * i + direction]) / dr) / voxel_size[0];
    a1 = (d1 * voxel_size[direction]) / (voxel_size[1] * dr);
    b1 = (lor_start[3 * i + 1] - image_origin[1] + d1 * (image_origin[direction] - lor_start[3 * i + direction]) / dr) / voxel_size[1];
  }

  for (std::size_t g = 0; g < num_images; g += T)
  {
    // sums over planes for the images g .. g+T-1
    float sum[T];
    PP_UNROLL
    for (int b = 0; b < T; ++b)
      sum[b] = 0.0f;

    if (direction == 0)
    {
      for (int i0 = istart; i0 <= iend; ++i0)
      {
        BilinearCorners c;
        bilinear_corners_fixed0(n0, n1, n2, i0, i0 * a1 + b1, i0 * a2 + b2, c);
        gather4_batch<T>(image, nvoxels, g, num_images, c, sum);
      }
    }
    else if (direction == 1)
    {
      for (int i1 = istart; i1 <= iend; ++i1)
      {
        BilinearCorners c;
        bilinear_corners_fixed1(n0, n1, n2, i1 * a0 + b0, i1, i1 * a2 + b2, c);
        gather4_batch<T>(image, nvoxels, g, num_images, c, sum);
      }
    }
    else
    {
      for (int i2 = istart; i2 <= iend; ++i2)
      {
        BilinearCorners c;
        bilinear_corners_fixed2(n0, n1, n2, i2 * a0 + b0, i2 * a1 + b1, i2, c);
        gather4_batch<T>(image, nvoxels, g, num_images, c, sum);
      }
    }

    PP_UNROLL
    for (int b = 0; b < T; ++b)
      if (g + b < num_images)
        projection_values[(g + b) * num_lors + i] = cf * sum[b];
  }
}
