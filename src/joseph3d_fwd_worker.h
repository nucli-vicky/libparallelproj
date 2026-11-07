#pragma once
#include "cuda_compat.h"
#include "utils.h"

WORKER_QUALIFIER inline void joseph3d_fwd_worker(std::size_t i,
                                                 const float *lor_start,
                                                 const float *lor_end,
                                                 const float *image,
                                                 const float *image_origin,
                                                 const float *voxel_size,
                                                 float *projection_values,
                                                 const int *image_dim)
{
  int n0 = image_dim[0];
  int n1 = image_dim[1];
  int n2 = image_dim[2];

  int direction;
  int i0, i1, i2;
  float i0_f, i1_f, i2_f;
  float cf;

  float a0, a1, a2;
  float b0, b1, b2;

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

  // if the ray does not intersect the image cube, the projection is 0
  // istart and iend are set to -1
  if (istart == -1)
  {
    projection_values[i] = 0.0f;
    return;
  }

  // sum over planes in a local variable and write the output once
  float sum = 0.0f;

  if (direction == 0)
  {
    dr = d0;

    a1 = (d1 * voxel_size[direction]) / (voxel_size[1] * dr);
    b1 = (lor_start[3 * i + 1] - image_origin[1] + d1 * (image_origin[direction] - lor_start[3 * i + direction]) / dr) / voxel_size[1];

    a2 = (d2 * voxel_size[direction]) / (voxel_size[2] * dr);
    b2 = (lor_start[3 * i + 2] - image_origin[2] + d2 * (image_origin[direction] - lor_start[3 * i + direction]) / dr) / voxel_size[2];

    for (i0 = istart; i0 <= iend; ++i0)
    {
      i1_f = i0 * a1 + b1;
      i2_f = i0 * a2 + b2;
      sum += bilinear_interp_fixed0(image, n0, n1, n2, i0, i1_f, i2_f);
    }
  }
  else if (direction == 1)
  {
    dr = d1;

    a0 = (d0 * voxel_size[direction]) / (voxel_size[0] * dr);
    b0 = (lor_start[3 * i + 0] - image_origin[0] + d0 * (image_origin[direction] - lor_start[3 * i + direction]) / dr) / voxel_size[0];

    a2 = (d2 * voxel_size[direction]) / (voxel_size[2] * dr);
    b2 = (lor_start[3 * i + 2] - image_origin[2] + d2 * (image_origin[direction] - lor_start[3 * i + direction]) / dr) / voxel_size[2];

    for (i1 = istart; i1 <= iend; ++i1)
    {
      i0_f = i1 * a0 + b0;
      i2_f = i1 * a2 + b2;
      sum += bilinear_interp_fixed1(image, n0, n1, n2, i0_f, i1, i2_f);
    }
  }
  else if (direction == 2)
  {
    dr = d2;

    a0 = (d0 * voxel_size[direction]) / (voxel_size[0] * dr);
    b0 = (lor_start[3 * i + 0] - image_origin[0] + d0 * (image_origin[direction] - lor_start[3 * i + direction]) / dr) / voxel_size[0];

    a1 = (d1 * voxel_size[direction]) / (voxel_size[1] * dr);
    b1 = (lor_start[3 * i + 1] - image_origin[1] + d1 * (image_origin[direction] - lor_start[3 * i + direction]) / dr) / voxel_size[1];

    for (i2 = istart; i2 <= iend; ++i2)
    {
      i0_f = i2 * a0 + b0;
      i1_f = i2 * a1 + b1;
      sum += bilinear_interp_fixed2(image, n0, n1, n2, i0_f, i1_f, i2);
    }
  }

  projection_values[i] = cf * sum;
}
