#pragma once
#include "cuda_compat.h"
#include "utils.h"

// TOF sinogram back projection of LOR i for a batch of num_images images.
// image holds the images one after the other ([num_images, n0, n1, n2]),
// projection_values the TOF sinograms one after the other
// ([num_images, num_lors, num_tof_bins]). Values are added to image.
// The ray geometry and TOF parameters are computed once per LOR. For every
// traversed plane, the bilinear corners and the TOF weights are computed once
// per group of T images and applied to all images in the group.
template <int T>
WORKER_QUALIFIER inline void joseph3d_tof_sino_back_batch_worker(std::size_t i,
                                                                const float *lor_start,
                                                                const float *lor_end,
                                                                float *image,
                                                                const float *image_origin,
                                                                const float *voxel_size,
                                                                const float *projection_values,
                                                                const int *image_dim,
                                                                float tof_bin_width,
                                                                const float *tof_sigma,
                                                                const float *tof_center_offset,
                                                                float num_sigmas,
                                                                short num_tof_bins,
                                                                unsigned char is_lor_dependent_tof_sigma,
                                                                unsigned char is_lor_dependent_tof_center_offset,
                                                                std::size_t num_lors,
                                                                std::size_t num_images)
{
  std::size_t proj_stride = num_lors * num_tof_bins;            // values per image
  std::size_t lor_offset = i * std::size_t(num_tof_bins);       // first TOF bin of LOR i

  // an image is skipped if the sum over the TOF bins of LOR i is 0
  // (as in the single-image worker); nothing to do if this holds for all images
  bool any_active = false;
  for (std::size_t b = 0; b < num_images && !any_active; ++b)
  {
    float tof_lor_sum = 0.0f;
    for (short j = 0; j < num_tof_bins; ++j)
      tof_lor_sum += projection_values[b * proj_stride + lor_offset + j];
    any_active = (tof_lor_sum != 0);
  }
  if (!any_active)
  {
    return;
  }

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

  if (istart == -1)
  {
    return;
  }

  ////// TOF parameters (as in the single-image TOF sinogram workers)
  float tof_weights[MAX_NUM_TOF_WEIGHTS];      // TOF weights of one plane, shared by all images
  float costheta = voxel_size[direction] / cf; // cosine of angle between ray and principal axis

  // get the tof_sigma and tof_center_offset for this LOR depending on whether they are constant or LOR-dependent
  float local_tof_sigma = is_lor_dependent_tof_sigma ? tof_sigma[i] : tof_sigma[0];
  float local_tof_center_offset = is_lor_dependent_tof_center_offset ? tof_center_offset[i] : tof_center_offset[0];

  // maximum number of TOF bins away from the current TOF bin to consider
  float max_tof_bin_diff = num_sigmas * local_tof_sigma / tof_bin_width;

  // sign variable that indicated whether TOF bin numbers increase or decrease when
  // through the image along the principal axis direction
  float sign = (lor_end[3 * i + direction] >= lor_start[3 * i + direction]) ? 1.0 : -1.0;

  // the center of the first TOF bin (TOF bin 0) projected onto the principal axis
  float tof_origin = 0.5 * (lor_start[3 * i + direction] + lor_end[3 * i + direction]) - sign * (0.5 * num_tof_bins - 0.5) * (tof_bin_width * costheta) + local_tof_center_offset * costheta;
  // slope of TOF bin number as a function of distance along the principal axis
  float tof_slope = sign * tof_bin_width * costheta;

  // the TOF bin number of intersection point of the ray with a given image plane along the principal axis is it_f = i*at + bt
  float at = sign * cf / tof_bin_width;
  float bt = (image_origin[direction] - tof_origin) / tof_slope;

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
    bool active[T];
    bool group_active = false;
    PP_UNROLL
    for (int b = 0; b < T; ++b)
    {
      float tof_lor_sum = 0.0f;
      if (g + b < num_images)
        for (short j = 0; j < num_tof_bins; ++j)
          tof_lor_sum += projection_values[(g + b) * proj_stride + lor_offset + j];
      active[b] = (tof_lor_sum != 0);
      group_active = group_active || active[b];
    }
    if (!group_active)
    {
      continue;
    }

    for (int k = istart; k <= iend; ++k)
    {
      BilinearCorners c;
      if (direction == 0)
        bilinear_corners_fixed0(n0, n1, n2, k, k * a1 + b1, k * a2 + b2, c);
      else if (direction == 1)
        bilinear_corners_fixed1(n0, n1, n2, k * a0 + b0, k, k * a2 + b2, c);
      else
        bilinear_corners_fixed2(n0, n1, n2, k * a0 + b0, k * a1 + b1, k, c);

      // TOF weights of this plane, shared by all images of the group
      int it_min, k_start, k_end;
      float sum_weights = tof_weight_window(k * at + bt, max_tof_bin_diff, tof_bin_width, local_tof_sigma,
                                            num_tof_bins, tof_weights, it_min, k_start, k_end);
      float norm = (sum_weights > 0.0f ? sum_weights : 1.0f);

      // TOF-weighted sum of the TOF bins of each image, normalized by the sum of the TOF weights
      float val[T];
      PP_UNROLL
      for (int b = 0; b < T; ++b)
      {
        float toAdd = 0.0f;
        if (active[b])
        {
          for (int kk = k_start; kk < k_end; ++kk)
            toAdd += projection_values[(g + b) * proj_stride + lor_offset + it_min + kk] * tof_weights[kk];
          toAdd /= norm;
        }
        val[b] = cf * toAdd;
      }

      scatter4_batch<T>(image, nvoxels, g, num_images, c, val);
    }
  }
}
