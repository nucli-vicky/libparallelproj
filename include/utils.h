#pragma once
#include <cstddef>
#include <cmath>
#include <climits>
#include <stdexcept>
#include "cuda_compat.h"

#ifndef __CUDACC__
#include <math.h>
#endif

#ifndef MAX_NUM_TOF_WEIGHTS
// default 64 can be changed at compile time
// for a TOF sinogram projector, we don't expect that more than 64 TOF bins will
// have a significant contribution
// if they have, bigger TOF bin widths should be used
#define MAX_NUM_TOF_WEIGHTS 64
#endif

template <typename T>
WORKER_QUALIFIER inline void atomic_sum(T *target, T value)
{
#ifdef __CUDACC__
  atomicAdd(target, value);
#else
#pragma omp atomic
  *target += value;
#endif
}

/**
 * @brief Ray/cube intersection + Joseph-specific parameters:
 *   - principal axis (0,1,2)
 *   - correction factor
 *   - start/end plane indices
 *
 * Implements the IEEE-754 slab method using fminf/fmaxf to handle +/-0 & inf.
 * Compiles under CUDA (__device__) and standard C++ compilers.
 *
 * If no intersection occurs, `direction` is set to -1 and all outputs
 * (correction, start_plane, end_plane) are reset to defaults.
 * Otherwise fills outputs appropriately.
 *
 * @param xstart       Ray origin (length-3)
 * @param xend         Ray end point (length-3)
 * @param img_origin   World coords of center of voxel (0,0,0)
 * @param voxsize      Voxel sizes (length-3)
 * @param img_dim      Image dimensions [n0,n1,n2]
 * @param[out] direction   Principal axis (0,1,2)
 * @param[out] correction  Correction factor = voxsize[dir]/cos[dir]
 * @param[out] start_plane First plane index to traverse, or -1 if no intersection
 * @param[out] end_plane   Last plane index to traverse, or -1 if no intersection
 */
WORKER_QUALIFIER inline void ray_cube_intersection_joseph(
    const float *xstart,
    const float *xend,
    const float *img_origin,
    const float *voxsize,
    const int *img_dim,
    int &direction,
    float &correction,
    int &start_plane,
    int &end_plane)
{

  // default values - assume no intersection
  direction = 0;
  correction = 1.0f;
  start_plane = -1;
  end_plane = -1;

  // build box bounds: outer faces
  float box_min[3], box_max[3];
  for (int k = 0; k < 3; ++k)
  {
    box_min[k] = img_origin[k] - 0.5f * voxsize[k];
    box_max[k] = box_min[k] + img_dim[k] * voxsize[k];
  }

  // ray vector
  float dr[3];
  for (int k = 0; k < 3; ++k)
    dr[k] = xend[k] - xstart[k];

  // degenerate zero-length LOR (xstart == xend): no ray direction is defined,
  // so treat it as "no intersection". Without this guard sum_sq below is 0 and
  // cos_sq = dr_sq / sum_sq = 0/0 = NaN. direction / correction / start_plane /
  // end_plane already hold their no-hit defaults set above.
  if (dr[0] == 0.0f && dr[1] == 0.0f && dr[2] == 0.0f)
    return;

  // slab intersection, parameter t in [0,1].
  // Rays parallel to an axis (dr[k] == 0, e.g. dr_z for in-plane / direct-plane
  // LORs) are handled explicitly rather than through 1/dr[k]: this keeps the
  // test well-defined and correct even under -ffast-math / -ffinite-math-only,
  // where the inf/NaN arithmetic the branchless slab method relied on is UB.
  float tmin = 0.0f;
  float tmax = 1.0f;
  for (int k = 0; k < 3; ++k)
  {
    if (dr[k] != 0.0f)
    {
      float inv = 1.0f / dr[k];
      float t1 = (box_min[k] - xstart[k]) * inv;
      float t2 = (box_max[k] - xstart[k]) * inv;
      tmin = fmaxf(tmin, fminf(t1, t2));
      tmax = fminf(tmax, fmaxf(t1, t2));
    }
    else if (xstart[k] < box_min[k] || xstart[k] > box_max[k])
    {
      // parallel to axis k and outside the slab -> no intersection.
      // direction / correction / start_plane / end_plane already hold their
      // no-hit defaults set above.
      return;
    }
    // else: parallel to axis k but inside the slab -> does not constrain t.
  }

  if (tmax < tmin)
  {
    // no intersection - direction = 0, start_plane = -1, end_plane = -1
    // correction = 1.0f, all set by default
    return;
  }

  // principal axis & correction
  float dr_sq[3] = {dr[0] * dr[0], dr[1] * dr[1], dr[2] * dr[2]};
  float sum_sq = dr_sq[0] + dr_sq[1] + dr_sq[2];
  // compute squared cosines
  float cos_sq[3] = {dr_sq[0] / sum_sq, dr_sq[1] / sum_sq, dr_sq[2] / sum_sq};

  // choose principal axis based on max squared cosine
  direction = 0;
  if (cos_sq[1] >= cos_sq[0] && cos_sq[1] >= cos_sq[2])
    direction = 1;
  else if (cos_sq[2] >= cos_sq[0] && cos_sq[2] >= cos_sq[1])
    direction = 2;

  // compute correction factor
  correction = voxsize[direction] / sqrtf(cos_sq[direction]);

  // get the first and last voxel planes that are intersected
  // computer the ray cube intersection points
  // note that if xstart or xend are inside the cube, these are xstart / xend
  float xi1 = xstart[direction] + tmin * dr[direction];
  float xi2 = xstart[direction] + tmax * dr[direction];

  // floating point plane values
  float f1 = (xi1 - img_origin[direction]) / voxsize[direction];
  float f2 = (xi2 - img_origin[direction]) / voxsize[direction];

  // Use floor (not truncation toward zero) so the entry-face case f1 in
  // [-0.5, 0) is handled consistently with the floorf-based start/end below.
  // If floor(f1) == floor(f2) the ray stays within a single voxel-plane
  // interval and no integer plane is crossed.
  if ((int)floorf(f1) != (int)floorf(f2))
  {
    if (f1 > f2)
    {
      float tmp = f1;
      f1 = f2;
      f2 = tmp;
    }
    // first full plane floor(f1)+1
    // ceil(f1) is not completely correct in case f1 is an integer
    start_plane = (int)floorf(f1) + 1;
    // last full plane is floor(exit)
    end_plane = (int)floorf(f2);
  }
}

// Throws std::invalid_argument if one image plane (n1*n2 voxels) does not
// fit in an int. The bilinear helpers below rely on this.
inline void check_image_plane_size(int n1, int n2)
{
  if (static_cast<long long>(n1) * n2 > INT_MAX)
    throw std::invalid_argument("image_dim[1] * image_dim[2] must not exceed INT_MAX");
}

/**
 * Bilinear interpolation in one image plane, in two steps:
 *   1. bilinear_corners_fixed{0,1,2}: compute the offsets of the 4 corner
 *      voxels, their bilinear weights and whether each corner lies inside
 *      the image.
 *   2. gather4 (forward) / scatter4 (back): read, or atomically add, the
 *      weighted values at these corners.
 * Step 1 depends only on the ray geometry and the image dimensions, not on
 * the voxel values.
 *
 * The fixed index is the plane index along the principal axis:
 *   - fixed0: plane at fixed i0, interpolate in (i1,i2)
 *   - fixed1: plane at fixed i1, interpolate in (i0,i2)
 *   - fixed2: plane at fixed i2, interpolate in (i0,i1)
 *
 * Offsets are relative to the first voxel of the image (C order, i2 fastest).
 * Index products within one plane use int, so n1*n2 must fit in an int
 * (see check_image_plane_size). The plane index term uses std::size_t.
 * Corner order is (lo,lo), (hi,lo), (lo,hi), (hi,hi) in the two interpolated
 * axes. The offset of a corner outside the image is not meaningful and must
 * not be used.
 *
 * If compiled with CUDA, these functions are device functions.
 */
struct BilinearCorners
{
  std::size_t off[4]; // voxel offsets
  float w[4];         // bilinear weights
  bool valid[4];      // corner lies inside the image
};

WORKER_QUALIFIER inline void bilinear_corners_fixed0(int n0, int n1, int n2,
                                                     int i0,
                                                     float i_f1,
                                                     float i_f2,
                                                     BilinearCorners &c)
{
  int i1_0 = int(floorf(i_f1));
  int i2_0 = int(floorf(i_f2));
  int i1_1 = i1_0 + 1;
  int i2_1 = i2_0 + 1;

  float w1 = i_f1 - i1_0;
  float w2 = i_f2 - i2_0;

  const int i1s[4] = {i1_0, i1_1, i1_0, i1_1};
  const int i2s[4] = {i2_0, i2_0, i2_1, i2_1};
  const float ws[4] = {(1 - w1) * (1 - w2), w1 * (1 - w2), (1 - w1) * w2, w1 * w2};

  // in-plane index products in int, plane offset in std::size_t
  const int P = n1 * n2;
  const std::size_t plane_base = std::size_t(i0) * P;
  for (int k = 0; k < 4; ++k)
  {
    c.w[k] = ws[k];
    c.valid[k] = (i1s[k] >= 0 && i1s[k] < n1 && i2s[k] >= 0 && i2s[k] < n2);
    c.off[k] = plane_base + std::size_t(i1s[k] * n2 + i2s[k]);
  }
}

WORKER_QUALIFIER inline void bilinear_corners_fixed1(int n0, int n1, int n2,
                                                     float i_f0,
                                                     int i1,
                                                     float i_f2,
                                                     BilinearCorners &c)
{
  int i0_0 = int(floorf(i_f0));
  int i2_0 = int(floorf(i_f2));
  int i0_1 = i0_0 + 1;
  int i2_1 = i2_0 + 1;

  float w0 = i_f0 - i0_0;
  float w2 = i_f2 - i2_0;

  const int i0s[4] = {i0_0, i0_1, i0_0, i0_1};
  const int i2s[4] = {i2_0, i2_0, i2_1, i2_1};
  const float ws[4] = {(1 - w0) * (1 - w2), w0 * (1 - w2), (1 - w0) * w2, w0 * w2};

  // in-plane index products in int; the offsets of planes i0_0 and i0_1
  // differ by n1*n2
  const int P = n1 * n2;
  const int row = i1 * n2;
  const std::size_t base0 = std::size_t(i0_0) * P;
  const std::size_t base1 = base0 + P;
  for (int k = 0; k < 4; ++k)
  {
    c.w[k] = ws[k];
    c.valid[k] = (i0s[k] >= 0 && i0s[k] < n0 && i2s[k] >= 0 && i2s[k] < n2);
    c.off[k] = ((k == 1 || k == 3) ? base1 : base0) + std::size_t(row + i2s[k]);
  }
}

WORKER_QUALIFIER inline void bilinear_corners_fixed2(int n0, int n1, int n2,
                                                     float i_f0,
                                                     float i_f1,
                                                     int i2,
                                                     BilinearCorners &c)
{
  int i0_0 = int(floorf(i_f0));
  int i1_0 = int(floorf(i_f1));
  int i0_1 = i0_0 + 1;
  int i1_1 = i1_0 + 1;

  float w0 = i_f0 - i0_0;
  float w1 = i_f1 - i1_0;

  const int i0s[4] = {i0_0, i0_1, i0_0, i0_1};
  const int i1s[4] = {i1_0, i1_0, i1_1, i1_1};
  const float ws[4] = {(1 - w0) * (1 - w1), w0 * (1 - w1), (1 - w0) * w1, w0 * w1};

  // in-plane index products in int; the offsets of planes i0_0 and i0_1
  // differ by n1*n2
  const int P = n1 * n2;
  const std::size_t col = std::size_t(i2);
  const std::size_t base0 = std::size_t(i0_0) * P;
  const std::size_t base1 = base0 + P;
  for (int k = 0; k < 4; ++k)
  {
    c.w[k] = ws[k];
    c.valid[k] = (i0s[k] >= 0 && i0s[k] < n0 && i1s[k] >= 0 && i1s[k] < n1);
    c.off[k] = ((k == 1 || k == 3) ? base1 : base0) + std::size_t(i1s[k] * n2) + col;
  }
}

// forward step: sum of the weighted image values at the 4 corners
// (corners outside the image count as 0)
template <typename T = float>
WORKER_QUALIFIER inline T gather4(const T *img, const BilinearCorners &c)
{
  T v = T(0);
#if defined(__CUDACC__)
#pragma unroll
#elif defined(__GNUC__)
#pragma GCC unroll 4
#endif
  for (int k = 0; k < 4; ++k)
    if (c.valid[k])
      v += img[c.off[k]] * c.w[k];
  return v;
}

// back step: atomically add val, times the corner weight, to the 4 corners
template <typename T = float>
WORKER_QUALIFIER inline void scatter4(T *img, const BilinearCorners &c, T val)
{
#if defined(__CUDACC__)
#pragma unroll
#elif defined(__GNUC__)
#pragma GCC unroll 4
#endif
  for (int k = 0; k < 4; ++k)
    if (c.valid[k])
      atomic_sum(&img[c.off[k]], val * c.w[k]);
}

/**
 * Batch versions of gather4 / scatter4 for a group of T images.
 * The images are stored one after the other, nvoxels apart; the group
 * starts at image g. Images with index >= num_images are skipped, so the
 * last group may be incomplete. T is a compile-time constant, so the loops
 * over the group are unrolled and sum[] / val[] stay in registers.
 */

// forward step: add the interpolated value of image g+b to sum[b]
template <int T>
WORKER_QUALIFIER inline void gather4_batch(const float *img, std::size_t nvoxels,
                                           std::size_t g, std::size_t num_images,
                                           const BilinearCorners &c, float *sum)
{
  PP_UNROLL
  for (int b = 0; b < T; ++b)
    if (g + b < num_images)
      sum[b] += gather4(img + (g + b) * nvoxels, c);
}

// back step: atomically add val[b], times the corner weights, to image g+b
template <int T>
WORKER_QUALIFIER inline void scatter4_batch(float *img, std::size_t nvoxels,
                                            std::size_t g, std::size_t num_images,
                                            const BilinearCorners &c, const float *val)
{
  PP_UNROLL
  for (int b = 0; b < T; ++b)
    if (g + b < num_images && val[b] != 0)
      scatter4(img + (g + b) * nvoxels, c, val[b]);
}

/**
 * Bilinear interpolation (forward) and its adjoint (back) in one plane of a
 * 3D image with dims (n0,n1,n2), in a single call:
 *   - bilinear_interp_fixed{0,1,2}: interpolated image value
 *   - bilinear_interp_adj_fixed{0,1,2}: atomically add val to the 4 corners
 * Same plane convention as bilinear_corners_fixed{0,1,2}.
 */

template <typename T = float>
WORKER_QUALIFIER inline T bilinear_interp_fixed0(const T *img,
                                                 int n0, int n1, int n2,
                                                 int i0,
                                                 float i_f1,
                                                 float i_f2)
{
  BilinearCorners c;
  bilinear_corners_fixed0(n0, n1, n2, i0, i_f1, i_f2, c);
  return gather4(img, c);
}

template <typename T = float>
WORKER_QUALIFIER inline T bilinear_interp_fixed1(const T *img,
                                                 int n0, int n1, int n2,
                                                 float i_f0,
                                                 int i1,
                                                 float i_f2)
{
  BilinearCorners c;
  bilinear_corners_fixed1(n0, n1, n2, i_f0, i1, i_f2, c);
  return gather4(img, c);
}

template <typename T = float>
WORKER_QUALIFIER inline T bilinear_interp_fixed2(const T *img,
                                                 int n0, int n1, int n2,
                                                 float i_f0,
                                                 float i_f1,
                                                 int i2)
{
  BilinearCorners c;
  bilinear_corners_fixed2(n0, n1, n2, i_f0, i_f1, i2, c);
  return gather4(img, c);
}

template <typename T = float>
WORKER_QUALIFIER inline void bilinear_interp_adj_fixed0(T *img,
                                                        int n0, int n1, int n2,
                                                        int i0,
                                                        float i_f1,
                                                        float i_f2,
                                                        T val)
{
  BilinearCorners c;
  bilinear_corners_fixed0(n0, n1, n2, i0, i_f1, i_f2, c);
  scatter4(img, c, val);
}

template <typename T = float>
WORKER_QUALIFIER inline void bilinear_interp_adj_fixed1(T *img,
                                                        int n0, int n1, int n2,
                                                        float i_f0,
                                                        int i1,
                                                        float i_f2,
                                                        T val)
{
  BilinearCorners c;
  bilinear_corners_fixed1(n0, n1, n2, i_f0, i1, i_f2, c);
  scatter4(img, c, val);
}

template <typename T = float>
WORKER_QUALIFIER inline void bilinear_interp_adj_fixed2(T *img,
                                                        int n0, int n1, int n2,
                                                        float i_f0,
                                                        float i_f1,
                                                        int i2,
                                                        T val)
{
  BilinearCorners c;
  bilinear_corners_fixed2(n0, n1, n2, i_f0, i_f1, i2, c);
  scatter4(img, c, val);
}

#ifndef __CUDACC__
// OpenMP build: adjoint bilinear interpolation that computes and adds one
// corner at a time. With GCC this is faster for the back projection than
// bilinear_corners_fixed* + scatter4, because fewer values must stay live
// across the atomic updates.
template <typename T = float>
WORKER_QUALIFIER inline void bilinear_interp_adj_direct_fixed0(
    T *img,
    int n0, int n1, int n2,
    int i0,
    float i_f1,
    float i_f2,
    T val)
{
  // pointer to the i0-th plane
  T *plane = img + std::size_t(i0) * n1 * n2;

  int i1_0 = int(floorf(i_f1));
  int i2_0 = int(floorf(i_f2));
  int i1_1 = i1_0 + 1;
  int i2_1 = i2_0 + 1;

  float w1 = i_f1 - i1_0;
  float w2 = i_f2 - i2_0;
  float w00 = (1 - w1) * (1 - w2);
  float w10 = (w1) * (1 - w2);
  float w01 = (1 - w1) * (w2);
  float w11 = (w1) * (w2);

  auto inject = [&](int i1, int i2, float w)
  {
    if (i1 < 0 || i1 >= n1 || i2 < 0 || i2 >= n2)
      return;
    atomic_sum(&plane[std::size_t(i1) * n2 + i2], val * w);
  };

  inject(i1_0, i2_0, w00);
  inject(i1_1, i2_0, w10);
  inject(i1_0, i2_1, w01);
  inject(i1_1, i2_1, w11);
}

template <typename T = float>
WORKER_QUALIFIER inline void bilinear_interp_adj_direct_fixed1(
    T *img,
    int n0, int n1, int n2,
    float i_f0,
    int i1,
    float i_f2,
    T val)
{
  int i0_0 = int(floorf(i_f0));
  int i2_0 = int(floorf(i_f2));
  int i0_1 = i0_0 + 1;
  int i2_1 = i2_0 + 1;

  float w0 = i_f0 - i0_0;
  float w2 = i_f2 - i2_0;
  float w00 = (1 - w0) * (1 - w2);
  float w10 = (w0) * (1 - w2);
  float w01 = (1 - w0) * (w2);
  float w11 = (w0) * (w2);

  auto inject = [&](int i0, int i2, float w)
  {
    if (i0 < 0 || i0 >= n0 || i2 < 0 || i2 >= n2)
      return;
    std::size_t idx = std::size_t(i0) * n1 * n2 + std::size_t(i1) * n2 + i2;
    atomic_sum(&img[idx], val * w);
  };

  inject(i0_0, i2_0, w00);
  inject(i0_1, i2_0, w10);
  inject(i0_0, i2_1, w01);
  inject(i0_1, i2_1, w11);
}

template <typename T = float>
WORKER_QUALIFIER inline void bilinear_interp_adj_direct_fixed2(
    T *img,
    int n0, int n1, int n2,
    float i_f0,
    float i_f1,
    int i2,
    T val)
{
  int i0_0 = int(floorf(i_f0));
  int i1_0 = int(floorf(i_f1));
  int i0_1 = i0_0 + 1;
  int i1_1 = i1_0 + 1;

  float w0 = i_f0 - i0_0;
  float w1 = i_f1 - i1_0;
  float w00 = (1 - w0) * (1 - w1);
  float w10 = (w0) * (1 - w1);
  float w01 = (1 - w0) * (w1);
  float w11 = (w0) * (w1);

  auto inject = [&](int i0, int i1, float w)
  {
    if (i0 < 0 || i0 >= n0 || i1 < 0 || i1 >= n1)
      return;
    std::size_t idx = std::size_t(i0) * n1 * n2 + std::size_t(i1) * n2 + i2;
    atomic_sum(&img[idx], val * w);
  };

  inject(i0_0, i1_0, w00);
  inject(i0_1, i1_0, w10);
  inject(i0_0, i1_1, w01);
  inject(i0_1, i1_1, w11);
}
#endif

/**
 * @brief Computes the effective Gaussian Time-of-Flight kernel given by the integral of a normalized
 * Gaussian over dx - tbin_width/2 to dx + tbin_width/2.
 *
 * @param dx         distance
 * @param sigma_t    standard deviation of the Gaussian (same units as dx)
 * @param tbin_width width of the TOF bin (same units as dx)
 * @return           the effective kernel value
 */
WORKER_QUALIFIER inline float effective_gaussian_tof_kernel(float dx, float sigma_t, float tbin_width)
{
  const float sqrt2 = 1.414213562373095f;
  float denom = sqrt2 * sigma_t;
  float x1 = (dx + 0.5f * tbin_width) / denom;
  float x2 = (dx - 0.5f * tbin_width) / denom;
  return 0.5f * (erff(x1) - erff(x2));
}

/**
 * @brief TOF weights of the TOF bins around the TOF position of one plane.
 *
 * Computes the weights of the TOF bins it_min .. it_min+n-1 that lie within
 * max_tof_bin_diff bins of the TOF position it_f into w[0 .. n-1]. The window
 * is clamped to MAX_NUM_TOF_WEIGHTS bins centered on it_f. Same computation as
 * in the TOF sinogram workers.
 *
 * @param it_f             TOF position of the plane (in TOF bin units)
 * @param max_tof_bin_diff half width of the TOF window (in TOF bin units)
 * @param tof_bin_width    width of a TOF bin
 * @param tof_sigma        standard deviation of the TOF kernel
 * @param num_tof_bins     number of TOF bins
 * @param w                output buffer of size MAX_NUM_TOF_WEIGHTS
 * @param it_min           output: TOF bin of w[0]
 * @param k_start          output: first k with it_min + k inside [0, num_tof_bins)
 * @param k_end            output: one past the last such k
 * @return sum of all n weights (including bins outside [0, num_tof_bins))
 */
WORKER_QUALIFIER inline float tof_weight_window(float it_f,
                                                float max_tof_bin_diff,
                                                float tof_bin_width,
                                                float tof_sigma,
                                                short num_tof_bins,
                                                float *w,
                                                int &it_min,
                                                int &k_start,
                                                int &k_end)
{
  it_min = static_cast<int>(floorf(it_f - max_tof_bin_diff));
  int it_max = static_cast<int>(ceilf(it_f + max_tof_bin_diff));
  int n_tof_weights = it_max + 1 - it_min;

  if (n_tof_weights > MAX_NUM_TOF_WEIGHTS)
  {
    it_min = static_cast<int>(lroundf(it_f)) - MAX_NUM_TOF_WEIGHTS / 2;
    n_tof_weights = MAX_NUM_TOF_WEIGHTS;
  }

  float sum_weights = 0.0f;
  for (int k = 0; k < n_tof_weights; ++k)
  {
    float dist = fabsf(it_f - it_min - k) * tof_bin_width;
    w[k] = effective_gaussian_tof_kernel(dist, tof_sigma, tof_bin_width);
    sum_weights += w[k];
  }

  k_start = (it_min < 0) ? -it_min : 0;
  k_end = ((it_min + n_tof_weights) > num_tof_bins) ? (num_tof_bins - it_min) : n_tof_weights;
  return sum_weights;
}
