#include "parallelproj.h"
#include "file_utils.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>
#include "cuda_test_utils.h"

int test_cuda_managed_arrays(int device_id, int threadsperblock)
{
  cudaSetDevice(device_id);
  bool all_passed = true;

  // CUDA-managed array test
  int *cm_img_dim;
  cudaMallocManaged(&cm_img_dim, 3 * sizeof(int));
  cm_img_dim[0] = 2;
  cm_img_dim[1] = 3;
  cm_img_dim[2] = 4;

  float *cm_voxsize;
  cudaMallocManaged(&cm_voxsize, 3 * sizeof(float));
  cm_voxsize[0] = 4.0f;
  cm_voxsize[1] = 3.0f;
  cm_voxsize[2] = 2.0f;

  float *cm_img_origin;
  cudaMallocManaged(&cm_img_origin, 3 * sizeof(float));
  for (int i = 0; i < 3; ++i)
  {
    cm_img_origin[i] = (-(float)cm_img_dim[i] / 2 + 0.5f) * cm_voxsize[i];
  }

  std::vector<float> h_img = readArrayFromFile<float>("img.txt");
  float *cm_img;
  cudaMallocManaged(&cm_img, h_img.size() * sizeof(float));
  std::copy(h_img.begin(), h_img.end(), cm_img);

  std::vector<float> h_vstart = readArrayFromFile<float>("vstart.txt");
  float *cm_vstart;
  cudaMallocManaged(&cm_vstart, h_vstart.size() * sizeof(float));
  std::copy(h_vstart.begin(), h_vstart.end(), cm_vstart);

  std::vector<float> h_vend = readArrayFromFile<float>("vend.txt");
  float *cm_vend;
  cudaMallocManaged(&cm_vend, h_vend.size() * sizeof(float));
  std::copy(h_vend.begin(), h_vend.end(), cm_vend);

  std::size_t nlors = h_vstart.size() / 3;

  float *cm_xstart, *cm_xend;
  cudaMallocManaged(&cm_xstart, 3 * nlors * sizeof(float));
  cudaMallocManaged(&cm_xend, 3 * nlors * sizeof(float));

  for (int ir = 0; ir < nlors; ir++)
  {
    for (int j = 0; j < 3; j++)
    {
      cm_xstart[ir * 3 + j] = cm_img_origin[j] + cm_vstart[ir * 3 + j] * cm_voxsize[j];
      cm_xend[ir * 3 + j] = cm_img_origin[j] + cm_vend[ir * 3 + j] * cm_voxsize[j];
    }
  }

  float *cm_img_fwd;
  cudaMallocManaged(&cm_img_fwd, nlors * sizeof(float));
  joseph3d_fwd(cm_xstart, cm_xend, cm_img, cm_img_origin, cm_voxsize, cm_img_fwd, nlors, cm_img_dim, device_id, threadsperblock);

  std::vector<float> h_expected_fwd_vals = readArrayFromFile<float>("expected_fwd_vals.txt");
  float fwd_diff = 0;
  float eps = 1e-7;

  for (int ir = 0; ir < nlors; ir++)
  {
    fwd_diff = std::abs(cm_img_fwd[ir] - h_expected_fwd_vals[ir]);
    if (fwd_diff > eps)
    {
      std::cerr << "CUDA-managed array test failed for ray " << ir << "\n";
      all_passed = false;
    }
  }

  // Test the back projection
  float *cm_bimg;
  cudaMallocManaged(&cm_bimg, cm_img_dim[0] * cm_img_dim[1] * cm_img_dim[2] * sizeof(float));
  std::fill(cm_bimg, cm_bimg + (cm_img_dim[0] * cm_img_dim[1] * cm_img_dim[2]), 0.0f);

  float *cm_ones;
  cudaMallocManaged(&cm_ones, nlors * sizeof(float));
  std::fill(cm_ones, cm_ones + nlors, 1.0f);

  joseph3d_back(cm_xstart, cm_xend, cm_bimg, cm_img_origin, cm_voxsize, cm_ones, nlors, cm_img_dim, device_id, threadsperblock);

  // Validate the back projection using adjointness
  float inner_product1 = 0.0f;
  float inner_product2 = 0.0f;

  for (std::size_t i = 0; i < h_img.size(); i++)
  {
    inner_product1 += cm_img[i] * cm_bimg[i];
  }

  for (std::size_t ir = 0; ir < nlors; ir++)
  {
    inner_product2 += cm_img_fwd[ir] * cm_ones[ir];
  }

  float ip_diff = fabs(inner_product1 - inner_product2);

  if (ip_diff > eps)
  {
    std::cerr << "CUDA-managed array back projection test failed: adjointness property violated.\n";
    all_passed = false;
  }

  cudaFree(cm_img_dim);
  cudaFree(cm_voxsize);
  cudaFree(cm_img_origin);
  cudaFree(cm_img);
  cudaFree(cm_vstart);
  cudaFree(cm_vend);
  cudaFree(cm_xstart);
  cudaFree(cm_xend);
  cudaFree(cm_img_fwd);
  cudaFree(cm_bimg);
  cudaFree(cm_ones);

  return all_passed ? 0 : 1;
}

int test_cuda_device_arrays(int device_id, int threadsperblock)
{
  cudaSetDevice(device_id);
  bool all_passed = true;

  // CUDA device array test
  int *d_img_dim;
  cudaMalloc(&d_img_dim, 3 * sizeof(int));
  int h_img_dim[3] = {2, 3, 4};
  cudaMemcpy(d_img_dim, h_img_dim, 3 * sizeof(int), cudaMemcpyHostToDevice);

  float *d_voxsize;
  cudaMalloc(&d_voxsize, 3 * sizeof(float));
  float h_voxsize[3] = {4.0f, 3.0f, 2.0f};
  cudaMemcpy(d_voxsize, h_voxsize, 3 * sizeof(float), cudaMemcpyHostToDevice);

  float h_img_origin[3];
  for (int i = 0; i < 3; ++i)
  {
    h_img_origin[i] = (-(float)h_img_dim[i] / 2 + 0.5f) * h_voxsize[i];
  }
  float *d_img_origin;
  cudaMalloc(&d_img_origin, 3 * sizeof(float));
  cudaMemcpy(d_img_origin, h_img_origin, 3 * sizeof(float), cudaMemcpyHostToDevice);

  std::vector<float> h_img = readArrayFromFile<float>("img.txt");
  float *d_img;
  cudaMalloc(&d_img, h_img.size() * sizeof(float));
  cudaMemcpy(d_img, h_img.data(), h_img.size() * sizeof(float), cudaMemcpyHostToDevice);

  std::vector<float> h_vstart = readArrayFromFile<float>("vstart.txt");
  float *d_vstart;
  cudaMalloc(&d_vstart, h_vstart.size() * sizeof(float));
  cudaMemcpy(d_vstart, h_vstart.data(), h_vstart.size() * sizeof(float), cudaMemcpyHostToDevice);

  std::vector<float> h_vend = readArrayFromFile<float>("vend.txt");
  float *d_vend;
  cudaMalloc(&d_vend, h_vend.size() * sizeof(float));
  cudaMemcpy(d_vend, h_vend.data(), h_vend.size() * sizeof(float), cudaMemcpyHostToDevice);

  std::size_t nlors = h_vstart.size() / 3;

  float *d_xstart, *d_xend;
  cudaMalloc(&d_xstart, 3 * nlors * sizeof(float));
  cudaMalloc(&d_xend, 3 * nlors * sizeof(float));

  for (int ir = 0; ir < nlors; ir++)
  {
    for (int j = 0; j < 3; j++)
    {
      float xstart_val = h_img_origin[j] + h_vstart[ir * 3 + j] * h_voxsize[j];
      float xend_val = h_img_origin[j] + h_vend[ir * 3 + j] * h_voxsize[j];
      cudaMemcpy(&d_xstart[ir * 3 + j], &xstart_val, sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(&d_xend[ir * 3 + j], &xend_val, sizeof(float), cudaMemcpyHostToDevice);
    }
  }
  float *d_img_fwd;
  cudaMalloc(&d_img_fwd, nlors * sizeof(float));
  joseph3d_fwd(d_xstart, d_xend, d_img, d_img_origin, d_voxsize, d_img_fwd, nlors, d_img_dim, device_id, threadsperblock);

  std::vector<float> h_img_fwd(nlors);
  cudaMemcpy(h_img_fwd.data(), d_img_fwd, nlors * sizeof(float), cudaMemcpyDeviceToHost);

  std::vector<float> h_expected_fwd_vals = readArrayFromFile<float>("expected_fwd_vals.txt");
  float fwd_diff = 0;
  float eps = 1e-7;

  for (int ir = 0; ir < nlors; ir++)
  {
    fwd_diff = std::abs(h_img_fwd[ir] - h_expected_fwd_vals[ir]);
    if (fwd_diff > eps)
    {
      std::cerr << "CUDA device array test failed for ray " << ir << "\n";
      all_passed = false;
    }
  }

  // Test the back projection
  float *d_bimg;
  cudaMalloc(&d_bimg, h_img_dim[0] * h_img_dim[1] * h_img_dim[2] * sizeof(float));
  cudaMemset(d_bimg, 0, h_img_dim[0] * h_img_dim[1] * h_img_dim[2] * sizeof(float));

  float *d_ones;
  cudaMalloc(&d_ones, nlors * sizeof(float));
  std::vector<float> h_ones(nlors, 1.0f);
  cudaMemcpy(d_ones, h_ones.data(), nlors * sizeof(float), cudaMemcpyHostToDevice);

  joseph3d_back(d_xstart, d_xend, d_bimg, d_img_origin, d_voxsize, d_ones, nlors, d_img_dim, device_id, threadsperblock);

  std::vector<float> h_bimg(h_img_dim[0] * h_img_dim[1] * h_img_dim[2]);
  cudaMemcpy(h_bimg.data(), d_bimg, h_bimg.size() * sizeof(float), cudaMemcpyDeviceToHost);

  // Validate the back projection using adjointness
  float inner_product1 = 0.0f;
  float inner_product2 = 0.0f;

  for (std::size_t i = 0; i < h_img.size(); i++)
  {
    inner_product1 += h_img[i] * h_bimg[i];
  }

  for (std::size_t ir = 0; ir < nlors; ir++)
  {
    inner_product2 += h_img_fwd[ir] * h_ones[ir];
  }

  float ip_diff = fabs(inner_product1 - inner_product2);

  if (ip_diff > eps)
  {
    std::cerr << "CUDA device array back projection test failed: adjointness property violated.\n";
    all_passed = false;
  }

  cudaFree(d_img_dim);
  cudaFree(d_voxsize);
  cudaFree(d_img);
  cudaFree(d_vstart);
  cudaFree(d_vend);
  cudaFree(d_xstart);
  cudaFree(d_xend);
  cudaFree(d_img_fwd);
  cudaFree(d_bimg);
  cudaFree(d_ones);

  return all_passed ? 0 : 1;
}

int test_box_projection_cuda_managed_arrays(int device_id, int threadsperblock)
{
  cudaSetDevice(device_id);
  bool all_passed = true;

  // Test parameters
  std::vector<float> voxel_size = {2.0f, 1.0f, 4.0f};
  std::vector<int> img_dim = {50, 100, 25};
  std::vector<float> img_origin = {-50.0f + 0.5f * voxel_size[0],
                                   -50.0f + 0.5f * voxel_size[1],
                                   -50.0f + 0.5f * voxel_size[2]};
  std::size_t nvox = img_dim[0] * img_dim[1] * img_dim[2];

  std::vector<std::vector<float>> xstart = {
      {100, 0, 0}, {50, 0, 0}, {0, 50, 0}, {0, 0, 50}, {40, 0, 0}, {0, 40, 0}, {0, 0, 40}, {50, 5, 0}, {0, 50, 5}, {5, 0, 50}, {50, 5, -2}, {-2, 50, 5}, {5, -2, 50}};
  std::vector<std::vector<float>> xend = {
      {-100, 0, 0}, {-50, 0, 0}, {0, -50, 0}, {0, 0, -50}, {-40, 0, 0}, {0, -40, 0}, {0, 0, -40}, {-50, -4, 0}, {0, -50, -4}, {-4, 0, -50}, {-50, -4, 3}, {3, -50, -4}, {-4, 3, -50}};

  float sqrt_100_9 = static_cast<float>(std::sqrt(100 * 100 + 9 * 9));
  float sqrt_100_9_5 = static_cast<float>(std::sqrt(100 * 100 + 9 * 9 + 5 * 5));

  std::vector<float> exp_vals = {
      100.0f, 100.0f, 100.0f, 100.0f,
      80.0f, 80.0f, 80.0f,
      sqrt_100_9, sqrt_100_9, sqrt_100_9,
      sqrt_100_9_5, sqrt_100_9_5, sqrt_100_9_5};

  std::size_t n_lors = xstart.size();
  std::vector<float> xstart_flat, xend_flat;
  for (std::size_t i = 0; i < n_lors; ++i)
  {
    xstart_flat.insert(xstart_flat.end(), xstart[i].begin(), xstart[i].end());
    xend_flat.insert(xend_flat.end(), xend[i].begin(), xend[i].end());
  }

  // Managed allocations
  float *d_xstart, *d_xend, *d_img, *d_img_origin, *d_voxel_size, *d_img_fwd;
  int *d_img_dim;
  cudaMallocManaged(&d_xstart, xstart_flat.size() * sizeof(float));
  cudaMallocManaged(&d_xend, xend_flat.size() * sizeof(float));
  cudaMallocManaged(&d_img, nvox * sizeof(float));
  cudaMallocManaged(&d_img_origin, 3 * sizeof(float));
  cudaMallocManaged(&d_voxel_size, 3 * sizeof(float));
  cudaMallocManaged(&d_img_fwd, n_lors * sizeof(float));
  cudaMallocManaged(&d_img_dim, 3 * sizeof(int));

  std::copy(xstart_flat.begin(), xstart_flat.end(), d_xstart);
  std::copy(xend_flat.begin(), xend_flat.end(), d_xend);
  std::fill_n(d_img, nvox, 1.0f);
  std::copy(img_origin.begin(), img_origin.end(), d_img_origin);
  std::copy(voxel_size.begin(), voxel_size.end(), d_voxel_size);
  std::copy(img_dim.begin(), img_dim.end(), d_img_dim);

  joseph3d_fwd(
      d_xstart, d_xend, d_img, d_img_origin, d_voxel_size, d_img_fwd,
      n_lors, d_img_dim, device_id, threadsperblock);

  cudaDeviceSynchronize();

  float eps = 1e-4f;
  for (std::size_t i = 0; i < n_lors; ++i)
  {
    if (std::abs(d_img_fwd[i] - exp_vals[i]) >= eps)
    {
      std::cerr << "[Managed] Forward box projection test failed at i=" << i
                << ": got " << d_img_fwd[i] << ", expected " << exp_vals[i] << std::endl;
      all_passed = false;
    }
  }

  cudaFree(d_xstart);
  cudaFree(d_xend);
  cudaFree(d_img);
  cudaFree(d_img_origin);
  cudaFree(d_voxel_size);
  cudaFree(d_img_fwd);
  cudaFree(d_img_dim);

  return all_passed ? 0 : 1;
}

int test_box_projection_cuda_device_arrays(int device_id, int threadsperblock)
{
  cudaSetDevice(device_id);
  bool all_passed = true;

  // Test parameters
  std::vector<float> voxel_size = {2.0f, 1.0f, 4.0f};
  std::vector<int> img_dim = {50, 100, 25};
  std::vector<float> img_origin = {-50.0f + 0.5f * voxel_size[0],
                                   -50.0f + 0.5f * voxel_size[1],
                                   -50.0f + 0.5f * voxel_size[2]};
  std::size_t nvox = img_dim[0] * img_dim[1] * img_dim[2];

  std::vector<std::vector<float>> xstart = {
      {100, 0, 0}, {50, 0, 0}, {0, 50, 0}, {0, 0, 50}, {40, 0, 0}, {0, 40, 0}, {0, 0, 40}, {50, 5, 0}, {0, 50, 5}, {5, 0, 50}, {50, 5, -2}, {-2, 50, 5}, {5, -2, 50}};
  std::vector<std::vector<float>> xend = {
      {-100, 0, 0}, {-50, 0, 0}, {0, -50, 0}, {0, 0, -50}, {-40, 0, 0}, {0, -40, 0}, {0, 0, -40}, {-50, -4, 0}, {0, -50, -4}, {-4, 0, -50}, {-50, -4, 3}, {3, -50, -4}, {-4, 3, -50}};

  float sqrt_100_9 = static_cast<float>(std::sqrt(100 * 100 + 9 * 9));
  float sqrt_100_9_5 = static_cast<float>(std::sqrt(100 * 100 + 9 * 9 + 5 * 5));

  std::vector<float> exp_vals = {
      100.0f, 100.0f, 100.0f, 100.0f,
      80.0f, 80.0f, 80.0f,
      sqrt_100_9, sqrt_100_9, sqrt_100_9,
      sqrt_100_9_5, sqrt_100_9_5, sqrt_100_9_5};

  std::size_t n_lors = xstart.size();
  std::vector<float> xstart_flat, xend_flat;
  for (std::size_t i = 0; i < n_lors; ++i)
  {
    xstart_flat.insert(xstart_flat.end(), xstart[i].begin(), xstart[i].end());
    xend_flat.insert(xend_flat.end(), xend[i].begin(), xend[i].end());
  }

  // Device allocations
  float *dxstart, *dxend, *dimg, *dimg_origin, *dvoxel_size, *dimg_fwd;
  int *dimg_dim;
  cudaMalloc(&dxstart, xstart_flat.size() * sizeof(float));
  cudaMalloc(&dxend, xend_flat.size() * sizeof(float));
  cudaMalloc(&dimg, nvox * sizeof(float));
  cudaMalloc(&dimg_origin, 3 * sizeof(float));
  cudaMalloc(&dvoxel_size, 3 * sizeof(float));
  cudaMalloc(&dimg_fwd, n_lors * sizeof(float));
  cudaMalloc(&dimg_dim, 3 * sizeof(int));

  cudaMemcpy(dxstart, xstart_flat.data(), xstart_flat.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dxend, xend_flat.data(), xend_flat.size() * sizeof(float), cudaMemcpyHostToDevice);
  std::vector<float> ones_img(nvox, 1.0f);
  cudaMemcpy(dimg, ones_img.data(), nvox * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dimg_origin, img_origin.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dvoxel_size, voxel_size.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dimg_dim, img_dim.data(), 3 * sizeof(int), cudaMemcpyHostToDevice);

  joseph3d_fwd(
      dxstart, dxend, dimg, dimg_origin, dvoxel_size, dimg_fwd,
      n_lors, dimg_dim, device_id, threadsperblock);

  std::vector<float> img_fwd2(n_lors, 0.0f);
  cudaMemcpy(img_fwd2.data(), dimg_fwd, n_lors * sizeof(float), cudaMemcpyDeviceToHost);

  float eps = 1e-4f;
  for (std::size_t i = 0; i < n_lors; ++i)
  {
    if (std::abs(img_fwd2[i] - exp_vals[i]) >= eps)
    {
      std::cerr << "[Device] Forward box projection test failed at i=" << i
                << ": got " << img_fwd2[i] << ", expected " << exp_vals[i] << std::endl;
      all_passed = false;
    }
  }

  cudaFree(dxstart);
  cudaFree(dxend);
  cudaFree(dimg);
  cudaFree(dimg_origin);
  cudaFree(dvoxel_size);
  cudaFree(dimg_fwd);
  cudaFree(dimg_dim);

  return all_passed ? 0 : 1;
}

// Batch projectors with CUDA managed arrays: the forward and back projection
// of each image in the batch must equal the single-image projectors.
int test_batch_cuda_managed_arrays(int device_id, int threadsperblock)
{
  cudaSetDevice(device_id);
  bool all_passed = true;

  const int num_images = 3;
  const std::size_t num_lors = 4;

  int *img_dim;
  float *voxsize, *img_origin, *xstart, *xend, *imgs, *proj, *proj1, *bimgs, *bimg1, *ones;
  cudaMallocManaged(&img_dim, 3 * sizeof(int));
  cudaMallocManaged(&voxsize, 3 * sizeof(float));
  cudaMallocManaged(&img_origin, 3 * sizeof(float));
  cudaMallocManaged(&xstart, 3 * num_lors * sizeof(float));
  cudaMallocManaged(&xend, 3 * num_lors * sizeof(float));

  img_dim[0] = 4;
  img_dim[1] = 5;
  img_dim[2] = 6;
  std::size_t nvox = std::size_t(img_dim[0]) * img_dim[1] * img_dim[2];
  for (int k = 0; k < 3; ++k)
  {
    voxsize[k] = 1.0f + 0.5f * k;
    img_origin[k] = (-0.5f * img_dim[k] + 0.5f) * voxsize[k];
  }

  // LORs along each axis and one oblique LOR
  const float s[4][3] = {{-20, 0.3f, 0.2f}, {0.1f, -20, 0.4f}, {0.2f, 0.3f, -20}, {-20, -15, -10}};
  const float e[4][3] = {{20, 0.3f, 0.2f}, {0.1f, 20, 0.4f}, {0.2f, 0.3f, 20}, {20, 15, 10}};
  for (std::size_t n = 0; n < num_lors; ++n)
    for (int k = 0; k < 3; ++k)
    {
      xstart[3 * n + k] = s[n][k];
      xend[3 * n + k] = e[n][k];
    }

  cudaMallocManaged(&imgs, num_images * nvox * sizeof(float));
  cudaMallocManaged(&proj, num_images * num_lors * sizeof(float));
  cudaMallocManaged(&proj1, num_lors * sizeof(float));
  cudaMallocManaged(&bimgs, num_images * nvox * sizeof(float));
  cudaMallocManaged(&bimg1, nvox * sizeof(float));
  cudaMallocManaged(&ones, num_images * num_lors * sizeof(float));

  for (std::size_t j = 0; j < num_images * nvox; ++j)
  {
    imgs[j] = float((j * 37) % 11) + 1.0f;
    bimgs[j] = 0.0f;
  }
  for (std::size_t j = 0; j < num_images * num_lors; ++j)
    ones[j] = 1.0f + float(j);

  joseph3d_fwd_batch(xstart, xend, imgs, img_origin, voxsize, proj, num_lors, img_dim, num_images, device_id, threadsperblock);
  joseph3d_back_batch(xstart, xend, bimgs, img_origin, voxsize, ones, num_lors, img_dim, num_images, device_id, threadsperblock);

  for (int b = 0; b < num_images; ++b)
  {
    joseph3d_fwd(xstart, xend, imgs + b * nvox, img_origin, voxsize, proj1, num_lors, img_dim, device_id, threadsperblock);
    for (std::size_t n = 0; n < num_lors; ++n)
    {
      if (std::abs(proj[b * num_lors + n] - proj1[n]) > 1e-6f * std::abs(proj1[n]) || proj1[n] == 0.0f)
      {
        std::cerr << "batch fwd mismatch: image " << b << " LOR " << n << ": " << proj[b * num_lors + n] << " vs " << proj1[n] << "\n";
        all_passed = false;
      }
    }

    for (std::size_t j = 0; j < nvox; ++j)
      bimg1[j] = 0.0f;
    joseph3d_back(xstart, xend, bimg1, img_origin, voxsize, ones + b * num_lors, num_lors, img_dim, device_id, threadsperblock);
    for (std::size_t j = 0; j < nvox; ++j)
    {
      if (std::abs(bimgs[b * nvox + j] - bimg1[j]) > 1e-5f * (1.0f + std::abs(bimg1[j])))
      {
        std::cerr << "batch back mismatch: image " << b << " voxel " << j << "\n";
        all_passed = false;
        break;
      }
    }
  }

  cudaFree(img_dim);
  cudaFree(voxsize);
  cudaFree(img_origin);
  cudaFree(xstart);
  cudaFree(xend);
  cudaFree(imgs);
  cudaFree(proj);
  cudaFree(proj1);
  cudaFree(bimgs);
  cudaFree(bimg1);
  cudaFree(ones);

  std::cout << (all_passed ? "batch managed arrays: passed" : "batch managed arrays: FAILED") << "\n";
  return all_passed ? 0 : 1;
}

int main()
{
  if (!cuda_device_available())
    return CUDA_TEST_SKIP_RETURN_CODE;

  bool all_tests_passed = true;

  int device_count;
  cudaGetDeviceCount(&device_count);
  int i = 0; // test on the first device

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, i);
  std::cout << "Device " << i << ": " << prop.name << "\n";

  std::cout << "\n--- Testing with CUDA Managed Arrays ---\n";
  if (test_cuda_managed_arrays(i, 64) != 0)
    all_tests_passed = false;

  std::cout << "\n--- Testing with CUDA Device Arrays ---\n";
  if (test_cuda_device_arrays(i, 64) != 0)
    all_tests_passed = false;

  std::cout << "\n--- Testing Box Projection with CUDA Managed Arrays ---\n";
  if (test_box_projection_cuda_managed_arrays(i, 64) != 0)
    all_tests_passed = false;

  std::cout << "\n--- Testing Box Projection with CUDA Device Arrays ---\n";
  if (test_box_projection_cuda_device_arrays(i, 64) != 0)
    all_tests_passed = false;

  std::cout << "\n--- Testing Batch Projectors with CUDA Managed Arrays ---\n";
  if (test_batch_cuda_managed_arrays(i, 64) != 0)
    all_tests_passed = false;

  return all_tests_passed ? 0 : 1;
}
