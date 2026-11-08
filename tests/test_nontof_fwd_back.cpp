#include "parallelproj.h"
#include "file_utils.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <numeric>
#include <stdexcept>

// In a CUDA build, joseph3d_fwd/back launch CUDA kernels, so this test needs
// a physical GPU even though it is a plain .cpp file. PARALLELPROJ_CUDA is a
// PUBLIC compile definition of the parallelproj library target.
#if defined(PARALLELPROJ_CUDA) && (PARALLELPROJ_CUDA == 1)
#include "cuda_test_utils.h"
#endif

int test_box_projection_cpp()
{
  bool all_passed = true;
  // Generate an image box full of ones with side length 100mm but non-uniform voxel size
  std::vector<float> voxel_size = {2.0f, 1.0f, 4.0f};
  std::vector<int> img_dim = {50, 100, 25};
  std::vector<float> img_origin = {-50.0f + 0.5f * voxel_size[0],
                                   -50.0f + 0.5f * voxel_size[1],
                                   -50.0f + 0.5f * voxel_size[2]};

  // Allocate memory for image
  std::size_t nvox = img_dim[0] * img_dim[1] * img_dim[2];
  std::vector<float> img(nvox, 1.0f);

  // xstart and xend arrays
  std::vector<std::vector<float>> xstart = {
      {100, 0, 0}, {50, 0, 0}, {0, 50, 0}, {0, 0, 50}, {40, 0, 0}, {0, 40, 0}, {0, 0, 40}, {50, 5, 0}, {0, 50, 5}, {5, 0, 50}, {50, 5, -2}, {-2, 50, 5}, {5, -2, 50}};
  std::vector<std::vector<float>> xend = {
      {-100, 0, 0}, {-50, 0, 0}, {0, -50, 0}, {0, 0, -50}, {-40, 0, 0}, {0, -40, 0}, {0, 0, -40}, {-50, -4, 0}, {0, -50, -4}, {-4, 0, -50}, {-50, -4, 3}, {3, -50, -4}, {-4, 3, -50}};

  std::vector<float> exp_vals = {
      100.0f, 100.0f, 100.0f, 100.0f,
      80.0f, 80.0f, 80.0f,
      static_cast<float>(std::sqrt(100 * 100 + 9 * 9)),
      static_cast<float>(std::sqrt(100 * 100 + 9 * 9)),
      static_cast<float>(std::sqrt(100 * 100 + 9 * 9)),
      static_cast<float>(std::sqrt(100 * 100 + 9 * 9 + 5 * 5)),
      static_cast<float>(std::sqrt(100 * 100 + 9 * 9 + 5 * 5)),
      static_cast<float>(std::sqrt(100 * 100 + 9 * 9 + 5 * 5))};

  std::size_t n_lors = xstart.size();
  std::vector<float> xstart_flat, xend_flat;
  for (std::size_t i = 0; i < n_lors; ++i)
  {
    xstart_flat.insert(xstart_flat.end(), xstart[i].begin(), xstart[i].end());
    xend_flat.insert(xend_flat.end(), xend[i].begin(), xend[i].end());
  }

  std::vector<float> img_fwd(n_lors, 0.0f);

  // Call the C++ joseph3d_fwd function
  joseph3d_fwd(
      xstart_flat.data(),
      xend_flat.data(),
      img.data(),
      img_origin.data(),
      voxel_size.data(),
      img_fwd.data(),
      n_lors,
      img_dim.data(),
      0, // device_id
      64 // threadsperblock
  );

  // Check results
  float eps = 1e-5f;
  for (std::size_t i = 0; i < n_lors; ++i)
  {
    if (std::abs(img_fwd[i] - exp_vals[i]) >= eps)
    {
      std::cerr << "Forward box projection test failed at i=" << i
                << ": got " << img_fwd[i] << ", expected " << exp_vals[i] << std::endl;
      all_passed = false;
    }
  }

  return all_passed ? 0 : 1;
}

int test_forward_and_back_projection_cpp()
{
  ////////////////////////////////////////////////////////
  // host array test cases
  ////////////////////////////////////////////////////////

  bool all_tests_passed = true;

  std::cout << "Host array test\n";

  std::vector<int> img_dim = {2, 3, 4};
  std::vector<float> voxsize = {4.0f, 3.0f, 2.0f};

  std::vector<float> img_origin(3);
  for (int i = 0; i < 3; ++i)
  {
    img_origin[i] = (-(float)img_dim[i] / 2 + 0.5f) * voxsize[i];
  }

  // Read the image from file
  std::vector<float> img = readArrayFromFile<float>("img.txt");

  // Read the ray start coordinates from file
  std::vector<float> vstart = readArrayFromFile<float>("vstart.txt");

  // Read the ray end coordinates from file
  std::vector<float> vend = readArrayFromFile<float>("vend.txt");

  std::size_t nlors = vstart.size() / 3;

  // Calculate the start and end coordinates in world coordinates
  std::vector<float> xstart(3 * nlors);
  std::vector<float> xend(3 * nlors);

  for (int ir = 0; ir < nlors; ir++)
  {
    xstart[ir * 3 + 0] = img_origin[0] + vstart[ir * 3 + 0] * voxsize[0];
    xstart[ir * 3 + 1] = img_origin[1] + vstart[ir * 3 + 1] * voxsize[1];
    xstart[ir * 3 + 2] = img_origin[2] + vstart[ir * 3 + 2] * voxsize[2];

    xend[ir * 3 + 0] = img_origin[0] + vend[ir * 3 + 0] * voxsize[0];
    xend[ir * 3 + 1] = img_origin[1] + vend[ir * 3 + 1] * voxsize[1];
    xend[ir * 3 + 2] = img_origin[2] + vend[ir * 3 + 2] * voxsize[2];
  }

  // Allocate memory for forward projection results
  std::vector<float> img_fwd(nlors);

  std::size_t nvoxels = img_dim[0] * img_dim[1] * img_dim[2];

  // Perform forward projection
  joseph3d_fwd(
      xstart.data(), xend.data(), img.data(),
      img_origin.data(), voxsize.data(), img_fwd.data(),
      nlors, img_dim.data(), 0, 64);

  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////

  // Read the expected forward values from file
  std::vector<float> expected_fwd_vals = readArrayFromFile<float>("expected_fwd_vals.txt");

  // Check if we got the expected results
  float fwd_diff = 0;
  float eps = 1e-7;

  printf("\nforward projection test\n");
  for (int ir = 0; ir < nlors; ir++)
  {
    printf("test ray %d: fwd projected: %.7e expected: %.7e\n", ir, img_fwd[ir], expected_fwd_vals[ir]);

    fwd_diff = std::abs(img_fwd[ir] - expected_fwd_vals[ir]);
    if (fwd_diff > eps)
    {
      std::cerr << "Forward projection test failed.\n";
      std::cerr << "Difference: " << fwd_diff << "\n";
      std::cerr << "Expected: " << expected_fwd_vals[ir] << "\n";
      std::cerr << "Actual: " << img_fwd[ir] << "\n";
      std::cerr << "Tolerance: " << eps << "\n";
      std::cerr << "Ray index: " << ir << "\n";

      all_tests_passed = false;
    }
  }

  ////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////

  // Test the back projection
  std::vector<float> bimg(img_dim[0] * img_dim[1] * img_dim[2], 0.0f);
  std::vector<float> ones(nlors, 1.0f);

  joseph3d_back(
      xstart.data(), xend.data(), bimg.data(),
      img_origin.data(), voxsize.data(), ones.data(),
      nlors, img_dim.data());

  printf("\nback projection of ones along all rays:\n");
  for (std::size_t i0 = 0; i0 < img_dim[0]; i0++)
  {
    for (std::size_t i1 = 0; i1 < img_dim[1]; i1++)
    {
      for (std::size_t i2 = 0; i2 < img_dim[2]; i2++)
      {
        printf("%.1f ", bimg[img_dim[1] * img_dim[2] * i0 + img_dim[2] * i1 + i2]);
      }
      printf("\n");
    }
    printf("\n");
  }

  // To test whether the back projection is correct, we test if the back projector is the adjoint
  // of the forward projector. This is more practical than checking a lot of single voxels in the
  // back projected image.

  float inner_product1 = std::inner_product(img.begin(), img.end(), bimg.begin(), 0.0f);
  float inner_product2 = std::inner_product(img_fwd.begin(), img_fwd.end(), ones.begin(), 0.0f);

  float ip_diff = fabs(inner_product1 - inner_product2);

  if (ip_diff > eps)
  {
    std::cerr << "Back projection test failed.\n";
    std::cerr << "Inner product 1: " << inner_product1 << "\n";
    std::cerr << "Inner product 2: " << inner_product2 << "\n";
    std::cerr << "Difference: " << ip_diff << "\n";
    std::cerr << "Tolerance: " << eps << "\n";

    all_tests_passed = false;
  }

  return all_tests_passed ? 0 : 1;
}

// joseph3d_fwd/back must reject images where one plane (n1*n2 voxels) does
// not fit in an int. The check runs before any image access, so no image
// memory is needed.
int test_image_plane_size_check_cpp()
{
  int img_dim[3] = {1, 50000, 50000};
  float img_origin[3] = {0.0f, 0.0f, 0.0f};
  float voxel_size[3] = {1.0f, 1.0f, 1.0f};
  float dummy = 0.0f;

  bool fwd_threw = false;
  try
  {
    joseph3d_fwd(&dummy, &dummy, &dummy, img_origin, voxel_size, &dummy, 0, img_dim);
  }
  catch (const std::invalid_argument &)
  {
    fwd_threw = true;
  }

  bool back_threw = false;
  try
  {
    joseph3d_back(&dummy, &dummy, &dummy, img_origin, voxel_size, &dummy, 0, img_dim);
  }
  catch (const std::invalid_argument &)
  {
    back_threw = true;
  }

  if (!fwd_threw || !back_threw)
  {
    std::cerr << "image plane size check failed: fwd threw " << fwd_threw
              << ", back threw " << back_threw << std::endl;
    return 1;
  }
  return 0;
}

int main()
{
#if defined(PARALLELPROJ_CUDA) && (PARALLELPROJ_CUDA == 1)
  if (!cuda_device_available())
    return CUDA_TEST_SKIP_RETURN_CODE;
#endif

  int result = 0;
  result |= test_box_projection_cpp();
  result |= test_forward_and_back_projection_cpp();
  result |= test_image_plane_size_check_cpp();
  return result;
}
