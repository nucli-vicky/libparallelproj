#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "parallelproj.h"

namespace nb = nanobind;
using namespace nb::literals;

// ND c-contiguous float arrays
using ConstFloatNDArray = nb::ndarray<const float, nb::c_contig>;
using ConstShortNDArray = nb::ndarray<const short, nb::c_contig>;
using FloatNDArray = nb::ndarray<float, nb::c_contig>;
// 3D c-contiguous float arrays
using ConstFloat3DArray = nb::ndarray<const float, nb::c_contig, nb::ndim<3>>;
using Float3DArray = nb::ndarray<float, nb::c_contig, nb::ndim<3>>;
// 4D c-contiguous float arrays (batch of 3D images)
using ConstFloat4DArray = nb::ndarray<const float, nb::c_contig, nb::ndim<4>>;
using Float4DArray = nb::ndarray<float, nb::c_contig, nb::ndim<4>>;
// 1D c-contiguous float arrays
using ConstFloat1D3ELArray = nb::ndarray<const float, nb::c_contig, nb::shape<3>>;

// Wrapper for joseph3d_fwd
void joseph3d_fwd_py(ConstFloatNDArray lor_start,
                     ConstFloatNDArray lor_end,
                     ConstFloat3DArray image,
                     ConstFloat1D3ELArray image_origin,
                     ConstFloat1D3ELArray voxel_size,
                     FloatNDArray projection_values,
                     int device_id = 0,
                     int threads_per_block = 64)
{
  // 1 check that ndim of lor_start and lor_end >=2 and last dim ==3
  if (lor_start.ndim() < 2 || lor_start.shape(lor_start.ndim() - 1) != 3)
    throw std::invalid_argument("lor_start must have at least 2 dims and shape (..., 3)");

  // 2 check that lor_start and lor_end have same ndim and shape
  if (lor_start.ndim() != lor_end.ndim())
    throw std::invalid_argument("lor_start and lor_end must have the same number of dimensions");
  for (std::size_t i = 0; i < lor_start.ndim(); ++i)
  {
    if (lor_start.shape(i) != lor_end.shape(i))
      throw std::invalid_argument("lor_start and lor_end must have the same shape");
  }

  // 3 check that the shape of projection_values matches lor_start.shape[:-1]
  if (projection_values.ndim() != lor_start.ndim() - 1)
    throw std::invalid_argument("projection_values must have a shape equal to lor_start.shape[:-1]");
  for (std::size_t i = 0; i < projection_values.ndim(); ++i)
  {
    if (projection_values.shape(i) != lor_start.shape(i))
      throw std::invalid_argument("projection_values must have a shape equal to lor_start.shape[:-1]");
  }

  // 4 check that lor_start, lor_end, image, image_origin, voxel_size, projection_values have the same device type
  if (lor_start.device_type() != lor_end.device_type() ||
      lor_start.device_type() != image.device_type() ||
      lor_start.device_type() != image_origin.device_type() ||
      lor_start.device_type() != voxel_size.device_type() ||
      lor_start.device_type() != projection_values.device_type())
  {
    throw std::invalid_argument("All input arrays must be on the same device type");
  }

  // 5 check that lor_start, lor_end, image, image_origin, voxel_size, projection_values have the same device ID
  if (lor_start.device_id() != lor_end.device_id() ||
      lor_start.device_id() != image.device_id() ||
      lor_start.device_id() != image_origin.device_id() ||
      lor_start.device_id() != voxel_size.device_id() ||
      lor_start.device_id() != projection_values.device_id())
  {
    throw std::invalid_argument("All input arrays must be on the same device ID");
  }

  // 7 compute the number of LORs as the product of all dimensions except the last
  // the last dimension must be 3 (3 floating point values per LOR endpoint)
  std::size_t num_lors = 1;
  for (std::size_t i = 0; i < lor_start.ndim() - 1; ++i)
  {
    num_lors *= lor_start.shape(i);
  }

  int image_dim[3] = {static_cast<int>(image.shape(0)),
                      static_cast<int>(image.shape(1)),
                      static_cast<int>(image.shape(2))};

  joseph3d_fwd(lor_start.data(), lor_end.data(), image.data(), image_origin.data(), voxel_size.data(), projection_values.data(), num_lors, image_dim, device_id, threads_per_block);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Wrapper for joseph3d_back
void joseph3d_back_py(ConstFloatNDArray lor_start,
                      ConstFloatNDArray lor_end,
                      Float3DArray image,
                      ConstFloat1D3ELArray image_origin,
                      ConstFloat1D3ELArray voxel_size,
                      ConstFloatNDArray projection_values,
                      int device_id = 0,
                      int threads_per_block = 64)
{
  // 1 check that ndim of lor_start and lor_end >=2 and last dim ==3
  if (lor_start.ndim() < 2 || lor_start.shape(lor_start.ndim() - 1) != 3)
    throw std::invalid_argument("lor_start must have at least 2 dims and shape (..., 3)");

  // 2 check that lor_start and lor_end have same ndim and shape
  if (lor_start.ndim() != lor_end.ndim())
    throw std::invalid_argument("lor_start and lor_end must have the same number of dimensions");
  for (std::size_t i = 0; i < lor_start.ndim(); ++i)
  {
    if (lor_start.shape(i) != lor_end.shape(i))
      throw std::invalid_argument("lor_start and lor_end must have the same shape");
  }

  // 3 check that the shape of projection_values matches lor_start.shape[:-1]
  if (projection_values.ndim() != lor_start.ndim() - 1)
    throw std::invalid_argument("projection_values must have a shape equal to lor_start.shape[:-1]");
  for (std::size_t i = 0; i < projection_values.ndim(); ++i)
  {
    if (projection_values.shape(i) != lor_start.shape(i))
      throw std::invalid_argument("projection_values must have a shape equal to lor_start.shape[:-1]");
  }

  // 4 check that lor_start, lor_end, image, image_origin, voxel_size, projection_values have the same device type
  if (lor_start.device_type() != lor_end.device_type() ||
      lor_start.device_type() != image.device_type() ||
      lor_start.device_type() != image_origin.device_type() ||
      lor_start.device_type() != voxel_size.device_type() ||
      lor_start.device_type() != projection_values.device_type())
  {
    throw std::invalid_argument("All input arrays must be on the same device type");
  }

  // 5 check that lor_start, lor_end, image, image_origin, voxel_size, projection_values have the same device ID
  if (lor_start.device_id() != lor_end.device_id() ||
      lor_start.device_id() != image.device_id() ||
      lor_start.device_id() != image_origin.device_id() ||
      lor_start.device_id() != voxel_size.device_id() ||
      lor_start.device_id() != projection_values.device_id())
  {
    throw std::invalid_argument("All input arrays must be on the same device ID");
  }

  // 7 compute the number of LORs as the product of all dimensions except the last
  // the last dimension must be 3 (3 floating point values per LOR endpoint)
  std::size_t num_lors = 1;
  for (std::size_t i = 0; i < lor_start.ndim() - 1; ++i)
  {
    num_lors *= lor_start.shape(i);
  }

  int image_dim[3] = {static_cast<int>(image.shape(0)),
                      static_cast<int>(image.shape(1)),
                      static_cast<int>(image.shape(2))};

  joseph3d_back(lor_start.data(), lor_end.data(), image.data(), image_origin.data(), voxel_size.data(), projection_values.data(), num_lors, image_dim, device_id, threads_per_block);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Shared argument checks for the batch projectors.
// image has shape (num_images, n0, n1, n2), projection_values has shape
// (num_images, *lor_start.shape[:-1]). Returns the number of LORs.
template <typename ImageArray, typename ProjArray>
std::size_t check_batch_args(const ConstFloatNDArray &lor_start,
                             const ConstFloatNDArray &lor_end,
                             const ImageArray &image,
                             const ConstFloat1D3ELArray &image_origin,
                             const ConstFloat1D3ELArray &voxel_size,
                             const ProjArray &projection_values)
{
  // 1 check that ndim of lor_start and lor_end >=2 and last dim ==3
  if (lor_start.ndim() < 2 || lor_start.shape(lor_start.ndim() - 1) != 3)
    throw std::invalid_argument("lor_start must have at least 2 dims and shape (..., 3)");

  // 2 check that lor_start and lor_end have same ndim and shape
  if (lor_start.ndim() != lor_end.ndim())
    throw std::invalid_argument("lor_start and lor_end must have the same number of dimensions");
  for (std::size_t i = 0; i < lor_start.ndim(); ++i)
  {
    if (lor_start.shape(i) != lor_end.shape(i))
      throw std::invalid_argument("lor_start and lor_end must have the same shape");
  }

  // 3 check that projection_values has shape (num_images, *lor_start.shape[:-1])
  if (projection_values.ndim() != lor_start.ndim())
    throw std::invalid_argument("projection_values must have shape (image.shape[0], *lor_start.shape[:-1])");
  if (projection_values.shape(0) != image.shape(0))
    throw std::invalid_argument("projection_values.shape[0] must be equal to image.shape[0] (number of images)");
  for (std::size_t i = 1; i < projection_values.ndim(); ++i)
  {
    if (projection_values.shape(i) != lor_start.shape(i - 1))
      throw std::invalid_argument("projection_values must have shape (image.shape[0], *lor_start.shape[:-1])");
  }

  // 4 check that all arrays have the same device type
  if (lor_start.device_type() != lor_end.device_type() ||
      lor_start.device_type() != image.device_type() ||
      lor_start.device_type() != image_origin.device_type() ||
      lor_start.device_type() != voxel_size.device_type() ||
      lor_start.device_type() != projection_values.device_type())
  {
    throw std::invalid_argument("All input arrays must be on the same device type");
  }

  // 5 check that all arrays have the same device ID
  if (lor_start.device_id() != lor_end.device_id() ||
      lor_start.device_id() != image.device_id() ||
      lor_start.device_id() != image_origin.device_id() ||
      lor_start.device_id() != voxel_size.device_id() ||
      lor_start.device_id() != projection_values.device_id())
  {
    throw std::invalid_argument("All input arrays must be on the same device ID");
  }

  // 6 the number of LORs is the product of all dimensions of lor_start except the last
  std::size_t num_lors = 1;
  for (std::size_t i = 0; i < lor_start.ndim() - 1; ++i)
  {
    num_lors *= lor_start.shape(i);
  }
  return num_lors;
}

// Wrapper for joseph3d_fwd_batch
void joseph3d_fwd_batch_py(ConstFloatNDArray lor_start,
                           ConstFloatNDArray lor_end,
                           ConstFloat4DArray image,
                           ConstFloat1D3ELArray image_origin,
                           ConstFloat1D3ELArray voxel_size,
                           FloatNDArray projection_values,
                           int device_id = 0,
                           int threads_per_block = 64)
{
  std::size_t num_lors = check_batch_args(lor_start, lor_end, image, image_origin, voxel_size, projection_values);

  int image_dim[3] = {static_cast<int>(image.shape(1)),
                      static_cast<int>(image.shape(2)),
                      static_cast<int>(image.shape(3))};

  joseph3d_fwd_batch(lor_start.data(), lor_end.data(), image.data(), image_origin.data(), voxel_size.data(), projection_values.data(), num_lors, image_dim, image.shape(0), device_id, threads_per_block);
}

// Wrapper for joseph3d_back_batch
void joseph3d_back_batch_py(ConstFloatNDArray lor_start,
                            ConstFloatNDArray lor_end,
                            Float4DArray image,
                            ConstFloat1D3ELArray image_origin,
                            ConstFloat1D3ELArray voxel_size,
                            ConstFloatNDArray projection_values,
                            int device_id = 0,
                            int threads_per_block = 64)
{
  std::size_t num_lors = check_batch_args(lor_start, lor_end, image, image_origin, voxel_size, projection_values);

  int image_dim[3] = {static_cast<int>(image.shape(1)),
                      static_cast<int>(image.shape(2)),
                      static_cast<int>(image.shape(3))};

  joseph3d_back_batch(lor_start.data(), lor_end.data(), image.data(), image_origin.data(), voxel_size.data(), projection_values.data(), num_lors, image_dim, image.shape(0), device_id, threads_per_block);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Wrapper for joseph3d_tof_sino_fwd
void joseph3d_tof_sino_fwd_py(ConstFloatNDArray lor_start,
                              ConstFloatNDArray lor_end,
                              ConstFloat3DArray image,
                              ConstFloat1D3ELArray image_origin,
                              ConstFloat1D3ELArray voxel_size,
                              FloatNDArray projection_values,
                              float tof_bin_width,
                              ConstFloatNDArray tof_sigma,
                              ConstFloatNDArray tof_center_offset,
                              short num_tof_bins,
                              float num_sigmas = 3.0f,
                              int device_id = 0,
                              int threads_per_block = 64)
{
  bool is_lor_dependent_tof_sigma;
  bool is_lor_dependent_tof_center_offset;

  // 1 check that ndim of lor_start and lor_end >=2 and last dim ==3
  if (lor_start.ndim() < 2 || lor_start.shape(lor_start.ndim() - 1) != 3)
    throw std::invalid_argument("lor_start must have at least 2 dims and shape (..., 3)");

  // 2 check that lor_start and lor_end have same ndim and shape
  if (lor_start.ndim() != lor_end.ndim())
    throw std::invalid_argument("lor_start and lor_end must have the same number of dimensions");
  for (std::size_t i = 0; i < lor_start.ndim(); ++i)
  {
    if (lor_start.shape(i) != lor_end.shape(i))
      throw std::invalid_argument("lor_start and lor_end must have the same shape");
  }

  // 3 check projection_values has same ndim as lor_start
  if (projection_values.ndim() != lor_start.ndim())
    throw std::invalid_argument("projection_values must have same number of dimensions as lor_start");
  for (std::size_t i = 0; i < (projection_values.ndim() - 1); ++i)
  {
    if (projection_values.shape(i) != lor_start.shape(i))
      throw std::invalid_argument("shape of projection_values[:-1] must match shape of lor_start[:-1]");
  }
  // check that projection_values.shape[-1] == num_tof_bins
  if (projection_values.shape(projection_values.ndim() - 1) != static_cast<std::size_t>(num_tof_bins))
    throw std::invalid_argument("last dimension of projection_values must equal num_tof_bins");

  // 4 check that lor_start, lor_end, image, image_origin, voxel_size, projection_values have the same device type
  if (lor_start.device_type() != lor_end.device_type() ||
      lor_start.device_type() != image.device_type() ||
      lor_start.device_type() != image_origin.device_type() ||
      lor_start.device_type() != voxel_size.device_type() ||
      lor_start.device_type() != projection_values.device_type())
  {
    throw std::invalid_argument("All input arrays must be on the same device type");
  }

  // 5 check that lor_start, lor_end, image, image_origin, voxel_size, projection_values have the same device ID
  if (lor_start.device_id() != lor_end.device_id() ||
      lor_start.device_id() != image.device_id() ||
      lor_start.device_id() != image_origin.device_id() ||
      lor_start.device_id() != voxel_size.device_id() ||
      lor_start.device_id() != projection_values.device_id())
  {
    throw std::invalid_argument("All input arrays must be on the same device ID");
  }

  // 7 compute the number of LORs as the product of all dimensions except the last
  // the last dimension must be 3 (3 floating point values per LOR endpoint)
  std::size_t num_lors = 1;
  for (std::size_t i = 0; i < lor_start.ndim() - 1; ++i)
  {
    num_lors *= lor_start.shape(i);
  }

  int image_dim[3] = {static_cast<int>(image.shape(0)),
                      static_cast<int>(image.shape(1)),
                      static_cast<int>(image.shape(2))};

  // check that the shape of tof_sigma is either [1,] or lor_start.shape[:-1]
  if (tof_sigma.ndim() == 1 && tof_sigma.shape(0) == 1)
  {
    is_lor_dependent_tof_sigma = false;
  }
  else if (tof_sigma.ndim() == lor_start.ndim() - 1)
  {
    for (std::size_t i = 0; i < (tof_sigma.ndim()); ++i)
    {
      if (tof_sigma.shape(i) != lor_start.shape(i))
        throw std::invalid_argument("shape of tof_sigma must match shape of lor_start[:-1] or be scalar");
    }
    is_lor_dependent_tof_sigma = true;
  }
  else
  {
    throw std::invalid_argument("shape of tof_sigma must match shape of lor_start[:-1] or be scalar");
  }

  // check that the shape of tof_center_offset is either [1,] or lor_start.shape[:-1]
  if (tof_center_offset.ndim() == 1 && tof_center_offset.shape(0) == 1)
  {
    is_lor_dependent_tof_center_offset = false;
  }
  else if (tof_center_offset.ndim() == lor_start.ndim() - 1)
  {
    for (std::size_t i = 0; i < (tof_center_offset.ndim()); ++i)
    {
      if (tof_center_offset.shape(i) != lor_start.shape(i))
        throw std::invalid_argument("shape of tof_center_offset must match shape of lor_start[:-1] or be scalar");
    }
    is_lor_dependent_tof_center_offset = true;
  }
  else
  {
    throw std::invalid_argument("shape of tof_center_offset must match shape of lor_start[:-1] or be scalar");
  }

  joseph3d_tof_sino_fwd(lor_start.data(),
                        lor_end.data(),
                        image.data(),
                        image_origin.data(),
                        voxel_size.data(),
                        projection_values.data(),
                        num_lors,
                        image_dim,
                        tof_bin_width,
                        tof_sigma.data(),
                        tof_center_offset.data(),
                        num_sigmas,
                        num_tof_bins,
                        static_cast<unsigned char>(is_lor_dependent_tof_sigma ? 1 : 0),
                        static_cast<unsigned char>(is_lor_dependent_tof_center_offset ? 1 : 0),
                        device_id,
                        threads_per_block);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Wrapper for joseph3d_tof_sino_back
void joseph3d_tof_back_fwd_py(ConstFloatNDArray lor_start,
                              ConstFloatNDArray lor_end,
                              Float3DArray image,
                              ConstFloat1D3ELArray image_origin,
                              ConstFloat1D3ELArray voxel_size,
                              ConstFloatNDArray projection_values,
                              float tof_bin_width,
                              ConstFloatNDArray tof_sigma,
                              ConstFloatNDArray tof_center_offset,
                              short num_tof_bins,
                              float num_sigmas = 3.0f,
                              int device_id = 0,
                              int threads_per_block = 64)
{
  bool is_lor_dependent_tof_sigma;
  bool is_lor_dependent_tof_center_offset;

  // 1 check that ndim of lor_start and lor_end >=2 and last dim ==3
  if (lor_start.ndim() < 2 || lor_start.shape(lor_start.ndim() - 1) != 3)
    throw std::invalid_argument("lor_start must have at least 2 dims and shape (..., 3)");

  // 2 check that lor_start and lor_end have same ndim and shape
  if (lor_start.ndim() != lor_end.ndim())
    throw std::invalid_argument("lor_start and lor_end must have the same number of dimensions");
  for (std::size_t i = 0; i < lor_start.ndim(); ++i)
  {
    if (lor_start.shape(i) != lor_end.shape(i))
      throw std::invalid_argument("lor_start and lor_end must have the same shape");
  }

  // 3 check projection_values has same ndim as lor_start
  if (projection_values.ndim() != lor_start.ndim())
    throw std::invalid_argument("projection_values must have same number of dimensions as lor_start");
  for (std::size_t i = 0; i < (projection_values.ndim() - 1); ++i)
  {
    if (projection_values.shape(i) != lor_start.shape(i))
      throw std::invalid_argument("shape of projection_values[:-1] must match shape of lor_start[:-1]");
  }
  // check that projection_values.shape[-1] == num_tof_bins
  if (projection_values.shape(projection_values.ndim() - 1) != static_cast<std::size_t>(num_tof_bins))
    throw std::invalid_argument("last dimension of projection_values must equal num_tof_bins");

  // 4 check that lor_start, lor_end, image, image_origin, voxel_size, projection_values have the same device type
  if (lor_start.device_type() != lor_end.device_type() ||
      lor_start.device_type() != image.device_type() ||
      lor_start.device_type() != image_origin.device_type() ||
      lor_start.device_type() != voxel_size.device_type() ||
      lor_start.device_type() != projection_values.device_type())
  {
    throw std::invalid_argument("All input arrays must be on the same device type");
  }

  // 5 check that lor_start, lor_end, image, image_origin, voxel_size, projection_values have the same device ID
  if (lor_start.device_id() != lor_end.device_id() ||
      lor_start.device_id() != image.device_id() ||
      lor_start.device_id() != image_origin.device_id() ||
      lor_start.device_id() != voxel_size.device_id() ||
      lor_start.device_id() != projection_values.device_id())
  {
    throw std::invalid_argument("All input arrays must be on the same device ID");
  }

  // 7 compute the number of LORs as the product of all dimensions except the last
  // the last dimension must be 3 (3 floating point values per LOR endpoint)
  std::size_t num_lors = 1;
  for (std::size_t i = 0; i < lor_start.ndim() - 1; ++i)
  {
    num_lors *= lor_start.shape(i);
  }

  int image_dim[3] = {static_cast<int>(image.shape(0)),
                      static_cast<int>(image.shape(1)),
                      static_cast<int>(image.shape(2))};

  // check that the shape of tof_sigma is either [1,] or lor_start.shape[:-1]
  if (tof_sigma.ndim() == 1 && tof_sigma.shape(0) == 1)
  {
    is_lor_dependent_tof_sigma = false;
  }
  else if (tof_sigma.ndim() == lor_start.ndim() - 1)
  {
    for (std::size_t i = 0; i < (tof_sigma.ndim()); ++i)
    {
      if (tof_sigma.shape(i) != lor_start.shape(i))
        throw std::invalid_argument("shape of tof_sigma must match shape of lor_start[:-1] or be scalar");
    }
    is_lor_dependent_tof_sigma = true;
  }
  else
  {
    throw std::invalid_argument("shape of tof_sigma must match shape of lor_start[:-1] or be scalar");
  }

  // check that the shape of tof_center_offset is either [1,] or lor_start.shape[:-1]
  if (tof_center_offset.ndim() == 1 && tof_center_offset.shape(0) == 1)
  {
    is_lor_dependent_tof_center_offset = false;
  }
  else if (tof_center_offset.ndim() == lor_start.ndim() - 1)
  {
    for (std::size_t i = 0; i < (tof_center_offset.ndim()); ++i)
    {
      if (tof_center_offset.shape(i) != lor_start.shape(i))
        throw std::invalid_argument("shape of tof_center_offset must match shape of lor_start[:-1] or be scalar");
    }
    is_lor_dependent_tof_center_offset = true;
  }
  else
  {
    throw std::invalid_argument("shape of tof_center_offset must match shape of lor_start[:-1] or be scalar");
  }

  joseph3d_tof_sino_back(lor_start.data(),
                         lor_end.data(),
                         image.data(),
                         image_origin.data(),
                         voxel_size.data(),
                         projection_values.data(),
                         num_lors,
                         image_dim,
                         tof_bin_width,
                         tof_sigma.data(),
                         tof_center_offset.data(),
                         num_sigmas,
                         num_tof_bins,
                         static_cast<unsigned char>(is_lor_dependent_tof_sigma ? 1 : 0),
                         static_cast<unsigned char>(is_lor_dependent_tof_center_offset ? 1 : 0),
                         device_id,
                         threads_per_block);
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Wrapper for joseph3d_tof_lm_fwd
void joseph3d_tof_lm_fwd_py(ConstFloatNDArray event_start,
                            ConstFloatNDArray event_end,
                            ConstFloat3DArray image,
                            ConstFloat1D3ELArray image_origin,
                            ConstFloat1D3ELArray voxel_size,
                            FloatNDArray projection_values,
                            float tof_bin_width,
                            ConstFloatNDArray tof_sigma,
                            ConstFloatNDArray tof_center_offset,
                            ConstShortNDArray tof_bin_index,
                            short num_tof_bins,
                            float num_sigmas = 3.0f,
                            int device_id = 0,
                            int threads_per_block = 64)
{
  bool is_lor_dependent_tof_sigma;
  bool is_lor_dependent_tof_center_offset;

  // 1 check that ndim of event_start and event_end are equal to 2
  if (event_start.ndim() != 2 || event_end.ndim() != 2)
    throw std::invalid_argument("event_start and event_end must have 2 dimensions");
  if (event_start.shape(1) != 3 || event_end.shape(1) != 3)
    throw std::invalid_argument("event_start and event_end must have shape (..., 3)");
  if (event_start.shape(0) != event_end.shape(0))
    throw std::invalid_argument("event_start and event_end must have the same number of events (shape[0])");

  std::size_t num_events = event_start.shape(0);

  // 2 check dims and shapes of projection_values and tof_bin_index
  if (projection_values.ndim() != 1)
    throw std::invalid_argument("projection_values.ndim must be 1");
  if (projection_values.shape(0) != num_events)
    throw std::invalid_argument("projection_values.shape[0] must match event_start.shape[0]");

  if (tof_bin_index.ndim() != 1)
    throw std::invalid_argument("tof_bin_index.ndim must be 1");
  if (tof_bin_index.shape(0) != num_events)
    throw std::invalid_argument("tof_bin_index.shape[0] must match event_start.shape[0]");

  // 3 check that all arrays have the same device type
  if (event_start.device_type() != event_end.device_type() ||
      event_start.device_type() != image.device_type() ||
      event_start.device_type() != image_origin.device_type() ||
      event_start.device_type() != voxel_size.device_type() ||
      event_start.device_type() != projection_values.device_type() ||
      event_start.device_type() != tof_bin_index.device_type())
  {
    throw std::invalid_argument("All input arrays must be on the same device type");
  }

  // 4 check that all arrays have the same device ID
  if (event_start.device_id() != event_end.device_id() ||
      event_start.device_id() != image.device_id() ||
      event_start.device_id() != image_origin.device_id() ||
      event_start.device_id() != voxel_size.device_id() ||
      event_start.device_id() != projection_values.device_id() ||
      event_start.device_id() != tof_bin_index.device_id())
  {
    throw std::invalid_argument("All input arrays must be on the same device ID");
  }

  int image_dim[3] = {static_cast<int>(image.shape(0)),
                      static_cast<int>(image.shape(1)),
                      static_cast<int>(image.shape(2))};

  // check that the shape of tof_sigma is either [1,] or [num_events,]
  if (tof_sigma.ndim() == 1 && tof_sigma.shape(0) == 1)
  {
    is_lor_dependent_tof_sigma = false;
  }
  else if (tof_sigma.ndim() == 1 && tof_sigma.shape(0) == num_events)
  {
    is_lor_dependent_tof_sigma = true;
  }
  else
  {
    throw std::invalid_argument("shape of tof_sigma must be [1,] or [num_events,]");
  }

  // check that the shape of tof_center_offset is either [1,] or [num_events,]
  if (tof_center_offset.ndim() == 1 && tof_center_offset.shape(0) == 1)
  {
    is_lor_dependent_tof_center_offset = false;
  }
  else if (tof_center_offset.ndim() == 1 && tof_center_offset.shape(0) == num_events)
  {
    is_lor_dependent_tof_center_offset = true;
  }
  else
  {
    throw std::invalid_argument("shape of tof_center_offset must be [1,] or [num_events,]");
  }

  joseph3d_tof_lm_fwd(event_start.data(),
                      event_end.data(),
                      image.data(),
                      image_origin.data(),
                      voxel_size.data(),
                      projection_values.data(),
                      num_events,
                      image_dim,
                      tof_bin_width,
                      tof_sigma.data(),
                      tof_center_offset.data(),
                      num_sigmas,
                      tof_bin_index.data(),
                      num_tof_bins,
                      static_cast<unsigned char>(is_lor_dependent_tof_sigma ? 1 : 0),
                      static_cast<unsigned char>(is_lor_dependent_tof_center_offset ? 1 : 0),
                      device_id,
                      threads_per_block);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Wrapper for joseph3d_tof_lm_back
void joseph3d_tof_lm_back_py(ConstFloatNDArray event_start,
                             ConstFloatNDArray event_end,
                             Float3DArray image,
                             ConstFloat1D3ELArray image_origin,
                             ConstFloat1D3ELArray voxel_size,
                             ConstFloatNDArray projection_values,
                             float tof_bin_width,
                             ConstFloatNDArray tof_sigma,
                             ConstFloatNDArray tof_center_offset,
                             ConstShortNDArray tof_bin_index,
                             short num_tof_bins,
                             float num_sigmas = 3.0f,
                             int device_id = 0,
                             int threads_per_block = 64)
{
  bool is_lor_dependent_tof_sigma;
  bool is_lor_dependent_tof_center_offset;

  // 1 check that ndim of event_start and event_end are equal to 2
  if (event_start.ndim() != 2 || event_end.ndim() != 2)
    throw std::invalid_argument("event_start and event_end must have 2 dimensions");
  if (event_start.shape(1) != 3 || event_end.shape(1) != 3)
    throw std::invalid_argument("event_start and event_end must have shape (..., 3)");
  if (event_start.shape(0) != event_end.shape(0))
    throw std::invalid_argument("event_start and event_end must have the same number of events (shape[0])");

  std::size_t num_events = event_start.shape(0);

  // 2 check dims and shapes of projection_values and tof_bin_index
  if (projection_values.ndim() != 1)
    throw std::invalid_argument("projection_values.ndim must be 1");
  if (projection_values.shape(0) != num_events)
    throw std::invalid_argument("projection_values.shape[0] must match event_start.shape[0]");

  if (tof_bin_index.ndim() != 1)
    throw std::invalid_argument("tof_bin_index.ndim must be 1");
  if (tof_bin_index.shape(0) != num_events)
    throw std::invalid_argument("tof_bin_index.shape[0] must match event_start.shape[0]");

  // 3 check that all arrays have the same device type
  if (event_start.device_type() != event_end.device_type() ||
      event_start.device_type() != image.device_type() ||
      event_start.device_type() != image_origin.device_type() ||
      event_start.device_type() != voxel_size.device_type() ||
      event_start.device_type() != projection_values.device_type() ||
      event_start.device_type() != tof_bin_index.device_type())
  {
    throw std::invalid_argument("All input arrays must be on the same device type");
  }

  // 4 check that all arrays have the same device ID
  if (event_start.device_id() != event_end.device_id() ||
      event_start.device_id() != image.device_id() ||
      event_start.device_id() != image_origin.device_id() ||
      event_start.device_id() != voxel_size.device_id() ||
      event_start.device_id() != projection_values.device_id() ||
      event_start.device_id() != tof_bin_index.device_id())
  {
    throw std::invalid_argument("All input arrays must be on the same device ID");
  }

  int image_dim[3] = {static_cast<int>(image.shape(0)),
                      static_cast<int>(image.shape(1)),
                      static_cast<int>(image.shape(2))};

  // check that the shape of tof_sigma is either [1,] or [num_events,]
  if (tof_sigma.ndim() == 1 && tof_sigma.shape(0) == 1)
  {
    is_lor_dependent_tof_sigma = false;
  }
  else if (tof_sigma.ndim() == 1 && tof_sigma.shape(0) == num_events)
  {
    is_lor_dependent_tof_sigma = true;
  }
  else
  {
    throw std::invalid_argument("shape of tof_sigma must be [1,] or [num_events,]");
  }

  // check that the shape of tof_center_offset is either [1,] or [num_events,]
  if (tof_center_offset.ndim() == 1 && tof_center_offset.shape(0) == 1)
  {
    is_lor_dependent_tof_center_offset = false;
  }
  else if (tof_center_offset.ndim() == 1 && tof_center_offset.shape(0) == num_events)
  {
    is_lor_dependent_tof_center_offset = true;
  }
  else
  {
    throw std::invalid_argument("shape of tof_center_offset must be [1,] or [num_events,]");
  }

  joseph3d_tof_lm_back(event_start.data(),
                       event_end.data(),
                       image.data(),
                       image_origin.data(),
                       voxel_size.data(),
                       projection_values.data(),
                       num_events,
                       image_dim,
                       tof_bin_width,
                       tof_sigma.data(),
                       tof_center_offset.data(),
                       num_sigmas,
                       tof_bin_index.data(),
                       num_tof_bins,
                       static_cast<unsigned char>(is_lor_dependent_tof_sigma ? 1 : 0),
                       static_cast<unsigned char>(is_lor_dependent_tof_center_offset ? 1 : 0),
                       device_id,
                       threads_per_block);
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

NB_MODULE(parallelproj_core, m)
{
  m.doc() = R"pbdoc(
    Python bindings for parallelproj backend.

    This module provides efficient C++/CUDA implementations of 3D Joseph
    forward and backward projectors with optional Time-of-Flight (TOF) support.

    Attributes
    ----------
    __version__ : str
        The version of the parallelproj backend library.
    version_major : int
      Major version number of the linked parallelproj library.
    version_minor : int
      Minor version number of the linked parallelproj library.
    version_patch : int
      Patch version number of the linked parallelproj library.
    cuda_enabled : int
        Flag indicating if the library was compiled with CUDA support.
        Returns 1 if CUDA is enabled, 0 otherwise.
  )pbdoc";

  // Expose the linked library version as __version__
  m.attr("__version__") = parallelproj_version();
  m.attr("version_major") = parallelproj_version_major();
  m.attr("version_minor") = parallelproj_version_minor();
  m.attr("version_patch") = parallelproj_version_patch();

  m.attr("cuda_enabled") = parallelproj_cuda_enabled();

  m.def("joseph3d_fwd", &joseph3d_fwd_py,
        "lor_start"_a.noconvert(), "lor_end"_a.noconvert(), "image"_a.noconvert(),
        "image_origin"_a.noconvert(), "voxel_size"_a.noconvert(), "projection_values"_a.noconvert(),
        "device_id"_a = 0, "threads_per_block"_a = 64,
        R"pbdoc(
        Forward projection using the Joseph 3D algorithm.

        Parameters
        ----------
        lor_start : ndarray
            Array of shape (..., 3) with coordinates of LOR start points.
        lor_end : ndarray
            Array of shape (..., 3) with coordinates of LOR end points.
        image : ndarray
            3D image array of shape (n0, n1, n2) for forward projection.
        image_origin : ndarray
            Array of shape (3,) with coordinates of [0,0,0] voxel center.
        voxel_size : ndarray
            Array of shape (3,) with voxel sizes.
        projection_values : ndarray
          Output array for projection results, shape must match lor_start.shape[:-1].
        device_id : int, optional
            Device ID for computation (default: 0).
        threads_per_block : int, optional
            Number of threads per block for GPU (default: 64).

        Notes
        -----
        All arrays must be on the same device (CPU or GPU) with matching device IDs.
        )pbdoc");

  m.def("joseph3d_back", &joseph3d_back_py,
        "lor_start"_a.noconvert(), "lor_end"_a.noconvert(), "image"_a.noconvert(),
        "image_origin"_a.noconvert(), "voxel_size"_a.noconvert(), "projection_values"_a.noconvert(),
        "device_id"_a = 0, "threads_per_block"_a = 64,
        R"pbdoc(
        Backprojection using the Joseph 3D algorithm.

        Parameters
        ----------
        lor_start : ndarray
            Array of shape (..., 3) with coordinates of LOR start points.
        lor_end : ndarray
            Array of shape (..., 3) with coordinates of LOR end points.
        image : ndarray
            3D image array of shape (n0, n1, n2) to accumulate backprojection into.
        image_origin : ndarray
            Array of shape (3,) with coordinates of [0,0,0] voxel center.
        voxel_size : ndarray
            Array of shape (3,) with voxel sizes.
        projection_values : ndarray
          Input values to backproject, shape must match lor_start.shape[:-1].
        device_id : int, optional
            Device ID for computation (default: 0).
        threads_per_block : int, optional
            Number of threads per block for GPU (default: 64).

        Notes
        -----
        Values are accumulated into the existing image array (not overwritten).
        All arrays must be on the same device (CPU or GPU) with matching device IDs.
        )pbdoc");

  m.def("joseph3d_fwd_batch", &joseph3d_fwd_batch_py,
        "lor_start"_a.noconvert(), "lor_end"_a.noconvert(), "image"_a.noconvert(),
        "image_origin"_a.noconvert(), "voxel_size"_a.noconvert(), "projection_values"_a.noconvert(),
        "device_id"_a = 0, "threads_per_block"_a = 64,
        R"pbdoc(
        Forward projection of a batch of images using the Joseph 3D algorithm.

        All images use the same LORs. The bilinear interpolation weights of
        each traversed plane are computed once and applied to all images.

        Parameters
        ----------
        lor_start : ndarray
            Array of shape (..., 3) with coordinates of LOR start points.
        lor_end : ndarray
            Array of shape (..., 3) with coordinates of LOR end points.
        image : ndarray
            Batch of 3D images, shape (num_images, n0, n1, n2).
        image_origin : ndarray
            Array of shape (3,) with coordinates of [0,0,0] voxel center.
        voxel_size : ndarray
            Array of shape (3,) with voxel sizes.
        projection_values : ndarray
            Output array for the projections, shape (num_images, *lor_start.shape[:-1]).
        device_id : int, optional
            Device ID for computation (default: 0).
        threads_per_block : int, optional
            Number of threads per block for GPU (default: 64).

        Notes
        -----
        All arrays must be on the same device (CPU or GPU) with matching device IDs.
        projection_values[b] is the forward projection of image[b].
        )pbdoc");

  m.def("joseph3d_back_batch", &joseph3d_back_batch_py,
        "lor_start"_a.noconvert(), "lor_end"_a.noconvert(), "image"_a.noconvert(),
        "image_origin"_a.noconvert(), "voxel_size"_a.noconvert(), "projection_values"_a.noconvert(),
        "device_id"_a = 0, "threads_per_block"_a = 64,
        R"pbdoc(
        Backprojection of a batch of projections using the Joseph 3D algorithm.

        All projections use the same LORs. The bilinear interpolation weights
        of each traversed plane are computed once and applied to all images.

        Parameters
        ----------
        lor_start : ndarray
            Array of shape (..., 3) with coordinates of LOR start points.
        lor_end : ndarray
            Array of shape (..., 3) with coordinates of LOR end points.
        image : ndarray
            Batch of 3D images to accumulate the backprojections into,
            shape (num_images, n0, n1, n2).
        image_origin : ndarray
            Array of shape (3,) with coordinates of [0,0,0] voxel center.
        voxel_size : ndarray
            Array of shape (3,) with voxel sizes.
        projection_values : ndarray
            Values to backproject, shape (num_images, *lor_start.shape[:-1]).
        device_id : int, optional
            Device ID for computation (default: 0).
        threads_per_block : int, optional
            Number of threads per block for GPU (default: 64).

        Notes
        -----
        Values are accumulated into the existing image array (not overwritten).
        All arrays must be on the same device (CPU or GPU) with matching device IDs.
        image[b] receives the backprojection of projection_values[b].
        )pbdoc");

  m.def("joseph3d_tof_sino_fwd", &joseph3d_tof_sino_fwd_py,
        "lor_start"_a.noconvert(), "lor_end"_a.noconvert(), "image"_a.noconvert(),
        "image_origin"_a.noconvert(), "voxel_size"_a.noconvert(), "projection_values"_a.noconvert(),
        "tof_bin_width"_a,
        "tof_sigma"_a.noconvert(),
        "tof_center_offset"_a.noconvert(),
        "num_tof_bins"_a,
        "num_sigmas"_a = 3.0f,
        "device_id"_a = 0, "threads_per_block"_a = 64,
        R"pbdoc(
        TOF sinogram forward projection using the Joseph 3D algorithm.

        Parameters
        ----------
        lor_start : ndarray
            Array of shape (..., 3) with coordinates of LOR start points.
        lor_end : ndarray
            Array of shape (..., 3) with coordinates of LOR end points.
        image : ndarray
            3D image array of shape (n0, n1, n2) for forward projection.
        image_origin : ndarray
            Array of shape (3,) with coordinates of [0,0,0] voxel center.
        voxel_size : ndarray
            Array of shape (3,) with voxel sizes.
        projection_values : ndarray
          Output array for TOF sinogram, shape (..., num_tof_bins).
        tof_bin_width : float
            Width of TOF bins in spatial units.
        tof_sigma : ndarray
            TOF resolution (sigma) in spatial units. Shape (1,) or (...,) for LOR-dependent.
        tof_center_offset : ndarray
            Offset of central TOF bin from LOR midpoint. Shape (1,) or (...,).
        num_tof_bins : int
            Number of TOF bins.
        num_sigmas : float, optional
            Number of sigmas for TOF kernel calculation (default: 3.0).
        device_id : int, optional
            Device ID for computation (default: 0).
        threads_per_block : int, optional
            Number of threads per block for GPU (default: 64).
        )pbdoc");

  m.def("joseph3d_tof_sino_back", &joseph3d_tof_back_fwd_py,
        "lor_start"_a.noconvert(), "lor_end"_a.noconvert(), "image"_a.noconvert(),
        "image_origin"_a.noconvert(), "voxel_size"_a.noconvert(), "projection_values"_a.noconvert(),
        "tof_bin_width"_a,
        "tof_sigma"_a.noconvert(),
        "tof_center_offset"_a.noconvert(),
        "num_tof_bins"_a,
        "num_sigmas"_a = 3.0f,
        "device_id"_a = 0, "threads_per_block"_a = 64,
        R"pbdoc(
        TOF sinogram backprojection using the Joseph 3D algorithm.

        Parameters
        ----------
        lor_start : ndarray
            Array of shape (..., 3) with coordinates of LOR start points.
        lor_end : ndarray
            Array of shape (..., 3) with coordinates of LOR end points.
        image : ndarray
            3D image array of shape (n0, n1, n2) to accumulate backprojection into.
        image_origin : ndarray
            Array of shape (3,) with coordinates of [0,0,0] voxel center.
        voxel_size : ndarray
            Array of shape (3,) with voxel sizes.
        projection_values : ndarray
          Input TOF sinogram to backproject, shape (..., num_tof_bins).
        tof_bin_width : float
            Width of TOF bins in spatial units.
        tof_sigma : ndarray
            TOF resolution (sigma) in spatial units. Shape (1,) or (...,) for LOR-dependent.
        tof_center_offset : ndarray
            Offset of central TOF bin from LOR midpoint. Shape (1,) or (...,).
        num_tof_bins : int
            Number of TOF bins.
        num_sigmas : float, optional
            Number of sigmas for TOF kernel calculation (default: 3.0).
        device_id : int, optional
            Device ID for computation (default: 0).
        threads_per_block : int, optional
            Number of threads per block for GPU (default: 64).

        Notes
        -----
        Values are accumulated into the existing image array (not overwritten).
        )pbdoc");

  m.def("joseph3d_tof_lm_fwd", &joseph3d_tof_lm_fwd_py,
        "event_start"_a.noconvert(), "event_end"_a.noconvert(), "image"_a.noconvert(),
        "image_origin"_a.noconvert(), "voxel_size"_a.noconvert(), "projection_values"_a.noconvert(),
        "tof_bin_width"_a,
        "tof_sigma"_a.noconvert(),
        "tof_center_offset"_a.noconvert(),
        "tof_bin_index"_a.noconvert(),
        "num_tof_bins"_a,
        "num_sigmas"_a = 3.0f,
        "device_id"_a = 0, "threads_per_block"_a = 64,
        R"pbdoc(
        TOF listmode forward projection using the Joseph 3D algorithm.

        Parameters
        ----------
        event_start : ndarray
          Array of shape (..., 3) with coordinates of event LOR start points.
        event_end : ndarray
          Array of shape (..., 3) with coordinates of event LOR end points.
        image : ndarray
            3D image array of shape (n0, n1, n2) for forward projection.
        image_origin : ndarray
            Array of shape (3,) with coordinates of [0,0,0] voxel center.
        voxel_size : ndarray
            Array of shape (3,) with voxel sizes.
        projection_values : ndarray
          Output array for event projections, shape matches event_start.shape[:-1].
        tof_bin_width : float
            Width of TOF bins in spatial units.
        tof_sigma : ndarray
            TOF resolution (sigma) in spatial units. Shape (1,) or (...,) for event-dependent.
        tof_center_offset : ndarray
            Offset of central TOF bin from LOR midpoint. Shape (1,) or (...,).
        tof_bin_index : ndarray
          TOF bin indices for each event, shape matches event_start.shape[:-1].
        num_tof_bins : int
            Number of TOF bins.
        num_sigmas : float, optional
            Number of sigmas for TOF kernel calculation (default: 3.0).
        device_id : int, optional
            Device ID for computation (default: 0).
        threads_per_block : int, optional
            Number of threads per block for GPU (default: 64).
        )pbdoc");

  m.def("joseph3d_tof_lm_back", &joseph3d_tof_lm_back_py,
        "event_start"_a.noconvert(), "event_end"_a.noconvert(), "image"_a.noconvert(),
        "image_origin"_a.noconvert(), "voxel_size"_a.noconvert(), "projection_values"_a.noconvert(),
        "tof_bin_width"_a,
        "tof_sigma"_a.noconvert(),
        "tof_center_offset"_a.noconvert(),
        "tof_bin_index"_a.noconvert(),
        "num_tof_bins"_a,
        "num_sigmas"_a = 3.0f,
        "device_id"_a = 0, "threads_per_block"_a = 64,
        R"pbdoc(
        TOF listmode backprojection using the Joseph 3D algorithm.

        Parameters
        ----------
        event_start : ndarray
          Array of shape (num_events, 3) with coordinates of event LOR start points.
        event_end : ndarray
          Array of shape (num_events, 3) with coordinates of event LOR end points.
        image : ndarray
            3D image array of shape (n0, n1, n2) to accumulate backprojection into.
        image_origin : ndarray
            Array of shape (3,) with coordinates of [0,0,0] voxel center.
        voxel_size : ndarray
            Array of shape (3,) with voxel sizes.
        projection_values : ndarray
            Input array to be backprojected, shape (num_events,).
        tof_bin_width : float
            Width of TOF bins in spatial units.
        tof_sigma : ndarray
            TOF resolution (sigma) in spatial units. Shape (1,) or (num_events,) for event-dependent.
        tof_center_offset : ndarray
            Offset of central TOF bin from LOR midpoint. Shape (1,) or (num_events,).
        tof_bin_index : ndarray
            TOF bin indices for each event, shape (num_events,).
        num_tof_bins : int
            Number of TOF bins.
        num_sigmas : float, optional
            Number of sigmas for TOF kernel calculation (default: 3.0).
        device_id : int, optional
            Device ID for computation (default: 0).
        threads_per_block : int, optional
            Number of threads per block for GPU (default: 64).

        Notes
        -----
        Values are accumulated into the existing image array (not overwritten).
        )pbdoc");
}
