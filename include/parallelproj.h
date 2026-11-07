/*!
@file parallelproj.h
*/

#pragma once
#include <cstddef>
// import parallelproj_export.h to get the PARALLELPROJ_API macro
// needed for __declspec(dllexport) and __declspec(dllimport)
// This is needed for Windows to export the functions in the DLL
#include "parallelproj_export.h"

#ifdef __cplusplus
extern "C"
{
#endif

  /*! @brief Query whether this libparallelproj library was compiled with CUDA support.

  @return 1 if compiled with CUDA support, 0 otherwise.
  */
  PARALLELPROJ_API int parallelproj_cuda_enabled(void);

  /*! @brief Query the version string of this libparallelproj library. Potentially includes extra suffix for "dirty" versions.

  @return Null-terminated version string.
  */
  PARALLELPROJ_API const char *parallelproj_version(void);

  /*! @brief Query the major version number of this libparallelproj library.

  @return Major version number.
  */
  PARALLELPROJ_API int parallelproj_version_major(void);

  /*! @brief Query the minor version number of this libparallelproj library.

  @return Minor version number.
  */
  PARALLELPROJ_API int parallelproj_version_minor(void);

  /*! @brief Query the patch version number of this libparallelproj library.

  @return Patch version number.
  */
  PARALLELPROJ_API int parallelproj_version_patch(void);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C"
{
#endif

  /*!
  @brief Forward projection using the Joseph 3D algorithm.

  \pp_note_pointer_types

  \pp_param_lor_start{num_lors}
  \pp_param_lor_end{num_lors}
  \pp_param_image_fwd
  \pp_note_image_layout
  \pp_param_image_origin
  \pp_param_voxel_size
  \pp_param_projection_values_fwd{num_lors}
  \pp_param_count{num_lors,geometric LORs}
  \pp_param_image_dim
  \pp_param_device_id
  \pp_param_threads
  */
  PARALLELPROJ_API void joseph3d_fwd(const float *lor_start,
                                     const float *lor_end,
                                     const float *image,
                                     const float *image_origin,
                                     const float *voxel_size,
                                     float *projection_values,
                                     std::size_t num_lors,
                                     const int *image_dim,
                                     int device_id = 0,
                                     int threads_per_block = 64);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C"
{
#endif

  /*!
  @brief Backprojection using the Joseph 3D algorithm.

  \pp_note_pointer_types

  \pp_param_lor_start{num_lors}
  \pp_param_lor_end{num_lors}
  \pp_param_image_back
  \pp_note_image_layout
  \pp_param_image_origin
  \pp_param_voxel_size
  \pp_param_projection_values_back{num_lors}
  \pp_param_count{num_lors,geometric LORs}
  \pp_param_image_dim
  \pp_param_device_id
  \pp_param_threads
   */
  PARALLELPROJ_API void joseph3d_back(const float *lor_start,
                                      const float *lor_end,
                                      float *image,
                                      const float *image_origin,
                                      const float *voxel_size,
                                      const float *projection_values,
                                      std::size_t num_lors,
                                      const int *image_dim,
                                      int device_id = 0,
                                      int threads_per_block = 64);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C"
{
#endif

  /*!
  @brief Forward projection of a batch of images using the Joseph 3D algorithm.

  All images use the same LORs. The bilinear interpolation weights of each
  traversed plane are computed once and applied to all images.

  \pp_note_pointer_types

  \pp_param_lor_start{num_lors}
  \pp_param_lor_end{num_lors}
  \pp_param_image_fwd_batch
  \pp_note_image_layout
  \pp_param_image_origin
  \pp_param_voxel_size
  \pp_param_projection_values_fwd_batch{num_lors}
  \pp_param_count{num_lors,geometric LORs}
  \pp_param_image_dim
  \pp_param_count{num_images,images in the batch}
  \pp_param_device_id
  \pp_param_threads
  */
  PARALLELPROJ_API void joseph3d_fwd_batch(const float *lor_start,
                                           const float *lor_end,
                                           const float *image,
                                           const float *image_origin,
                                           const float *voxel_size,
                                           float *projection_values,
                                           std::size_t num_lors,
                                           const int *image_dim,
                                           std::size_t num_images,
                                           int device_id = 0,
                                           int threads_per_block = 64);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C"
{
#endif

  /*!
  @brief Backprojection of a batch of projections using the Joseph 3D algorithm.

  All projections use the same LORs. The bilinear interpolation weights of
  each traversed plane are computed once and applied to all images.

  \pp_note_pointer_types

  \pp_param_lor_start{num_lors}
  \pp_param_lor_end{num_lors}
  \pp_param_image_back_batch
  \pp_note_image_layout
  \pp_param_image_origin
  \pp_param_voxel_size
  \pp_param_projection_values_back_batch{num_lors}
  \pp_param_count{num_lors,geometric LORs}
  \pp_param_image_dim
  \pp_param_count{num_images,images in the batch}
  \pp_param_device_id
  \pp_param_threads
   */
  PARALLELPROJ_API void joseph3d_back_batch(const float *lor_start,
                                            const float *lor_end,
                                            float *image,
                                            const float *image_origin,
                                            const float *voxel_size,
                                            const float *projection_values,
                                            std::size_t num_lors,
                                            const int *image_dim,
                                            std::size_t num_images,
                                            int device_id = 0,
                                            int threads_per_block = 64);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C"
{
#endif

  /*! @brief 3D sinogram TOF Joseph forward projector

  \pp_note_pointer_types

  \pp_param_lor_start{num_lors}
  \pp_param_lor_end{num_lors}
  \pp_param_image_fwd
  \pp_note_image_layout
  \pp_param_image_origin
  \pp_param_voxel_size
  \pp_param_projection_values_tof_sino_out{num_lors,num_tof_bins}
  \pp_note_tof_sino_layout
  \pp_param_count{num_lors,geometric LORs}
  \pp_param_tof_image_dim
  \pp_param_tof_bin_width
  \pp_param_tof_sigma{num_lors,is_lor_dependent_tof_sigma}
  \pp_param_tof_center_offset{num_lors,is_lor_dependent_tof_center_offset}
  \pp_param_tof_num_sigmas
  \pp_param_tof_num_bins
  \pp_param_tof_dep_sigma{LORs}
  \pp_param_tof_dep_center{LORs}
  \pp_param_device_id
  \pp_param_threads
  */

  PARALLELPROJ_API void joseph3d_tof_sino_fwd(const float *lor_start,
                                              const float *lor_end,
                                              const float *image,
                                              const float *image_origin,
                                              const float *voxel_size,
                                              float *projection_values,
                                              std::size_t num_lors,
                                              const int *image_dim,
                                              float tof_bin_width,
                                              const float *tof_sigma,
                                              const float *tof_center_offset,
                                              float num_sigmas,
                                              short num_tof_bins,
                                              unsigned char is_lor_dependent_tof_sigma,
                                              unsigned char is_lor_dependent_tof_center_offset,
                                              int device_id = 0,
                                              int threads_per_block = 64);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C"
{
#endif

  /*!
  @brief TOF sinogram backprojection using the Joseph 3D algorithm.

  \pp_note_pointer_types

  @details The function backprojects a TOF sinogram into a 3D image volume using the Joseph
           ray-driven algorithm. Values from @p projection_values are distributed into @p image
           (accumulated, not overwritten).

  \pp_param_lor_start{num_lors}
  \pp_param_lor_end{num_lors}
  \pp_param_image_back
  \pp_note_image_layout
  \pp_param_image_origin
  \pp_param_voxel_size
  \pp_param_projection_values_tof_sino_in{num_lors,num_tof_bins}
  \pp_note_tof_sino_layout
  \pp_param_count{num_lors,geometric LORs}
  \pp_param_tof_image_dim
  \pp_param_tof_bin_width
  \pp_param_tof_sigma{num_lors,is_lor_dependent_tof_sigma}
  \pp_param_tof_center_offset{num_lors,is_lor_dependent_tof_center_offset}
  \pp_param_tof_num_sigmas
  \pp_param_tof_num_bins
  \pp_param_tof_dep_sigma{LORs}
  \pp_param_tof_dep_center{LORs}
  \pp_param_device_id
  \pp_param_threads

  */
  PARALLELPROJ_API void joseph3d_tof_sino_back(const float *lor_start,
                                               const float *lor_end,
                                               float *image,
                                               const float *image_origin,
                                               const float *voxel_size,
                                               const float *projection_values,
                                               std::size_t num_lors,
                                               const int *image_dim,
                                               float tof_bin_width,
                                               const float *tof_sigma,
                                               const float *tof_center_offset,
                                               float num_sigmas,
                                               short num_tof_bins,
                                               unsigned char is_lor_dependent_tof_sigma,
                                               unsigned char is_lor_dependent_tof_center_offset,
                                               int device_id = 0,
                                               int threads_per_block = 64);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C"
{
#endif

  /*! @brief 3D listmode TOF Joseph forward projector

  \pp_note_pointer_types

  \pp_param_lor_start{num_events}
  \pp_param_lor_end{num_events}
  \pp_param_image_fwd
  \pp_note_image_layout
  \pp_param_image_origin
  \pp_param_voxel_size
  \pp_param_projection_values_tof_lm_out{num_events}
  \pp_param_count{num_events,events}
  \pp_param_tof_image_dim
  \pp_param_tof_bin_width
  \pp_param_tof_sigma{num_events,is_lor_dependent_tof_sigma}
  \pp_param_tof_center_offset{num_events,is_lor_dependent_tof_center_offset}
  \pp_param_tof_num_sigmas
  \pp_param_tof_bin_index{num_events}
  \pp_param_tof_num_bins
  \pp_param_tof_dep_sigma{events}
  \pp_param_tof_dep_center{events}
  \pp_param_device_id
  \pp_param_threads
  */

  PARALLELPROJ_API void joseph3d_tof_lm_fwd(const float *lor_start,
                                            const float *lor_end,
                                            const float *image,
                                            const float *image_origin,
                                            const float *voxel_size,
                                            float *projection_values,
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
                                            int device_id = 0,
                                            int threads_per_block = 64);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C"
{
#endif

  /*! @brief 3D listmode TOF Joseph back projector

  \pp_note_pointer_types

  \pp_param_lor_start{num_events}
  \pp_param_lor_end{num_events}
  \pp_param_image_back
  \pp_note_image_layout
  \pp_param_image_origin
  \pp_param_voxel_size
  \pp_param_projection_values_tof_lm_back{num_events}
  \pp_param_count{num_events,events}
  \pp_param_tof_image_dim
  \pp_param_tof_bin_width
  \pp_param_tof_sigma{num_events,is_lor_dependent_tof_sigma}
  \pp_param_tof_center_offset{num_events,is_lor_dependent_tof_center_offset}
  \pp_param_tof_num_sigmas
  \pp_param_tof_bin_index{num_events}
  \pp_param_tof_num_bins
  \pp_param_tof_dep_sigma{events}
  \pp_param_tof_dep_center{events}
  \pp_param_device_id
  \pp_param_threads
  */

  PARALLELPROJ_API void joseph3d_tof_lm_back(const float *lor_start,
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
                                             int device_id = 0,
                                             int threads_per_block = 64);

#ifdef __cplusplus
}
#endif
