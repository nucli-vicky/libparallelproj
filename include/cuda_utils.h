#pragma once
#include "utils.h"

#include <cstddef>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// CudaDevicePtr: RAII wrapper for CUDA device memory.
//
// RAII ("Resource Acquisition Is Initialization") ties a resource — here a
// device memory allocation — to the lifetime of a stack object. C++
// guarantees that the destructor of a stack object runs whenever the object
// goes out of scope, on EVERY exit path: normal return, early return, or an
// exception unwinding the stack. Putting cudaFree() in the destructor
// therefore guarantees the device buffer is released even when an error is
// thrown halfway through a projector function.
//
// Why this replaced the old "T* d_x; bool free_x; ... if (free_x)
// cudaFree(d_x);" pattern: the manual cudaFree() calls at the end of each
// projector only ran if execution reached the end. Since the projectors now
// throw std::runtime_error on CUDA errors (failed H2D copy, kernel launch
// error, failed cudaDeviceSynchronize, ...), a throw would have skipped the
// cleanup block and leaked every device buffer allocated up to that point.
// With this wrapper no explicit free is needed anywhere — the compiler
// inserts the destructor calls on all paths.
//
// Ownership semantics (the `owns` flag, set by handle_cuda_input_array):
//   * owns == true  : the input was a host array; a temporary device copy
//                     was allocated with cudaMalloc and must be freed.
//                     The destructor calls cudaFree(ptr).
//   * owns == false : the input was already device or managed memory; `ptr`
//                     just borrows the caller's pointer. The destructor does
//                     nothing — we must not free memory we do not own.
//
// The wrapper is non-copyable: two copies owning the same pointer would
// both call cudaFree on it (double free). It is movable, so ownership can
// be transferred; the moved-from object is set to {nullptr, false} so its
// destructor is a no-op.
//
// Note: cudaFree in a destructor must not throw (destructors that throw
// during stack unwinding terminate the program), so its return value is
// deliberately ignored. This is the same trade-off std::unique_ptr makes.
// ---------------------------------------------------------------------------
template <typename T>
struct CudaDevicePtr {
    T*   ptr  = nullptr;  // device pointer used in kernel launches
    bool owns = false;    // true only if we cudaMalloc'ed ptr ourselves

    CudaDevicePtr() = default;

    // Non-copyable: copies would double-free the same allocation.
    CudaDevicePtr(const CudaDevicePtr&)            = delete;
    CudaDevicePtr& operator=(const CudaDevicePtr&) = delete;

    // Movable: transfers ownership and disarms the source object.
    CudaDevicePtr(CudaDevicePtr&& other) noexcept
        : ptr(other.ptr), owns(other.owns)
    { other.ptr = nullptr; other.owns = false; }
    CudaDevicePtr& operator=(CudaDevicePtr&& other) noexcept {
        if (this != &other) {
            if (owns && ptr) cudaFree(ptr); // release current allocation first
            ptr = other.ptr; owns = other.owns;
            other.ptr = nullptr; other.owns = false;
        }
        return *this;
    }

    // Runs automatically on scope exit (including exception unwinding).
    ~CudaDevicePtr() { if (owns && ptr) cudaFree(ptr); }

    T* get() const { return ptr; }
};

// Checked device-to-host copy. Throws std::runtime_error on failure.
inline void cuda_memcpy_d2h(void* dst, const void* src, std::size_t size)
{
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        cudaGetLastError(); // clear last-error state before throwing
        throw std::runtime_error(
            std::string("cudaMemcpy (D2H) failed: ") + cudaGetErrorString(err));
    }
}

// Derive nvoxels = img_dim[0]*img_dim[1]*img_dim[2] when img_dim may be a
// host, CUDA managed or device pointer. Throws std::invalid_argument or
// std::runtime_error on error; returns computed nvoxels on success.
inline std::size_t cuda_nvoxels_from_img_dim(const int *img_dim_ptr)
{
    if (!img_dim_ptr)
        throw std::invalid_argument("nvoxels_from_img_dim: img_dim_ptr is null");

    int h_img_dim[3] = {0, 0, 0};
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, static_cast<const void *>(img_dim_ptr));
    if (err != cudaSuccess)
    {
        // Expected for plain host pointers on older CUDA versions. Clear the
        // last-error state so it cannot surface in a later kernel launch check.
        cudaGetLastError();
    }

    // If pointer known to CUDA and points to device/managed memory, copy to host.
    if (err == cudaSuccess && (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged))
    {
        err = cudaMemcpy(h_img_dim, img_dim_ptr, 3 * sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            cudaGetLastError(); // clear last-error state before throwing
            throw std::runtime_error(std::string("nvoxels_from_img_dim: cudaMemcpy failed: ") + cudaGetErrorString(err));
        }
    }
    else
    {
        // Treat as host pointer (or pointer attributes not available) — read directly.
        h_img_dim[0] = img_dim_ptr[0];
        h_img_dim[1] = img_dim_ptr[1];
        h_img_dim[2] = img_dim_ptr[2];
    }

    if (h_img_dim[0] <= 0 || h_img_dim[1] <= 0 || h_img_dim[2] <= 0)
        throw std::invalid_argument("nvoxels_from_img_dim: invalid img_dim values");
    check_image_plane_size(h_img_dim[1], h_img_dim[2]);

    return static_cast<std::size_t>(h_img_dim[0]) *
           static_cast<std::size_t>(h_img_dim[1]) *
           static_cast<std::size_t>(h_img_dim[2]);
}

// Overload for constant input_ptr (const T*)
template <typename T>
void handle_cuda_input_array(const T *input_ptr, T **device_ptr, std::size_t size, bool &free_flag, int device_id, cudaMemoryAdvise memory_hint);

// Overload for non-constant input_ptr (T*)
template <typename T>
void handle_cuda_input_array(T *input_ptr, T **device_ptr, std::size_t size, bool &free_flag, int device_id, cudaMemoryAdvise memory_hint);
