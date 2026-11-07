#pragma once

#ifdef __CUDACC__
#define WORKER_QUALIFIER __device__
#else
#define WORKER_QUALIFIER
#endif

// PP_UNROLL: ask the compiler to fully unroll the next loop (used for loops
// over a compile-time number of images)
#if defined(__CUDACC__)
#define PP_UNROLL _Pragma("unroll")
#elif defined(__GNUC__)
#define PP_UNROLL _Pragma("GCC unroll 8")
#else
#define PP_UNROLL
#endif
