Changelog
=========

Unreleased
----------

Changed
^^^^^^^

* **Non-TOF Joseph forward projector: faster inner loop.** The forward worker
  now sums the plane contributions in a local variable and writes
  ``projection_values[i]`` once per LOR, instead of updating global memory for
  every traversed plane. Together with the changes below, the CUDA forward
  projection of sinogram-ordered LORs is about 20% faster in our tests;
  LORs in random order are not slower. CUDA back projection speed is
  unchanged. The CPU (OpenMP) back projection is about 5% slower.

* **Bilinear interpolation split into two steps.** New helpers
  ``bilinear_corners_fixed{0,1,2}`` compute the corner voxel offsets, bilinear
  weights and in-image flags of one plane; ``gather4`` / ``scatter4`` apply them
  to an image. The existing ``bilinear_interp_*`` and ``bilinear_interp_adj_*``
  functions keep their signatures and are now wrappers. The forward weights are
  precomputed per corner, which changes forward projections by about 1 ulp.

* **Index arithmetic within one image plane uses** ``int``. All projectors now
  throw ``std::invalid_argument`` if one image plane (``image_dim[1] *
  image_dim[2]`` voxels) exceeds ``INT_MAX``.

v2.0.7 (2026-07-08)
-------------------

Fixed
^^^^^

* **Joseph projectors: eliminate single-precision drift along the ray.** The
  forward/back workers now recompute the ray/voxel-plane intersection
  coordinates directly per plane (``i_f = i*a + b``) instead of accumulating
  them incrementally (``i_f += a``). The incremental update accumulated float32
  round-off that grew from the entry toward the exit of the image (up to
  ~1e-2 voxel on a large field of view). Projected values can therefore shift
  very slightly (~1e-3 relative) versus previous versions. Applied to all six
  workers (non-TOF and TOF, sinogram and listmode, forward and back), preserving
  the forward/back adjoint relationship.

* **TOF sinogram projectors: fixed a possible buffer overflow.** The per-plane
  TOF-weight loop could write past the fixed ``tof_weights[MAX_NUM_TOF_WEIGHTS]``
  buffer when the requested TOF window exceeded ``MAX_NUM_TOF_WEIGHTS`` bins
  (e.g. very fine TOF binning) -- undefined behavior. The window is now clamped
  to at most ``MAX_NUM_TOF_WEIGHTS`` bins centered on the ray's TOF position.

* **Forward projectors now always initialize their output.** For LORs/events
  that miss the image volume the output element is now set to 0 (non-TOF, TOF
  sinogram and TOF listmode). Previously a missed LOR left the output element
  unwritten, so it retained whatever was in the (possibly uninitialized) output
  buffer unless the caller pre-zeroed it.

* ``ray_cube_intersection_joseph`` hardening:

  - the single-plane guard uses ``floorf`` consistently (was ``(int)``
    truncation), so grazing rays entering at the image boundary face are no
    longer dropped;

  - rays parallel to an axis (``dr[k] == 0``, e.g. in-plane / direct-plane LORs)
    are handled explicitly instead of via ``1/0`` -> inf/NaN arithmetic, making
    the test well-defined also under ``-ffast-math`` / ``-ffinite-math-only``;

  - zero-length LORs (``xstart == xend``) are detected and treated as
    "no intersection" instead of producing ``0/0 = NaN``.

* **TOF listmode projectors: guard against degenerate TOF parameters.**
  Non-positive or non-finite ``tof_bin_width`` / ``sigma`` / ``num_sigmas`` are
  now rejected per event (they previously could divide by zero and yield
  out-of-range plane indices).

* The adjoint bilinear interpolation helpers no longer ``reinterpret_cast`` the
  image pointer to ``float *`` (which silently assumed ``T == float``);
  ``atomic_sum`` is now a template, so the helpers are correct for any
  floating-point element type.

v2.0.6 (2026-06-10)
-------------------

Changed
^^^^^^^

* The project version is now defined in a single ``VERSION`` file at the repo
  root (instead of being derived from ``git describe``); release tags must
  match it, enforced via the ``tag-release`` pixi task and a CI check.

Fixed
^^^^^

* Better checking of CUDA errors: failures of ``cudaMalloc``, ``cudaMemcpy``,
  kernel launches and ``cudaDeviceSynchronize`` now raise exceptions (visible
  as ``RuntimeError`` in Python), device memory is freed via RAII on all error
  paths, and stale CUDA error states can no longer be misattributed to later
  calls.

v2.0.5 (2026-03-25)
-------------------

Fixed
^^^^^

* Fix bug in TOF listmode forward and backprojector such that only image planes are taken that where the ray intersects the image volume. 
  Now the behavior of sinogram and LM TOF projectors should be very close for all LORs and the sum over TOF should be very close to non-TOF.

v2.0.4 (2026-03-13)
-------------------

Fixed
^^^^^

* remove cuda arch auto detection magic (conda-forge sets CUDAARCHS env variable)

* improve install docs

v2.0.3 (2026-03-05)
-------------------

Fixed
^^^^^

* bugfix to support all cuda 12.x versions (cudaMemLocation)

* improved auto version detection from git in build process


v2.0.2 (2026-03-04)
-------------------

Fixed
^^^^^

* better default install directories of python bindings compatible with conda-forge


v2.0.1 (2026-03-04)
-------------------

Fixed
^^^^^

* add VERSION and SOVERSION number to build of shared library libparallelproj

v2.0.0 (2026-03-03)
-------------------

Added
^^^^^

* direct support for host arrays, cuda managed arrays and cuda device arrays in the C API

* support for even number of TOF bins in TOF projectors

* nanobind Python API using python abi3 (python 3.12+ required) for better compatibility and easier installation

Breaking Changes
^^^^^^^^^^^^^^^^

* **listmode TOF** forward and backward projectors now require the **extra argument** ``num_tof_bins`` to specify the number of TOF bins, which can be even or odd.

* renaming: ``joseph3d_fwd_tof_sino`` to ``joseph3d_tof_sino_fwd``

* renaming: ``joseph3d_back_tof_sino`` to ``joseph3d_tof_sino_back``

* renaming: ``joseph3d_fwd_tof_lm`` to ``joseph3d_tof_lm_fwd``

* renaming: ``joseph3d_back_tof_lm`` to ``joseph3d_tof_lm_back``

* In the **Python API** to all **TOF projectors**, the **order of the TOF related arguments** has changed. ``num_sigmas`` is now last and has a default value of ``3.0``
