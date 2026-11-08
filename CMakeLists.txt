cmake_minimum_required(VERSION 3.31)

# ------------------------------------------------------------------------------
# OPTIONS
# ------------------------------------------------------------------------------

option(USE_CUDA       "Build with CUDA support" OFF)
option(BUILD_PYTHON   "Build the Python interface using nanobind" OFF)
option(BUILD_TESTS    "Build the tests" OFF)
option(BUILD_LIBRARY  "Build the C++ shared library (parallelproj)" ON)
option(INSTALL_DEVEL  "Install headers + CMake package config files" ON)
option(DEBUG          "Enable custom debug print statements" OFF)

if(BUILD_TESTS AND NOT BUILD_LIBRARY)
  message(FATAL_ERROR "BUILD_TESTS=ON requires BUILD_LIBRARY=ON")
endif()

if(DEBUG)
  add_compile_definitions(DEBUG)
endif()

# ------------------------------------------------------------------------------
# Determine version: the single source of truth is the VERSION file at the
# repo root (PEP 440 style, e.g. "2.1.0" or "2.0.6.dev0").
# - CMake's project() only accepts numeric X.Y.Z, so the numeric prefix goes
#   there; the full string is exported as PARALLELPROJ_VERSION_STRING.
# - Release tags must match the VERSION file exactly; this is enforced by
#   the tag-release pixi task and .github/workflows/check_version.yml.
# ------------------------------------------------------------------------------

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" PARALLELPROJ_VERSION_STRING LIMIT_COUNT 1)
string(STRIP "${PARALLELPROJ_VERSION_STRING}" PARALLELPROJ_VERSION_STRING)
# re-run the configure step automatically when the VERSION file changes
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
             "${CMAKE_CURRENT_SOURCE_DIR}/VERSION")

if(NOT PARALLELPROJ_VERSION_STRING MATCHES "^([0-9]+\\.[0-9]+\\.[0-9]+)(\\.dev[0-9]+)?$")
  message(FATAL_ERROR
    "Invalid version '${PARALLELPROJ_VERSION_STRING}' in the VERSION file. "
    "Expected X.Y.Z or X.Y.Z.devN (e.g. 2.1.0 or 2.0.6.dev0)."
  )
endif()
set(PARALLELPROJ_VERSION "${CMAKE_MATCH_1}")

project(PARALLELPROJ VERSION ${PARALLELPROJ_VERSION} LANGUAGES CXX)

message(STATUS "PARALLELPROJ_VERSION: ${PARALLELPROJ_VERSION}")
message(STATUS "PARALLELPROJ_VERSION_STRING: ${PARALLELPROJ_VERSION_STRING}")

include(CheckLanguage)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)


# ------------------------------------------------------------------------------
# Toolchain / platform tweaks
# ------------------------------------------------------------------------------
# Auto-export symbols helps when you don't explicitly annotate exports on Windows.
# Only affects SHARED libs.
if(WIN32 AND BUILD_LIBRARY)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif()

# Prefer position-independent code for libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ------------------------------------------------------------------------------
# C++ library target: parallelproj
# ------------------------------------------------------------------------------
set(PARALLELPROJ_CUDA 0)

if(BUILD_LIBRARY AND USE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit 12.0 REQUIRED)

  # nvcc>=12.8 init scripts on conda-forge set CUDAARCHS which is used to init CMAKE_CUDA_ARCHITECTURES
  # for local builds it is best to set CMAKE_CUDA_ARCHITECTURES explicitly to "local" (as being done in the preset)
  message(STATUS "ENV{CUDAARCHS}=$ENV{CUDAARCHS}")
  message(STATUS "CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")

  message(STATUS "CUDA library build enabled.")
  set(PARALLELPROJ_CUDA 1)

  set(LIB_SOURCES
    src/joseph3d_fwd.cu
    src/joseph3d_back.cu
    src/joseph3d_fwd_batch.cu
    src/joseph3d_back_batch.cu
    src/joseph3d_tof_sino_fwd.cu
    src/joseph3d_tof_sino_back.cu
    src/joseph3d_tof_sino_fwd_batch.cu
    src/joseph3d_tof_sino_back_batch.cu
    src/joseph3d_tof_lm_fwd.cu
    src/joseph3d_tof_lm_back.cu
    src/cuda_utils.cu
    src/parallelproj_info.cpp
  )

  add_library(parallelproj SHARED ${LIB_SOURCES})
  target_link_libraries(parallelproj PRIVATE CUDA::cudart)
  target_compile_definitions(parallelproj
    PUBLIC  PARALLELPROJ_CUDA=1
    PRIVATE PARALLELPROJ_EXPORTS
            PROJECT_VERSION="${PROJECT_VERSION}"
            PARALLELPROJ_VERSION_STRING="${PARALLELPROJ_VERSION_STRING}"
            PROJECT_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
            PROJECT_VERSION_MINOR=${PROJECT_VERSION_MINOR}
            PROJECT_VERSION_PATCH=${PROJECT_VERSION_PATCH}
  )

elseif(BUILD_LIBRARY)
  message(STATUS "CUDA library build disabled.")
  set(PARALLELPROJ_CUDA 0)

  set(LIB_SOURCES
    src/joseph3d_fwd.cpp
    src/joseph3d_back.cpp
    src/joseph3d_fwd_batch.cpp
    src/joseph3d_back_batch.cpp
    src/joseph3d_tof_sino_fwd.cpp
    src/joseph3d_tof_sino_back.cpp
    src/joseph3d_tof_sino_fwd_batch.cpp
    src/joseph3d_tof_sino_back_batch.cpp
    src/joseph3d_tof_lm_fwd.cpp
    src/joseph3d_tof_lm_back.cpp
    src/parallelproj_info.cpp
  )

  find_package(OpenMP REQUIRED)

  add_library(parallelproj SHARED ${LIB_SOURCES})
  target_link_libraries(parallelproj PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(parallelproj
    PUBLIC  PARALLELPROJ_CUDA=0
    PRIVATE PARALLELPROJ_EXPORTS
            PROJECT_VERSION="${PROJECT_VERSION}"
            PARALLELPROJ_VERSION_STRING="${PARALLELPROJ_VERSION_STRING}"
            PROJECT_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
            PROJECT_VERSION_MINOR=${PROJECT_VERSION_MINOR}
            PROJECT_VERSION_PATCH=${PROJECT_VERSION_PATCH}
  )
endif()

if(TARGET parallelproj)
  target_include_directories(parallelproj
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

  # Versioned shared library + SONAME (Linux/macOS). Windows ignores VERSION/SOVERSION.
  set_target_properties(parallelproj PROPERTIES
    VERSION   "${PROJECT_VERSION}"
    SOVERSION "${PROJECT_VERSION_MAJOR}"
    OUTPUT_NAME "parallelproj"
  )
endif()

# ------------------------------------------------------------------------------
# Python extension: parallelproj_core (nanobind)
# ------------------------------------------------------------------------------
if(BUILD_PYTHON)
  # Require interpreter + module development; request stable-ABI support when available.
  find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module
               OPTIONAL_COMPONENTS Development.SABIModule)

  # nanobind
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT
  )
  find_package(nanobind CONFIG REQUIRED)

  nanobind_add_module(parallelproj_core
    STABLE_ABI
    python/bindings.cpp
  )

  # Link to in-tree library if it exists, otherwise to installed package config.
  if(TARGET parallelproj)
    target_link_libraries(parallelproj_core PRIVATE parallelproj)
    add_dependencies(parallelproj_core parallelproj)
  else()
    find_package(parallelproj CONFIG REQUIRED)
    target_link_libraries(parallelproj_core PRIVATE parallelproj::parallelproj)
  endif()

  message(STATUS "Installing Python extension to ${Python_SITEARCH}")

  install(TARGETS parallelproj_core
  LIBRARY DESTINATION "${Python_SITEARCH}"   # Unix/macOS
  RUNTIME DESTINATION "${Python_SITEARCH}"   # Windows (.pyd)
  ARCHIVE DESTINATION "${Python_SITEARCH}"   # harmless
  )
endif()

# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------
if(BUILD_TESTS)
  enable_testing()

  set(TEST_SOURCES
    tests/test_ray_cube_intersection.cpp
    tests/test_bilinear_interp.cpp
    tests/test_bilinear_interp_adj.cpp
    tests/test_nontof_fwd_back.cpp
  )

  if(USE_CUDA)
    foreach(TEST_SOURCE IN LISTS TEST_SOURCES)
      get_filename_component(TEST_NAME "${TEST_SOURCE}" NAME_WE)
      set(CUDA_TEST_SOURCE "${CMAKE_SOURCE_DIR}/tests/${TEST_NAME}_cuda.cu")
      if(EXISTS "${CUDA_TEST_SOURCE}")
        list(APPEND TEST_SOURCES "${CUDA_TEST_SOURCE}")
      else()
        message(WARNING "CUDA test source not found: ${CUDA_TEST_SOURCE}")
      endif()
    endforeach()
  endif()

  set(TEST_DATA_DIR "${CMAKE_SOURCE_DIR}/tests")
  set(TEST_DATA_FILES
    img.txt
    vstart.txt
    vend.txt
    expected_fwd_vals.txt
  )

  set(TEST_DATA_OUTPUTS "")
  foreach(TEST_FILE IN LISTS TEST_DATA_FILES)
    add_custom_command(
      OUTPUT "${CMAKE_BINARY_DIR}/${TEST_FILE}"
      COMMAND "${CMAKE_COMMAND}" -E copy
              "${TEST_DATA_DIR}/${TEST_FILE}"
              "${CMAKE_BINARY_DIR}/${TEST_FILE}"
      DEPENDS "${TEST_DATA_DIR}/${TEST_FILE}"
    )
    list(APPEND TEST_DATA_OUTPUTS "${CMAKE_BINARY_DIR}/${TEST_FILE}")
  endforeach()

  add_custom_target(copy_test_data ALL DEPENDS ${TEST_DATA_OUTPUTS})

  foreach(TEST_SOURCE IN LISTS TEST_SOURCES)
    get_filename_component(TEST_NAME "${TEST_SOURCE}" NAME_WE)
    add_executable("${TEST_NAME}" "${TEST_SOURCE}")
    target_include_directories("${TEST_NAME}" PRIVATE
      "${CMAKE_SOURCE_DIR}/include"
      "${CMAKE_SOURCE_DIR}/tests"
    )
    add_dependencies("${TEST_NAME}" copy_test_data)

    if(USE_CUDA)
      # CMAKE_DL_LIBS is needed because tests/cuda_test_utils.h uses dlopen
      # to probe for the CUDA driver library (empty on Windows).
      target_link_libraries("${TEST_NAME}" PRIVATE parallelproj CUDA::cudart ${CMAKE_DL_LIBS})
    else()
      target_link_libraries("${TEST_NAME}" PRIVATE parallelproj OpenMP::OpenMP_CXX)
    endif()

    add_test(NAME "${TEST_NAME}" COMMAND "${TEST_NAME}")
  endforeach()

  # ----------------------------------------------------------------------------
  # CUDA-only tests
  # ----------------------------------------------------------------------------
  if(USE_CUDA)
    # White-box unit test for the internal handle_cuda_input_array() utility.
    # The function is deliberately NOT exported from the shared library (no
    # PARALLELPROJ_API), so src/cuda_utils.cu is compiled directly into the
    # test executable. This works identically on Linux, Windows and WSL and
    # avoids throwing C++ exceptions across a DLL boundary on Windows.
    add_executable(test_handle_cuda_input_array
      tests/test_handle_cuda_input_array.cu
      src/cuda_utils.cu
    )
    target_include_directories(test_handle_cuda_input_array PRIVATE
      "${CMAKE_SOURCE_DIR}/include"
      "${CMAKE_SOURCE_DIR}/tests"
    )
    target_link_libraries(test_handle_cuda_input_array PRIVATE CUDA::cudart ${CMAKE_DL_LIBS})
    add_test(NAME test_handle_cuda_input_array COMMAND test_handle_cuda_input_array)

    # Building with the CUDA toolkit does not imply a physical GPU is present
    # at runtime (e.g. CI runners, build-only machines). Every test that needs
    # a GPU self-detects this at startup (see tests/cuda_test_utils.h) and
    # exits with code 77; SKIP_RETURN_CODE makes CTest report such runs as
    # "Skipped" instead of "Failed".
    # test_nontof_fwd_back is included because in a CUDA build its
    # joseph3d_fwd/back calls launch CUDA kernels, even though it is a .cpp.
    set_tests_properties(
      test_ray_cube_intersection_cuda
      test_bilinear_interp_cuda
      test_bilinear_interp_adj_cuda
      test_nontof_fwd_back_cuda
      test_nontof_fwd_back
      test_handle_cuda_input_array
      PROPERTIES SKIP_RETURN_CODE 77
    )
  endif()
endif()

# ------------------------------------------------------------------------------
# Install: runtime library
# ------------------------------------------------------------------------------
if(TARGET parallelproj)
  install(TARGETS parallelproj
    EXPORT parallelprojTargets
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  )
endif()

# ------------------------------------------------------------------------------
# Install: devel (headers + CMake package config)
# ------------------------------------------------------------------------------
if(INSTALL_DEVEL AND TARGET parallelproj)
  # Headers
  install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/"
          DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

  # Export targets
  install(EXPORT parallelprojTargets
    FILE parallelprojTargets.cmake
    NAMESPACE parallelproj::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/parallelproj"
  )

  # Config + version files
  write_basic_package_version_file(
    "${CMAKE_BINARY_DIR}/parallelprojConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY AnyNewerVersion
  )

  configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/parallelprojConfig.cmake.in"
    "${CMAKE_BINARY_DIR}/parallelprojConfig.cmake"
    @ONLY
  )

  install(FILES
    "${CMAKE_BINARY_DIR}/parallelprojConfig.cmake"
    "${CMAKE_BINARY_DIR}/parallelprojConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/parallelproj"
  )
endif()

# ------------------------------------------------------------------------------
# Uninstall target
# ------------------------------------------------------------------------------
if(NOT TARGET uninstall)
  configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
    "${CMAKE_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY
  )
  add_custom_target(uninstall
    COMMAND "${CMAKE_COMMAND}" -P "${CMAKE_BINARY_DIR}/cmake_uninstall.cmake"
  )
endif()

# ------------------------------------------------------------------------------
# Doxygen
# ------------------------------------------------------------------------------
option(BUILD_DOCS "Build Doxygen documentation" OFF)
if(BUILD_DOCS)
  find_package(Doxygen REQUIRED)

  set(DOXYGEN_INPUT_FILE  "${CMAKE_SOURCE_DIR}/Doxyfile.in")
  set(DOXYGEN_OUTPUT_FILE "${CMAKE_BINARY_DIR}/Doxyfile")
  set(DOXYGEN_OUTPUT_DIR  "${CMAKE_BINARY_DIR}/docs")

  configure_file("${DOXYGEN_INPUT_FILE}" "${DOXYGEN_OUTPUT_FILE}" @ONLY)

  add_custom_target(doxygen-docs
    COMMAND "${DOXYGEN_EXECUTABLE}" "${DOXYGEN_OUTPUT_FILE}"
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    COMMENT "Generating Doxygen documentation..."
    VERBATIM
  )

  add_custom_target(doxygen-clean
    COMMAND "${CMAKE_COMMAND}" -E remove_directory "${DOXYGEN_OUTPUT_DIR}"
    COMMENT "Cleaning Doxygen documentation..."
    VERBATIM
  )
endif()
