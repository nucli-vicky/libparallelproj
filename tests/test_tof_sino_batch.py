import numpy as np
import parallelproj_core as pp
import pytest

from types import ModuleType

from .config import pytestmark
from .test_nontof_batch import BATCH_SIZES, max_abs, setup_geometry, to_numpy

TOF_BIN_WIDTH = 3.0
NUM_TOF_BINS = 11
NUM_SIGMAS = 3.0


def tof_parameters(xp: ModuleType, dev: str, lor_shape, lor_dependent: bool, rng):
    """TOF resolution and center offset, either one value or one value per LOR"""
    if lor_dependent:
        sigma = rng.uniform(3.0, 6.0, size=lor_shape).astype(np.float32)
        offset = rng.uniform(-2.0, 2.0, size=lor_shape).astype(np.float32)
    else:
        sigma = np.array([4.5], dtype=np.float32)
        offset = np.array([0.7], dtype=np.float32)
    return xp.asarray(sigma, device=dev), xp.asarray(offset, device=dev)


@pytest.mark.parametrize("lor_dependent", [False, True])
@pytest.mark.parametrize("num_images", BATCH_SIZES)
def test_tof_sino_fwd_batch_matches_single(xp: ModuleType, dev: str, num_images: int, lor_dependent: bool):
    """TOF forward projection of each image in a batch equals the single-image TOF forward projection"""
    img_dim, voxel_size, img_origin, lor_start, lor_end, rng = setup_geometry(xp, dev)
    lor_shape = tuple(lor_start.shape[:-1])
    sigma, offset = tof_parameters(xp, dev, lor_shape, lor_dependent, rng)

    images_np = rng.uniform(0, 1, size=(num_images,) + img_dim).astype(np.float32)
    proj_batch = xp.zeros((num_images,) + lor_shape + (NUM_TOF_BINS,), dtype=xp.float32, device=dev)
    pp.joseph3d_tof_sino_fwd_batch(lor_start, lor_end, xp.asarray(images_np, device=dev), img_origin, voxel_size,
                                   proj_batch, TOF_BIN_WIDTH, sigma, offset, NUM_TOF_BINS, NUM_SIGMAS)
    proj_batch_np = to_numpy(xp, proj_batch)

    for b in range(num_images):
        proj = xp.zeros(lor_shape + (NUM_TOF_BINS,), dtype=xp.float32, device=dev)
        pp.joseph3d_tof_sino_fwd(lor_start, lor_end, xp.asarray(images_np[b], device=dev), img_origin, voxel_size,
                                 proj, TOF_BIN_WIDTH, sigma, offset, NUM_TOF_BINS, NUM_SIGMAS)
        proj_np = to_numpy(xp, proj)
        assert max_abs(proj_np) > 0
        assert max_abs(proj_batch_np[b] - proj_np) <= 1e-6 * max_abs(proj_np)

    # the LOR that misses the image has a zero projection for all images
    assert max_abs(proj_batch_np[:, 0, 0, :]) == 0


@pytest.mark.parametrize("lor_dependent", [False, True])
@pytest.mark.parametrize("num_images", BATCH_SIZES)
def test_tof_sino_back_batch_matches_single(xp: ModuleType, dev: str, num_images: int, lor_dependent: bool):
    """TOF back projection of each sinogram in a batch equals the single TOF back projection,
    and values are added to the existing image"""
    img_dim, voxel_size, img_origin, lor_start, lor_end, rng = setup_geometry(xp, dev)
    lor_shape = tuple(lor_start.shape[:-1])
    sigma, offset = tof_parameters(xp, dev, lor_shape, lor_dependent, rng)

    proj_np = rng.uniform(0, 1, size=(num_images,) + lor_shape + (NUM_TOF_BINS,)).astype(np.float32)
    # the sinogram of image 1 is zero: its image must stay unchanged
    if num_images > 1:
        proj_np[1] = 0
    init_np = rng.uniform(0, 1, size=(num_images,) + img_dim).astype(np.float32)

    images = xp.asarray(init_np, device=dev)
    pp.joseph3d_tof_sino_back_batch(lor_start, lor_end, images, img_origin, voxel_size, xp.asarray(proj_np, device=dev),
                                    TOF_BIN_WIDTH, sigma, offset, NUM_TOF_BINS, NUM_SIGMAS)
    images_np = to_numpy(xp, images)

    for b in range(num_images):
        image = xp.asarray(init_np[b], device=dev)
        pp.joseph3d_tof_sino_back(lor_start, lor_end, image, img_origin, voxel_size, xp.asarray(proj_np[b], device=dev),
                                  TOF_BIN_WIDTH, sigma, offset, NUM_TOF_BINS, NUM_SIGMAS)
        image_np = to_numpy(xp, image)
        assert max_abs(images_np[b] - image_np) <= 1e-5 * max_abs(image_np)

    if num_images > 1:
        assert max_abs(images_np[1] - init_np[1]) == 0


@pytest.mark.parametrize("num_images", [3, 8])
def test_tof_sino_batch_adjointness(xp: ModuleType, dev: str, num_images: int):
    """<A x_b, y_b> == <x_b, A^T y_b> for every image b in the batch"""
    img_dim, voxel_size, img_origin, lor_start, lor_end, rng = setup_geometry(xp, dev, seed=1)
    lor_shape = tuple(lor_start.shape[:-1])
    sigma, offset = tof_parameters(xp, dev, lor_shape, True, rng)

    x_np = rng.uniform(0, 1, size=(num_images,) + img_dim).astype(np.float32)
    y_np = rng.uniform(0, 1, size=(num_images,) + lor_shape + (NUM_TOF_BINS,)).astype(np.float32)

    x_fwd = xp.zeros((num_images,) + lor_shape + (NUM_TOF_BINS,), dtype=xp.float32, device=dev)
    pp.joseph3d_tof_sino_fwd_batch(lor_start, lor_end, xp.asarray(x_np, device=dev), img_origin, voxel_size,
                                   x_fwd, TOF_BIN_WIDTH, sigma, offset, NUM_TOF_BINS, NUM_SIGMAS)
    y_back = xp.zeros((num_images,) + img_dim, dtype=xp.float32, device=dev)
    pp.joseph3d_tof_sino_back_batch(lor_start, lor_end, y_back, img_origin, voxel_size, xp.asarray(y_np, device=dev),
                                    TOF_BIN_WIDTH, sigma, offset, NUM_TOF_BINS, NUM_SIGMAS)

    x_fwd_np = to_numpy(xp, x_fwd).astype(np.float64)
    y_back_np = to_numpy(xp, y_back).astype(np.float64)
    for b in range(num_images):
        ip1 = float(np.sum(x_fwd_np[b] * y_np[b]))
        ip2 = float(np.sum(x_np[b] * y_back_np[b]))
        assert abs(ip1 - ip2) / abs(ip1) < 1e-5


def test_tof_sino_batch_invalid_shapes(xp: ModuleType, dev: str):
    """the TOF batch projectors reject sinogram arrays that do not match the image batch"""
    img_dim, voxel_size, img_origin, lor_start, lor_end, rng = setup_geometry(xp, dev)
    lor_shape = tuple(lor_start.shape[:-1])
    sigma, offset = tof_parameters(xp, dev, lor_shape, False, rng)
    images = xp.ones((3,) + img_dim, dtype=xp.float32, device=dev)

    bad_shapes = [
        (2,) + lor_shape + (NUM_TOF_BINS,),  # wrong number of images
        lor_shape + (NUM_TOF_BINS,),  # missing batch dimension
        (3,) + lor_shape + (NUM_TOF_BINS + 1,),  # wrong number of TOF bins
    ]
    for shape in bad_shapes:
        proj = xp.zeros(shape, dtype=xp.float32, device=dev)
        with pytest.raises(ValueError):
            pp.joseph3d_tof_sino_fwd_batch(lor_start, lor_end, images, img_origin, voxel_size, proj,
                                           TOF_BIN_WIDTH, sigma, offset, NUM_TOF_BINS, NUM_SIGMAS)
        with pytest.raises(ValueError):
            pp.joseph3d_tof_sino_back_batch(lor_start, lor_end, images, img_origin, voxel_size, proj,
                                            TOF_BIN_WIDTH, sigma, offset, NUM_TOF_BINS, NUM_SIGMAS)
