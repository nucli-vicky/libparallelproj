import numpy as np
import parallelproj_core as pp
import pytest

from types import ModuleType

from .config import pytestmark

# batch sizes: single image, every group size (1, 2, 4, 8) and incomplete groups
BATCH_SIZES = [1, 2, 3, 5, 8, 11]


def setup_geometry(xp: ModuleType, dev: str, seed: int = 0):
    """small anisotropic image and random LORs, some of which miss the image"""
    rng = np.random.default_rng(seed)

    img_dim = (7, 9, 5)
    voxel_size = np.array([2.0, 1.5, 3.0], dtype=np.float32)
    img_origin = ((-0.5 * np.array(img_dim, dtype=np.float32) + 0.5) * voxel_size).astype(np.float32)

    # LOR array with 2 leading dimensions, end points in a cube around the image
    lor_start = rng.uniform(-20, 20, size=(4, 25, 3)).astype(np.float32)
    lor_end = rng.uniform(-20, 20, size=(4, 25, 3)).astype(np.float32)
    # one LOR that misses the image and one parallel to each axis
    lor_start[0, 0] = [100.0, 100.0, 100.0]
    lor_end[0, 0] = [101.0, 100.0, 100.0]
    for k in range(3):
        lor_start[1, k] = 0.3
        lor_end[1, k] = 0.3
        lor_start[1, k, k] = -50.0
        lor_end[1, k, k] = 50.0

    return (
        img_dim,
        xp.asarray(voxel_size, device=dev),
        xp.asarray(img_origin, device=dev),
        xp.asarray(lor_start, device=dev),
        xp.asarray(lor_end, device=dev),
        rng,
    )


def max_abs(x) -> float:
    return float(np.max(np.abs(np.asarray(x))))


def to_numpy(xp: ModuleType, x) -> np.ndarray:
    if hasattr(x, "get"):  # cupy
        return x.get()
    if hasattr(x, "cpu"):  # torch
        return x.cpu().numpy()
    return np.asarray(x)


@pytest.mark.parametrize("num_images", BATCH_SIZES)
def test_fwd_batch_matches_single(xp: ModuleType, dev: str, num_images: int):
    """forward projection of each image in a batch equals the single-image forward projection"""
    img_dim, voxel_size, img_origin, lor_start, lor_end, rng = setup_geometry(xp, dev)

    images_np = rng.uniform(0, 1, size=(num_images,) + img_dim).astype(np.float32)
    images = xp.asarray(images_np, device=dev)

    proj_batch = xp.zeros((num_images,) + lor_start.shape[:-1], dtype=xp.float32, device=dev)
    pp.joseph3d_fwd_batch(lor_start, lor_end, images, img_origin, voxel_size, proj_batch)
    proj_batch_np = to_numpy(xp, proj_batch)

    for b in range(num_images):
        proj = xp.zeros(lor_start.shape[:-1], dtype=xp.float32, device=dev)
        pp.joseph3d_fwd(lor_start, lor_end, xp.asarray(images_np[b], device=dev), img_origin, voxel_size, proj)
        proj_np = to_numpy(xp, proj)
        assert max_abs(proj_np) > 0
        assert max_abs(proj_batch_np[b] - proj_np) <= 1e-6 * max_abs(proj_np)

    # the LOR that misses the image has a zero projection for all images
    assert max_abs(proj_batch_np[:, 0, 0]) == 0


@pytest.mark.parametrize("num_images", BATCH_SIZES)
def test_back_batch_matches_single(xp: ModuleType, dev: str, num_images: int):
    """back projection of each projection in a batch equals the single back projection,
    and values are added to the existing image"""
    img_dim, voxel_size, img_origin, lor_start, lor_end, rng = setup_geometry(xp, dev)

    proj_np = rng.uniform(-1, 1, size=(num_images,) + tuple(lor_start.shape[:-1])).astype(np.float32)
    # the projection of image 1 is zero: its image must stay unchanged
    if num_images > 1:
        proj_np[1] = 0
    init_np = rng.uniform(0, 1, size=(num_images,) + img_dim).astype(np.float32)

    images = xp.asarray(init_np, device=dev)
    pp.joseph3d_back_batch(lor_start, lor_end, images, img_origin, voxel_size, xp.asarray(proj_np, device=dev))
    images_np = to_numpy(xp, images)

    for b in range(num_images):
        image = xp.asarray(init_np[b], device=dev)
        pp.joseph3d_back(lor_start, lor_end, image, img_origin, voxel_size, xp.asarray(proj_np[b], device=dev))
        image_np = to_numpy(xp, image)
        assert max_abs(images_np[b] - image_np) <= 1e-5 * max_abs(image_np)

    if num_images > 1:
        assert max_abs(images_np[1] - init_np[1]) == 0


@pytest.mark.parametrize("num_images", [3, 8])
def test_batch_adjointness(xp: ModuleType, dev: str, num_images: int):
    """<A x_b, y_b> == <x_b, A^T y_b> for every image b in the batch"""
    img_dim, voxel_size, img_origin, lor_start, lor_end, rng = setup_geometry(xp, dev, seed=1)

    x_np = rng.uniform(0, 1, size=(num_images,) + img_dim).astype(np.float32)
    y_np = rng.uniform(0, 1, size=(num_images,) + tuple(lor_start.shape[:-1])).astype(np.float32)

    x_fwd = xp.zeros((num_images,) + lor_start.shape[:-1], dtype=xp.float32, device=dev)
    pp.joseph3d_fwd_batch(lor_start, lor_end, xp.asarray(x_np, device=dev), img_origin, voxel_size, x_fwd)
    y_back = xp.zeros((num_images,) + img_dim, dtype=xp.float32, device=dev)
    pp.joseph3d_back_batch(lor_start, lor_end, y_back, img_origin, voxel_size, xp.asarray(y_np, device=dev))

    x_fwd_np = to_numpy(xp, x_fwd).astype(np.float64)
    y_back_np = to_numpy(xp, y_back).astype(np.float64)
    for b in range(num_images):
        ip1 = float(np.sum(x_fwd_np[b] * y_np[b]))
        ip2 = float(np.sum(x_np[b] * y_back_np[b]))
        assert abs(ip1 - ip2) / abs(ip1) < 1e-5


def test_batch_invalid_shapes(xp: ModuleType, dev: str):
    """the batch projectors reject projection arrays that do not match the image batch"""
    img_dim, voxel_size, img_origin, lor_start, lor_end, rng = setup_geometry(xp, dev)
    images = xp.ones((3,) + img_dim, dtype=xp.float32, device=dev)

    bad_shapes = [
        (2,) + tuple(lor_start.shape[:-1]),  # wrong number of images
        tuple(lor_start.shape[:-1]),  # missing batch dimension
        (3, 4, 24),  # wrong LOR shape
    ]
    for shape in bad_shapes:
        proj = xp.zeros(shape, dtype=xp.float32, device=dev)
        with pytest.raises(ValueError):
            pp.joseph3d_fwd_batch(lor_start, lor_end, images, img_origin, voxel_size, proj)
        with pytest.raises(ValueError):
            pp.joseph3d_back_batch(lor_start, lor_end, images, img_origin, voxel_size, proj)
