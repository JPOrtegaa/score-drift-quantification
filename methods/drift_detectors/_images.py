from io import BytesIO

import numpy as np
from matplotlib.image import imsave
from PIL import Image

IMAGE_BACKENDS = ("jpeg", "array")


# Convert a window of examples (rows) x features (columns) into the IBDD image:
# one row per feature, one column per example.
#
# "jpeg" reproduces the original IBDD pipeline, which wrote each window to disk
# with plt.imsave(..., cmap='Greys') and read it back with skimage.io.imread: the
# window is min-max normalized as a whole, mapped through the Greys colormap and
# JPEG compressed, giving a (features, examples, 3) uint8 array. Same calls,
# in memory instead of through temporary files.
#
# "array" skips the colormap and the lossy compression: the min-max normalized
# window itself, as float64 in [0, 1].
def window_to_image(X, backend="jpeg"):
    data = np.asarray(X, dtype=np.float64).T

    if backend == "jpeg":
        buffer = BytesIO()
        imsave(buffer, data, cmap="Greys", format="jpeg", dpi=100)
        buffer.seek(0)
        with Image.open(buffer) as image:
            return np.asarray(image)

    if backend == "array":
        lo, hi = np.nanmin(data), np.nanmax(data)
        if hi == lo:
            return np.zeros_like(data)
        return (data - lo) / (hi - lo)

    raise ValueError(f"image backend must be one of {IMAGE_BACKENDS}, got {backend!r}")


# Mean-Squared Deviation between two images of the same shape (Eq. 1 of the
# paper; same as skimage's compare_mse / mean_squared_error).
def msd(image_a, image_b):
    if image_a.shape != image_b.shape:
        raise ValueError(f"images must have the same shape, got {image_a.shape} and {image_b.shape}")
    diff = image_a.astype(np.float64) - image_b.astype(np.float64)
    return float(np.mean(diff ** 2))
