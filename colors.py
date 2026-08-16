# colors.py
import numpy as np
try:
    import cv2
    def rgb2lab(rgb):
        """
        Comparison-compatible rgb2lab using OpenCV.
        Args:
            rgb: Input array of shape (..., 3) with float values in [0, 1].
        Returns:
            lab: Output array of same shape with Lab values.
        """
        inp = np.asanyarray(rgb, dtype=np.float32)
        orig_shape = inp.shape
        
        # Flatten to (1, N, 3) for OpenCV if needed
        if len(orig_shape) >= 2:
            pixels = inp.reshape(1, -1, 3)
        elif len(orig_shape) == 1:
            pixels = inp.reshape(1, 1, 3)
        else:
            pixels = inp
            
        # Convert
        lab_pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2Lab)
        
        # Reshape back to original shape
        return lab_pixels.reshape(orig_shape)
except ImportError:
    from skimage.color import rgb2lab


def distinguishable_colors(n_colors, bg=None, func=None):
    if bg is None:
        bg = np.array([1, 1, 1])
    else:
        bg = np.array(bg)
        if bg.ndim == 1:
            bg = bg.reshape((1, 3))
    n_grid = 30
    x = np.linspace(0, 1, n_grid)
    R, G, B = np.meshgrid(x, x, x)
    rgb = np.column_stack((R.ravel(), G.ravel(), B.ravel()))
    if func is None:
        func = rgb2lab
    lab = func(rgb)
    bglab = func(bg)
    if bglab.ndim == 1:
        bglab = bglab.reshape((1, 3))
    mindist2 = np.full((rgb.shape[0],), np.inf)
    for i in range(bglab.shape[0]):
        dX = lab - bglab[i, :]
        dist2 = np.sum(dX ** 2, axis=1)
        mindist2 = np.minimum(dist2, mindist2)
    colors = np.zeros((n_colors, 3))
    lastlab = bglab[-1, :]
    for i in range(n_colors):
        dX = lab - lastlab
        dist2 = np.sum(dX ** 2, axis=1)
        mindist2 = np.minimum(dist2, mindist2)
        index = np.argmax(mindist2)
        colors[i, :] = rgb[index, :]
        lastlab = lab[index, :]
    return colors
