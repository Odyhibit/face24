"""Face24 similarity alignment: five landmarks to a 64x80 grayscale crop."""
import numpy as np
from PIL import Image

BACKGROUND = 128 / 255
VARIANTS = {"Face20": 20., "Face24": 24., "Face28": 28.}


def _checked_alpha(values, shape=None):
    alpha = np.asarray(values, dtype=np.float32)
    if alpha.ndim != 2 or not alpha.size or not np.isfinite(alpha).all() or alpha.min() < 0 or alpha.max() > 1:
        raise ValueError("Alpha must be a finite nonempty HxW array in [0,1]")
    if shape is not None and alpha.shape != tuple(shape):
        raise ValueError("Alpha/image dimensions differ")
    return alpha


def alpha_bytes(alpha):
    return np.rint(_checked_alpha(alpha) * 255).astype(np.uint8)


def resize_float(values, size=(64, 80)):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Invalid image plane")
    return np.asarray(Image.fromarray(values).resize(tuple(size), Image.Resampling.LANCZOS), dtype=np.float32)


def eye_coordinates(points):
    p = np.asarray(points, dtype=float)
    if p.shape != (5, 2) or not np.isfinite(p).all():
        raise ValueError("Invalid five-point landmarks")
    eye = .5 * (p[0] + p[1])
    vector = p[1] - p[0]
    distance = float(np.linalg.norm(vector))
    if distance <= 1 or vector[0] <= 0:
        raise ValueError("Invalid interocular geometry")
    u = vector / distance
    rotation = np.array([[u[0], u[1]], [-u[1], u[0]]])
    return eye, distance, rotation


def apply_transform(points, matrix):
    p = np.asarray(points, dtype=float)
    return p @ np.asarray(matrix)[:, :2].T + np.asarray(matrix)[:, 2]


def similarity(points, requested, eye_y):
    if requested not in VARIANTS.values() or not np.isfinite(eye_y):
        raise ValueError("Unsupported crop specification")
    eye, d, rotation = eye_coordinates(points)
    scale = requested / d
    matrix = np.column_stack([scale * rotation, [31.5, eye_y] - scale * rotation @ eye])
    actual = float(np.linalg.norm(np.diff(apply_transform(np.asarray(points)[:2], matrix), axis=0)))
    return {"matrix_source_to_output": matrix.tolist(), "scale": scale,
            "requested_interocular_pixels": requested, "actual_interocular_pixels": actual,
            "target_eye_midpoint": [31.5, float(eye_y)], "nonuniform_scaling": False,
            "head_fit_limited": False, "output_size": [64, 80]}


def render_crop(image, alpha, transform, supersample=4):
    import cv2
    a = np.asarray(alpha, dtype=np.float32)
    if a.shape != (image.height, image.width) or not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
        raise ValueError("Alpha/source mismatch")
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255
    composite = a * gray + (1 - a) * BACKGROUND
    matrix = np.asarray(transform["matrix_source_to_output"], dtype=np.float64)
    linear = matrix[:, :2]
    if not np.allclose(linear.T @ linear, np.eye(2) * np.linalg.det(linear), rtol=1e-7, atol=1e-9):
        raise ValueError("Crop must be a uniform similarity transform")
    high = matrix.copy()
    high[:, :2] *= supersample
    high[:, 2] = high[:, 2] * supersample + (supersample - 1) / 2
    size = (64 * supersample, 80 * supersample)
    # Composite once before filtering, with a gray border. Alpha is transformed
    # independently for metadata only, never multiplied into the target again.
    target_hi = cv2.warpAffine(composite, high, size, flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=BACKGROUND)
    alpha_hi = cv2.warpAffine(a, high, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    support = cv2.warpAffine(np.ones_like(a), high, size, flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    target = np.clip(resize_float(target_hi), 0, 1)
    out_alpha = np.clip(resize_float(alpha_hi), 0, 1)
    return Image.fromarray(alpha_bytes(target)), Image.fromarray(alpha_bytes(out_alpha)), {
        "padding_fraction": float(1 - support.mean()), "foreground_area": float(out_alpha.mean()),
        "interpolation": "OpenCV cubic to 4x pixel-center-aligned canvas, Pillow F LANCZOS to 64x80",
        "composite": "native uint8-alpha portrait on 128/255, before filtering; gray outside source"}
