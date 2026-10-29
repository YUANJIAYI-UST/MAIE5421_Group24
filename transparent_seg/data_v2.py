"""V2 category metadata only. Never use these labels as V1 binary targets."""
from pathlib import Path
import numpy as np
from PIL import Image

# Original author's class order and display palette, revision 3972916bba7f985ca1aabc047fea56bdec9e9e5d.
# Keep upstream names (including 'Glass Bow'); prose may explain their paper aliases.
CLASS_NAMES = ('Background', 'Shelf', 'Jar or Tank', 'Freezer', 'Window',
               'Glass Door', 'Eyeglass', 'Cup', 'Floor Glass', 'Glass Bow',
               'Water Bottle', 'Storage Box')
PALETTE = ((0, 0, 0), (120, 120, 70), (235, 255, 7), (6, 230, 230),
           (204, 255, 4), (120, 120, 120), (140, 140, 140), (255, 51, 7),
           (224, 5, 255), (204, 5, 255), (150, 5, 61), (4, 250, 7))


def category_array(image, encoding):
    """Return native-resolution IDs and explicit unknown codes, without guessing.

    Indexed and rendered-color data are separate explicit encodings. For color
    data, alpha must be 255; unknown colors become -1 for quarantine, not void.
    """
    raw = np.asarray(image)
    if encoding == 'author_index_0_11':
        if image.mode not in ('P', 'L') or raw.ndim != 2:
            raise ValueError('Indexed V2 masks require P/L mode')
        ids = raw.astype(np.int16)
        bad = (ids < 0) | (ids >= len(CLASS_NAMES))
        unknown = {str(int(k)): int(n) for k, n in zip(*np.unique(ids[bad], return_counts=True))}
        ids[bad] = -1
    elif encoding == 'author_rgb_palette':
        if image.mode not in ('RGB', 'RGBA'):
            raise ValueError('Rendered V2 masks require RGB/RGBA mode')
        rgb = raw[..., :3].astype(np.uint32)
        code = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
        ids = np.full(raw.shape[:2], -1, np.int16)
        opaque = raw[..., 3] == 255 if raw.shape[-1] == 4 else np.ones(ids.shape, bool)
        for i, (r, g, b) in enumerate(PALETTE):
            ids[(code == (r << 16 | g << 8 | b)) & opaque] = i
        unknown = {f'#{int(k):06x}': int(n) for k, n in zip(*np.unique(code[ids < 0], return_counts=True))}
        if not opaque.all():
            unknown['nonopaque_pixels'] = int((~opaque).sum())
    else:
        raise ValueError(f'Unknown V2 encoding: {encoding}')
    return ids, unknown


def read_category_mask(path, encoding):
    with Image.open(Path(path)) as image:
        ids, unknown = category_array(image, encoding)
    if unknown:
        raise ValueError(f'Unresolved V2 category codes in {path}: {unknown}')
    return ids
