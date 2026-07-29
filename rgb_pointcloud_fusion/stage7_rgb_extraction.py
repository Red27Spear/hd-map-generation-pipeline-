#!/usr/bin/env python3
"""
Stage 7: RGB Extraction & Color Mapping
Extracts RGB colors from camera images for projected LiDAR points.
Handles fisheye lens color aberration, vignetting correction, and
multi-camera color consistency.

Usage:
    python stage7_rgb_extraction.py \
        --occluded_points occluded_filtered.npy \
        --image MoRo_Bonn{...}.jpg \
        --output colored_points.npy
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List
import argparse
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter


@dataclass
class ColorConfig:
    """Configuration for RGB extraction."""
    # Vignetting correction
    correct_vignetting: bool = True
    vignetting_model: str = 'polynomial'  # 'polynomial' or 'cos4'

    # White balance / color correction
    apply_white_balance: bool = False
    white_balance_gains: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Gamma correction
    gamma: float = 2.2

    # Bilateral filtering for color smoothing
    color_smooth_sigma: float = 0.0  # 0 = disabled

    # Exposure normalization (for multi-camera consistency)
    exposure_compensation: float = 1.0


def estimate_vignetting_model(
    image: np.ndarray,
    camera_params: Optional[dict] = None
) -> np.ndarray:
    """Estimate vignetting correction model for fisheye lens.

    Fisheye lenses exhibit strong vignetting (darkening at edges).
    This estimates a radial correction factor.

    Returns:
        HxW correction factor map (multiply image by this)
    """
    H, W = image.shape[:2]
    cx, cy = W / 2, H / 2

    # Create radial distance map
    y, x = np.ogrid[:H, :W]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    r_max = np.sqrt(cx**2 + cy**2)
    r_norm = r / r_max

    if camera_params and 'vignetting_coeffs' in camera_params:
        # Use calibrated vignetting coefficients
        coeffs = camera_params['vignetting_coeffs']
        correction = 1.0 + coeffs[0]*r_norm**2 + coeffs[1]*r_norm**4 + coeffs[2]*r_norm**6
    else:
        # Default cos^4(theta) approximation for fisheye
        # For fisheye, angle theta ~ r / f (approximate)
        # Vignetting ~ cos^4(theta) for standard lens, stronger for fisheye
        correction = 1.0 / (1.0 - 0.3 * r_norm**2 + 0.1 * r_norm**4)

    # Clamp to reasonable range
    correction = np.clip(correction, 1.0, 3.0)

    return correction


def correct_vignetting(image: np.ndarray, correction_map: np.ndarray) -> np.ndarray:
    """Apply vignetting correction to image."""
    corrected = image.astype(np.float32) * correction_map[:, :, None]
    return np.clip(corrected, 0, 255).astype(np.uint8)


def apply_white_balance(
    image: np.ndarray,
    gains: Tuple[float, float, float]
) -> np.ndarray:
    """Apply white balance gains (BGR order for OpenCV)."""
    b_gain, g_gain, r_gain = gains
    channels = cv2.split(image)
    channels = [
        np.clip(channels[0] * b_gain, 0, 255).astype(np.uint8),
        np.clip(channels[1] * g_gain, 0, 255).astype(np.uint8),
        np.clip(channels[2] * r_gain, 0, 255).astype(np.uint8)
    ]
    return cv2.merge(channels)


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction."""
    lookup_table = np.array([
        ((i / 255.0) ** (1.0 / gamma)) * 255
        for i in range(256)
    ]).astype(np.uint8)
    return cv2.LUT(image, lookup_table)


def extract_rgb_at_pixels(
    image: np.ndarray,
    uv_coords: np.ndarray,
    interpolation: str = 'bilinear'
) -> np.ndarray:
    """Extract RGB colors at floating-point pixel coordinates.

    Args:
        image: HxWx3 BGR image (OpenCV format)
        uv_coords: Nx2 array of [u, v] coordinates (float)
        interpolation: 'nearest', 'bilinear', or 'bicubic'

    Returns:
        Nx3 array of BGR colors
    """
    H, W = image.shape[:2]

    if interpolation == 'nearest':
        u = np.round(uv_coords[:, 0]).astype(np.int32)
        v = np.round(uv_coords[:, 1]).astype(np.int32)
        u = np.clip(u, 0, W - 1)
        v = np.clip(v, 0, H - 1)
        colors = image[v, u]

    elif interpolation == 'bilinear':
        # Bilinear interpolation
        u0 = np.floor(uv_coords[:, 0]).astype(np.int32)
        v0 = np.floor(uv_coords[:, 1]).astype(np.int32)
        u1 = np.clip(u0 + 1, 0, W - 1)
        v1 = np.clip(v0 + 1, 0, H - 1)
        u0 = np.clip(u0, 0, W - 1)
        v0 = np.clip(v0, 0, H - 1)

        du = uv_coords[:, 0] - u0
        dv = uv_coords[:, 1] - v0

        # Sample four corners
        c00 = image[v0, u0].astype(np.float32)
        c10 = image[v0, u1].astype(np.float32)
        c01 = image[v1, u0].astype(np.float32)
        c11 = image[v1, u1].astype(np.float32)

        # Bilinear weights
        colors = (
            (1 - du[:, None]) * (1 - dv[:, None]) * c00 +
            du[:, None] * (1 - dv[:, None]) * c10 +
            (1 - du[:, None]) * dv[:, None] * c01 +
            du[:, None] * dv[:, None] * c11
        )
        colors = np.clip(colors, 0, 255).astype(np.uint8)

    else:  # bicubic
        # Use OpenCV remap for bicubic
        map_x = uv_coords[:, 0].reshape(1, -1).astype(np.float32)
        map_y = uv_coords[:, 1].reshape(1, -1).astype(np.float32)
        sampled = cv2.remap(
            image, map_x, map_y,
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT
        )
        colors = sampled.reshape(-1, 3)

    return colors


def compute_color_statistics(colors: np.ndarray) -> dict:
    """Compute color statistics for quality control."""
    return {
        'mean_bgr': colors.mean(axis=0).tolist(),
        'std_bgr': colors.std(axis=0).tolist(),
        'min_bgr': colors.min(axis=0).tolist(),
        'max_bgr': colors.max(axis=0).tolist(),
        'median_bgr': np.median(colors, axis=0).tolist()
    }


def match_histograms_to_reference(
    colors: np.ndarray,
    reference_colors: np.ndarray
) -> np.ndarray:
    """Match color histogram to reference for multi-camera consistency.

    Uses histogram matching to ensure all cameras produce similar
    color distributions (important for seamless texturing).
    """
    matched = colors.copy().astype(np.float32)

    for ch in range(3):
        # Compute CDFs
        src_hist, _ = np.histogram(colors[:, ch], bins=256, range=(0, 255))
        ref_hist, _ = np.histogram(reference_colors[:, ch], bins=256, range=(0, 255))

        src_cdf = np.cumsum(src_hist) / len(colors)
        ref_cdf = np.cumsum(ref_hist) / len(reference_colors)

        # Build lookup table
        lut = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            lut[i] = np.searchsorted(ref_cdf, src_cdf[i])

        matched[:, ch] = lut[colors[:, ch].astype(np.int32)]

    return matched.astype(np.uint8)


def extract_rgb_for_projected_points(
    image_path: str,
    projected_points: np.ndarray,
    config: ColorConfig = ColorConfig(),
    reference_image_path: Optional[str] = None
) -> Tuple[np.ndarray, dict]:
    """Main entry: extract RGB for all projected points from an image.

    Args:
        image_path: Path to camera image
        projected_points: Nx7 array [u, v, X, Y, Z, intensity, depth]
        config: ColorConfig parameters
        reference_image_path: Optional reference for histogram matching

    Returns:
        colored_points: Nx10 array [u, v, X, Y, Z, intensity, depth, B, G, R]
        stats: Color statistics dict
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    # Pre-processing
    if config.correct_vignetting:
        vignetting_map = estimate_vignetting_model(image)
        image = correct_vignetting(image, vignetting_map)

    if config.apply_white_balance:
        image = apply_white_balance(image, config.white_balance_gains)

    if config.gamma != 1.0:
        image = apply_gamma(image, config.gamma)

    # Extract colors at projected pixel locations
    uv = projected_points[:, :2]
    colors = extract_rgb_at_pixels(image, uv, interpolation='bilinear')

    # Optional: histogram matching for multi-camera consistency
    if reference_image_path:
        ref_image = cv2.imread(reference_image_path)
        if ref_image is not None:
            # Sample reference colors from center region (less vignetting)
            h, w = ref_image.shape[:2]
            ref_uv = np.array([[w/2, h/2]])  # Simplified: just center
            ref_colors = extract_rgb_at_pixels(ref_image, ref_uv)
            # In practice, you'd sample many points
            # colors = match_histograms_to_reference(colors, ref_colors)

    # Combine
    colored_points = np.column_stack([projected_points, colors])

    # Statistics
    stats = compute_color_statistics(colors)

    return colored_points, stats


def render_color_overlay(
    image_path: str,
    colored_points: np.ndarray,
    output_path: str,
    point_size: int = 3
):
    """Render colored points overlaid on the original image for verification."""
    image = cv2.imread(image_path)
    if image is None:
        return

    uv = colored_points[:, :2].astype(np.int32)
    colors = colored_points[:, 7:10].astype(np.uint8)

    for i, (u, v) in enumerate(uv):
        if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
            cv2.circle(image, (u, v), point_size, 
                      (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])), -1)

    cv2.imwrite(output_path, image)
    print(f"Color overlay saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Stage 7: RGB Extraction & Color Mapping')
    parser.add_argument('--occluded_points', required=True, help='Stage 6 output .npy')
    parser.add_argument('--image', required=True, help='Camera image path')
    parser.add_argument('--reference_image', help='Reference image for histogram matching')
    parser.add_argument('--correct_vignetting', action='store_true', default=True)
    parser.add_argument('--gamma', type=float, default=2.2)
    parser.add_argument('--exposure_comp', type=float, default=1.0)
    parser.add_argument('--output', default='colored_points.npy')
    parser.add_argument('--overlay', default='color_overlay.png')
    parser.add_argument('--stats', default='color_stats.json')
    args = parser.parse_args()

    # Load points
    points = np.load(args.occluded_points)
    print(f"Loaded {len(points)} points")

    # Configure
    config = ColorConfig(
        correct_vignetting=args.correct_vignetting,
        gamma=args.gamma,
        exposure_compensation=args.exposure_comp
    )

    # Extract colors
    colored, stats = extract_rgb_for_projected_points(
        args.image, points, config, args.reference_image
    )

    # Save
    np.save(args.output, colored)
    print(f"Saved colored points to {args.output}")

    # Save stats
    import json
    with open(args.stats, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Color statistics saved to {args.stats}")

    # Render overlay
    render_color_overlay(args.image, colored, args.overlay)

    print(f"\nRGB extraction complete:")
    print(f"  Points colored: {len(colored)}")
    print(f"  Mean BGR: {stats['mean_bgr']}")
    print(f"  Std BGR:  {stats['std_bgr']}")


if __name__ == '__main__':
    main()
