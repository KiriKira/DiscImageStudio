"""Experimental DVD image preprocessing with a passive reciprocal effective BRDF.

This models texture patches, NOT individual pits or measured Verbatim materials.
The output is an opaque binary PNG accepted by the existing DVD dispersion path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter, map_coordinates


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, name, minimum=0, strict=False):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and np.isfinite(value) and (value > minimum if strict else value >= minimum),
            f"Invalid {name}")
    return float(value)


def validate(profile):
    require(profile.get('schema') == 1, 'Unsupported profile schema')
    require(profile.get('provenance', {}).get('status') in ('hypothetical', 'measured'),
            'Profile must declare hypothetical or measured provenance')
    if profile['provenance']['status'] == 'measured':
        require(bool(profile['provenance'].get('evidence')), 'Measured profile needs evidence')
    finite(profile['track_pitch_nm'], 'track pitch', strict=True)
    radius = finite(profile['canvas_radius_mm'], 'canvas radius', strict=True)
    inner = finite(profile['inner_radius_mm'], 'inner radius')
    outer = finite(profile['outer_radius_mm'], 'outer radius', strict=True)
    require(inner < outer <= radius, 'Require inner < outer <= canvas radius')
    require(len(profile['textures']) == 2, 'Exactly two textures are required')
    require([t['scrambled_q'] for t in profile['textures']] == ['A5', '92'],
            'PNG bridge requires texture 0=A5, texture 1=92')
    for texture in profile['textures']:
        budget = sum(finite(texture[k], k) for k in ('diffuse', 'specular', 'diffraction'))
        require(budget <= 1, 'Reflectance budgets must sum to <= 1')
        for key in ('specular_width', 'diffraction_width_radial', 'diffraction_width_tangent'):
            finite(texture[key], key, strict=True)
    require(1 <= len(profile['wavelengths']) <= 32, 'Require 1..32 spectral samples')
    for wavelength, weight in profile['wavelengths']:
        finite(wavelength, 'wavelength', strict=True)
        finite(weight, 'spectral weight', strict=True)
    require(1 <= len(profile['views']) <= 8, 'Require 1..8 view conditions')
    for view in profile['views']:
        require(isinstance(view['name'], str) and bool(view['name']), 'View needs a name')
        finite(view['weight'], 'view weight', strict=True)
        camera = np.asarray(view['camera_mm'], dtype=float)
        require(camera.shape == (3,) and np.isfinite(camera).all() and camera[2] > 0,
                'Camera must be above the disc')
        require(bool(view['lights']), 'View needs a light')
        for light in view['lights']:
            center = np.asarray(light['center_mm'], dtype=float)
            size = np.asarray(light['size_mm'], dtype=float)
            require(center.shape == (3,) and np.isfinite(center).all() and center[2] > 0,
                    'Light must be above the disc')
            require(size.shape == (2,) and np.isfinite(size).all() and (size > 0).all(),
                    'Light dimensions must be positive')
            samples = light['samples_per_axis']
            require(isinstance(samples, int) and not isinstance(samples, bool) and 1 <= samples <= 32,
                    'Light quadrature must be 1..32')
            finite(light['radiance'], 'radiance', strict=True)
    return profile


def coordinates(size, radius):
    axis = np.linspace(-radius, radius, size)
    # Match RasterImage.SampleDisc: row zero is positive mathematical y.
    x, y = np.meshgrid(axis, -axis)
    return x, y, np.hypot(x, y)


def gaussian_lobe(qr, qt, radial_width, tangent_width):
    return np.exp(-.5*((qr/radial_width)**2+(qt/tangent_width)**2)) / (
        2*np.pi*radial_width*tangent_width)


def brdf(texture, light_direction, view_direction, radial, wavelength_nm, pitch_nm):
    """Directions point outwards. Integral f*cos(theta_o)*dOmega_o <= budget.

    Projection u=(sx,sy) gives cos(theta)*dOmega=du_x*du_y. Each Gaussian
    integrates to one on R^2, hence <=1 on the unit disk. q=s_xy+v_xy
    makes every term reciprocal; equal +/- order weights preserve symmetry.
    """
    sx, sy, sz = light_direction
    vx, vy, vz = view_direction
    ex, ey = radial
    qx, qy = sx+vx, sy+vy
    qr, qt = qx*ex+qy*ey, -qx*ey+qy*ex
    spec = gaussian_lobe(qr, qt, texture['specular_width'], texture['specular_width'])
    offset = wavelength_nm/pitch_nm
    dif = .5*(gaussian_lobe(qr-offset, qt, texture['diffraction_width_radial'],
                           texture['diffraction_width_tangent'])
              + gaussian_lobe(qr+offset, qt, texture['diffraction_width_radial'],
                              texture['diffraction_width_tangent']))
    value = texture['diffuse']/np.pi + texture['specular']*spec + texture['diffraction']*dif
    return np.where((np.asarray(sz) > 0) & (np.asarray(vz) > 0), value, 0)


def responses(profile, size=97):
    """Two full-texture radiance maps per view, integrated over finite area lights.

    All emitters are horizontal rectangles facing the disc. Incoherent wavelengths
    are sensor-weighted. Effective texture parameters are constant over wavelength.
    """
    x, y, radius = coordinates(size, profile['canvas_radius_mm'])
    ex = x/np.maximum(radius, 1e-12)
    ey = y/np.maximum(radius, 1e-12)
    ex[radius == 0] = 1
    result = np.zeros((len(profile['views']), 2, size, size), dtype=float)
    spectra = np.asarray(profile['wavelengths'], dtype=float)
    spectra[:, 1] /= spectra[:, 1].sum()
    for vi, view in enumerate(profile['views']):
        cx, cy, cz = view['camera_mm']
        distance = np.sqrt((cx-x)**2+(cy-y)**2+cz*cz)
        camera = ((cx-x)/distance, (cy-y)/distance, cz/distance)
        for light in view['lights']:
            lx, ly, height = light['center_mm']
            width, depth = light['size_mm']
            n = light['samples_per_axis']
            offsets = (np.arange(n)+.5)/n-.5
            area = width*depth/(n*n)
            for ox in offsets:
                for oy in offsets:
                    dx, dy = lx+ox*width-x, ly+oy*depth-y
                    r2 = dx*dx+dy*dy+height*height
                    distance = np.sqrt(r2)
                    direction = (dx/distance, dy/distance, height/distance)
                    # cos(source normal)*cos(disc normal)*dArea / distance^2.
                    irradiance = light['radiance']*height*height*area/(r2*r2)
                    for wavelength, weight in spectra:
                        for ti, texture in enumerate(profile['textures']):
                            result[vi, ti] += weight*irradiance*brdf(
                                texture, direction, camera, (ex, ey), wavelength,
                                profile['track_pitch_nm'])
    return result


def resize_maps(maps, size):
    rows, cols = np.meshgrid(np.linspace(0, maps.shape[-2]-1, size),
                             np.linspace(0, maps.shape[-1]-1, size), indexing='ij')
    return np.asarray([map_coordinates(m, [rows, cols], order=1, mode='nearest')
                       for m in maps.reshape(-1, *maps.shape[-2:])], dtype=np.float32).reshape(
                           *maps.shape[:-2], size, size)


def solve_fraction(r0, r1, target, weights):
    """Exact bounded per-cell least squares across all viewing conditions."""
    delta = r1-r0
    w = np.asarray(weights)[:, None, None]
    gain2 = np.sum(w*delta*delta, axis=0)
    floor = max(float(gain2.max())*1e-12, 1e-30)
    numerator = np.sum(w*delta*(target-r0), axis=0)
    raw = np.divide(numerator, gain2, out=np.full_like(gain2, .5), where=gain2 > floor)
    return np.clip(raw, 0, 1), gain2 > floor, (raw < 0) | (raw > 1)


def optical_error_diffusion(fraction, r0, r1, mask, weights, seed=43523):
    """Serpentine error diffusion on equal-mm x/y cells in radiance-error space.

    Store residual vectors per view. Select textures in weighted radiance space
    with a small non-tiled threshold perturbation to break regular locking.
    Only diffuse the component parallel to the local texture
    difference. Uncorrectable components are not allowed to accumulate.
    This is noise-shaped error diffusion, not a claim of isotropic blue noise.
    """
    height, width = mask.shape
    views = r0.shape[0]
    delta = np.asarray(r1-r0, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights /= weights.sum()
    errors = np.zeros((views, 2, width), dtype=float)
    output = np.zeros(mask.shape, dtype=np.uint8)
    for y in range(height):
        direction = 1 if y % 2 == 0 else -1
        positions = range(width) if direction == 1 else range(width-1, -1, -1)
        current, following = y % 2, (y+1) % 2
        for x in positions:
            if not mask[y, x]:
                continue
            d = delta[:, y, x]
            norm = float(np.dot(weights*d, d))
            accumulated = errors[:, current, x]
            adjusted = d*fraction[y, x]+accumulated
            # Perturb the radiance-space bisector, not the input tone curve.
            # Feedback retains the residual; there is no periodic threshold tile.
            phase = ((x*0x9E3779B1) ^ (y*0x85EBCA77) ^ seed) & 0xFFFFFFFF
            phase ^= phase >> 16
            phase = (phase*0x7FEB352D) & 0xFFFFFFFF
            phase ^= phase >> 15
            threshold = norm*(1+.2*(phase/4294967296.-.5))
            code = int(2*np.dot(weights*d, adjusted) >= threshold) if norm > 1e-30 else int(fraction[y, x] >= .5)
            output[y, x] = code
            if norm <= 1e-30:
                continue
            residual = adjusted-code*d
            projected = d*(np.dot(weights*d, residual)/norm)
            neighbors = ((y, x+direction, 7), (y+1, x-direction, 3),
                         (y+1, x, 5), (y+1, x+direction, 1))
            valid = [(ny, nx, weight) for ny, nx, weight in neighbors
                     if 0 <= ny < height and 0 <= nx < width and mask[ny, nx]]
            total = sum(weight for _, _, weight in valid)
            for ny, nx, weight in valid:
                errors[:, ny % 2, nx] += projected*(weight/total)
        errors[:, current, :] = 0
    return output


def srgb_linear(rgb):
    return np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)


def load_target(path, size, tone_exponent):
    # The input must already be a square disc-layout canvas: no silent cropping.
    im = ImageOps.exif_transpose(Image.open(path)).convert('RGBA')
    require(im.width == im.height, 'Input must be a square disc layout; place/crop the image explicitly first')
    rgba = np.asarray(im, dtype=float)/255
    rgb = srgb_linear(rgba[:, :, :3])
    luminance = rgb @ np.array([.2126, .7152, .0722])
    # White compositing is explicit and precedes tone mapping in linear light.
    luminance = luminance*rgba[:, :, 3]+1-rgba[:, :, 3]
    image = Image.fromarray(luminance.astype(np.float32))
    fitted = np.asarray(image.resize((size, size), Image.Resampling.LANCZOS)).copy()
    return np.clip(fitted, 0, 1)**tone_exponent


def evaluation(codes, target, r0, r1, mask, weights, cell_mm, sigmas_mm):
    radiance = r0+(r1-r0)*codes[None, :, :]
    records = []
    for sigma_mm in sigmas_mm:
        sigma = sigma_mm/cell_mm
        # Reflect boundary preserves constant radiance; evaluation excludes the
        # physical edge by a PSF-radius margin to avoid invented edge lighting.
        valid = mask & (gaussian_filter(mask.astype(float), sigma, truncate=3, mode='constant') > .999)
        if not valid.any():
            continue
        per_view = []
        for vi in range(len(weights)):
            error = gaussian_filter(radiance[vi]-target[vi], sigma, mode='reflect', truncate=3)
            per_view.append(float(np.mean(error[valid]**2)))
        records.append({'sigma_mm': sigma_mm, 'per_view_radiance_mse': per_view,
                        'weighted_mse': float(np.dot(weights, per_view))})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cell-um', type=float, default=80, help='Physical sample spacing; not a proven optimal scale')
    parser.add_argument('--map-size', type=int, default=97, help='Coarse optical-map grid; check convergence for narrow lobes')
    parser.add_argument('--tone-exponent', type=float, default=1, help='Explicit curve on linear source luminance')
    parser.add_argument('--seed', type=int, default=43523, help='Deterministic threshold-jitter and comparison seed')
    args = parser.parse_args()
    finite(args.cell_um, 'cell size', strict=True)
    finite(args.tone_exponent, 'tone exponent', strict=True)
    require(0 <= args.seed <= 0xFFFFFFFF, 'Seed must fit an unsigned 32-bit integer')
    require(17 <= args.map_size <= 513, 'Map size must be 17..513')
    profile = validate(json.loads(args.profile.read_text(encoding='utf-8')))
    size = int(round(2*profile['canvas_radius_mm']/(args.cell_um/1000)))+1
    require(33 <= size <= 4097, 'Prototype supports 33..4097 pixels; choose an appropriate cell size')
    cell_mm = 2*profile['canvas_radius_mm']/(size-1)
    require(not args.output_dir.exists() or not any(args.output_dir.iterdir()), 'Output directory must be empty')
    source = load_target(args.image, size, args.tone_exponent)
    _, _, radius = coordinates(size, profile['canvas_radius_mm'])
    mask = (radius >= profile['inner_radius_mm']) & (radius <= profile['outer_radius_mm'])
    require(mask.sum() > 32, 'Too few disc cells for this geometry')
    print(f'Optical maps: {args.map_size}x{args.map_size}; output: {size}x{size}, {cell_mm*1000:.3f} um', flush=True)
    maps = resize_maps(responses(profile, args.map_size), size)
    r0, r1 = maps[:, 0], maps[:, 1]
    targets = []
    endpoints = []
    weights = np.array([view['weight'] for view in profile['views']], dtype=float)
    weights /= weights.sum()
    for vi in range(len(weights)):
        # Uniform photographic tone target per view, relative to its global
        # median texture endpoints. Local illumination variation is not baked in.
        low, high = sorted([float(np.median(r0[vi][mask])), float(np.median(r1[vi][mask]))])
        endpoints.append([low, high])
        targets.append(low+(high-low)*source)
    target = np.asarray(targets)
    fraction, observable, clipped = solve_fraction(r0, r1, target, weights)
    print('Quantizing in physical radiance-error space...', flush=True)
    codes = optical_error_diffusion(fraction, r0, r1, mask, weights, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    random_codes = (rng.random(mask.shape) < fraction).astype(np.uint8)
    random_codes[~mask] = 0
    source_codes = (source >= .5).astype(np.uint8)
    source_codes[~mask] = 0
    sigmas = [cell_mm, 2*cell_mm, 4*cell_mm]
    metrics = {name: evaluation(values, target, r0, r1, mask, weights, cell_mm, sigmas)
               for name, values in [('physical', codes), ('random_same_fraction', random_codes),
                                    ('source_threshold', source_codes)]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Opaque, including outside the annulus: the existing reader only preserves
    # nearest-neighbor sampling for fully opaque binary grayscale images.
    Image.fromarray(codes*255).save(args.output_dir/'binary.png')
    Image.fromarray(random_codes*255).save(args.output_dir/'random-baseline.png')
    Image.fromarray(np.uint8(np.clip(fraction, 0, 1)*255)).save(args.output_dir/'mixture-preview.png')
    Image.fromarray(np.uint8(mask)*255).save(args.output_dir/'drawing-mask.png')
    for vi in range(len(weights)):
        raw = r0[vi]+(r1[vi]-r0[vi])*codes
        blurred = gaussian_filter(raw, 2, mode='reflect', truncate=3)
        # SAME display scale for target and output within a view. No per-image auto exposure.
        maximum = max(float(np.max(r0[vi][mask])), float(np.max(r1[vi][mask])), 1e-20)
        for name, values in [('predicted', blurred), ('target', target[vi])]:
            pixels = np.uint8(np.clip(values/maximum, 0, 1)*255)
            pixels[~mask] = 0
            Image.fromarray(pixels).save(args.output_dir/f'view-{vi}-{name}.png')
    radiance_fit = r0+(r1-r0)*fraction
    residual = np.sqrt(np.sum(weights[:, None, None]*(radiance_fit-target)**2, axis=0))
    report = {
        'algorithm': 'physical-coordinate-radiance-error-diffusion-v1',
        'profile': profile, 'source_sha256': hashlib.sha256(args.image.read_bytes()).hexdigest(),
        'size': size, 'cell_um': cell_mm*1000, 'map_size': args.map_size,
        'tone_exponent': args.tone_exponent, 'seed': args.seed,
        'threshold_jitter_fraction_range': [-0.05, 0.05],
        'endpoints_per_view': endpoints,
        'unobservable_fraction': float(np.mean(~observable[mask])),
        'clipped_mixture_fraction': float(np.mean(clipped[mask])),
        'continuous_fit_rms': float(np.sqrt(np.mean(residual[mask]**2))),
        'white_code_fraction': float(np.mean(codes[mask])), 'metrics': metrics,
        'camera_model': {'psf': 'isotropic Gaussian, evaluation only', 'sigma_mm': sigmas},
        'bridge': {'zero_q': 'A5', 'one_q': '92', 'canvas_radius_mm': profile['canvas_radius_mm'],
                   'opaque': True, 'disable_tone_and_resize': True, 'exact_ecc_replay': False},
        'limitations': ['Effective BRDF, not Maxwell/individual-pit simulation.',
                        'No material validation without measured profile evidence.',
                        'Mixture linearity and independent cells are assumptions.',
                        'PSF is evaluated after quantization, not globally optimized.',
                        'Not proof of improved physical contrast or wider viewing angle.'],
    }
    (args.output_dir/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote {args.output_dir}/binary.png and report.json', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f'Invalid input/profile: {exc}') from exc
