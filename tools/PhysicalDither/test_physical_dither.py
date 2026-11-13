import copy
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.ndimage import gaussian_filter

from physical_dither import (brdf, optical_error_diffusion, responses, resize_maps,
                             solve_fraction, validate)


class PhysicalDitherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = validate(json.loads(Path(__file__).with_name('example-profile.json').read_text()))

    def test_reciprocity(self):
        a = (.3, .2, np.sqrt(.87))
        b = (-.6, .1, np.sqrt(.63))
        radial = (.6, .8)
        for texture in self.profile['textures']:
            self.assertAlmostEqual(float(brdf(texture, a, b, radial, 550, 740)),
                                   float(brdf(texture, b, a, radial, 550, 740)), places=14)

    def test_hemispherical_energy_bound(self):
        # Projected disk quadrature directly integrates f*cos(theta)*dOmega.
        axis = (np.arange(501)+.5)*2/501-1
        u, v = np.meshgrid(axis, axis)
        inside = u*u+v*v < 1
        z = np.sqrt(np.maximum(1-u*u-v*v, 0))
        for incident in [(0., 0., 1.), (.8, 0., .6), (.1, .85, np.sqrt(.2675))]:
            for texture in self.profile['textures']:
                f = brdf(texture, incident, (u, v, z), (1., 0.), 550, 740)
                energy = np.sum(f[inside])*(2/501)**2
                self.assertGreaterEqual(energy, 0)
                self.assertLessEqual(energy, sum(texture[k] for k in ('diffuse', 'specular', 'diffraction'))+1e-4)

    def test_profile_rejects_energy_gain_and_nonfinite(self):
        p = copy.deepcopy(self.profile)
        p['textures'][0]['diffuse'] = 1
        with self.assertRaises(ValueError): validate(p)
        p = copy.deepcopy(self.profile)
        p['textures'][0]['specular_width'] = float('nan')
        with self.assertRaises(ValueError): validate(p)

    def test_physical_forward_model_light_quadrature_convergence(self):
        p = copy.deepcopy(self.profile)
        a = responses(p, 33)
        for view in p['views']:
            view['lights'][0]['samples_per_axis'] = 13
        b = responses(p, 33)
        self.assertTrue(np.isfinite(a).all())
        self.assertGreater(float(a.min()), 0)
        self.assertLess(float(np.max(abs(a-b))/np.max(b)), .003)

    def test_coarse_map_interpolation_convergence(self):
        coarse = resize_maps(responses(self.profile, 65), 129)
        fine = responses(self.profile, 129)
        self.assertLess(float(np.max(abs(coarse-fine))/np.max(fine)), .01)

    def test_mixture_recovers_target_and_handles_inverted_texture_order(self):
        p = np.linspace(0, 1, 144).reshape(12, 12)
        r0 = np.ones((2, 12, 12))*np.array([.1, .6])[:, None, None]
        r1 = np.ones_like(r0)*np.array([.8, .2])[:, None, None]
        got, observable, clipped = solve_fraction(r0, r1, r0+(r1-r0)*p, [.5, .5])
        np.testing.assert_allclose(got, p, atol=1e-14)
        self.assertTrue(observable.all())
        # Tiny numerical boundary errors may mark an endpoint clipped.
        self.assertFalse(clipped[1:-1, 1:-1].any())

    def test_opposed_views_cannot_magically_both_match_white(self):
        r0 = np.array([0., 1.])[:, None, None]
        r1 = 1-r0
        got, _, _ = solve_fraction(r0, r1, np.ones_like(r0), [.5, .5])
        self.assertAlmostEqual(float(got[0, 0]), .5)
        self.assertGreater(float(np.mean((r0+(r1-r0)*got-1)**2)), .2)

    def test_no_contrast_is_unobservable(self):
        r = np.ones((2, 8, 8))*.2
        got, observable, _ = solve_fraction(r, r, r, [.5, .5])
        self.assertFalse(observable.any())
        np.testing.assert_array_equal(got, .5)

    def test_flat_tone_is_deterministic_and_reduces_low_frequency_noise(self):
        shape = (96, 96)
        p = np.full(shape, .37)
        r0 = np.full((2, *shape), .1)
        r1 = np.full((2, *shape), .8)
        mask = np.ones(shape, dtype=bool)
        a = optical_error_diffusion(p, r0, r1, mask, [.5, .5])
        b = optical_error_diffusion(p, r0, r1, mask, [.5, .5])
        np.testing.assert_array_equal(a, b)
        self.assertLess(abs(float(a.mean())-.37), .003)
        random = (np.random.default_rng(42).random(shape) < .37).astype(float)
        error = lambda q: np.mean(gaussian_filter(q.astype(float)-p, 2)[8:-8, 8:-8]**2)
        self.assertLess(error(a), error(random)*.35)

    def test_hole_is_not_written_and_endpoints_remain_binary(self):
        yy, xx = np.mgrid[:32, :32]
        mask = (xx-16)**2+(yy-16)**2 > 25
        r0 = np.zeros((1, 32, 32)); r1 = np.ones_like(r0)
        for value in [0., 1.]:
            out = optical_error_diffusion(np.full(mask.shape, value), r0, r1, mask, [1.])
            np.testing.assert_array_equal(out[mask], int(value))
            np.testing.assert_array_equal(out[~mask], 0)


if __name__ == '__main__':
    unittest.main()
