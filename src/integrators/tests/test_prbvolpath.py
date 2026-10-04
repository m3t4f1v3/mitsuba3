import pytest
import drjit as dr
import mitsuba as mi
import numpy as np


def furnace(spp, **integrator):
    """A non-absorbing scattering sphere under a uniform sky: every pixel on it
    reads one, after walks of hundreds of scattering events."""
    res = 32
    scene = mi.load_dict({
        'type': 'scene',
        'integrator': {'type': 'prbvolpath', 'max_depth': -1, **integrator},
        'sky': {'type': 'constant', 'radiance': 1.0},
        'sensor': {
            'type': 'perspective',
            'fov': 30,
            'to_world': mi.ScalarTransform4f().look_at(origin=[0, -5, 0],
                                                       target=[0, 0, 0],
                                                       up=[0, 0, 1]),
            'film': {'type': 'hdrfilm', 'width': res, 'height': res,
                     'pixel_format': 'luminance',
                     'rfilter': {'type': 'box'}},
            'sampler': {'type': 'independent', 'sample_count': spp},
        },
        'ball': {
            'type': 'sphere',
            'bsdf': {'type': 'dielectric', 'int_ior': 1.4, 'ext_ior': 1.0},
            'interior': {'type': 'homogeneous', 'albedo': 1.0, 'sigma_t': 20.0,
                         'phase': {'type': 'isotropic'},
                         'sample_emitters': False},
        },
    })
    image = np.array(mi.render(scene, seed=0))
    lo, hi = res * 3 // 8, res * 5 // 8
    center = image[lo:hi, lo:hi]
    return float(center.mean()), float(center.std())


def test01_rr_threshold(variants_vec_backends_once_rgb):
    default_mean, default_noise = furnace(256, rr_depth=4)
    mean, noise = furnace(256, rr_depth=4, rr_threshold=0.3)

    # Same image ...
    assert abs(default_mean - 1.0) < 0.06
    assert abs(mean - 1.0) < 0.04
    # ... without the survivors of hundreds of roulettes
    assert noise < 0.7 * default_noise


def test02_rr_threshold_default(variants_vec_backends_once_rgb):
    integrator = mi.load_dict({'type': 'prbvolpath'})
    assert integrator.rr_threshold == 0.0
    assert 'rr_threshold = 0.3' in str(mi.load_dict({'type': 'prbvolpath',
                                                     'rr_threshold': 0.3}))
