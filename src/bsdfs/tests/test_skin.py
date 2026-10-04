import pytest
import drjit as dr
import mitsuba as mi
import numpy as np


def furnace(bsdf, integrator='volpath', res=32, spp=256, **kwargs):
    """A non-absorbing scattering sphere under a uniform sky must vanish: every
    pixel on it reads one."""
    scene = mi.load_dict({
        'type': 'scene',
        'integrator': {'type': integrator, 'max_depth': -1, **kwargs},
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
            'bsdf': bsdf,
            'interior': {'type': 'homogeneous', 'albedo': 1.0, 'sigma_t': 20.0,
                         'phase': {'type': 'isotropic'},
                         'sample_emitters': False},
        },
    })
    image = np.array(mi.render(scene, seed=0))
    # The sphere spans about a third of the frame: average its central part
    lo, hi = res * 3 // 8, res * 5 // 8
    return float(image[lo:hi, lo:hi].mean())


def constant(value):
    return {'type': 'bitmap', 'raw': True,
            'data': dr.full(mi.TensorXf, value, (2, 2, 1))}


def test01_roughdielectric_loses_energy(variants_vec_backends_once_rgb):
    # The loss these plugins exist to restore
    lost = furnace({'type': 'roughdielectric', 'distribution': 'ggx', 'alpha': 0.3,
                    'int_ior': 1.4, 'ext_ior': 1.0}, rr_depth=1000)
    assert lost < 0.85


@pytest.mark.parametrize('plugin', ['skindielectric', 'skinsurface'])
@pytest.mark.parametrize('alpha', [0.05, 0.3])
def test02_white_furnace(variants_vec_backends_once_rgb, plugin, alpha):
    value = furnace({'type': plugin, 'alpha': alpha}, rr_depth=1000)
    assert abs(value - 1.0) < 0.04


def test03_other_index(variants_vec_backends_once_rgb):
    # No shipped table for this index: tabulated on first use
    value = furnace({'type': 'skinsurface', 'eta': 1.33, 'alpha': 0.2}, rr_depth=1000)
    assert abs(value - 1.0) < 0.04


def test04_tint_and_gloss(variants_vec_backends_once_rgb):
    plain = furnace({'type': 'skinsurface', 'alpha': 0.2}, rr_depth=1000)
    # Without the reflection, and with half of what comes out
    tinted = furnace({'type': 'skinsurface', 'alpha': 0.2, 'tint': constant(0.5),
                      'gloss': constant(0.0)}, rr_depth=1000)
    assert 0.40 * plain < tinted < 0.50 * plain


def test05_traverse(variants_vec_backends_once_rgb):
    bsdf = mi.load_dict({'type': 'skinsurface', 'alpha': constant(0.2),
                         'tint': constant(1.0)})
    params = mi.traverse(bsdf)
    assert 'eta' in params and 'alpha.data' in params and 'tint.data' in params
    assert 'gloss.data' not in params

    flags = bsdf.flags()
    assert mi.has_flag(flags, mi.BSDFFlags.Smooth)
    assert mi.has_flag(flags, mi.BSDFFlags.DiffuseTransmission)
    assert mi.has_flag(mi.load_dict({'type': 'skindielectric'}).flags(),
                       mi.BSDFFlags.GlossyTransmission)
