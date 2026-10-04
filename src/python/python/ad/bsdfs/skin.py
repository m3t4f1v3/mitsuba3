from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

from . import skin_tables

# Tabulated directional albedos of `roughdielectric`, by relative index of refraction
_TABLES = {}

def tabulate(eta=1.4, per_mu=1 << 18):
    """
    Directional albedo of the GGX ``roughdielectric`` at relative index ``eta``:
    ``(alphas, mus, reflected, transmitted)`` with the last two indexed
    ``[side (0 outside, 1 inside)][alpha][mu]``, in energy (importance) units.
    """
    import numpy as np
    alphas, mus = np.array(skin_tables.ALPHAS), np.array(skin_tables.MUS)
    ctx = mi.BSDFContext(mi.TransportMode.Importance)
    reflected = np.zeros((2, len(alphas), len(mus)))
    transmitted = np.zeros_like(reflected)
    mu = np.repeat(np.maximum(mus, 1e-3), per_mu)
    sampler = mi.load_dict({'type': 'independent'})
    for a, alpha in enumerate(alphas):
        bsdf = mi.load_dict({'type': 'roughdielectric', 'distribution': 'ggx',
                             'alpha': float(alpha), 'int_ior': eta, 'ext_ior': 1.0})
        for side in (0, 1):
            s = 1.0 if side == 0 else -1.0
            sampler.seed(2 * a + side, len(mu))
            si = dr.zeros(mi.SurfaceInteraction3f, len(mu))
            si.sh_frame = mi.Frame3f(mi.Vector3f(0, 0, 1))
            si.n = mi.Vector3f(0, 0, 1)
            si.wi = mi.Vector3f(mi.Float(np.sqrt(1 - mu * mu)), 0.0, mi.Float(s * mu))
            bs, w = bsdf.sample(ctx, si, sampler.next_1d(), sampler.next_2d())
            w = np.nan_to_num(np.array(w[0]), nan=0.0, posinf=0.0).reshape(len(mus), per_mu)
            same = (np.array(bs.wo.z) * s > 0).reshape(len(mus), per_mu)
            reflected[side, a] = np.where(same, w, 0.0).mean(1)
            transmitted[side, a] = np.where(same, 0.0, w).mean(1)
    return alphas, mus, reflected, transmitted

def tables(eta):
    """The tabulated albedos for ``eta``: shipped for 1.4, computed once otherwise."""
    import numpy as np
    key = round(float(eta), 6)
    if key not in _TABLES:
        if abs(key - skin_tables.ETA) < 1e-6:
            _TABLES[key] = (np.array(skin_tables.ALPHAS), np.array(skin_tables.MUS),
                            np.array(skin_tables.REFLECTED), np.array(skin_tables.TRANSMITTED))
        else:
            _TABLES[key] = tabulate(key)
    return _TABLES[key]


class _RoughBoundary(mi.BSDF):
    """A GGX ``roughdielectric`` and the energy bookkeeping of its tabulated albedos."""

    def __init__(self, props):
        mi.BSDF.__init__(self, props)
        import numpy as np
        self.eta = props.get('eta', 1.4)
        self.alpha = props.get_texture('alpha', 0.3)
        self.inner = mi.load_dict({'type': 'roughdielectric', 'distribution': 'ggx',
                                   'alpha': self.alpha, 'int_ior': self.eta, 'ext_ior': 1.0})
        alphas, mus, reflected, transmitted = tables(self.eta)
        lost = np.clip(1.0 - reflected - transmitted, 0.0, 1.0)
        mean = lambda x: np.trapezoid(x * 2 * mus, mus, axis=-1)  # cosine-weighted
        G = np.maximum(mean(lost), 1e-6)  # [side, alpha]
        eta2 = self.eta ** 2
        # The share of the dropped energy that is transmitted: the single-scattered share
        # from outside, as far as reciprocity leaves the inside's share below one
        share = mean(transmitted[0]) / np.maximum(mean(reflected[0] + transmitted[0]), 1e-6)
        tau_out = np.clip(np.minimum(share, 0.999 * eta2 * G[1] / G[0]), 0.0, 1.0)
        tau = np.stack([tau_out, tau_out * G[0] / (eta2 * G[1])])

        self.n_a, self.n_m = len(alphas), len(mus)
        self.log_a0 = float(np.log(alphas[0]))
        self.log_da = float(np.log(alphas[1] / alphas[0]))
        self.np_tables = (mus, reflected, transmitted, lost, G, tau)

    @staticmethod
    def flat(x):
        import numpy as np
        return mi.Float(np.ascontiguousarray(x, dtype=np.float32).ravel())

    def coords(self, si, active):
        """Position on the roughness axis: (index, fraction)"""
        alpha = self.alpha.eval_1(si, active)
        fa = dr.clip((dr.log(dr.maximum(alpha, 1e-4)) - self.log_a0) / self.log_da,
                     0, self.n_a - 1.001)
        ia = mi.UInt32(fa)
        return ia, fa - mi.Float(ia)

    def table(self, values, ia, ta, mu_signed, active):
        """Bilinear lookup of a [side, alpha, mu] table; the sign of mu picks the side"""
        side = mi.UInt32(dr.select(mu_signed >= 0, 0, 1))
        fm = dr.clip(dr.abs(mu_signed) * (self.n_m - 1), 0, self.n_m - 1.001)
        im = mi.UInt32(fm)
        tm = fm - mi.Float(im)
        base = side * (self.n_a * self.n_m)
        at = lambda a, m: dr.gather(mi.Float, values, base + a * self.n_m + m, active)
        return ((1 - ta) * ((1 - tm) * at(ia, im) + tm * at(ia, im + 1))
                + ta * ((1 - tm) * at(ia + 1, im) + tm * at(ia + 1, im + 1)))

    def coefficient(self, values, k, ia, ta, active):
        """Linear lookup of row k of a [k, alpha] table"""
        return ((1 - ta) * dr.gather(mi.Float, values, k * self.n_a + ia, active)
                + ta * dr.gather(mi.Float, values, k * self.n_a + ia + 1, active))

    def eval(self, ctx, si, wo, active=True):
        return self.eval_pdf(ctx, si, wo, active)[0]

    def pdf(self, ctx, si, wo, active=True):
        return self.eval_pdf(ctx, si, wo, active)[1]

    def traverse(self, callback):
        callback.put('eta', self.eta, mi.ParamFlags.NonDifferentiable)
        callback.put('alpha', self.alpha, mi.ParamFlags.NonDifferentiable)


class SkinDielectric(_RoughBoundary):
    r"""
    .. _bsdf-skindielectric:

    Energy-conserving rough dielectric (:monosp:`skindielectric`)
    -------------------------------------------------------------

    .. pluginparameters::

     * - eta
       - |float|
       - Relative index of refraction of the interior. The energy tables for 1.4 (skin)
         are shipped; any other value is tabulated when first used. (Default: 1.4)

     * - alpha
       - |texture| or |float|
       - GGX roughness. (Default: 0.3)
       - |exposed|

    A :ref:`roughdielectric <bsdf-roughdielectric>` that does not lose light. GGX with Smith
    masking counts only light that leaves a microfacet after one bounce; what meets a second
    facet is dropped. Seen from outside that is a few per cent at moderate roughness, but
    light scattered back up through a medium meets the boundary from inside, where most of
    it is totally internally reflected, and meets it several times before it escapes: a
    slab of skin under ``roughdielectric`` returns 10-25 % too little, and a non-absorbing
    sphere under a uniform sky renders at 0.58-0.98 instead of 1.

    Following Kulla and Conty, the model's own directional albedo is tabulated on each side
    and the missing energy is handed back as a cosine lobe, split between reflection and
    transmission the way the single-scattered light splits. With :math:`g_s(\mu)` the
    energy lost on side :math:`s`:

    .. math::

        f_{ss} = a_s \frac{g_s(\mu_i)\, g_s(\mu_o)}{\pi}, \qquad
        f_{\mathrm{out}\to\mathrm{in}} = b \frac{g_\mathrm{out}(\mu_l)\, g_\mathrm{in}(\mu_v)}{\pi}, \qquad
        f_{\mathrm{in}\to\mathrm{out}} = f_{\mathrm{out}\to\mathrm{in}} / \eta^2

    where energy balance from either side fixes :math:`a_\mathrm{out}`,
    :math:`a_\mathrm{in}` and :math:`b`, and the :math:`\eta^2` is generalized reciprocity
    (a lobe that merely conserves energy renders the white furnace at 1.15).

    The lobe is implemented for radiance transport only.

    .. tabs::

        .. code-tab:: python

            'type': 'skindielectric',
            'alpha': 0.3
    """

    def __init__(self, props):
        super().__init__(props)
        import numpy as np
        mus, reflected, transmitted, lost, G, tau = self.np_tables
        self.g = self.flat(lost)
        # a_out, a_in, b, and the share of each side's lobe that is reflected
        self.coef = self.flat(np.stack([(1 - tau[0]) / G[0], (1 - tau[1]) / G[1], tau[0] / G[1],
                                        1 - tau[0], 1 - tau[1]]))
        both = mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        self.m_components = [mi.BSDFFlags.GlossyReflection | both,
                             mi.BSDFFlags.GlossyTransmission | both | mi.BSDFFlags.NonSymmetric]
        self.m_flags = self.m_components[0] | self.m_components[1]

    def comp(self, si, wo, active):
        """The compensation lobe: (f |cos wo|, its own pdf, probability of picking it,
        its reflected share)"""
        ia, ta = self.coords(si, active)
        v, l = si.wi.z, wo.z  # light arrives along wo and leaves along wi
        gv = self.table(self.g, ia, ta, v, active)
        gl = self.table(self.g, ia, ta, l, active)
        same = v * l > 0
        a = self.coefficient(self.coef, mi.UInt32(dr.select(v > 0, 0, 1)), ia, ta, active)
        b = self.coefficient(self.coef, mi.UInt32(2), ia, ta, active)
        b = dr.select(l > 0, b, b / (self.eta * self.eta))
        f = dr.select(same, a, b) * gv * gl / dr.pi
        q = self.coefficient(self.coef, mi.UInt32(dr.select(v > 0, 3, 4)), ia, ta, active)
        pdf = dr.select(same, q, 1 - q) * dr.abs(l) / dr.pi
        return f * dr.abs(l), pdf, dr.clip(gv, 0.0, 1.0), q

    def sample(self, ctx, si, sample1, sample2, active=True):
        # One lobe per sample, each weighted by its own density: `roughdielectric.pdf()`
        # disagrees with its `sample()` for some directions seen from inside, while its
        # sample weight is right. `bs.pdf` is the mixture, which keeps the MIS pair with
        # emitter sampling consistent.
        _, _, p, q = self.comp(si, si.wi, active)  # p, q depend on wi only
        pick = sample1 < p
        u = sample1 / dr.maximum(p, 1e-6)
        d = mi.warp.square_to_cosine_hemisphere(sample2)
        side = dr.select(si.wi.z >= 0, 1.0, -1.0)
        d.z = dr.select(u < q, d.z * side, -d.z * side)
        bs_inner, w_inner = self.inner.sample(ctx, si, (sample1 - p) / dr.maximum(1 - p, 1e-6),
                                              sample2, active & ~pick)
        bs = mi.BSDFSample3f(bs_inner)
        bs.wo = dr.select(pick, d, bs_inner.wo)
        f_c, pdf_c, _, _ = self.comp(si, bs.wo, active)
        same = si.wi.z * bs.wo.z > 0
        bs.pdf = (1 - p) * self.inner.pdf(ctx, si, bs.wo, active) + p * pdf_c
        bs.eta = dr.select(same, 1.0, dr.select(si.wi.z > 0, self.eta, 1.0 / self.eta))
        bs.sampled_type = dr.select(same, mi.UInt32(+mi.BSDFFlags.GlossyReflection),
                                    mi.UInt32(+mi.BSDFFlags.GlossyTransmission))
        bs.sampled_component = dr.select(same, mi.UInt32(0), mi.UInt32(1))
        weight = dr.select(pick, f_c / dr.maximum(p * pdf_c, 1e-12),
                           w_inner / dr.maximum(1 - p, 1e-6))
        ok = active & (bs.pdf > 0) & dr.all(dr.isfinite(weight))
        return bs, dr.select(ok, weight, 0.0)

    def eval_pdf(self, ctx, si, wo, active=True):
        f_in, pdf_in = self.inner.eval_pdf(ctx, si, wo, active)
        f_c, pdf_c, p, _ = self.comp(si, wo, active)
        return f_in + f_c, (1 - p) * pdf_in + p * pdf_c

    def to_string(self):
        return f'SkinDielectric[eta = {self.eta}]'


class SkinSurface(_RoughBoundary):
    r"""
    .. _bsdf-skinsurface:

    Rough boundary of a scattering medium, connectable to lights (:monosp:`skinsurface`)
    -------------------------------------------------------------------------------------

    .. pluginparameters::

     * - eta
       - |float|
       - Relative index of refraction of the interior. The energy tables for 1.4 (skin)
         are shipped; any other value is tabulated when first used. (Default: 1.4)

     * - alpha
       - |texture| or |float|
       - GGX roughness of the reflection lobe. (Default: 0.3)
       - |exposed|

     * - tint
       - |texture|
       - Multiplies the light that leaves the medium toward the sensor, at the point
         where it leaves: albedo detail finer than the medium's own resolution.
         (Default: none)
       - |exposed|

     * - gloss
       - |texture|
       - Multiplies the reflection lobe seen from outside, e.g. a cavity map.
         (Default: none)
       - |exposed|

    The boundary of a dense scattering medium, such as skin, for scenes lit by small
    light sources.

    Under a (rough) dielectric boundary, a random walk that reaches the surface from
    inside refracts into one narrow lobe, and only contributes if that lobe happens to
    point at a light: next event estimation is performed at that vertex, but finds
    nothing in a refraction lobe. With a softbox covering 5 % of the hemisphere, skin
    renders with six times the noise of a diffuse surface beside it.

    Light under the boundary of such a medium is all but isotropic, and for isotropic
    light only the boundary's directional transmittance decides what leaves in which
    direction. This plugin therefore replaces the refraction lobe by the separable

    .. math::

        f_{\mathrm{out}\to\mathrm{in}} = \frac{T_\mathrm{out}(\mu_l)\, T_\mathrm{in}(\mu_v)}{\pi\, \bar T_\mathrm{in}},
        \qquad f_{\mathrm{in}\to\mathrm{out}} = f_{\mathrm{out}\to\mathrm{in}} / \eta^2

    where :math:`T` is the tabulated transmittance of the rough boundary on each side
    (single scattering plus its share of the energy the microfacet model drops, as in
    :ref:`skindielectric <bsdf-skindielectric>`) and :math:`\bar T_\mathrm{in}` its
    cosine-weighted mean. The lobe carries exactly the energy the refraction did, at
    every angle from either side, and every light sample lands in it. Reflection keeps
    ``roughdielectric``'s GGX lobe (the highlight, and total internal reflection) and a
    cosine lobe for the reflected share of the dropped energy.

    A slab of skin seen head-on renders within 6 % of the same slab under
    ``skindielectric``, a white furnace at 1.00-1.01, and a face under studio softboxes
    with a third of the noise.

    The lobes are implemented for radiance transport only.

    .. tabs::

        .. code-tab:: python

            'type': 'skinsurface',
            'alpha': 0.3
    """

    # Share of the samples sent into the medium that go inside the refraction cone
    CONE = 0.8

    def __init__(self, props):
        super().__init__(props)
        import numpy as np
        self.tint = props.get('tint', None)
        self.gloss = props.get('gloss', None)
        self.reflection = mi.BSDFContext()
        self.reflection.component = 0  # roughdielectric's reflection alone

        mus, reflected, transmitted, lost, G, tau = self.np_tables
        through = transmitted + tau[..., None] * lost
        self.t_spec, self.t_lost, self.t_through = (self.flat(x) for x in (reflected, lost, through))
        # share of the dropped energy that is transmitted, and the reflected lobe's scale
        self.c_side = self.flat(np.stack([tau[0], tau[1], (1 - tau[0]) / G[0], (1 - tau[1]) / G[1]]))
        # cosine-weighted mean transmittance from inside
        self.c_norm = self.flat(np.maximum(np.trapezoid(through[1] * 2 * mus, mus, axis=-1), 1e-6))
        self.mu_c = float(np.sqrt(1.0 - 1.0 / self.eta ** 2))  # cosine of the critical angle

        both = mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        self.m_components = [mi.BSDFFlags.GlossyReflection | both,
                             mi.BSDFFlags.DiffuseReflection | both,
                             mi.BSDFFlags.DiffuseTransmission | both | mi.BSDFFlags.NonSymmetric]
        self.m_flags = self.m_components[0] | self.m_components[1] | self.m_components[2]

    def shares(self, ia, ta, v, active):
        """How light arriving at cosine v splits: GGX reflection, cosine reflection lobe,
        transmission"""
        spec = dr.clip(self.table(self.t_spec, ia, ta, v, active), 0.0, 1.0)
        tau = self.coefficient(self.c_side, mi.UInt32(dr.select(v >= 0, 0, 1)), ia, ta, active)
        back = dr.clip((1 - tau) * self.table(self.t_lost, ia, ta, v, active), 0.0, 1 - spec)
        return spec, back, dr.maximum(1 - spec - back, 0.0)

    def inward_pdf(self, mu):
        """Directions into the medium: mostly inside the cone refraction would have filled"""
        cone = dr.select(mu > self.mu_c, self.CONE / (1.0 - self.mu_c * self.mu_c), 0.0)
        return (cone + (1 - self.CONE)) * mu / dr.pi

    def lobes(self, si, wo, active):
        """The two cosine lobes toward wo: (f |cos wo|, pdf within the mixture)"""
        ia, ta = self.coords(si, active)
        v, l = si.wi.z, wo.z  # light arrives along wo and leaves along wi
        same = v * l > 0
        _, back, through = self.shares(ia, ta, v, active)
        a = self.coefficient(self.c_side, mi.UInt32(dr.select(v >= 0, 2, 3)), ia, ta, active)
        f_back = (a * self.table(self.t_lost, ia, ta, v, active)
                  * self.table(self.t_lost, ia, ta, l, active))
        f_through = (self.table(self.t_through, ia, ta, v, active)
                     * self.table(self.t_through, ia, ta, l, active)
                     / self.coefficient(self.c_norm, mi.UInt32(0), ia, ta, active)
                     * dr.select(l > 0, 1.0, 1.0 / (self.eta * self.eta)))
        if self.tint is not None:  # the light leaving toward the sensor
            f_through *= dr.select((v > 0) & (l < 0), self.tint.eval_1(si, active), 1.0)
        f = dr.select(same, f_back, f_through) * dr.abs(l) / dr.pi
        pdf = dr.select(same, back * dr.abs(l) / dr.pi,
                        through * dr.select(l < 0, self.inward_pdf(-l), l / dr.pi))
        return f, pdf

    def shine(self, si, active):
        """Scale of the reflection lobe: the gloss map, for light arriving from outside"""
        if self.gloss is None:
            return 1.0
        return dr.select(si.wi.z > 0, self.gloss.eval_1(si, active), 1.0)

    def sample(self, ctx, si, sample1, sample2, active=True):
        # One lobe per sample, each weighted by its own density (see SkinDielectric);
        # `bs.pdf` is the mixture
        ia, ta = self.coords(si, active)
        spec, back, _ = self.shares(ia, ta, si.wi.z, active)
        pick_spec = sample1 < spec
        pick_back = ~pick_spec & (sample1 < spec + back)
        side = dr.select(si.wi.z >= 0, 1.0, -1.0)
        d = mi.warp.square_to_cosine_hemisphere(sample2)
        # Into the medium: CONE of the samples inside the refraction cone (a smaller disc)
        u = (sample1 - spec - back) / dr.maximum(1 - spec - back, 1e-6)
        shrink = dr.select(~pick_spec & ~pick_back & (si.wi.z > 0) & (u < self.CONE),
                           dr.sqrt(1.0 - self.mu_c * self.mu_c), 1.0)
        x, y = d.x * shrink, d.y * shrink
        d = mi.Vector3f(x, y, dr.safe_sqrt(1 - x * x - y * y))
        d.z = dr.select(pick_back, d.z * side, -d.z * side)
        bs_inner, w_inner = self.inner.sample(self.reflection, si, sample1, sample2,
                                              active & pick_spec)
        bs = mi.BSDFSample3f(bs_inner)
        bs.wo = dr.select(pick_spec, bs_inner.wo, d)
        f, pdf = self.lobes(si, bs.wo, active)
        same = si.wi.z * bs.wo.z > 0
        bs.pdf = spec * self.inner.pdf(self.reflection, si, bs.wo, active) + pdf
        bs.eta = dr.select(same, 1.0, dr.select(si.wi.z > 0, self.eta, 1.0 / self.eta))
        bs.sampled_type = dr.select(
            pick_spec, mi.UInt32(+mi.BSDFFlags.GlossyReflection),
            dr.select(pick_back, mi.UInt32(+mi.BSDFFlags.DiffuseReflection),
                      mi.UInt32(+mi.BSDFFlags.DiffuseTransmission)))
        bs.sampled_component = dr.select(pick_spec, mi.UInt32(0),
                                         dr.select(pick_back, mi.UInt32(1), mi.UInt32(2)))
        weight = dr.select(pick_spec,
                           w_inner * self.shine(si, active) / dr.maximum(spec, 1e-6),
                           f / dr.maximum(pdf, 1e-12))
        ok = active & (bs.pdf > 0) & dr.all(dr.isfinite(weight)) & (pick_spec | (pdf > 0))
        return bs, dr.select(ok, weight, 0.0)

    def eval_pdf(self, ctx, si, wo, active=True):
        ia, ta = self.coords(si, active)
        spec, _, _ = self.shares(ia, ta, si.wi.z, active)
        f_in, pdf_in = self.inner.eval_pdf(self.reflection, si, wo, active)
        f, pdf = self.lobes(si, wo, active)
        return f_in * self.shine(si, active) + f, spec * pdf_in + pdf

    def traverse(self, callback):
        super().traverse(callback)
        if self.tint is not None:
            callback.put('tint', self.tint, mi.ParamFlags.NonDifferentiable)
        if self.gloss is not None:
            callback.put('gloss', self.gloss, mi.ParamFlags.NonDifferentiable)

    def to_string(self):
        return f'SkinSurface[eta = {self.eta}]'


mi.register_bsdf('skindielectric', lambda props: SkinDielectric(props))
mi.register_bsdf('skinsurface', lambda props: SkinSurface(props))
