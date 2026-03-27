"""Compute functions for simple flux-compression ITG proxies."""

from desc.backend import jax, jnp

from ..utils import cross, dot, safediv
from .data_index import register_compute_fun

_FLUX_COMPRESSION_K_DEFAULT = None
_FLUX_COMPRESSION_SURFACE_AVERAGE_DEFAULT = "clebsch"


def _get_flux_compression_surface_average_weight_key(surface_average):
    """Return the weight quantity used for flux-compression surface averages."""
    surface_average = (
        _FLUX_COMPRESSION_SURFACE_AVERAGE_DEFAULT
        if surface_average is None
        else str(surface_average).lower()
    )
    if surface_average == "clebsch":
        return "sqrt(g)_Clebsch"
    if surface_average == "pest":
        return "sqrt(g)_PEST"
    if surface_average == "desc":
        return "sqrt(g)"
    raise ValueError(
        "Unknown surface_average='{}', expected 'clebsch', 'pest', or 'desc'.".format(
            surface_average
        )
    )


def _heaviside(x, flux_compression_k):
    """Return hard or smooth Heaviside for the flux-compression proxy."""
    if flux_compression_k is None:
        return jnp.heaviside(x, 0.0)
    return jax.nn.sigmoid(flux_compression_k * x)


@register_compute_fun(
    name="flux compression B_cross_kappa_dot_grad_alpha",
    label="\\mathbf{B} \\times \\kappa \\cdot \\nabla \\alpha",
    units="T / m^2",
    units_long="Tesla per square meter",
    description="DESC-native bad-curvature metric used by the simple flux-"
    "compression ITG proxy.",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["|B|", "b", "kappa", "grad(alpha)"],
    public=False,
)
def _flux_compression_b_cross_kappa_dot_grad_alpha(
    params, transforms, profiles, data, **kwargs
):
    B = data["|B|"][:, jnp.newaxis] * data["b"]
    data["flux compression B_cross_kappa_dot_grad_alpha"] = dot(
        cross(B, data["kappa"]), data["grad(alpha)"]
    )
    return data


@register_compute_fun(
    name="flux compression bad-curvature metric",
    label="\\mathrm{cvdrift}",
    units="1 / Wb",
    units_long="Inverse webers",
    description="DESC-native bad-curvature surrogate used in the flux-compression "
    "ITG "
    "proxy. For hard masks this is sign-equivalent to "
    "B x kappa · grad(alpha), so no additional sign(Psi) correction is applied.",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["cvdrift"],
    public=False,
)
def _flux_compression_bad_curvature_metric(
    params, transforms, profiles, data, **kwargs
):
    data["flux compression bad-curvature metric"] = data["cvdrift"]
    return data


@register_compute_fun(
    name="flux compression mask",
    label="H_k(\\mathrm{cvdrift})",
    units="~",
    units_long="dimensionless",
    description="Hard or smooth bad-curvature mask for the flux-compression ITG "
    "proxy.",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["flux compression bad-curvature metric"],
    public=False,
    flux_compression_k="float or None: smoothing parameter for the flux-"
    "compression bad-curvature mask. ``None`` uses a hard Heaviside, otherwise "
    "``sigmoid(flux_compression_k * x)``. "
    f"Default is {_FLUX_COMPRESSION_K_DEFAULT}.",
)
def _flux_compression_mask(params, transforms, profiles, data, **kwargs):
    flux_compression_k = kwargs.get(
        "flux_compression_k", _FLUX_COMPRESSION_K_DEFAULT
    )

    data["flux compression mask"] = _heaviside(
        data["flux compression bad-curvature metric"], flux_compression_k
    )
    return data


@register_compute_fun(
    name="flux compression integrand",
    label="H_k(\\mathrm{cvdrift}) a^2 g^{rr}",
    units="~",
    units_long="dimensionless",
    description="Local flux-compression ITG proxy integrand. The default hard mask "
    "uses cvdrift > 0 as a DESC-native surrogate for bad curvature.",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["flux compression mask", "a", "g^rr"],
    flux_compression_k="float or None: smoothing parameter for the flux-"
    "compression bad-curvature mask. ``None`` uses a hard Heaviside, otherwise "
    "``sigmoid(flux_compression_k * x)``. "
    f"Default is {_FLUX_COMPRESSION_K_DEFAULT}.",
)
def _flux_compression_integrand(params, transforms, profiles, data, **kwargs):
    data["flux compression integrand"] = data["flux compression mask"] * data[
        "a"
    ] ** 2 * data["g^rr"]
    return data


@register_compute_fun(
    name="flux compression proxy",
    label="\\chi_{\\nabla r}",
    units="~",
    units_long="dimensionless",
    description="Flux-compression ITG proxy averaged over a DESC field-line mesh "
    "with a selectable surface-average Jacobian.",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["flux compression integrand", "sqrt(g)", "sqrt(g)_PEST", "sqrt(g)_Clebsch"],
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
    flux_compression_k="float or None: smoothing parameter for the flux-"
    "compression bad-curvature mask. ``None`` uses a hard Heaviside, otherwise "
    "``sigmoid(flux_compression_k * x)``. "
    f"Default is {_FLUX_COMPRESSION_K_DEFAULT}.",
    surface_average="str: Jacobian used in the surface average. One of "
    "``'clebsch'`` (DESC-native default), ``'pest'``, or ``'desc'``. "
    f"Default is ``'{_FLUX_COMPRESSION_SURFACE_AVERAGE_DEFAULT}'``.",
)
def _flux_compression_proxy(params, transforms, profiles, data, **kwargs):
    grid = transforms["grid"].source_grid
    weight_key = _get_flux_compression_surface_average_weight_key(
        kwargs.get("surface_average", _FLUX_COMPRESSION_SURFACE_AVERAGE_DEFAULT)
    )
    integrand = grid.meshgrid_reshape(data["flux compression integrand"], "raz")
    weights = jnp.abs(grid.meshgrid_reshape(data[weight_key], "raz"))
    numerator = jnp.mean(integrand * weights, axis=(1, 2))
    denominator = jnp.mean(weights, axis=(1, 2))
    data["flux compression proxy"] = transforms["grid"].expand(
        safediv(numerator, denominator)
    )
    return data
