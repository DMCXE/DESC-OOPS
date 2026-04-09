"""Compute functions for Goodman-style quasi-isodynamicity."""

from desc.backend import jnp, scan
from desc.batching import vmap_chunked

from .data_index import register_compute_fun


def _project_increasing(x, eps):
    """Project a 1D array onto a strictly increasing sequence."""

    def body(prev, xi):
        nxt = jnp.maximum(prev + eps, xi)
        return nxt, nxt

    _, tail = scan(body, x[0], x[1:])
    return jnp.concatenate([x[:1], tail])


def _cumextrema(x, op):
    """Cumulative extrema with a JAX-safe scan."""

    def body(prev, xi):
        nxt = op(prev, xi)
        return nxt, nxt

    _, tail = scan(body, x[0], x[1:])
    return jnp.concatenate([x[:1], tail])


def _cummin(x):
    """Cumulative minimum with a JAX-safe scan."""
    return _cumextrema(x, jnp.minimum)


def _safe_div(num, den, eps):
    """Safe division with a fixed denominator floor."""
    return num / jnp.where(jnp.abs(den) > eps, den, 1.0)


def _squash_well(B):
    """Apply the Goodman squash step to a sampled field line."""
    nphi = B.size
    idx = jnp.arange(nphi)
    min_idx = jnp.argmin(B)
    left_mask = idx <= min_idx
    right_mask = idx >= min_idx

    left_argmax = jnp.argmax(jnp.where(left_mask, B, -jnp.inf))
    right_argmax = jnp.argmax(jnp.where(right_mask, B, -jnp.inf))

    left_flat = jnp.where(idx < left_argmax, B[left_argmax], B)
    right_flat = jnp.where(idx > right_argmax, B[right_argmax], B)

    left = jnp.where(left_mask, _cummin(left_flat), 0.0)
    right = jnp.where(right_mask, jnp.flip(_cummin(jnp.flip(right_flat))), 0.0)
    squashed = jnp.where(idx < min_idx, left, right)
    return squashed, min_idx


def _stretch_well(B, eps=1e-14):
    """Apply the Goodman stretch step to a squashed, normalized well."""
    squashed, min_idx = _squash_well(B)
    idx = jnp.arange(B.size)
    min_val = squashed[min_idx]

    left = _safe_div(squashed - min_val, squashed[0] - min_val, eps)
    right = _safe_div(squashed - min_val, squashed[-1] - min_val, eps)
    stretched = jnp.where(idx < min_idx, left, right)
    stretched = stretched.at[min_idx].set(0.0)
    stretched = stretched.at[0].set(1.0)
    stretched = stretched.at[-1].set(1.0)
    return stretched, min_idx


def _crossing_points(zeta, B, levels, min_idx, eps):
    """Invert a monotone well at fixed field-strength levels."""
    zeta = jnp.asarray(zeta)
    levels = jnp.asarray(levels)
    idx = jnp.arange(B.size - 1)
    left_mask = idx < min_idx
    right_mask = idx >= min_idx

    z0 = zeta[:-1]
    z1 = zeta[1:]
    B0 = B[:-1]
    B1 = B[1:]
    dB = B1 - B0
    nonflat = jnp.abs(dB) > eps

    levels2d = levels[:, None]
    z = z0 + _safe_div(levels2d - B0, dB, eps) * (z1 - z0)

    left_hits = (
        left_mask[None, :]
        & nonflat[None, :]
        & (B0[None, :] >= levels2d)
        & (levels2d >= B1[None, :])
    )
    right_hits = (
        right_mask[None, :]
        & nonflat[None, :]
        & (B0[None, :] <= levels2d)
        & (levels2d <= B1[None, :])
    )

    left = jnp.max(jnp.where(left_hits, z, -jnp.inf), axis=-1)
    right = jnp.min(jnp.where(right_hits, z, jnp.inf), axis=-1)

    zeta_min = zeta[min_idx]
    left = jnp.where(levels <= eps, zeta_min, left)
    right = jnp.where(levels <= eps, zeta_min, right)
    left = jnp.where(levels >= 1.0 - eps, zeta[0], left)
    right = jnp.where(levels >= 1.0 - eps, zeta[-1], right)
    return left, right


def _project_knots(left, right, zeta0, zeta1, eps):
    """Project shuffled bounce-point knots back to a valid ordered set."""
    left = left[::-1]
    left = left.at[0].set(zeta0)
    left = _project_increasing(left, eps)

    right = right.at[0].set(left[-1])
    right = _project_increasing(right, eps)
    right = right.at[-1].set(zeta1)
    return jnp.concatenate([left, right[1:]])


def _linear_interp_monotone(xq, x, y, eps):
    """Piecewise-linear interpolation on an ordered knot vector."""
    idx = jnp.searchsorted(x, xq, side="right") - 1
    idx = jnp.clip(idx, 0, x.size - 2)
    x0 = x[idx]
    x1 = x[idx + 1]
    y0 = y[idx]
    y1 = y[idx + 1]
    t = _safe_div(xq - x0, x1 - x0, eps)
    return y0 + t * (y1 - y0)


def _evaluate_boozer_fieldline(alpha, zeta, iota, basis, B_mn):
    """Evaluate Boozer harmonics on one straight Boozer field line."""
    theta = alpha + iota * zeta
    nodes = jnp.column_stack([jnp.zeros_like(zeta), theta, zeta])
    return basis.evaluate(nodes) @ B_mn


def _evaluate_boozer_fieldlines(basis, B_mn, iota, alpha, zeta, fieldline_batch_size):
    """Evaluate Boozer harmonics on straight Boozer field lines."""

    def eval_one(alpha0):
        return _evaluate_boozer_fieldline(alpha0, zeta, iota, basis, B_mn)

    return vmap_chunked(eval_one, chunk_size=fieldline_batch_size)(alpha)


def _surface_diagnostics(B, zeta, levels, eps, fieldline_batch_size):
    """Construct Goodman diagnostics on one flux surface."""
    B = jnp.asarray(B)
    zeta = jnp.asarray(zeta)
    levels = jnp.asarray(levels)
    Bmin = jnp.min(B)
    Bmax = jnp.max(B)
    Bnorm = _safe_div(B - Bmin, Bmax - Bmin, eps)

    constructed, min_idx = vmap_chunked(
        lambda well: _stretch_well(well, eps), chunk_size=fieldline_batch_size
    )(Bnorm)
    mismatch = jnp.mean((Bnorm - constructed) ** 2, axis=-1)
    weights = 1.0 / (mismatch + eps)
    weights = weights / jnp.sum(weights)

    left, right = vmap_chunked(
        lambda well, idx: _crossing_points(zeta, well, levels, idx, eps),
        in_axes=(0, 0),
        chunk_size=fieldline_batch_size,
    )(constructed, min_idx)
    bounce_distances = right - left
    mean_bounce = jnp.sum(weights[:, None] * bounce_distances, axis=0)
    shift = 0.5 * (bounce_distances - mean_bounce)

    shuffled_knots = vmap_chunked(
        lambda l, r: _project_knots(l, r, zeta[0], zeta[-1], eps),
        in_axes=(0, 0),
        chunk_size=fieldline_batch_size,
    )(left + shift, right - shift)
    values = jnp.concatenate([levels[::-1], levels[1:]])
    target = vmap_chunked(
        lambda knots: _linear_interp_monotone(zeta, knots, values, eps),
        chunk_size=fieldline_batch_size,
    )(shuffled_knots)
    bounce_points = jnp.concatenate([left[:, ::-1], right[:, 1:]], axis=-1)
    residual = Bnorm - target
    return (
        Bnorm,
        constructed,
        target,
        weights,
        bounce_distances,
        bounce_points,
        shuffled_knots,
        residual,
    )


def _goodman_poloidal_target(B, zeta, levels, eps, fieldline_batch_size=None):
    """Construct the Goodman QI target on a single flux surface."""
    return _surface_diagnostics(B, zeta, levels, eps, fieldline_batch_size)[2]


def _surface_residual(B, zeta, levels, eps, fieldline_batch_size=None):
    """Construct the Goodman residual on a single flux surface."""
    return _surface_diagnostics(B, zeta, levels, eps, fieldline_batch_size)[-1]


def _compute_qimetric_data(transforms, data, alpha, zeta, levels, eps, chunk_sizes):
    """Compute all qimetric diagnostic arrays."""
    grid = transforms["grid"]
    basis = transforms["B"].basis
    B_mn = data["|B|_mn_B"].reshape((grid.num_rho, -1))
    iota = data["iota"][grid.unique_rho_idx]
    fieldline_batch_size, surf_batch_size = chunk_sizes

    def surface_fun(Bmn, iot):
        B = _evaluate_boozer_fieldlines(
            basis, Bmn, iot, alpha, zeta, fieldline_batch_size
        )
        return _surface_diagnostics(B, zeta, levels, eps, fieldline_batch_size)

    return vmap_chunked(
        surface_fun, in_axes=(0, 0), chunk_size=surf_batch_size
    )(B_mn, iota)


_QIMETRIC_DOC = {
    "alpha": "ndarray: Boozer field-line labels used in the qimetric diagnostic.",
    "zeta": "ndarray: Boozer toroidal samples spanning one field period.",
    "levels": "ndarray: Normalized field-strength levels used for the Goodman shuffle.",
    "eps": "float: Regularization used in divisions and ordering projections.",
    "fieldline_batch_size": "int: Number of field lines processed simultaneously.",
    "surf_batch_size": "int: Number of flux surfaces processed simultaneously.",
    "M_booz": "int: Maximum poloidal mode number for Boozer harmonics. Default 2*eq.M",
    "N_booz": "int: Maximum toroidal mode number for Boozer harmonics. Default 2*eq.N",
}


@register_compute_fun(
    name="qimetric residual",
    label="qimetric residual",
    units="~",
    units_long="None",
    description="Goodman quasi-isodynamicity residual on sampled Boozer field lines.",
    dim=1,
    params=[],
    transforms={"B": [[0, 0, 0]], "grid": []},
    profiles=[],
    coordinates="r",
    data=["|B|_mn_B", "iota"],
    public=False,
    **_QIMETRIC_DOC,
)
def _qimetric_residual(params, transforms, profiles, data, **kwargs):
    alpha = kwargs["alpha"]
    zeta = kwargs["zeta"]
    levels = kwargs["levels"]
    eps = kwargs.get("eps", 1e-12)
    fieldline_batch_size = kwargs.get("fieldline_batch_size", None)
    surf_batch_size = kwargs.get("surf_batch_size", None)

    (
        B,
        constructed,
        target,
        weights,
        bounce_distances,
        bounce_points,
        shuffled_knots,
        residual,
    ) = _compute_qimetric_data(
        transforms,
        data,
        alpha,
        zeta,
        levels,
        eps,
        (fieldline_batch_size, surf_batch_size),
    )
    data["qimetric |B|"] = B.reshape(-1)
    data["qimetric constructed |B|"] = constructed.reshape(-1)
    data["qimetric target |B|"] = target.reshape(-1)
    data["qimetric weights"] = weights.reshape(-1)
    data["qimetric bounce distances"] = bounce_distances.reshape(-1)
    data["qimetric bounce points"] = bounce_points.reshape(-1)
    data["qimetric shuffled knots"] = shuffled_knots.reshape(-1)
    data["qimetric residual"] = residual.reshape(-1)
    return data


def _qimetric_passthrough(params, transforms, profiles, data, **kwargs):
    """No-op passthrough for qimetric diagnostics computed by qimetric residual."""
    return data


@register_compute_fun(
    name="qimetric |B|",
    label="qimetric |B|",
    units="~",
    units_long="None",
    description="Normalized Boozer field-line samples used by the qimetric diagnostic.",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["qimetric residual"],
    public=False,
)
def _qimetric_B(params, transforms, profiles, data, **kwargs):
    return _qimetric_passthrough(params, transforms, profiles, data, **kwargs)


@register_compute_fun(
    name="qimetric constructed |B|",
    label="qimetric constructed |B|",
    units="~",
    units_long="None",
    description="Goodman squash-stretch field-line construction.",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["qimetric residual"],
    public=False,
)
def _qimetric_constructed(params, transforms, profiles, data, **kwargs):
    return _qimetric_passthrough(params, transforms, profiles, data, **kwargs)


@register_compute_fun(
    name="qimetric target |B|",
    label="qimetric target |B|",
    units="~",
    units_long="None",
    description="Constructed Goodman quasi-isodynamic target field.",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["qimetric residual"],
    public=False,
)
def _qimetric_target(params, transforms, profiles, data, **kwargs):
    return _qimetric_passthrough(params, transforms, profiles, data, **kwargs)


@register_compute_fun(
    name="qimetric weights",
    label="qimetric weights",
    units="~",
    units_long="None",
    description="Field-line weights used by the Goodman shuffle.",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["qimetric residual"],
    public=False,
)
def _qimetric_weights(params, transforms, profiles, data, **kwargs):
    return _qimetric_passthrough(params, transforms, profiles, data, **kwargs)


@register_compute_fun(
    name="qimetric bounce distances",
    label="qimetric bounce distances",
    units="rad",
    units_long="radians",
    description="Bounce distances of the squash-stretch wells.",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["qimetric residual"],
    public=False,
)
def _qimetric_bounce_distances(params, transforms, profiles, data, **kwargs):
    return _qimetric_passthrough(params, transforms, profiles, data, **kwargs)


@register_compute_fun(
    name="qimetric bounce points",
    label="qimetric bounce points",
    units="rad",
    units_long="radians",
    description="Bounce-point knots before the Goodman shuffle projection.",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["qimetric residual"],
    public=False,
)
def _qimetric_bounce_points(params, transforms, profiles, data, **kwargs):
    return _qimetric_passthrough(params, transforms, profiles, data, **kwargs)


@register_compute_fun(
    name="qimetric shuffled knots",
    label="qimetric shuffled knots",
    units="rad",
    units_long="radians",
    description="Projected knot locations after the Goodman shuffle.",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["qimetric residual"],
    public=False,
)
def _qimetric_shuffled_knots(params, transforms, profiles, data, **kwargs):
    return _qimetric_passthrough(params, transforms, profiles, data, **kwargs)
