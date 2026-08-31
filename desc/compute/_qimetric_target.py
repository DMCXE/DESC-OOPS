"""Compute functions for the target-action quasi-isodynamicity metric."""

from functools import partial

from desc.backend import jit, jnp, vmap

from ..batching import batch_map
from ..integrals._bounce_utils import _broadcast_for_bounce, bounce_points
from ..integrals._interp_utils import polyder_vec
from ..integrals.bounce_integral import Bounce2D
from ..integrals.quad_utils import chebgauss2
from ._neoclassical import _bounce_doc
from ._qimetric import _stretch_well
from .data_index import register_compute_fun


def _construct_target_field(B, min_B, max_B):
    """Squash and affinely stretch sampled wells to common surface extrema."""
    shape = B.shape
    stretched, _ = vmap(lambda x: _stretch_well(x), in_axes=0)(
        B.reshape((-1, shape[-1]))
    )
    stretched = stretched.reshape(shape)
    return min_B[:, None, None] + stretched * (max_B - min_B)[:, None, None]


def _linear_coefficients(knots, values):
    """Return local-power coefficients for a piecewise-linear function."""
    slope = jnp.diff(values, axis=-1) / jnp.diff(knots)
    return jnp.stack([slope, values[..., :-1]], axis=-1)


def _linear_interp(zeta, knots, coefficients):
    """Evaluate per-field-line piecewise-linear polynomials at ``zeta``."""
    shape = zeta.shape
    zeta = zeta.reshape((*zeta.shape[:2], -1))
    idx = jnp.searchsorted(knots, zeta, side="right") - 1
    idx = jnp.clip(idx, 0, knots.size - 2)
    c = jnp.take_along_axis(
        coefficients,
        jnp.broadcast_to(idx[..., None], (*idx.shape, coefficients.shape[-1])),
        axis=-2,
    )
    local_zeta = zeta - knots[idx]
    values = c[..., -2] * local_zeta + c[..., -1]
    return values.reshape(shape)


def _all_pairs_residual(J_I, J_C, pitch_weight):
    """Factor the PRX all-pairs action mismatch without a pair matrix."""
    nalpha = J_I.shape[-1]
    mean_I = jnp.mean(J_I, axis=-1)
    mean_C = jnp.mean(J_C, axis=-1)
    residual = jnp.concatenate(
        [
            (J_I - mean_I[..., None]) / jnp.sqrt(nalpha),
            (J_C - mean_C[..., None]) / jnp.sqrt(nalpha),
            (mean_I - mean_C)[..., None],
        ],
        axis=-1,
    )
    pitch_weight = pitch_weight / jnp.sum(pitch_weight, axis=-1, keepdims=True)
    residual *= jnp.sqrt(pitch_weight)[..., None]
    mean_action = jnp.abs(jnp.mean(J_I + J_C, axis=(-2, -1)))
    mean_action = jnp.maximum(mean_action, jnp.finfo(mean_action.dtype).eps)
    return residual / mean_action[:, None, None]


_TARGET_J_DOC = {
    "B_samples": """jnp.ndarray :
        Shape (num rho, num alpha, nphi). Magnetic-field strength sampled in
        general DESC/Clebsch field-line coordinates over one field period.
        """,
    "zeta": """jnp.ndarray :
        Shape (nphi,). Strictly increasing toroidal samples for ``B_samples``.
        """,
}


@register_compute_fun(
    name="QI target J residual",
    label="r_{\\mathrm{QI},J}",
    units="~",
    units_long="None",
    description="Constructed-field all-pairs quasi-isodynamic action residual.",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["min_tz |B|", "max_tz |B|"] + Bounce2D.required_names,
    resolution_requirement="tz",
    grid_requirement={"can_fft2": True},
    public=False,
    theta=_bounce_doc["theta"],
    Y_B=_bounce_doc["Y_B"],
    alpha=_bounce_doc["alpha"],
    num_quad=_bounce_doc["num_quad"],
    num_pitch=_bounce_doc["num_pitch"],
    pitch_batch_size=_bounce_doc["pitch_batch_size"],
    surf_batch_size=_bounce_doc["surf_batch_size"],
    quad=_bounce_doc["quad"],
    nufft_eps=_bounce_doc["nufft_eps"],
    spline=_bounce_doc["spline"],
    _vander=_bounce_doc["_vander"],
    **_TARGET_J_DOC,
)
@partial(
    jit,
    static_argnames=[
        "Y_B",
        "num_quad",
        "num_pitch",
        "pitch_batch_size",
        "surf_batch_size",
        "nufft_eps",
        "spline",
    ],
)
def _qi_target_j_residual(params, transforms, profiles, data, **kwargs):
    """Compute the target-action QI residual using one real ``Bounce2D``."""
    # noqa: unused dependency
    theta = kwargs["theta"]
    B_samples = kwargs["B_samples"]
    zeta = kwargs["zeta"]
    Y_B = kwargs.get("Y_B", theta.shape[-1] * 2)
    alpha = kwargs.get("alpha", jnp.array([0.0]))
    num_pitch = kwargs.get("num_pitch", 51)
    pitch_batch_size = kwargs.get("pitch_batch_size", None)
    surf_batch_size = kwargs.get("surf_batch_size", 1)
    quad = (
        kwargs["quad"] if "quad" in kwargs else chebgauss2(kwargs.get("num_quad", 32))
    )
    nufft_eps = kwargs.get("nufft_eps", 1e-6)
    spline = kwargs.get("spline", True)
    vander = kwargs.get("_vander", None)

    grid = transforms["grid"]
    fun_data = {
        name: Bounce2D.fourier(Bounce2D.reshape(grid, data[name]))
        for name in Bounce2D.required_names
        if name != "iota"
    }
    fun_data["iota"] = grid.compress(data["iota"])
    fun_data["theta"] = theta
    fun_data["B_samples"] = B_samples
    fun_data["min_B"] = grid.compress(data["min_tz |B|"])
    fun_data["max_B"] = grid.compress(data["max_tz |B|"])
    fun_data["pitch_inv"], fun_data["pitch_weight"] = Bounce2D.get_pitch_inv_quad(
        fun_data["min_B"], fun_data["max_B"], num_pitch
    )

    def surface_fun(data):
        B_C = _construct_target_field(data["B_samples"], data["min_B"], data["max_B"])
        B_C_coeff = _linear_coefficients(zeta, B_C)
        dB_C_dz = polyder_vec(B_C_coeff)
        bounce = Bounce2D(
            grid,
            data,
            data["theta"],
            Y_B,
            alpha,
            1,
            quad,
            nufft_eps=nufft_eps,
            is_fourier=True,
            spline=spline,
            vander=vander,
        )

        def integrate_pitch(pitch_inv):
            pitch_inv = jnp.swapaxes(pitch_inv, 0, 1)
            points = bounce_points(
                _broadcast_for_bounce(pitch_inv),
                zeta,
                B_C_coeff,
                dB_C_dz,
                1,
            )

            def signed_actual(data, B, pitch):
                parallel_energy = 1 - pitch * B
                return jnp.sign(parallel_energy) * jnp.sqrt(jnp.abs(parallel_energy))

            def target(data, B, pitch):
                B_C_quad = _linear_interp(data["zeta"], zeta, B_C_coeff)
                return jnp.sqrt(jnp.abs(1 - pitch * B_C_quad))

            J_I, J_C = bounce.integrate(
                [signed_actual, target],
                pitch_inv,
                points=points,
                nufft_eps=nufft_eps,
                is_fourier=True,
            )
            J_I = jnp.swapaxes(J_I[..., 0], 1, 2)
            J_C = jnp.swapaxes(J_C[..., 0], 1, 2)
            return jnp.swapaxes(J_I, 0, 1), jnp.swapaxes(J_C, 0, 1)

        J_I, J_C = batch_map(
            integrate_pitch,
            jnp.swapaxes(data["pitch_inv"], 0, 1),
            pitch_batch_size,
        )
        J_I = jnp.swapaxes(J_I, 0, 1)
        J_C = jnp.swapaxes(J_C, 0, 1)
        return _all_pairs_residual(J_I, J_C, data["pitch_weight"]).reshape(
            (J_I.shape[0], -1)
        )

    data["QI target J residual"] = batch_map(
        surface_fun, fun_data, surf_batch_size
    ).reshape(-1)
    return data
