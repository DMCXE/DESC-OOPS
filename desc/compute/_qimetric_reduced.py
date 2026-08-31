"""Compute functions for a reduced quasi-isodynamicity objective."""

from functools import partial

from desc.backend import jit, jnp

from ..batching import batch_map
from ..integrals.bounce_integral import Bounce2D
from ..integrals.quad_utils import chebgauss2
from ._neoclassical import _bounce_doc, _compute
from .data_index import register_compute_fun


def _second_adiabatic_invariant(data, B, pitch):
    """Integrand of J with the parallel-speed normalization omitted."""
    return jnp.sqrt(jnp.abs(1 - pitch * B))


@register_compute_fun(
    name="J alpha residual",
    label="J(\\alpha, B^*) - \\langle J(B^*) \\rangle_\\alpha",
    units="m",
    units_long="meters",
    description="Field-line variation of each individual-well second adiabatic "
    "invariant.",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["min_tz |B|", "max_tz |B|"] + Bounce2D.required_names,
    resolution_requirement="tz",
    grid_requirement={"can_fft2": True},
    public=False,
    **_bounce_doc,
)
@partial(
    jit,
    static_argnames=[
        "Y_B",
        "num_transit",
        "num_well",
        "num_quad",
        "num_pitch",
        "pitch_batch_size",
        "surf_batch_size",
        "nufft_eps",
        "spline",
    ],
)
def _J_alpha_residual(params, transforms, profiles, data, **kwargs):
    """Compute the per-well residual J(α,B*) - mean_α J(B*).

    This is a reduced omnigenity condition. It does not impose the magnetic-well
    topology, common extrema, or straight maximum-|B| contours required for full
    quasi-isodynamicity.
    """
    # noqa: unused dependency
    theta = kwargs["theta"]
    grid = transforms["grid"]
    Y_B = kwargs.get("Y_B", theta.shape[-1] * 2)
    alpha = kwargs.get("alpha", 2 * jnp.pi * jnp.arange(16, dtype=theta.dtype) / 16)
    num_transit = kwargs.get("num_transit", 1)
    num_well = kwargs.get("num_well", grid.NFP)
    num_pitch = kwargs.get("num_pitch", 51)
    pitch_batch_size = kwargs.get("pitch_batch_size", None)
    surf_batch_size = kwargs.get("surf_batch_size", 1)
    assert (
        surf_batch_size == 1 or pitch_batch_size is None
    ), f"Expected pitch_batch_size to be None, got {pitch_batch_size}."
    quad = (
        kwargs["quad"] if "quad" in kwargs else chebgauss2(kwargs.get("num_quad", 32))
    )
    nufft_eps = kwargs.get("nufft_eps", 1e-6)
    spline = kwargs.get("spline", True)
    vander = kwargs.get("_vander", None)

    def reduced_residual(data):
        bounce = Bounce2D(
            grid,
            data,
            data["theta"],
            Y_B,
            alpha,
            num_transit,
            quad,
            nufft_eps=nufft_eps,
            is_fourier=True,
            spline=spline,
            vander=vander,
        )

        def integrate_pitch(pitch_by_surface):
            # batch_map chunks the leading pitch axis. Bounce2D expects
            # (num surface, num pitch), so transpose before finding roots.
            pitch_inv = pitch_by_surface.T
            points = bounce.points(pitch_inv, num_well)
            J = bounce.integrate(
                _second_adiabatic_invariant,
                pitch_inv,
                points=points,
                nufft_eps=nufft_eps,
                is_fourier=True,
            )
            # Bounce2D pads absent wells with the degenerate interval (0, 0).
            # Its primal integral is zero there, but masking also gives that padding
            # an exact zero tangent before taking the mean over field lines.
            J = jnp.where(points[1] > points[0], J, 0.0)
            residual = J - jnp.mean(J, axis=1, keepdims=True)
            # Keep the batched pitch axis leading so batch_map concatenates it.
            return jnp.moveaxis(residual, 2, 0)

        residual = batch_map(integrate_pitch, data["pitch_inv"].T, pitch_batch_size)
        residual = jnp.moveaxis(residual, 0, 1)
        pitch_weight = data["pitch_inv weight"]
        pitch_weight = pitch_weight / jnp.sum(pitch_weight, axis=-1, keepdims=True)
        return (
            residual * jnp.sqrt(pitch_weight)[:, :, None, None] / jnp.sqrt(alpha.size)
        )

    data["J alpha residual"] = _compute(
        reduced_residual,
        {},
        data,
        theta,
        grid,
        num_pitch,
        surf_batch_size,
        simp=True,
        expand_out=False,
    ).reshape(-1)
    return data
