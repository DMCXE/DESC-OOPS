"""Tests for the reduced, bounce-action quasi-isodynamicity objective."""

import numpy as np
import pytest

from desc.backend import jax, jnp
from desc.compute._qimetric_reduced import _J_alpha_residual
from desc.compute.data_index import data_index
from desc.examples import get
from desc.grid import LinearGrid
from desc.integrals import Bounce2D
from desc.integrals._interp_utils import fourier_pts
from desc.integrals.quad_utils import chebgauss2
from desc.objectives import QuasiIsodynamicReduced


def _synthetic_grid(rho=(1.0,), num_theta=8, num_zeta=32):
    """Return a tensor grid and the exact iota=0 Clebsch theta map."""
    grid = LinearGrid(
        rho=np.asarray(rho),
        theta=np.linspace(0, 2 * np.pi, num_theta, endpoint=False),
        zeta=np.linspace(0, 2 * np.pi, num_zeta, endpoint=False),
        NFP=1,
        sym=False,
    )
    theta = jnp.broadcast_to(
        fourier_pts(num_theta)[None, :, None],
        (grid.num_rho, num_theta, 2 * num_theta),
    )
    return grid, theta


def _surface_data(grid, B):
    """Construct only the declared direct dependencies of the residual."""
    B = jnp.asarray(B)
    B_rzt = Bounce2D.reshape(grid, B)
    Bmin = grid.expand(jnp.min(B_rzt, axis=(-2, -1)))
    Bmax = grid.expand(jnp.max(B_rzt, axis=(-2, -1)))
    return {
        "min_tz |B|": Bmin,
        "max_tz |B|": Bmax,
        "B^zeta": B,
        "|B|": B,
        "iota": jnp.zeros_like(B),
    }


def _compute_residual(
    grid,
    theta,
    B,
    *,
    alpha,
    num_well=1,
    num_pitch=5,
    pitch_batch_size=None,
    surf_batch_size=1,
    Y_B=64,
    num_quad=8,
):
    """Evaluate the registered kernel using an analytic field and real Bounce2D."""
    data = _J_alpha_residual(
        {},
        {"grid": grid},
        {},
        _surface_data(grid, B),
        theta=theta,
        alpha=jnp.asarray(alpha),
        Y_B=Y_B,
        num_transit=1,
        num_well=num_well,
        num_quad=num_quad,
        num_pitch=num_pitch,
        pitch_batch_size=pitch_batch_size,
        surf_batch_size=surf_batch_size,
        nufft_eps=0,
        spline=True,
        quad=chebgauss2(num_quad),
    )
    return data["J alpha residual"].reshape(
        grid.num_rho, num_pitch, np.size(alpha), num_well
    )


@pytest.mark.unit
def test_qi_reduced_registry_contract():
    """The route owns one private result and only declares physical inputs."""
    entry = data_index["desc.equilibrium.equilibrium.Equilibrium"]["J alpha residual"]
    assert not entry["public"]
    assert entry["dependencies"]["data"] == [
        "min_tz |B|",
        "max_tz |B|",
        *Bounce2D.required_names,
    ]


@pytest.mark.unit
@pytest.mark.parametrize("alpha", [np.array([]), np.array([0.0]), np.array(0.0)])
def test_qi_reduced_requires_multiple_field_lines(alpha):
    """An alpha-variance objective requires at least two field lines."""
    with pytest.raises(ValueError, match="alpha must have size >= 2"):
        QuasiIsodynamicReduced(get("DSHAPE"), alpha=alpha, nufft_eps=0)


@pytest.mark.unit
@pytest.mark.parametrize("num_pitch", [3, 4])
def test_qi_reduced_requires_valid_simpson_pitch_count(num_pitch):
    """The public output shape must equal the requested Simpson pitch count."""
    with pytest.raises(ValueError, match="odd integer >= 5"):
        QuasiIsodynamicReduced(get("DSHAPE"), num_pitch=num_pitch, nufft_eps=0)


@pytest.mark.unit
def test_qi_reduced_identical_and_different_width_wells_and_pitch_batching():
    """Identical wells vanish, while alpha-dependent widths are detected."""
    grid, theta = _synthetic_grid()
    alpha = np.array([0.0, np.pi / 2])
    vartheta = grid.nodes[:, 1]
    zeta = grid.nodes[:, 2]

    B_same = 1 + 0.3 * (1 + jnp.cos(zeta))
    same = _compute_residual(grid, theta, B_same, alpha=alpha)
    np.testing.assert_allclose(same, 0, atol=2e-11)

    width = 0.11 * (1 + jnp.cos(vartheta))
    B_different = 1 + 0.3 * (1 + jnp.cos(zeta + width * jnp.sin(zeta)))
    unbatched = _compute_residual(grid, theta, B_different, alpha=alpha)
    batched = _compute_residual(
        grid, theta, B_different, alpha=alpha, pitch_batch_size=2
    )
    assert np.linalg.norm(unbatched) > 1e-3
    np.testing.assert_allclose(batched, unbatched, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(unbatched.sum(axis=2), 0, atol=2e-12)


@pytest.mark.unit
def test_qi_reduced_keeps_well_axis_before_alpha_average():
    """Opposite errors in two wells must not cancel before forming residuals."""
    grid, theta = _synthetic_grid(num_theta=16, num_zeta=64)
    alpha = np.array([0.0, np.pi])
    vartheta = grid.nodes[:, 1]
    zeta = grid.nodes[:, 2]
    phase = zeta + vartheta + 0.23
    # A shift by pi swaps two unequal wells. The total action is unchanged, but
    # the two positional well residuals are equal and opposite.
    B = 1 + 0.35 * jnp.cos(2 * phase) + 0.08 * jnp.sin(phase) ** 3
    residual = _compute_residual(
        grid,
        theta,
        B,
        alpha=alpha,
        num_well=2,
        num_pitch=7,
        Y_B=96,
        num_quad=12,
    )
    assert np.linalg.norm(residual) > 5e-2
    np.testing.assert_allclose(residual.sum(axis=-1), 0, atol=2e-8)


@pytest.mark.unit
def test_qi_reduced_surface_batching_and_multisurface_shape():
    """Surface chunks preserve ordering and concatenate to the advertised shape."""
    grid, theta = _synthetic_grid(rho=(0.6, 1.0))
    alpha = np.array([0.0, np.pi / 2])
    rho, vartheta, zeta = grid.nodes.T
    width = (0.04 + 0.08 * rho) * (1 + jnp.cos(vartheta))
    B = 1 + (0.2 + 0.1 * rho) * (1 + jnp.cos(zeta + width * jnp.sin(zeta)))

    together = _compute_residual(grid, theta, B, alpha=alpha, surf_batch_size=None)
    batched = _compute_residual(grid, theta, B, alpha=alpha, surf_batch_size=1)
    assert together.shape == (2, 5, 2, 1)
    assert np.isfinite(together).all()
    np.testing.assert_allclose(batched, together, rtol=2e-12, atol=2e-12)


@pytest.mark.unit
def test_qi_reduced_jvp_matches_centered_difference():
    """Bounce-point and action differentiation has a finite directional JVP."""
    grid, theta = _synthetic_grid()
    alpha = np.array([0.0, np.pi / 2])
    vartheta = grid.nodes[:, 1]
    zeta = grid.nodes[:, 2]
    width = 0.11 * (1 + jnp.cos(vartheta))
    B = (
        1
        + 0.3 * (1 + jnp.cos(zeta + width * jnp.sin(zeta)))
        + 0.015 * (1 + jnp.cos(vartheta + 0.17))
    )
    tangent = 1e-4 * (0.3 + jnp.sin(zeta + 0.2) * jnp.cos(vartheta))

    def fun(B):
        # Request an extra well so Bounce2D's fixed-size output contains padding.
        return _compute_residual(grid, theta, B, alpha=alpha, num_well=2).reshape(-1)

    _, jvp = jax.jvp(fun, (B,), (tangent,))
    step = 1e-3
    finite_difference = (fun(B + step * tangent) - fun(B - step * tangent)) / (2 * step)
    assert np.isfinite(jvp).all()
    assert np.isfinite(finite_difference).all()
    np.testing.assert_allclose(jvp, finite_difference, rtol=2e-2, atol=2e-7)


@pytest.mark.unit
def test_quasi_isodynamic_reduced_objective_nfp4_smoke():
    """The public objective builds and returns its documented NFP=4 shape."""
    eq = get("WISTELL-A")
    grid = LinearGrid(rho=np.array([0.8]), M=4, N=4, NFP=eq.NFP, sym=False)
    alpha = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    objective = QuasiIsodynamicReduced(
        eq,
        grid=grid,
        X=4,
        Y=8,
        Y_B=16,
        alpha=alpha,
        num_well=eq.NFP,
        num_pitch=5,
        num_quad=4,
        nufft_eps=0,
    )
    objective.build(use_jit=True, verbose=0)
    residual = objective.compute(eq.params_dict)
    assert residual.shape == (5 * alpha.size * eq.NFP,)
    assert residual.shape == (objective.dim_f,)
    assert np.isfinite(residual).all()
