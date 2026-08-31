"""Tests for the constructed-field target-action QI objective."""

import numpy as np
import pytest

from desc.backend import jax, jnp
from desc.compute._qimetric_target import (
    _all_pairs_residual,
    _construct_target_field,
    _linear_coefficients,
    _linear_interp,
    _qi_target_j_residual,
)
from desc.compute.data_index import data_index
from desc.examples import get
from desc.grid import LinearGrid
from desc.integrals import Bounce2D
from desc.integrals._bounce_utils import _broadcast_for_bounce, bounce_points
from desc.integrals._interp_utils import fourier_pts, polyder_vec
from desc.integrals.quad_utils import chebgauss2
from desc.objectives import QuasiIsodynamic


def _synthetic_target_residual(
    nphi,
    *,
    defect=0.0,
    rho=(1.0,),
    pitch_batch_size=None,
    surf_batch_size=1,
):
    """Evaluate the registered kernel on smooth one-period analytic fields."""
    rho = np.asarray(rho)
    num_theta = 8
    num_zeta = 64
    grid = LinearGrid(
        rho=rho,
        theta=np.linspace(0, 2 * np.pi, num_theta, endpoint=False),
        zeta=np.linspace(0, 2 * np.pi, num_zeta, endpoint=False),
        NFP=1,
        sym=False,
    )
    theta = jnp.broadcast_to(
        fourier_pts(num_theta)[None, :, None],
        (grid.num_rho, num_theta, 2 * num_theta),
    )
    radial, vartheta, zeta = grid.nodes.T
    amplitude = 0.4 + 0.1 * radial
    B = 1.5 + amplitude * jnp.cos(zeta) + defect * 0.08 * jnp.cos(vartheta + 2 * zeta)
    B_rtz = grid.meshgrid_reshape(B, "rtz")
    data = {
        "min_tz |B|": grid.expand(jnp.min(B_rtz, axis=(-2, -1))),
        "max_tz |B|": grid.expand(jnp.max(B_rtz, axis=(-2, -1))),
        # This synthetic choice makes dℓ=dζ and isolates the action construction.
        "B^zeta": B,
        "|B|": B,
        "iota": jnp.zeros_like(B),
    }
    alpha = jnp.array([0.0, jnp.pi / 2])
    knots = jnp.linspace(0, 2 * jnp.pi, nphi)
    B_samples = (
        1.5
        + (0.4 + 0.1 * jnp.asarray(rho))[:, None, None] * jnp.cos(knots)[None, None, :]
        + defect * 0.08 * jnp.cos(alpha[None, :, None] + 2 * knots[None, None, :])
    )
    out = _qi_target_j_residual(
        {},
        {"grid": grid},
        {},
        data,
        theta=theta,
        B_samples=B_samples,
        zeta=knots,
        alpha=alpha,
        Y_B=96,
        num_quad=12,
        num_pitch=7,
        pitch_batch_size=pitch_batch_size,
        surf_batch_size=surf_batch_size,
        nufft_eps=0,
        spline=True,
        quad=chebgauss2(12),
        _vander=None,
    )
    return out["QI target J residual"].reshape(grid.num_rho, 7, 2 * alpha.size + 1)


@pytest.mark.unit
def test_qi_target_registry_contract():
    """The route registers one private residual with physical dependencies."""
    entry = data_index["desc.equilibrium.equilibrium.Equilibrium"][
        "QI target J residual"
    ]
    assert not entry["public"]
    assert entry["dependencies"]["data"] == [
        "min_tz |B|",
        "max_tz |B|",
        *Bounce2D.required_names,
    ]


@pytest.mark.unit
def test_qi_target_squash_stretch_and_piecewise_linear_roots():
    """The target uses affine field values and DESC's low-level root path."""
    knots = jnp.linspace(0, 2, 5)
    B = jnp.array([[[2.0, 1.5, 1.0, 1.5, 2.0], [3.0, 2.0, 1.0, 2.0, 3.0]]])
    B_C = _construct_target_field(B, jnp.array([1.0]), jnp.array([3.0]))
    expected = jnp.array([[[3.0, 2.0, 1.0, 2.0, 3.0]] * 2])
    np.testing.assert_allclose(B_C, expected)

    coefficients = _linear_coefficients(knots, B_C)
    query = jnp.broadcast_to(knots[None, None], B_C.shape)
    np.testing.assert_allclose(_linear_interp(query, knots, coefficients), B_C)
    pitch_inv = jnp.array([[1.5, 2.5]])
    z1, z2 = bounce_points(
        _broadcast_for_bounce(pitch_inv),
        knots,
        coefficients,
        polyder_vec(coefficients),
        1,
    )
    np.testing.assert_allclose(z1[0, :, :, 0], [[0.75, 0.25]] * 2)
    np.testing.assert_allclose(z2[0, :, :, 0], [[1.25, 1.75]] * 2)

    plateau_knots = jnp.linspace(0, 1, 7)
    plateau = jnp.array([[[3.0, 2.0, 2.0, 1.0, 2.0, 2.0, 3.0]]])
    plateau_coefficients = _linear_coefficients(plateau_knots, plateau)
    z1, z2 = bounce_points(
        _broadcast_for_bounce(jnp.array([[2.0]])),
        plateau_knots,
        plateau_coefficients,
        polyder_vec(plateau_coefficients),
        1,
    )
    np.testing.assert_allclose(z1.item(), 1 / 3)
    np.testing.assert_allclose(z2.item(), 2 / 3)


@pytest.mark.unit
def test_qi_target_factored_all_pairs_matches_explicit_pairs():
    """Variance blocks reproduce the explicit alpha-pair objective."""
    J_I = jnp.array([[[1.0, 2.0, 3.0], [2.0, 4.0, 7.0]]])
    J_C = jnp.array([[[2.0, 2.0, 2.0], [1.0, 4.0, 8.0]]])
    weight = jnp.array([[1.0, 3.0]])
    residual = _all_pairs_residual(J_I, J_C, weight)
    mean_action = np.mean(np.asarray(J_I + J_C))
    explicit = 0.0
    for k in range(J_I.shape[1]):
        pair = np.asarray(J_I[0, k])[:, None] - np.asarray(J_C[0, k])[None, :]
        explicit += float(weight[0, k] / weight.sum()) * np.mean(pair**2)
    np.testing.assert_allclose(
        np.sum(np.asarray(residual) ** 2), explicit / mean_action**2
    )

    same_but_alpha_dependent = _all_pairs_residual(J_I, J_I, weight)
    assert np.linalg.norm(same_but_alpha_dependent) > 0
    padded = _all_pairs_residual(J_I.at[:, 0].set(0), J_C.at[:, 0].set(0), weight)
    assert np.isfinite(padded).all()
    zero_mean_action = _all_pairs_residual(
        -jnp.ones((1, 1, 2)), jnp.ones((1, 1, 2)), jnp.ones((1, 1))
    )
    assert np.isfinite(zero_mean_action).all()
    assert np.linalg.norm(zero_mean_action) > 0


@pytest.mark.unit
def test_qi_target_exact_well_converges_and_topology_defect_is_nonzero():
    """A represented target converges to zero while extra trapping is penalized."""
    coarse = _synthetic_target_residual(17)
    fine = _synthetic_target_residual(65)
    defect = _synthetic_target_residual(65, defect=1.0)
    assert np.linalg.norm(fine) < 0.1 * np.linalg.norm(coarse)
    assert np.linalg.norm(fine) < 4e-4
    assert np.linalg.norm(defect) > 0.1
    assert np.isfinite(defect).all()


@pytest.mark.unit
def test_qi_target_pitch_and_surface_batching():
    """Pitch and surface chunks preserve the residual axis order."""
    unbatched_pitch = _synthetic_target_residual(33, defect=1.0)
    batched_pitch = _synthetic_target_residual(33, defect=1.0, pitch_batch_size=3)
    np.testing.assert_allclose(batched_pitch, unbatched_pitch, rtol=2e-12, atol=2e-12)

    together = _synthetic_target_residual(
        33, defect=1.0, rho=(0.7, 1.0), surf_batch_size=None
    )
    batched = _synthetic_target_residual(
        33, defect=1.0, rho=(0.7, 1.0), surf_batch_size=1
    )
    assert together.shape == (2, 7, 5)
    np.testing.assert_allclose(batched, together, rtol=2e-12, atol=2e-12)


@pytest.mark.unit
@pytest.mark.parametrize(
    "well",
    [
        np.arange(9.0),
        np.array([3.0, 2.5, 2.0, 1.5, 1.0, 1.5, 2.0, 1.0, 1.0]),
    ],
)
def test_qi_target_boundary_minimum_fails_build(monkeypatch, well):
    """An initial sampled field with a boundary minimum is rejected explicitly."""
    eq = get("DSHAPE")
    grid = LinearGrid(rho=np.array([0.8]), M=4, N=4, NFP=eq.NFP, sym=False)
    objective = QuasiIsodynamic(
        eq,
        grid=grid,
        X=4,
        Y=6,
        Y_B=8,
        alpha=np.array([3.0, 3.5]),
        nphi=9,
        num_quad=4,
        num_pitch=5,
        nufft_eps=0,
    )
    boundary_minimum = jnp.broadcast_to(jnp.asarray(well)[None, None], (1, 2, 9))
    monkeypatch.setattr(
        objective,
        "_sample_field_strength",
        lambda params, data, constants: boundary_minimum,
    )
    with pytest.raises(ValueError, match="interior"):
        objective.build(use_jit=False, verbose=0)


@pytest.mark.unit
def test_qi_target_objective_ad_matches_centered_difference():
    """The complete target path has finite forward and reverse derivatives."""
    eq = get("DSHAPE")
    grid = LinearGrid(rho=np.array([0.8]), M=4, N=4, NFP=eq.NFP, sym=False)
    objective = QuasiIsodynamic(
        eq,
        grid=grid,
        X=4,
        Y=6,
        Y_B=8,
        alpha=np.array([3.0, 3.5]),
        nphi=21,
        num_quad=4,
        num_pitch=5,
        nufft_eps=0,
    )
    objective.build(use_jit=True, verbose=0)
    params = eq.params_dict
    tangent = jax.tree.map(
        lambda x: (
            jnp.linspace(-1, 1, x.size).reshape(x.shape) * 1e-4
            if x.size
            else jnp.zeros_like(x)
        ),
        params,
    )
    _, jvp = jax.jvp(lambda x: objective.compute(x), (params,), (tangent,))
    step = 1e-3
    plus = jax.tree.map(lambda x, dx: x + step * dx, params, tangent)
    minus = jax.tree.map(lambda x, dx: x - step * dx, params, tangent)
    finite_difference = (objective.compute(plus) - objective.compute(minus)) / (
        2 * step
    )
    assert np.isfinite(jvp).all()
    assert np.isfinite(finite_difference).all()
    np.testing.assert_allclose(jvp, finite_difference, rtol=5e-4, atol=5e-9)

    gradient = jax.grad(lambda x: 0.5 * jnp.sum(objective.compute(x) ** 2))(params)
    assert jax.tree.all(jax.tree.map(lambda x: jnp.isfinite(x).all(), gradient))


@pytest.mark.unit
def test_qi_target_objective_nfp4_shape_and_jit():
    """The public objective follows one NFP=4 period and advertises its shape."""
    eq = get("WISTELL-A")
    grid = LinearGrid(rho=np.array([0.8]), M=4, N=4, NFP=eq.NFP, sym=False)
    alpha = np.array([1.71, 2.25])
    objective = QuasiIsodynamic(
        eq,
        grid=grid,
        X=4,
        Y=8,
        Y_B=16,
        alpha=alpha,
        nphi=33,
        num_quad=4,
        num_pitch=5,
        nufft_eps=0,
    )
    objective.build(use_jit=True, verbose=0)
    residual = objective.compute_scaled(*objective.xs(eq))
    assert residual.shape == (5 * (2 * alpha.size + 1),)
    assert residual.shape == (objective.dim_f,)
    assert np.isfinite(residual).all()
