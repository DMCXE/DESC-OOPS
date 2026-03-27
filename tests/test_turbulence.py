"""Tests for turbulence compute functions."""

import numpy as np
import pytest

from desc.examples import get
from desc.grid import Grid


@pytest.mark.unit
def test_flux_compression_b_cross_kappa_dot_grad_alpha_formula():
    """DESC flux-compression bad-curvature metric matches B x kappa · grad(alpha)."""
    eq = get("ESTELL")
    rho = np.array([0.4, 0.8])
    alpha = np.linspace(0, (1 + eq.sym) * np.pi, (1 + eq.sym) * 4, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi / eq.NFP, 12, endpoint=False)
    grid = Grid.create_meshgrid([rho, alpha, zeta], coordinates="raz")

    data = eq.compute(
        [
            "flux compression B_cross_kappa_dot_grad_alpha",
            "|B|",
            "b",
            "kappa",
            "grad(alpha)",
        ],
        grid=grid,
    )
    B = data["|B|"][:, None] * data["b"]
    expected = np.sum(np.cross(B, data["kappa"]) * data["grad(alpha)"], axis=1)

    np.testing.assert_allclose(
        data["flux compression B_cross_kappa_dot_grad_alpha"], expected
    )


@pytest.mark.unit
def test_flux_compression_bad_curvature_metric_uses_cvdrift():
    """Flux-compression bad-curvature metric uses DESC-native cvdrift."""
    eq = get("ESTELL")
    rho = np.array([0.4, 0.8])
    alpha = np.linspace(0, (1 + eq.sym) * np.pi, (1 + eq.sym) * 4, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi / eq.NFP, 12, endpoint=False)
    grid = Grid.create_meshgrid([rho, alpha, zeta], coordinates="raz")

    data = eq.compute(
        ["flux compression bad-curvature metric", "cvdrift"],
        grid=grid,
    )

    np.testing.assert_allclose(
        data["flux compression bad-curvature metric"], data["cvdrift"]
    )


@pytest.mark.unit
def test_flux_compression_cvdrift_mask_matches_b_cross_kappa_sign():
    """Hard bad-curvature masks from cvdrift and B x kappa · grad(alpha) agree."""
    eq = get("ESTELL")
    rho = np.array([0.4, 0.8])
    alpha = np.linspace(0, (1 + eq.sym) * np.pi, (1 + eq.sym) * 4, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi / eq.NFP, 12, endpoint=False)
    grid = Grid.create_meshgrid([rho, alpha, zeta], coordinates="raz")

    data = eq.compute(
        ["cvdrift", "flux compression B_cross_kappa_dot_grad_alpha"],
        grid=grid,
    )

    np.testing.assert_array_equal(
        np.heaviside(data["cvdrift"], 0.0),
        np.heaviside(data["flux compression B_cross_kappa_dot_grad_alpha"], 0.0),
    )


@pytest.mark.unit
def test_flux_compression_metrics_flip_with_psi_sign():
    """Flipping Psi flips both DESC bad-curvature metrics, without extra correction."""
    eq = get("ESTELL")
    rho = np.array([0.4, 0.8])
    alpha = np.linspace(0, (1 + eq.sym) * np.pi, (1 + eq.sym) * 4, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi / eq.NFP, 12, endpoint=False)
    grid = Grid.create_meshgrid([rho, alpha, zeta], coordinates="raz")

    data_pos = eq.compute(
        ["cvdrift", "flux compression B_cross_kappa_dot_grad_alpha"],
        grid=grid,
        params=eq.params_dict,
    )
    params_neg = dict(eq.params_dict)
    params_neg["Psi"] = -params_neg["Psi"]
    data_neg = eq.compute(
        ["cvdrift", "flux compression B_cross_kappa_dot_grad_alpha"],
        grid=grid,
        params=params_neg,
    )

    np.testing.assert_allclose(data_neg["cvdrift"], -data_pos["cvdrift"])
    np.testing.assert_allclose(
        data_neg["flux compression B_cross_kappa_dot_grad_alpha"],
        -data_pos["flux compression B_cross_kappa_dot_grad_alpha"],
    )


@pytest.mark.unit
def test_flux_compression_integrand_formula():
    """Flux-compression ITG integrand matches its defining expression."""
    eq = get("ESTELL")
    rho = np.array([0.4, 0.8])
    alpha = np.linspace(0, (1 + eq.sym) * np.pi, (1 + eq.sym) * 4, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi / eq.NFP, 12, endpoint=False)
    grid = Grid.create_meshgrid([rho, alpha, zeta], coordinates="raz")

    data = eq.compute(
        ["flux compression integrand", "cvdrift", "g^rr", "a"],
        grid=grid,
        flux_compression_k=10.0,
    )
    metric = data["cvdrift"]
    mask = 1.0 / (1.0 + np.exp(-10.0 * metric))
    expected = mask * data["a"] ** 2 * data["g^rr"]

    np.testing.assert_allclose(data["flux compression integrand"], expected)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("surface_average", "weight_key"),
    [
        ("clebsch", "sqrt(g)_Clebsch"),
        ("pest", "sqrt(g)_PEST"),
        ("desc", "sqrt(g)"),
    ],
)
def test_flux_compression_proxy_weighted_average(surface_average, weight_key):
    """Flux-compression ITG proxy matches the selected weighted mean on the raz mesh."""
    eq = get("ESTELL")
    rho = np.array([0.35, 0.75])
    alpha = np.linspace(0, (1 + eq.sym) * np.pi, (1 + eq.sym) * 4, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi / eq.NFP, 16, endpoint=False)
    grid = Grid.create_meshgrid([rho, alpha, zeta], coordinates="raz")

    data = eq.compute(
        [
            "flux compression proxy",
            "flux compression integrand",
            "sqrt(g)",
            "sqrt(g)_PEST",
            "sqrt(g)_Clebsch",
        ],
        grid=grid,
        flux_compression_k=10.0,
        surface_average=surface_average,
    )

    integrand = grid.meshgrid_reshape(data["flux compression integrand"], "raz")
    weights = np.abs(grid.meshgrid_reshape(data[weight_key], "raz"))
    expected = np.mean(integrand * weights, axis=(1, 2)) / np.mean(
        weights, axis=(1, 2)
    )

    np.testing.assert_allclose(grid.compress(data["flux compression proxy"]), expected)


@pytest.mark.unit
def test_flux_compression_default_uses_hard_heaviside():
    """Default flux-compression ITG behavior uses a hard Heaviside mask."""
    eq = get("ESTELL")
    rho = np.array([0.45, 0.8])
    alpha = np.linspace(0, (1 + eq.sym) * np.pi, (1 + eq.sym) * 3, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi / eq.NFP, 10, endpoint=False)
    grid = Grid.create_meshgrid([rho, alpha, zeta], coordinates="raz")

    data = eq.compute(
        ["flux compression integrand", "cvdrift", "g^rr", "a"],
        grid=grid,
    )
    metric = data["cvdrift"]
    expected = np.heaviside(metric, 0.0) * data["a"] ** 2 * data["g^rr"]

    np.testing.assert_allclose(data["flux compression integrand"], expected)
