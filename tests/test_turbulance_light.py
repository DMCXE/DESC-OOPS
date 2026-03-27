"""Tests for the lightweight VMEC bad-curvature proxy helper."""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from turbulance import (  # noqa: E402
    VmecFieldLightResults,
    _set_up_surface_grid,
    compute_flux_compression_in_regions_of_bad_curvature,
)


@pytest.mark.unit
def test_vmec_light_surface_grid_matches_simple_proxy_ranges():
    """Surface-grid setup matches the STELLOPT-style simple proxy ranges."""
    alpha, theta = _set_up_surface_grid(
        n_field_periods=5,
        n_field_lines=4,
        n_toroidal_points=6,
        is_stellarator_symmetric=True,
    )

    alpha_extent = 2 * np.pi / 5 / 2
    expected_alpha = np.linspace(-alpha_extent, alpha_extent, 4, endpoint=False)
    expected_theta = np.linspace(-np.pi, np.pi, 6, endpoint=False)

    np.testing.assert_allclose(alpha, expected_alpha)
    np.testing.assert_allclose(theta, expected_theta)


@pytest.mark.unit
def test_vmec_light_proxy_matches_weighted_average(monkeypatch):
    """Proxy helper reduces vmec_field_light outputs with the expected weights."""
    alpha = np.array([-0.2, 0.0])
    theta = np.array([-np.pi, 0.0])

    def fake_surface_grid(**kwargs):
        return alpha, theta

    data = VmecFieldLightResults(
        sqrt_g_vmec=np.array([[[1.0, -2.0], [3.0, -4.0]]]),
        sqrt_g_pest=np.array([[[2.0, -1.0], [4.0, -3.0]]]),
        grad_s_dot_grad_s=np.array([[[2.0, 1.0], [4.0, 3.0]]]),
        B_cross_kappa_dot_grad_alpha=np.array([[[-1.0, 2.0], [-3.0, 4.0]]]),
        nalpha=2,
        nl=2,
        alpha=alpha,
        theta1d=theta,
        phi1d=None,
    )

    def fake_vmec_field_light(vs, s, alpha, theta1d):
        return data

    monkeypatch.setattr(
        "turbulance._set_up_surface_grid",
        fake_surface_grid,
    )
    monkeypatch.setattr(
        "turbulance.vmec_field_light",
        fake_vmec_field_light,
    )

    equilibrium = SimpleNamespace(
        wout=SimpleNamespace(
            lasym=False,
            Aminor_p=2.0,
            phipf=np.array([0.0, -1.0]),
            nfp=5,
        )
    )
    s = np.array([0.25])

    result = compute_flux_compression_in_regions_of_bad_curvature(
        equilibrium=equilibrium,
        normalized_toroidal_flux=s,
        n_field_lines=alpha.size,
        n_toroidal_points=theta.size,
    )

    weights = np.abs(data.sqrt_g_vmec)
    weights = weights / weights.mean(axis=(1, 2))[:, None, None]
    grad = equilibrium.wout.Aminor_p**2 / (4 * s[:, None, None]) * data.grad_s_dot_grad_s
    curv = np.sign(equilibrium.wout.phipf[-1]) * data.B_cross_kappa_dot_grad_alpha
    expected = np.mean(np.heaviside(curv, 0.0) * grad * weights, axis=(1, 2))

    np.testing.assert_allclose(result, expected)


@pytest.mark.unit
def test_vmec_light_proxy_supports_pest_weights(monkeypatch):
    """Proxy helper can use sqrt_g_pest weights for PEST-style averaging."""
    alpha = np.array([-0.2, 0.0])
    theta = np.array([-np.pi, 0.0])

    def fake_surface_grid(**kwargs):
        return alpha, theta

    data = VmecFieldLightResults(
        sqrt_g_vmec=np.array([[[1.0, -2.0], [3.0, -4.0]]]),
        sqrt_g_pest=np.array([[[2.0, -1.0], [4.0, -3.0]]]),
        grad_s_dot_grad_s=np.array([[[2.0, 1.0], [4.0, 3.0]]]),
        B_cross_kappa_dot_grad_alpha=np.array([[[-1.0, 2.0], [-3.0, 4.0]]]),
        nalpha=2,
        nl=2,
        alpha=alpha,
        theta1d=theta,
        phi1d=None,
    )

    def fake_vmec_field_light(vs, s, alpha, theta1d):
        return data

    monkeypatch.setattr("turbulance._set_up_surface_grid", fake_surface_grid)
    monkeypatch.setattr("turbulance.vmec_field_light", fake_vmec_field_light)

    equilibrium = SimpleNamespace(
        wout=SimpleNamespace(
            lasym=False,
            Aminor_p=2.0,
            phipf=np.array([0.0, -1.0]),
            nfp=5,
        )
    )
    s = np.array([0.25])

    result = compute_flux_compression_in_regions_of_bad_curvature(
        equilibrium=equilibrium,
        normalized_toroidal_flux=s,
        n_field_lines=alpha.size,
        n_toroidal_points=theta.size,
        surface_average="pest",
    )

    weights = np.abs(data.sqrt_g_pest)
    weights = weights / weights.mean(axis=(1, 2))[:, None, None]
    grad = equilibrium.wout.Aminor_p**2 / (4 * s[:, None, None]) * data.grad_s_dot_grad_s
    curv = np.sign(equilibrium.wout.phipf[-1]) * data.B_cross_kappa_dot_grad_alpha
    expected = np.mean(np.heaviside(curv, 0.0) * grad * weights, axis=(1, 2))

    np.testing.assert_allclose(result, expected)


@pytest.mark.unit
def test_vmec_light_proxy_weight_alias_matches_surface_average(monkeypatch):
    """Legacy weight alias matches the new surface_average keyword."""
    alpha = np.array([-0.2, 0.0])
    theta = np.array([-np.pi, 0.0])

    def fake_surface_grid(**kwargs):
        return alpha, theta

    data = VmecFieldLightResults(
        sqrt_g_vmec=np.array([[[1.0, -2.0], [3.0, -4.0]]]),
        sqrt_g_pest=np.array([[[2.0, -1.0], [4.0, -3.0]]]),
        grad_s_dot_grad_s=np.array([[[2.0, 1.0], [4.0, 3.0]]]),
        B_cross_kappa_dot_grad_alpha=np.array([[[-1.0, 2.0], [-3.0, 4.0]]]),
        nalpha=2,
        nl=2,
        alpha=alpha,
        theta1d=theta,
        phi1d=None,
    )

    def fake_vmec_field_light(vs, s, alpha, theta1d):
        return data

    monkeypatch.setattr("turbulance._set_up_surface_grid", fake_surface_grid)
    monkeypatch.setattr("turbulance.vmec_field_light", fake_vmec_field_light)

    equilibrium = SimpleNamespace(
        wout=SimpleNamespace(
            lasym=False,
            Aminor_p=2.0,
            phipf=np.array([0.0, -1.0]),
            nfp=5,
        )
    )
    s = np.array([0.25])

    result_surface_average = compute_flux_compression_in_regions_of_bad_curvature(
        equilibrium=equilibrium,
        normalized_toroidal_flux=s,
        n_field_lines=alpha.size,
        n_toroidal_points=theta.size,
        surface_average="pest",
    )
    result_weight = compute_flux_compression_in_regions_of_bad_curvature(
        equilibrium=equilibrium,
        normalized_toroidal_flux=s,
        n_field_lines=alpha.size,
        n_toroidal_points=theta.size,
        weight="pest",
    )

    np.testing.assert_allclose(result_surface_average, result_weight)
