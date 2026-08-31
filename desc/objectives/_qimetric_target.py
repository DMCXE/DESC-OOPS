"""Constructed-field target-action quasi-isodynamicity objective."""

import warnings

import numpy as np
from orthax.legendre import leggauss

from desc.backend import jnp
from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import LinearGrid
from desc.integrals._interp_utils import cheb_pts, fourier_pts
from desc.integrals.quad_utils import chebgauss2
from desc.utils import check_posint, errorif, warnif

from ._neoclassical import _bounce_overwrite, _get_vander
from .objective_funs import _Objective, collect_docs


class QuasiIsodynamic(_Objective):
    """Constructed-field, target-action quasi-isodynamicity error.

    The input field is sampled on general DESC/Clebsch field lines over one field
    period. Each sampled well is transformed by the Goodman squash followed by a
    linear affine stretch of its left and right field-strength branches. The result
    is restored to the common surface extrema to form an objective-local ``B_C``.
    Bounce points of this constructed field are passed to the real ``Bounce2D``
    geometry, so both actions use ``dℓ = |B|/|B^zeta| dζ`` from the input field.

    The residual is the all-pairs L2 target of Goodman et al., PRX Energy 3, 023010
    (2024), factored into the variances of the signed input action and constructed
    action plus their mean mismatch. This avoids constructing an alpha-pair matrix.

    Notes
    -----
    The initial field must have an interior minimum on every sampled field line.
    This objective assumes one constructed well with common surface extrema. Full
    quasi-isodynamicity also requires the intended well topology and straight
    maximum-field contours to remain valid during optimization.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to optimize.
    grid : Grid, optional
        Tensor-product ``(rho, theta, zeta)`` grid. Defaults to the boundary surface
        at the equilibrium grid resolution.
    X : int, optional
        Poloidal Fourier resolution for the Clebsch mapping. Default is 16.
    Y : int, optional
        Toroidal Chebyshev resolution for the Clebsch mapping. Default is 32.
    Y_B : int, optional
        Resolution used by ``Bounce2D`` to approximate the real field. Default is 64.
    alpha : ndarray, optional
        Field-line labels. Defaults to 16 uniform labels on ``[0, 2π)``.
    nphi : int, optional
        Samples used to construct ``B_C`` over one field period. Default is 201.
    zeta0 : float, optional
        Toroidal origin of the sampled field period. Default is 0.
    num_quad : int, optional
        Bounce quadrature resolution. Default is 32.
    num_pitch : int, optional
        Open pitch quadrature resolution. Default is 51.
    pitch_batch_size : int or None, optional
        Number of pitch values computed together. Default computes all pitches.
    surf_batch_size : int or None, optional
        Number of surfaces computed together. Default is 1.
    nufft_eps : float, optional
        Requested NUFFT precision. Values below ``1e-14`` use the matrix path.
    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
        normalize_detail=" Note: Has no effect; normalization is action-local.",
        normalize_target_detail=" Note: Has no effect for this objective.",
        overwrite=_bounce_overwrite,
    )

    _coordinates = "r"
    _units = "~"
    _print_value_fmt = "Quasi-isodynamic target-action error: "
    _static_attrs = _Objective._static_attrs + ["_hyperparam", "_nphi", "_zeta0"]

    def __init__(
        self,
        eq,
        *,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        jac_chunk_size=None,
        name="Quasi-isodynamic target action",
        grid=None,
        X=16,
        Y=32,
        Y_B=64,
        alpha=None,
        nphi=201,
        zeta0=0.0,
        num_quad=32,
        num_pitch=51,
        pitch_batch_size=None,
        surf_batch_size=1,
        nufft_eps=1e-6,
    ):
        try:
            import jax_finufft  # noqa: F401
        except:  # noqa: E722
            warnif(
                nufft_eps >= 1e-14,
                msg="\njax-finufft is not installed properly.\n"
                "Setting parameter nufft_eps to zero.\n"
                "Performance will deteriorate significantly.\n",
            )
            nufft_eps = 0.0

        if target is None and bounds is None:
            target = 0.0
        alpha = np.asarray(
            2 * np.pi * np.arange(16) / 16 if alpha is None else alpha,
            dtype=float,
        )
        errorif(
            alpha.ndim != 1 or alpha.size < 2, ValueError, "alpha must have size >= 2"
        )
        self._grid = grid
        self._nphi = check_posint(nphi, "nphi", False)
        errorif(self._nphi < 3, ValueError, "nphi must be at least 3")
        self._zeta0 = float(zeta0)
        self._constants = {
            "quad_weights": 1.0,
            "alpha": alpha,
            "X": fourier_pts(check_posint(X, "X", False)),
            "Y": cheb_pts(check_posint(Y, "Y", False), (0, 2 * np.pi))[::-1],
        }
        self._hyperparam = {
            "Y_B": check_posint(Y_B, "Y_B", False),
            "num_quad": check_posint(num_quad, "num_quad", False),
            "num_pitch": check_posint(num_pitch, "num_pitch", False),
            "pitch_batch_size": check_posint(pitch_batch_size, "pitch_batch_size"),
            "surf_batch_size": check_posint(surf_batch_size, "surf_batch_size"),
            "nufft_eps": nufft_eps,
        }
        super().__init__(
            things=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            name=name,
            jac_chunk_size=jac_chunk_size,
        )

    def _sample_field_strength(self, params, data, constants):
        """Evaluate ``|B|`` on the constructed-field sampling grid."""
        eq = self.things[0]
        grid = constants["transforms"]["grid"]
        fieldline_grid = eq._get_rtz_grid(
            constants["rho"],
            constants["alpha"],
            constants["zeta"],
            coordinates="raz",
            iota=grid.compress(data["iota"]),
            params=params,
        )
        B = compute_fun(
            eq,
            "|B|",
            params,
            get_transforms("|B|", eq, fieldline_grid, jitable=True),
            constants["profiles"],
            data={"iota": fieldline_grid.copy_data_from_other(data["iota"], grid)},
        )["|B|"]
        return fieldline_grid.source_grid.meshgrid_reshape(B, "raz")

    def build(self, use_jit=True, verbose=1):
        """Build Clebsch, Bounce2D, and constructed-field constants."""
        eq = self.things[0]
        if self._grid is None:
            self._grid = LinearGrid(
                rho=np.array([1.0]),
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=False,
            )
        assert self._grid.can_fft2

        rho = self._grid.compress(self._grid.nodes[:, 0])
        self._constants["rho"] = rho
        self._constants["zeta"] = jnp.linspace(
            self._zeta0,
            self._zeta0 + 2 * np.pi / eq.NFP,
            self._nphi,
        )
        x, _ = leggauss(self._hyperparam["Y_B"] // 2)
        self._constants["_vander"] = _get_vander(self, x)
        self._constants["quad"] = chebgauss2(self._hyperparam.pop("num_quad"))
        self._data_keys = ["QI target J residual"]
        self._constants["profiles"] = get_profiles(self._data_keys, eq, grid=self._grid)
        self._constants["transforms"] = get_transforms(
            self._data_keys, eq, grid=self._grid
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Unequal number of field periods")
            self._constants["lambda"] = get_transforms(
                "lambda",
                eq,
                grid=LinearGrid(rho=rho, M=eq.L_basis.M, zeta=self._constants["Y"]),
            )["L"]
        assert self._constants["lambda"].basis.NFP == eq.NFP

        data = compute_fun(
            eq,
            "iota",
            eq.params_dict,
            self._constants["transforms"],
            self._constants["profiles"],
        )
        B_samples = np.asarray(
            self._sample_field_strength(eq.params_dict, data, self._constants)
        )
        min_B = np.min(B_samples, axis=-1)
        if np.any((B_samples[..., 0] == min_B) | (B_samples[..., -1] == min_B)):
            raise ValueError(
                "QuasiIsodynamic requires an interior |B| minimum on every "
                "sampled field line."
            )

        self._dim_f = (
            self._grid.num_rho
            * self._hyperparam["num_pitch"]
            * (2 * self._constants["alpha"].size + 1)
        )
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute the constructed-field all-pairs action residual."""
        if constants is None:
            constants = self.constants
        eq = self.things[0]
        data = compute_fun(
            eq,
            "iota",
            params,
            constants["transforms"],
            constants["profiles"],
        )
        theta = eq._map_clebsch_coordinates(
            iota=constants["transforms"]["grid"].compress(data["iota"]),
            alpha=constants["X"],
            zeta=constants["Y"],
            L_lmn=params["L_lmn"],
            lmbda=constants["lambda"],
            tol=1e-7,
        )[..., ::-1]
        B_samples = self._sample_field_strength(params, data, constants)
        data = compute_fun(
            eq,
            self._data_keys,
            params,
            constants["transforms"],
            constants["profiles"],
            data,
            theta=theta,
            B_samples=B_samples,
            zeta=constants["zeta"],
            alpha=constants["alpha"],
            quad=constants["quad"],
            _vander=constants["_vander"],
            **self._hyperparam,
        )
        return data["QI target J residual"]
