"""Reduced quasi-isodynamicity objective based on bounce actions."""

import warnings

import numpy as np
from orthax.legendre import leggauss

from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import LinearGrid
from desc.integrals._interp_utils import cheb_pts, fourier_pts
from desc.integrals.quad_utils import chebgauss2
from desc.utils import check_posint, errorif, setdefault, warnif

from ._neoclassical import _bounce_overwrite, _get_vander
from .normalization import compute_scaling_factors
from .objective_funs import _Objective, collect_docs


class QuasiIsodynamicReduced(_Objective):
    r"""Reduced omnigenity error from field-line variation of bounce actions.

    For every flux surface, open pitch level, and detected magnetic well, this
    objective evaluates

    .. math::

        J_{\alpha k w} = \int_w \sqrt{|1-\lambda_k B|}\,d\ell,

    using DESC's :class:`~desc.integrals.Bounce2D`, and returns
    :math:`J_{\alpha k w}-\langle J_{kw}\rangle_\alpha`. The wells remain a
    residual axis; they are not summed before the field-line average.

    This is only a reduced omnigenity condition. Full quasi-isodynamicity also
    requires suitable single-well topology, common extrema across field lines,
    and straight maximum-|B| contours. The fixed well axis uses ``Bounce2D``'s
    positional ordering and assumes that topology remains consistent; no dynamic
    well matching is performed.

    Parameters
    ----------
    eq : Equilibrium
        ``Equilibrium`` to be optimized.
    grid : Grid, optional
        Tensor-product grid in ``(rho, theta, zeta)`` with uniformly spaced
        angular nodes. Only its flux surfaces are scored. By default, the boundary
        surface is sampled at the equilibrium's standard grid resolution.
    X : int, optional
        Poloidal Fourier resolution of the Clebsch-coordinate map. Default is 16.
    Y : int, optional
        Toroidal Chebyshev resolution of the Clebsch-coordinate map. Default is 32.
    Y_B : int, optional
        Number of knots per toroidal transit used by the bounce-point algorithm.
        Default is 64.
    alpha : ndarray, optional
        Field-line labels. Default is 16 uniformly spaced labels on ``[0, 2 pi)``.
    num_transit : int, optional
        Number of toroidal transits followed by each field line. Default is 1.
    num_well : int, optional
        Fixed upper bound on the number of wells per pitch and field line. Missing
        wells are zero padded by ``Bounce2D``. Default is ``eq.NFP``.
    num_quad : int, optional
        Bounce-integral quadrature resolution. Default is 32.
    num_pitch : int, optional
        Odd number (at least 5) of open, surface-global ``B*=1/lambda`` levels
        for Simpson quadrature. Default is 51.
    pitch_batch_size : int or None, optional
        Number of pitch values computed simultaneously. Default is all pitches.
    surf_batch_size : int or None, optional
        Number of flux surfaces computed simultaneously. Default is 1.
    nufft_eps : float, optional
        Requested NUFFT precision. Values below ``1e-14`` use DESC's matrix
        transform path. Default is ``1e-6``.
    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
        normalize_detail=" The action is normalized by the equilibrium major radius.",
        overwrite=_bounce_overwrite,
    )

    _static_attrs = _Objective._static_attrs + ["_hyperparam"]
    _coordinates = "r"
    _units = "(m)"
    _print_value_fmt = "Reduced quasi-isodynamicity action error: "

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
        name="reduced quasi-isodynamicity",
        grid=None,
        X=16,
        Y=32,
        Y_B=64,
        alpha=None,
        num_transit=1,
        num_well=None,
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
            setdefault(alpha, 2 * np.pi * np.arange(16, dtype=float) / 16),
            dtype=float,
        )
        errorif(
            alpha.ndim != 1 or alpha.size < 2,
            ValueError,
            "alpha must have size >= 2",
        )
        num_pitch = check_posint(num_pitch, "num_pitch", False)
        errorif(
            num_pitch < 5 or num_pitch % 2 == 0,
            ValueError,
            "num_pitch must be an odd integer >= 5",
        )
        Y_B = setdefault(Y_B, 2 * Y)
        self._grid = grid
        self._constants = {
            "quad_weights": 1.0,
            "alpha": alpha,
            "X": fourier_pts(X),
            "Y": cheb_pts(Y, (0, 2 * np.pi))[::-1],
        }
        self._hyperparam = {
            "Y_B": Y_B,
            "num_transit": num_transit,
            "num_well": setdefault(num_well, eq.NFP),
            "num_quad": num_quad,
            "num_pitch": num_pitch,
            "pitch_batch_size": pitch_batch_size,
            "surf_batch_size": surf_batch_size,
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

    def build(self, use_jit=True, verbose=1):
        """Build DESC transforms and Bounce2D interpolation constants."""
        eq = self.things[0]
        if self._grid is None:
            self._grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False)
        assert self._grid.can_fft2

        rho = self._grid.compress(self._grid.nodes[:, 0])
        x, _ = leggauss(self._hyperparam["Y_B"] // 2)
        self._constants["_vander"] = _get_vander(self, x)
        self._constants["quad"] = chebgauss2(self._hyperparam.pop("num_quad"))
        self._constants["profiles"] = get_profiles(
            "J alpha residual", eq, grid=self._grid
        )
        self._constants["transforms"] = get_transforms(
            "J alpha residual", eq, grid=self._grid
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Unequal number of field periods")
            self._constants["lambda"] = get_transforms(
                "lambda",
                eq,
                grid=LinearGrid(rho=rho, M=eq.L_basis.M, zeta=self._constants["Y"]),
            )["L"]
        assert self._constants["lambda"].basis.NFP == eq.NFP

        self._data_keys = ["J alpha residual"]
        self._dim_f = (
            self._grid.num_rho
            * self._hyperparam["num_pitch"]
            * self._constants["alpha"].size
            * self._hyperparam["num_well"]
        )
        if self._normalize:
            self._normalization = compute_scaling_factors(eq)["R0"]
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute weighted per-well action residuals."""
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
        data = compute_fun(
            eq,
            self._data_keys,
            params,
            constants["transforms"],
            constants["profiles"],
            data,
            theta=theta,
            alpha=constants["alpha"],
            quad=constants["quad"],
            _vander=constants["_vander"],
            **self._hyperparam,
        )
        return data["J alpha residual"]
