"""Objectives for Goodman-style quasi-isodynamicity."""

import numpy as np

from desc.backend import jnp
from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import LinearGrid
from desc.utils import Timer, check_posint

from .objective_funs import _Objective, collect_docs


class QuasiIsodynamicity(_Objective):
    """Goodman-style quasi-isodynamicity error for poloidally closed |B| contours.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the objective.
    grid : Grid, optional
        Grid specifying which flux surfaces to evaluate on. Only the unique ``rho``
        locations are used; Boozer harmonics are internally built on a non-symmetric
        linear grid.
    alpha : ndarray, optional
        Field-line labels to sample in Boozer coordinates. Defaults to 16 uniformly
        spaced values on ``[0, 2π)``.
    nphi : int, optional
        Number of Boozer toroidal samples per field line over one field period.
    nB : int, optional
        Number of normalized field-strength levels used for the Goodman shuffle.
    M_booz : int, optional
        Poloidal Boozer resolution. Defaults to ``2 * eq.M``.
    N_booz : int, optional
        Toroidal Boozer resolution. Defaults to ``2 * eq.N``.
    zeta0 : float, optional
        Boozer toroidal origin of the sampled period.
    eps : float, optional
        Small regularization used in divisions and ordering projections.
    fieldline_batch_size : int or None, optional
        Number of field lines processed simultaneously. Default is ``None``,
        which processes all field lines at once.
    surf_batch_size : int or None, optional
        Number of flux surfaces processed simultaneously. Default is ``None``,
        which processes all surfaces at once.
    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
        normalize_detail=" Note: Has no effect for this objective.",
        normalize_target_detail=" Note: Has no effect for this objective.",
    )

    _coordinates = "rtz"
    _units = "(dimensionless)"
    _print_value_fmt = "Quasi-isodynamicity error: "
    _static_attrs = _Objective._static_attrs + [
        "_hyperparam",
        "_M_booz",
        "_N_booz",
        "_alpha",
        "_nB",
        "_nphi",
        "_zeta0",
        "_eps",
    ]

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        alpha=None,
        nphi=200,
        nB=81,
        M_booz=None,
        N_booz=None,
        zeta0=0.0,
        eps=1e-12,
        fieldline_batch_size=None,
        surf_batch_size=None,
        name="quasi-isodynamicity",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0.0
        self._grid = grid
        self._alpha = np.asarray(
            2 * np.pi * np.arange(16) / 16 if alpha is None else alpha,
            dtype=float,
        )
        self._nphi = int(nphi)
        self._nB = int(nB)
        self._M_booz = M_booz
        self._N_booz = N_booz
        self._zeta0 = float(zeta0)
        self._eps = float(eps)
        self._hyperparam = {
            "fieldline_batch_size": check_posint(
                fieldline_batch_size, "fieldline_batch_size"
            ),
            "surf_batch_size": check_posint(surf_batch_size, "surf_batch_size"),
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
        """Build constant arrays."""
        eq = self.things[0]
        M_booz = self._M_booz or 2 * eq.M
        N_booz = self._N_booz or 2 * eq.N

        rho = (
            np.array([1.0])
            if self._grid is None
            else self._grid.nodes[self._grid.unique_rho_idx, 0]
        )
        grid = LinearGrid(rho=rho, M=2 * M_booz, N=2 * N_booz, NFP=eq.NFP, sym=False)

        self._data_keys = ["qimetric residual"]
        self._dim_f = grid.num_rho * self._alpha.size * self._nphi

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        profiles = get_profiles(self._data_keys, obj=eq, grid=grid)
        transforms = get_transforms(
            self._data_keys,
            obj=eq,
            grid=grid,
            M_booz=M_booz,
            N_booz=N_booz,
        )

        zeta = jnp.linspace(
            self._zeta0,
            self._zeta0 + 2 * np.pi / eq.NFP,
            self._nphi,
        )
        self._constants = {
            "profiles": profiles,
            "transforms": transforms,
            "alpha": jnp.asarray(self._alpha),
            "zeta": zeta,
            "levels": jnp.linspace(0.0, 1.0, self._nB),
            "quad_weights": 1.0,
            "eps": self._eps,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute Goodman-style quasi-isodynamicity residuals."""
        if constants is None:
            constants = self.constants
        data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
            alpha=constants["alpha"],
            zeta=constants["zeta"],
            levels=constants["levels"],
            eps=constants["eps"],
            **self._hyperparam,
        )
        residual = data["qimetric residual"]
        residual = residual / jnp.sqrt(constants["alpha"].size * constants["zeta"].size)
        return residual
