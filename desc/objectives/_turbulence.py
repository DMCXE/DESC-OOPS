"""Objectives for turbulence optimization."""

import numpy as np

from desc.backend import jnp
from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import Grid, LinearGrid
from desc.utils import Timer, check_posint, safediv, setdefault

from .objective_funs import _Objective, collect_docs

_FLUX_COMPRESSION_SURFACE_AVERAGE_DEFAULT = "clebsch"


def _get_flux_compression_surface_average_key(surface_average):
    """Return the compute quantity used as flux-compression surface-average weight."""
    surface_average = (
        _FLUX_COMPRESSION_SURFACE_AVERAGE_DEFAULT
        if surface_average is None
        else str(surface_average).lower()
    )
    if surface_average == "clebsch":
        return "sqrt(g)_Clebsch"
    if surface_average == "pest":
        return "sqrt(g)_PEST"
    if surface_average == "desc":
        return "sqrt(g)"
    raise ValueError(
        "Unknown surface_average='{}', expected 'clebsch', 'pest', or 'desc'.".format(
            surface_average
        )
    )


class FluxCompressionITGProxy(_Objective):
    """Flux-compression ITG turbulence proxy.

    This objective evaluates a DESC-native simple bad-curvature ITG proxy on one
    or more flux surfaces. The proxy is the flux-surface average of

    H_k(cvdrift) * a^2 * g^rr

    over a DESC field-line mesh with a selectable surface-average Jacobian.
    For a hard mask, DESC ``cvdrift`` is sign-equivalent to
    ``B x kappa · grad(alpha)``, so no additional ``sign(Psi)`` correction is
    applied here. This is not the later ``xi95`` paper target.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to be optimized.
    rho : float or array-like
        Flux surface(s) to evaluate on.
    alpha : float or array-like, optional
        Field line labels to evaluate. Default is 8 field lines in
        ``[0, (1 + eq.sym) * pi]`` for non-axisymmetric cases, or ``alpha=0``
        for axisymmetric equilibria.
    nzeta : int, optional
        Number of toroidal points in one field period. Default is 64.
    flux_compression_k : float or None, optional
        Smoothing parameter for the bad-curvature mask. ``None`` uses a hard
        Heaviside, otherwise ``sigmoid(flux_compression_k * x)``. Default is
        ``None``.
    surface_average : {"clebsch", "pest", "desc"}, optional
        Jacobian used in the surface average. ``"clebsch"`` is the DESC-native
        default, ``"pest"`` uses ``|sqrt(g)_PEST|``, and ``"desc"`` uses
        ``|sqrt(g)|`` to match the constellaration-style discrete average.
    use_PEST : bool, optional
        If ``False`` (default), sample one field period uniformly in ``zeta`` on
        a ``raz`` field-line mesh. If ``True``, sample uniformly in
        ``theta_PEST`` and map those points onto a corresponding non-mesh
        ``raz`` grid. In this branch ``nzeta`` is interpreted as the number of
        ``theta_PEST`` points.
    fieldline_batch_size : int or None, optional
        Number of field lines to process simultaneously. Default is ``None``,
        which processes all field lines at once.
    surf_batch_size : int or None, optional
        Number of flux surfaces to process simultaneously. Default is ``None``,
        which processes all surfaces at once.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
        normalize_detail=" Note: Has no effect for this objective.",
        normalize_target_detail=" Note: Has no effect for this objective.",
    )

    _static_attrs = _Objective._static_attrs + [
        "_rho",
        "_alpha",
        "_nzeta",
        "_flux_compression_k",
        "_surface_average",
        "_use_PEST",
        "_fieldline_batch_size",
        "_surf_batch_size",
        "_iota_keys",
        "_add_lcfs",
    ]

    _coordinates = "r"
    _units = "~"
    _print_value_fmt = "Flux Compression ITG Proxy: "

    def __init__(
        self,
        eq,
        rho=0.5,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        alpha=None,
        nzeta=64,
        flux_compression_k=None,
        surface_average=_FLUX_COMPRESSION_SURFACE_AVERAGE_DEFAULT,
        use_PEST=False,
        fieldline_batch_size=None,
        surf_batch_size=None,
        name="Flux Compression ITG Proxy",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0

        self._nzeta = check_posint(nzeta, "nzeta", False)
        self._rho = np.atleast_1d(rho)
        self._alpha = setdefault(
            alpha,
            (
                jnp.linspace(0, (1 + eq.sym) * jnp.pi, (1 + eq.sym) * 8)
                if eq.N
                else jnp.array([0.0])
            ),
        )
        self._flux_compression_k = flux_compression_k
        self._surface_average = str(surface_average).lower()
        _get_flux_compression_surface_average_key(self._surface_average)
        self._use_PEST = bool(use_PEST)
        self._fieldline_batch_size = check_posint(
            fieldline_batch_size, "fieldline_batch_size"
        )
        self._surf_batch_size = check_posint(surf_batch_size, "surf_batch_size")
        self._add_lcfs = np.all(self._rho < 0.97)

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
        self._iota_keys = ["iota", "iota_r", "a"]

        eq = self.things[0]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        iota_grid = LinearGrid(
            rho=np.append(self._rho, 1) if self._add_lcfs else self._rho,
            M=eq.M_grid,
            N=eq.N_grid,
            NFP=eq.NFP,
            sym=eq.sym,
        )
        assert not iota_grid.axis.size
        self._dim_f = iota_grid.num_rho - self._add_lcfs

        transforms = get_transforms(self._iota_keys, eq, iota_grid)
        profiles = get_profiles(
            self._iota_keys + ["flux compression proxy"], eq, iota_grid
        )
        zeta = (
            jnp.linspace(0, 2 * jnp.pi / eq.NFP, self._nzeta, endpoint=False)
            if eq.N
            else jnp.array([0.0])
        )
        theta_pest = (
            jnp.linspace(-jnp.pi, jnp.pi, self._nzeta, endpoint=False)
            if eq.N
            else jnp.array([0.0])
        )
        self._constants = {
            "rho": self._rho,
            "alpha": self._alpha,
            "zeta": zeta,
            "theta_pest": theta_pest,
            "iota_transforms": transforms,
            "profiles": profiles,
            "quad_weights": 1.0,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    @staticmethod
    def _chunk_slices(size, batch_size):
        """Return a static list of slices covering an axis."""
        if batch_size is None or batch_size >= size:
            return [slice(0, size)]
        return [
            slice(start, min(start + batch_size, size))
            for start in range(0, size, batch_size)
        ]

    def _get_iota_data(self, eq, params, constants):
        """Compute iota-like surface quantities on the 1D radial grid."""
        iota_data = compute_fun(
            eq,
            self._iota_keys,
            params,
            constants["iota_transforms"],
            constants["profiles"],
        )
        iota_grid = constants["iota_transforms"]["grid"]

        def get(key):
            x = iota_grid.compress(iota_data[key])
            return x[:-1] if self._add_lcfs else x

        return iota_data, get("iota"), get("iota_r")

    @staticmethod
    def _build_pest_grid(rho, alpha, theta_pest, iota, NFP):
        """Build a jitable custom raz grid that samples uniform theta_PEST points."""
        rho_grid, alpha_grid, theta_grid = jnp.meshgrid(
            rho, alpha, theta_pest, indexing="ij"
        )
        zeta_grid = (theta_grid - alpha_grid) / iota[:, jnp.newaxis, jnp.newaxis]
        nodes = jnp.column_stack(
            [rho_grid.reshape(-1), alpha_grid.reshape(-1), zeta_grid.reshape(-1)]
        )
        nr = rho.shape[0]
        na = alpha.shape[0]
        nt = theta_pest.shape[0]
        num_nodes = nr * na * nt
        return Grid(
            nodes=nodes,
            coordinates="raz",
            period=(jnp.inf, jnp.inf, jnp.inf),
            NFP=NFP,
            sort=False,
            jitable=True,
            _unique_rho_idx=jnp.arange(nr) * na * nt,
            _inverse_rho_idx=jnp.repeat(jnp.arange(nr), na * nt),
            _unique_poloidal_idx=jnp.arange(na) * nt,
            _inverse_poloidal_idx=jnp.tile(jnp.repeat(jnp.arange(na), nt), nr),
            _unique_zeta_idx=jnp.arange(num_nodes),
            _inverse_zeta_idx=jnp.arange(num_nodes),
        )

    def _compute_batched(self, eq, params, constants, iota_data, iota, iota_r):
        """Compute the flux-compression ITG proxy in surface and field-line batches."""
        rho = constants["rho"]
        alpha = constants["alpha"]
        zeta = constants["zeta"]
        out = []
        weight_key = _get_flux_compression_surface_average_key(self._surface_average)
        keys = ["flux compression integrand", weight_key]

        for rho_sl in self._chunk_slices(rho.size, self._surf_batch_size):
            rho_chunk = rho[rho_sl]
            iota_chunk = iota[rho_sl]
            iota_r_chunk = iota_r[rho_sl]
            numerator = jnp.zeros(rho_chunk.shape[0], dtype=iota_chunk.dtype)
            denominator = jnp.zeros(rho_chunk.shape[0], dtype=iota_chunk.dtype)

            for alpha_sl in self._chunk_slices(alpha.size, self._fieldline_batch_size):
                alpha_chunk = alpha[alpha_sl]
                grid = eq._get_rtz_grid(
                    rho_chunk,
                    alpha_chunk,
                    zeta,
                    coordinates="raz",
                    iota=iota_chunk,
                    params=params,
                )
                data = {
                    "iota": grid.expand(iota_chunk),
                    "iota_r": grid.expand(iota_r_chunk),
                    "a": iota_data["a"],
                }
                data = compute_fun(
                    eq,
                    keys,
                    params,
                    transforms=get_transforms(keys, eq, grid, jitable=True),
                    profiles=constants["profiles"],
                    data=data,
                    flux_compression_k=self._flux_compression_k,
                )
                source_grid = grid.source_grid
                integrand = source_grid.meshgrid_reshape(
                    data["flux compression integrand"], "raz"
                )
                weights = jnp.abs(source_grid.meshgrid_reshape(data[weight_key], "raz"))
                numerator = numerator + jnp.sum(integrand * weights, axis=(1, 2))
                denominator = denominator + jnp.sum(weights, axis=(1, 2))

            out.append(safediv(numerator, denominator))

        return out[0] if len(out) == 1 else jnp.concatenate(out)

    def _compute_use_pest(self, eq, params, constants, iota_data, iota, iota_r):
        """Compute the flux-compression ITG proxy on a uniform theta_PEST sampling."""
        rho = constants["rho"]
        alpha = constants["alpha"]
        theta_pest = constants["theta_pest"]
        out = []
        weight_key = _get_flux_compression_surface_average_key(self._surface_average)
        keys = ["flux compression integrand", weight_key]

        for rho_sl in self._chunk_slices(rho.size, self._surf_batch_size):
            rho_chunk = rho[rho_sl]
            iota_chunk = iota[rho_sl]
            iota_r_chunk = iota_r[rho_sl]
            numerator = jnp.zeros(rho_chunk.shape[0], dtype=iota_chunk.dtype)
            denominator = jnp.zeros(rho_chunk.shape[0], dtype=iota_chunk.dtype)

            for alpha_sl in self._chunk_slices(alpha.size, self._fieldline_batch_size):
                alpha_chunk = alpha[alpha_sl]
                source_grid = self._build_pest_grid(
                    rho_chunk, alpha_chunk, theta_pest, iota_chunk, eq.NFP
                )
                grid = eq._get_rtz_grid_from_source(
                    source_grid,
                    jitable=True,
                    point_cloud=True,
                    iota=iota_chunk,
                    params=params,
                )
                data = {
                    "iota": grid.expand(iota_chunk),
                    "iota_r": grid.expand(iota_r_chunk),
                    "a": iota_data["a"],
                }
                data = compute_fun(
                    eq,
                    keys,
                    params,
                    transforms=get_transforms(keys, eq, grid, jitable=True),
                    profiles=constants["profiles"],
                    data=data,
                    flux_compression_k=self._flux_compression_k,
                )
                shape = (rho_chunk.size, alpha_chunk.size, theta_pest.size)
                integrand = data["flux compression integrand"].reshape(shape)
                weights = jnp.abs(data[weight_key].reshape(shape))
                numerator = numerator + jnp.sum(integrand * weights, axis=(1, 2))
                denominator = denominator + jnp.sum(weights, axis=(1, 2))

            out.append(safediv(numerator, denominator))

        return out[0] if len(out) == 1 else jnp.concatenate(out)

    def compute(self, params, constants=None):
        """Compute the flux-compression ITG proxy."""
        if constants is None:
            constants = self.constants

        eq = self.things[0]
        iota_data, iota, iota_r = self._get_iota_data(eq, params, constants)

        if self._use_PEST:
            return self._compute_use_pest(eq, params, constants, iota_data, iota, iota_r)

        if (self._fieldline_batch_size is not None) or (
            self._surf_batch_size is not None
        ):
            return self._compute_batched(eq, params, constants, iota_data, iota, iota_r)

        grid = eq._get_rtz_grid(
            constants["rho"],
            constants["alpha"],
            constants["zeta"],
            coordinates="raz",
            iota=iota,
            params=params,
        )
        data = {
            "iota": grid.expand(iota),
            "iota_r": grid.expand(iota_r),
            "a": iota_data["a"],
        }
        data = compute_fun(
            eq,
            ["flux compression proxy"],
            params,
            transforms=get_transforms(
                ["flux compression proxy"], eq, grid, jitable=True
            ),
            profiles=constants["profiles"],
            data=data,
            flux_compression_k=self._flux_compression_k,
            surface_average=self._surface_average,
        )
        return grid.compress(data["flux compression proxy"])
