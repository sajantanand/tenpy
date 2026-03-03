"""Kitaev's exactly solvable anisotropic model on a honeycomb lattice.

"""
# Copyright (C) TeNPy Developers, Apache license

import numpy as np

from ..networks.site import SpinSite
from .lattice import Honeycomb
from ..tools.params import asConfig
from .model import CouplingMPOModel

__all__ = ['KitaevHoneycomb']


class KitaevHoneycomb(CouplingMPOModel):
    r"""Kitaev Honeycomb model.

    The Hamiltonian reads:

    .. math ::
        H = \mathtt{J_x} \sum_{\langle i, j \rangle_x} S^x_i S^x_j
            + \mathtt{J_y} \sum_{\langle i, j \rangle_y} S^y_i S^y_j
            + \mathtt{J_z} \sum_{\langle i, j \rangle_y} S^z_i S^z_j
            + \sum_i \mathtt{h_x} S^x_i + \mathtt{h_y} S^y_i + \mathtt{h_z} S^z_i

    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`KitaevHoneycomb` below.

    Options
    -------
    .. cfg:config :: Kitaev
        :include: CouplingMPOModel

        Lx, Ly: int
            Dimension of the lattice, number of unit cells in each direction.
        sort_charge : bool
            Whether to sort by charges of physical legs. `True` by default.
        Jx, Jy, Jz, hx, hy, hz : float | array
            Couplings as defined for the Hamiltonian above.
        order : str
            The order of the lattice sites in the lattice, see :class:`Honeycomb`.
        bc_y : ``"open" | "periodic"``
            The boundary conditions in y-direction.
        bc_x : ``"open" | "periodic"``
            Can be used to force "periodic" boundaries for the lattice,
            i.e., for the couplings in the Hamiltonian, even if the MPS is finite.
            Defaults to ``"open"`` for ``bc_MPS="finite"`` and
            ``"periodic"`` for ``bc_MPS="infinite``.
            If you are not aware of the consequences, you should probably
            *not* use "periodic" boundary conditions:
            The MPS is still "open", so this will introduce long-range couplings between the
            first and last sites of the MPS, and require **squared** MPS bond-dimensions.

    """

    default_lattice = Honeycomb
    force_default_lattice = True

    def init_sites(self, model_params):
        S = model_params.get('S', 0.5, 'real')
        conserve = None
        sort_charge = model_params.get('sort_charge', True, bool)
        site = SpinSite(S, conserve, sort_charge)
        return site

    def init_terms(self, model_params):
        Jx = np.asarray(model_params.get('Jx', 1.0, 'real_or_array'))
        Jy = np.asarray(model_params.get('Jy', 1.0, 'real_or_array'))
        Jz = np.asarray(model_params.get('Jz', 1.0, 'real_or_array'))
        hx = np.asarray(model_params.get('hx', 0.0, 'real_or_array'))
        hy = np.asarray(model_params.get('hy', 0.0, 'real_or_array'))
        hz = np.asarray(model_params.get('hz', 0.0, 'real_or_array'))

        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(hx, u, 'Sx')
            self.add_onsite(hy, u, 'Sy')
            self.add_onsite(hz, u, 'Sz')

        for J, op, (u1, u2, dx) in zip([Jx, Jy, Jz], ['Sx', 'Sy', 'Sz'], self.lat.pairs['nearest_neighbors']):
            self.add_coupling(J, u1, op, u2, op, dx, plus_hc=False)
