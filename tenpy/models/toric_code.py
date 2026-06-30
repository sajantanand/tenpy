"""Kitaev's exactly solvable toric code model.

As we put the model on a cylinder, the name "toric code" is a bit misleading, but it is the
established name for this model...
"""
# Copyright (C) TeNPy Developers, Apache license

import numpy as np

from ..networks.site import SpinHalfSite
from .lattice import Lattice, _parse_sites, get_order
from .model import CouplingMPOModel

__all__ = ['DualSquare', 'ToricCode']


class DualSquare(Lattice):
    """The dual lattice of the square lattice (again square).

    The sites in this lattice correspond to the vertical and horizontal (nearest neighbor) bonds
    of a common :class:`~tenpy.models.lattice.Square` lattice with the same dimensions `Lx, Ly`.

    .. plot ::

        import matplotlib.pyplot as plt
        from tenpy.models.toric_code import DualSquare
        from tenpy.models.lattice import Square
        plt.figure(figsize=(5, 5))
        ax = plt.gca()
        lat = DualSquare(4, 4, None, bc='periodic')
        sq = Square(4, 4, None, bc='periodic')
        sq.plot_coupling(ax, linewidth=3.)
        lat.plot_order(ax, linestyle=':')
        lat.plot_sites(ax)
        lat.plot_basis(ax, origin=-0.5*(lat.basis[0] + lat.basis[1]))
        ax.set_aspect('equal')
        ax.set_xlim(-1)
        ax.set_ylim(-1)
        plt.show()

    Parameters
    ----------
    Lx, Ly : int
        Dimensions of the original lattice. This lattice has `2*Lx*Ly` sites.
    sites : :class:`~tenpy.networks.site.Site`
        The sites for the horizontal (first entry) and vertical (second entry) bonds.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `basis`, `pos` and `pairs` are set accordingly.

    """

    dim = 2  #: the dimension of the lattice

    def __init__(self, Lx, Ly, sites, **kwargs):
        sites = _parse_sites(sites, 2)
        basis = np.eye(2)
        pos = np.array([[0.0, 0.5], [0.5, 0.0]])
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)
        NN = [(1, 0, np.array([0, 0])), (1, 0, np.array([1, 0])), (0, 1, np.array([-1, 1])), (0, 1, np.array([0, 1]))]
        nNN = [(i, i, dx) for i in [0, 1] for dx in [np.array([1, 0]), np.array([0, 1])]]
        nnNN = [(i, i, dx) for i in [0, 1] for dx in [np.array([1, 1]), np.array([-1, 1])]]
        kwargs.setdefault('pairs', {})
        kwargs['pairs'].setdefault('nearest_neighbors', NN)
        kwargs['pairs'].setdefault('next_nearest_neighbors', nNN)
        kwargs['pairs'].setdefault('next_next_nearest_neighbors', nnNN)
        super().__init__([Lx, Ly], sites, **kwargs)

    def ordering(self, order):
        """Provide possible orderings of the `N` lattice sites.

        The following orders are defined in this method compared to
        :meth:`tenpy.models.lattice.Lattice.ordering`:

        ================== =========================== =============================
        `order`            equivalent `priority`       equivalent ``snake_winding``
        ================== =========================== =============================
        ``'default'``      (0, 2, 1)                   (False, False, False)
        ================== =========================== =============================
        """
        if isinstance(order, str):
            if order == 'default':
                priority = (0, 2, 1)
                snake_winding = (False, False, False)
                return get_order(self.shape, snake_winding, priority)
        return super().ordering(order)


class ToricCode(CouplingMPOModel):
    r"""Toric code model.

    The Hamiltonian reads:

    .. math ::
        H = - \mathtt{Jv} \sum_{vertices v} \prod_{i \in v}  \sigma^x_i
            - \mathtt{Jp} \sum_{plaquettes p} \prod_{i \in p} \sigma^z_i

    (Note that this are Pauli matrices, not spin-1/2 operators.)
    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    .. versionchanged :: 0.7.2-98
        There was a bug that the terms for Jv and Jp were added with a positive instead of
        a negative sign.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`ToricCode` below.

    Options
    -------
    .. cfg:config :: ToricCode
        :include: CouplingMPOModel

        Lx, Ly: int
            Dimension of the lattice, number of plaquettes around the cylinder.
        conserve : 'parity' | None
            What should be conserved. See :class:`~tenpy.networks.Site.SpinHalfSite`.
        sort_charge : bool
            Whether to sort by charges of physical legs. `True` by default.
        Jv, Jp : float | array
            Couplings as defined for the Hamiltonian above.
        order : str
            The order of the lattice sites in the lattice, see :class:`DualSquare`.
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

    default_lattice = DualSquare
    force_default_lattice = True

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'parity', str)
        sort_charge = model_params.get('sort_charge', True, bool)
        site = SpinHalfSite(conserve, sort_charge=sort_charge)
        return site

    def init_terms(self, model_params):
        Jv = np.asarray(model_params.get('Jv', 1.0, 'real_or_array'))
        Jp = np.asarray(model_params.get('Jp', 1.0, 'real_or_array'))
        # vertex/star term
        self.add_multi_coupling(
            -Jv, [('Sigmax', [0, 0], 1), ('Sigmax', [0, 0], 0), ('Sigmax', [-1, 0], 1), ('Sigmax', [0, -1], 0)]
        )
        # plaquette term
        self.add_multi_coupling(
            -Jp, [('Sigmaz', [0, 0], 1), ('Sigmaz', [0, 0], 0), ('Sigmaz', [0, 1], 1), ('Sigmaz', [1, 0], 0)]
        )
        # done


class TwoColorSquare(Lattice):
    """Square lattice with 2x2 unit cell, with red and blue plaquettes.

    For the surface code, we define a square lattice ith a 2 x 2 unit cell, which allows for easy
    definition of the plaquette operators on both red and blue plaquettes. Additionally, one can
    define the boundary terms that either give a topological qubit or a unique ground state.

    The sites are on the vertices, as usual for the square lattice.

    .. plot ::

        import matplotlib.pyplot as plt
        from tenpy.models.toric_code import TwoColorSquare
        plt.figure(figsize=(5, 5))
        ax = plt.gca()
        sq = TwoColorSquare(4, 4, None, bc='periodic')
        sq.plot_coupling(ax, linewidth=3.)
        sq.plot_order(ax, linestyle=':')
        sq.plot_sites(ax)
        sq.plot_basis(ax, origin=-0.5*(lat.basis[0] + lat.basis[1]))
        ax.set_aspect('equal')
        ax.set_xlim(-1)
        ax.set_ylim(-1)
        plt.show()

    Parameters
    ----------
    Lx, Ly : int
        The square lattice is of size `2*Lx, 2*Ly`, with each primitive unit cell containing 4 sites.
    sites : :class:`~tenpy.networks.site.Site`
        The sites for the horizontal (first entry) and vertical (second entry) bonds.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `basis`, `pos` and `pairs` are set accordingly.

    """

    dim = 2  #: the dimension of the lattice

    def __init__(self, Lx, Ly, sites, **kwargs):
        sites = _parse_sites(sites, 4)
        basis = np.eye(2) * 2
        #   2   3
        #
        #   0   1
        pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)
        NN = [(0, 1, np.array([0, 0])), (2, 3, np.array([0, 0])), (0, 2, np.array([0, 0])), (1, 3, np.array([0, 0])),
              (1, 0, np.array([1, 0])), (3, 2, np.array([1, 0])),
              (2, 0, np.array([0, 1])), (3, 1, np.array([0, 1]))]
        kwargs.setdefault('pairs', {})
        kwargs['pairs'].setdefault('nearest_neighbors', NN)
        super().__init__([Lx, Ly], sites, **kwargs)

    def ordering(self, order):
        """Provide possible orderings of the `N` lattice sites.

        The following orders are defined in this method compared to
        :meth:`tenpy.models.lattice.Lattice.ordering`:

        ================== =========================== =============================
        `order`            equivalent `priority`       equivalent ``snake_winding``
        ================== =========================== =============================
        ``'default'``      (0, 1, 2)                   (False, False, False)
        ================== =========================== =============================

        Wind first through unit cell, then y direction, then x direction.
        """
        if isinstance(order, str):
            if order == 'default':
                priority = (0, 1, 2)
                snake_winding = (False, False, False)
                return get_order(self.shape, snake_winding, priority)
        return super().ordering(order)


class SurfaceCode(CouplingMPOModel):
    r"""Surface code model, i.e. 45 degree rotated Toric code

    The Hamiltonian reads:

    .. math ::
        H = - \mathtt{Jx} \sum_{red plaquette p} \prod_{i \in p}  \sigma^x_i
            - \mathtt{Jz} \sum_{blue plaquettes p} \prod_{i \in p} \sigma^z_i
            + \sum_i (h_x \sigma^x_i + h_y \sigma^y_i + h_z \sigma^z_i)

    (Note that this are Pauli matrices, not spin-1/2 operators.)
    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`SurfaceCode` below.

    Options
    -------
    .. cfg:config :: ToricCode
        :include: CouplingMPOModel

        Lx, Ly: int
            Dimension of the lattice is `2*Lx, 2*Ly`. If periodic BCs, then the number of
            plaquettes is `4*Ly*Ly`.
        conserve : 'parity' | None
            What should be conserved. See :class:`~tenpy.networks.Site.SpinHalfSite`.
        sort_charge : bool
            Whether to sort by charges of physical legs. `True` by default.
        Jx, Jz, Jx_boundary, Jz_boundary, hx, hy, hz : float | array
            Couplings as defined for the Hamiltonian above.
        order : str
            The order of the lattice sites in the lattice, see :class:`TwoColorSquare`.
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

    default_lattice = TwoColorSquare
    force_default_lattice = True

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'best', str)
        assert conserve != 'Sz'
        if conserve == 'best':
            # check how much we can conserve
            if not model_params.any_nonzero(['hx', 'hy'], 'check parity conservation'):
                conserve = 'parity'
            else:
                conserve = None
            self.logger.info('%s: set conserve to %s', self.name, conserve)
            print(conserve)
        sort_charge = model_params.get('sort_charge', True, bool)
        site = SpinHalfSite(conserve, sort_charge=sort_charge)
        return site

    def init_terms(self, model_params):
        Jx = np.asarray(model_params.get('Jx', 1.0, 'real_or_array'))
        Jz = np.asarray(model_params.get('Jz', 1.0, 'real_or_array'))
        
        hx = np.asarray(model_params.get('hx', 0.0, 'real_or_array'))
        hy = np.asarray(model_params.get('hy', 0.0, 'real_or_array'))
        hz = np.asarray(model_params.get('hz', 0.0, 'real_or_array'))

        # Onsite fields
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(hx, u, 'Sigmax')
            self.add_onsite(hy, u, 'Sigmay')
            self.add_onsite(hz, u, 'Sigmaz')
        
        # Plaquette within unit cell - X type
        self.add_multi_coupling(
            -Jx, [('Sigmax', [0, 0], 0), ('Sigmax', [0, 0], 1), ('Sigmax', [0, 0], 2), ('Sigmax', [0, 0], 3)], category='XXXX'
        )
        # Plaquette at top right corner - X type
        self.add_multi_coupling(
            -Jx, [('Sigmax', [0, 0], 3), ('Sigmax', [1, 0], 2), ('Sigmax', [0, 1], 1), ('Sigmax', [1, 1], 0)], category='XXXX'
        )
        # Plaquette to the right - Z type
        self.add_multi_coupling(
            -Jz, [('Sigmaz', [0, 0], 1), ('Sigmaz', [1, 0], 0), ('Sigmaz', [0, 0], 3), ('Sigmaz', [1, 0], 2)], category='ZZZZ'
        )
        # Plaquette above - Z type
        self.add_multi_coupling(
            -Jz, [('Sigmaz', [0, 0], 2), ('Sigmaz', [0, 0], 3), ('Sigmaz', [0, 1], 0), ('Sigmaz', [0, 1], 1)], category='ZZZZ'
        )
        

        # 'open', 'periodic', 'cylinder', or integer (bc_shift)
        bc_x = model_params['bc_x']
        bc_y = model_params['bc_y']
        assert type(bc_x) == type(bc_y) == str
        Lx = model_params['Lx']
        Ly = model_params['Ly']

        sort = lambda i, j: (i, j) if i <= j else (j, i)

        # Place two qubit Z operators on the X plaquettes in the first plaquette column and the last
        if bc_x == 'open':
            Jx_boundary = model_params.get('Jx_vert_boundary', Jx, 'real_or_array')
            Jz_boundary = model_params.get('Jz_vert_boundary', Jz, 'real_or_array')
            
            # First column
            for i in range(Ly):
                # ZZ - within unit cell
                s0 = 4*i
                s1 = s0+2
                s0, s1 = sort(s0, s1)
                print(s0, s1)
                self.add_coupling_term(-Jz_boundary, s0, s1, 'Sigmaz', 'Sigmaz', category='ZZ')
            
            for i in range(Ly - (bc_y == 'open')):
                # XX - between unit cells
                s0 = 4*i+2
                s1 = (s0+2) % (4*Ly)
                s0, s1 = sort(s0, s1)
                print(s0, s1)
                self.add_coupling_term(-Jx_boundary, s0, s1, 'Sigmax', 'Sigmax', category='XX')
            print("Column 2")
            # Last column
            for i in range(Ly):
                # ZZ - within unit cell
                s0 = (Lx-1)*4*Ly+4*i+1
                s1 = s0+2
                s0, s1 = sort(s0, s1)
                print(s0, s1)
                self.add_coupling_term(-Jz_boundary, s0, s1, 'Sigmaz', 'Sigmaz', category='ZZ')
            
            for i in range(Ly - (bc_y == 'open')):
                # XX - between unit cells
                s0 = (Lx-1)*4*Ly+4*i+3
                s1 = (Lx-1)*4*Ly+(4*i+3+2) % (4*Ly)
                s0, s1 = sort(s0, s1)
                print(s0, s1)
                self.add_coupling_term(-Jx_boundary, s0, s1, 'Sigmax', 'Sigmax', category='XX')
        
        # Either place two qubit X or two qubit Z on top and bottom rows, depending on whether we want
        # a topological qubit in the GS subspace.
        if bc_y == 'open':
            Jx_boundary = model_params.get('Jx_hor_boundary', Jx, 'real_or_array')
            Jz_boundary = model_params.get('Jz_hor_boundary', Jz, 'real_or_array')
            
            # Bottom row
            for i in range(Lx):
                # ZZ - within unit cell
                s0 = (4*Ly)*i
                s1 = s0+1
                s0, s1 = sort(s0, s1)
                self.add_coupling_term(-Jz_boundary, s0, s1, 'Sigmaz', 'Sigmaz', category='ZZ')
            
            for i in range(Lx - (bc_x == 'open')):
                # XX - between unit cells
                s0 = (4*Ly)*i+1
                s1 = ((4*Ly)*(i+1)) % (4*Lx*Ly)
                s0, s1 = sort(s0, s1)
                self.add_coupling_term(-Jx_boundary, s0, s1, 'Sigmax', 'Sigmax', category='XX')

            # Top row
            for i in range(Lx):
                # ZZ - within unit cell
                s0 = (4*Ly)*(i+1)-2
                s1 = s0+1
                s0, s1 = sort(s0, s1)
                self.add_coupling_term(-Jz_boundary, s0, s1, 'Sigmaz', 'Sigmaz', category='ZZ')
            
            for i in range(Ly - (bc_y == 'open')):
                # XX - between unit cells
                s0 = (4*Ly)*(i+1)-1
                s1 = ((4*Ly)*(i+2)-2)%(4*Lx*Ly)
                s0, s1 = sort(s0, s1)
                self.add_coupling_term(-Jx_boundary, s0, s1, 'Sigmax', 'Sigmax', category='XX')

        # done

