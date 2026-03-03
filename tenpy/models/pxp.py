"""Implementation of the PXP model and variants on a chain."""
# Copyright (C) TeNPy Developers, Apache license

import numpy as np
import scipy as sp
from collections import defaultdict
import bisect

from ..networks.site import SpinHalfSite
from .lattice import Chain
from .model import CouplingMPOModel

__all__ = ['PXPChain', 'GeneralizedPXPModel', 'PXXZPChain', 'GeneralizedPXXZPModel', 'PExpPChain']


class PXPChain(CouplingMPOModel):
    r"""The PXP model as (approximately) implemented by a chain of Rydberg-blockaded atoms.

    The Hamiltonian reads:

    .. math ::
        H = \mathtt{J} \sum_{i} P_{i-1} X_i P_{i+1}
            + \mathtt{J_boundary} X_0 P_1 + P_{L-2} X_{L-1}

    where we only add the boundary terms for open boundaries with `J_boundary` defaulting to `J`.
    `P` is the projector onto the up state of the site, which corresponds to the ground state
    of the atom.

    The model arises from the strong-interaction limit of Rydberg atom chains in the seminal
    experiment of :doi:`10.1038/nature24622`, which found long oscillations now attributed to
    quantum many-body scars in the PXP model.

    Options
    -------
    .. cfg:config :: PXPChain
        :include: CouplingMPOModel

        conserve : 'best' | 'parity' | None
            What should be conserved. See :class:`~tenpy.networks.Site.SpinHalfSite`.
        J, J_boundary : float | array
            Couplings as defined for the Hamiltonian above.
    """

    default_lattice = Chain
    force_default_lattice = True  # we implicitly assume a 1D chain,
    # otherwise more P's need to be added

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'best', None)
        if conserve == 'best':
            conserve = 'parity'
        assert conserve != 'Sz'
        s = SpinHalfSite(conserve=conserve)
        s.add_op('X', s.get_op('Sigmax'), hc='X')  # X is already defined under other name
        # P is defined as P0, the projector onto the state 0, i.e. the up spin
        # The projector onto the state 1 is P1
        return s

    def init_terms(self, model_params):
        J = model_params.get('J', 2.0, 'real_or_array')             # PXP
        delta = model_params.get('delta', 0.0, 'real_or_array')     # PNP
        xi = model_params.get('xi', 0.0, 'real_or_array')           # PNPNP
        alpha = model_params.get('alpha', 0.0, 'real_or_array')     # PPP
        beta = model_params.get('beta', 0.0, 'real_or_array')       # PXXP
        h = model_params.get('h', 0.0, 'real_or_array')             # Z

        # PXP
        self.add_multi_coupling(J, [('P0', [-1], 0), ('X', [0], 0), ('P0', [1], 0)])
        # PNP
        self.add_multi_coupling(delta, [('P0', [-1], 0), ('P1', [0], 0), ('P0', [1], 0)])
        # PNPNP
        self.add_multi_coupling(xi, [('P0', [-2], 0), ('P1', [-1], 0), ('P0', [0], 0), ('P1', [1], 0), ('P0', [2], 0)])
        # PPP
        self.add_multi_coupling(alpha, [('P0', [-1], 0), ('P0', [0], 0), ('P0', [1], 0)])
        # PXXP - Note that this VIOLATES the Rydberg blockade constraint. It becomes a XY constraint if we
        # want to stay in the no blockade sector.
        self.add_multi_coupling(beta, [('P0', [-1], 0), ('X', [0], 0), ('X', [1], 0), ('P0', [2], 0)])
        # Z
        self.add_onsite(h, 0, 'Sigmaz')

        if model_params['bc_x'] == 'open':
            L = model_params['L']
            # If J is an array, I am not sure what J_boundary will do.
            J_boundary = model_params.get('J_boundary', J, 'real_or_array')
            delta_boundary = model_params.get('delta_boundary', delta, 'real_or_array')
            xi_boundary = model_params.get('xi_boundary', xi, 'real_or_array')
            alpha_boundary = model_params.get('alpha_boundary', alpha, 'real_or_array')
            beta_boundary = model_params.get('beta_boundary', beta, 'real_or_array')
            
            # J - XP and PX
            self.add_coupling_term(J_boundary, 0, 1, 'X', 'P0')
            self.add_coupling_term(J_boundary, L - 2, L - 1, 'P0', 'X')

            # delta - NP and PN
            self.add_coupling_term(delta_boundary, 0, 1, 'P1', 'P0')
            self.add_coupling_term(delta_boundary, L - 2, L - 1, 'P0', 'P1')

            # xi - PNP (0), NPNP (1) and PNPN (L-2), PNP (L-1)
            self.add_multi_coupling_term(xi_boundary, [0, 1, 2], ['P0', 'P1', 'P0'], ['Id', 'Id'], category='PNPNP')
            self.add_multi_coupling_term(xi_boundary, [0, 1, 2, 3], ['P1', 'P0', 'P1', 'P0'], ['Id', 'Id', 'Id'], category='PNPNP')
            
            self.add_multi_coupling_term(xi_boundary, [L-4, L-3, L-2, L-1], ['P0', 'P1', 'P0', 'P1'], ['Id', 'Id', 'Id'], category='PNPNP')
            self.add_multi_coupling_term(xi_boundary, [L-3, L-2, L-1], ['P0', 'P1', 'P0'], ['Id', 'Id'], category='PNPNP')

            # alpha - PP and PP
            self.add_coupling_term(alpha_boundary, 0, 1, 'P0', 'P0')
            self.add_coupling_term(alpha_boundary, L - 2, L - 1, 'P0', 'P0')

            # beta - XXP and PXX
            self.add_multi_coupling_term(beta_boundary, [0, 1, 2], ['X', 'X', 'P0'], ['Id', 'Id'], category='PXXP')
            self.add_multi_coupling_term(beta_boundary, [L-3, L-2, L-1], ['P0', 'X', 'X'], ['Id', 'Id'], category='PXXP')

class GeneralizedPXPModel(CouplingMPOModel):
    r"""The PXP model on arbitrary lattices with arbitrary blockade radius.

    The Hamiltonian reads:

    .. math ::
        H = \mathtt{J} \sum_{i} \prod_{j \in \mathcal{N}(i)} P_{j} X_i

    where the projectors are applied to the neighbors of the current site. The blockade radius
    is specified by how the neighborhood :math:`\mathcal{N}(i)` is defined. The standard PXP chain
    uses nearest neighbors on a chain.

    The user MUST define the blockade radius, either by specifying the keys for the neighbors
    (e.g. "nearest_neighbors" or whatever is defined for the lattice) or by specifying a dictionary
    of all neighbors for each site in the lattice.
    
    Boundaries are included if a dictionary of neighbors is specified. Any coupling that includes
    less than the max number of site couplings is treated as a boundary.

    `P` is the projector onto the up state (|0>) of the site, which corresponds to the ground state
    of the atom.

    The model arises from the strong-interaction limit of Rydberg atom chains in the seminal
    experiment of :doi:`10.1038/nature24622`, which found long oscillations now attributed to
    quantum many-body scars in the PXP model.

    Note that this model can be used to generate the PXP model on the chain by `neighbor_keys='NN'`.

    Options
    -------
    .. cfg:config :: GeneralizedPXPModel
        :include: CouplingMPOModel

        conserve : 'best' | 'parity' | None
            What should be conserved. See :class:`~tenpy.networks.Site.SpinHalfSite`.
        staggered : bool
            Do we generate the staggered Hamiltonian, where a minus sign is placed on
            all terms that END on the right half of the lattice?
        J, J_boundary : float | array
            Couplings as defined for the Hamiltonian above.
        neighbor_keys : str
            Keys for generating neighbors, used to define Rydberg blockade.
        sum_over_lattice_sites : book
            If True, we add the terms via functions that sum over lattice sites. This is needed
            for infinite MPS. Only completed terms are added; partial terms on the boundary are
            omitted.
            If False, we add the terms for each site separately. This allows us to handle
            boundary couplings with partial terms.
    """

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'best', None)
        sort_charge = model_params.get('sort_charge', True, bool)
        if conserve == 'best':
            conserve = 'parity'
        assert conserve != 'Sz'
        s = SpinHalfSite(conserve=conserve, sort_charge=sort_charge)
        s.add_op('X', s.get_op('Sigmax'), hc='X')  # X is already defined under other name
        # P is defined as P0, the projector onto the state 0, i.e. the up spin
        return s

    def init_terms(self, model_params):
        J = model_params.get('J', 2.0, 'real_or_array')             # PXP
        delta = model_params.get('delta', 0.0, 'real_or_array')     # PNP
        alpha = model_params.get('alpha', 0.0, 'real_or_array')     # PPP
        h = model_params.get('h', 0.0, 'real_or_array')             # Z
        
        sum_over_lattice_sites = model_params.get('sum_over_lattice_sites', True, bool)
        neighbor_keys = model_params.get('neighbor_keys', 'nearest_neighbors')
        neighbor_keys = neighbor_keys.split('-')

        neighbor_dict = defaultdict(list)

        if sum_over_lattice_sites:
            # Build neighbor_dict based on keys for neighbors
            neighbor_dict = _build_neighbor_dict_via_couplings(neighbor_keys, self.lat)
        else:
            assert model_params['bc_MPS'] != 'infinite', "For infinite MPS, we cannot enumerate pairs between sites."
            from ..algorithms.dmt_utils import generate_pairs, neighbors_from_pairs
            pairs = []
            for nk in neighbor_keys:
                pairs.extend(generate_pairs(self.lat, nk))
            neighbor_dict = neighbors_from_pairs(pairs)

        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(h, u, 'Sigmaz')

        if sum_over_lattice_sites:
            # We need to add terms using the function that sums over lattice sites.
            # For each site in the unit cell, we need to the term that is projector on all neighbors and X on the site.
            # NO BOUNDARIES WILL BE INCLUDED
            for ucs in neighbor_dict.keys():
                self.add_multi_coupling(J, neighbor_dict[ucs] + [('X', np.array([0] * self.lat.dim), ucs)])
                self.add_multi_coupling(delta, neighbor_dict[ucs] + [('P1', np.array([0] * self.lat.dim), ucs)])
                self.add_multi_coupling(alpha, neighbor_dict[ucs] + [('P0', np.array([0] * self.lat.dim), ucs)])
        else:
            # Add each term separately
            len_terms = [len(neighbor_dict[ucs]) for ucs in neighbor_dict.keys()]
            # The max number of neighbors a site couples to defines the bulk coupling.
            # For any coupling to fewer sites, we use the boundary coupling strength.
            bulk = np.max(len_terms)
            
            J_boundary = model_params.get('J_boundary', J, 'real_or_array')
            delta_boundary = model_params.get('delta_boundary', delta, 'real_or_array')
            alpha_boundary = model_params.get('alpha_boundary', alpha, 'real_or_array')
            # Do we want the staggered model, with half of the terms negated.
            staggered = model_params.get('staggered', False, bool)
            L = len(self.lat.mps_sites())

            # We need to insert X on the center site to the list of operators.
            # This must be inserted IN ORDER; so we created a sorted list of sites and operators.
            for ucs in neighbor_dict.keys():
                op_inds = neighbor_dict[ucs]
                op_inds.sort()
                ops = ['P0'] * len(op_inds)
                bisect.insort(op_inds, ucs)
                X_ind = op_inds.index(ucs)

                opsX = ops.copy()
                opsX.insert(X_ind, 'X')
                J_term = J if len(op_inds) == (bulk + 1) else J_boundary
                self.add_multi_coupling_term(J_term * ((-1)**(ucs >= L // 2) if staggered else 1), op_inds, opsX, ['Id'] * (len(op_inds)-1))
                
                opsN = ops.copy()
                opsN.insert(X_ind, 'P1')
                delta_term = delta if len(op_inds) == (bulk + 1) else delta_boundary
                self.add_multi_coupling_term(delta_term * ((-1)**(ucs >= L // 2) if staggered else 1), op_inds, opsN, ['Id'] * (len(op_inds)-1))
                
                opsP = ops.copy()
                opsP.insert(X_ind, 'P0')
                alpha_term = alpha if len(op_inds) == (bulk + 1) else alpha_boundary
                self.add_multi_coupling_term(alpha_term * ((-1)**(ucs >= L // 2) if staggered else 1), op_inds, opsP, ['Id'] * (len(op_inds)-1))

class PXXZPChain(CouplingMPOModel):
    r"""The PXXZP model, i.e. a chain of Rydberg-blockaded atoms with a U(1) symmetric XXZ interaction.

    The Hamiltonian reads:

    .. math ::
        H =  \mathtt{J} \sum_{i} P_{i-1} (X_i X_{i+1} + Y_{i} Y_{i+1} + Z_{i} Z_{i+1}) P_{i+1}
            + \mathtt{J_z} \sum_{i} P_{i-1} Z_{i} Z_{i+1} P_{i+1}
            + \mathtt{J_{zzp}} \sum_{i} P_{i-1} Z_{i} Z_{i+2} P_{i+3}
            + \mathtt{J_boundary} (X_0 X_1 + Y_0 Y_1 ) P_2 
            + P_{L-3} (X_{L-2} X_{L-1} + Y_{L-2} Y_{L-1})
            + \mathtt{J_boundaryz} Z_0 Z_1 P_2 + P_{L-3} Z_{L-2} Z_{L-1}
            + \mathtt{J_boundaryzzp} Z_0 Z_2 P_3 + P_{L-4} Z_{L-3} Z_{L-1}

    where we only add the boundary terms for open boundaries with `J_boundary` defaulting to `J`.
    `P` is the projector onto the up state of the site, which corresponds to the ground state
    of the atom.

    This model respects both the U(1) magnetization symmetry and the Rydberg blockade. In principle,
    this model can be realized in dipolar Rydberg simulators, where the native interaction is a dipolar
    XY flip-flop and Van der Wals ZZ; however, due to the long-range dipolar (alpha=3) tails of the 
    interaction, it is not immediately clear how to get a NN model with an infinitely strong blockade. 
    So this model is more of theoretical interest.

    Options
    -------
    .. cfg:config :: PXXZPChain
        :include: CouplingMPOModel

        conserve : 'best' | 'parity' | None
            What should be conserved. See :class:`~tenpy.networks.Site.SpinHalfSite`.
        J, J_z, J_boundary J_boundary_z : float | array
            Couplings as defined for the Hamiltonian above.
    """

    default_lattice = Chain
    force_default_lattice = True  # we implicitly assume a 1D chain,
    # otherwise more P's need to be added

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'best', None)
        if conserve == 'best':
            # This model has U(1) symmetry.
            conserve = 'Sz'
        s = SpinHalfSite(conserve=conserve)
        # P is defined as P0, the projector onto the state 0, i.e. the up spin
        return s

    def init_terms(self, model_params):
        J = model_params.get('J', 2.0, 'real_or_array')         # P(XX+YY)P
        Jz = model_params.get('Jz', 0.0, 'real_or_array')       # PZZP; non-integrable anisotropic perturbation
        Jzzp = model_params.get('Jzzp', 0.0, 'real_or_array')   # PZIZP; integrable anisotropic perturbation

        self.add_multi_coupling(Jz*4.0, [('P0', [-1], 0), ('Sz', [0], 0), ('Sz', [1], 0), ('P0', [2], 0)])
        self.add_multi_coupling(Jzzp*4.0, [('P0', [-1], 0), ('Sz', [0], 0), ('Sz', [2], 0), ('P0', [3], 0)])
        # Sp = Sx + i Sy; Sx = 1/2 Sigma x
        # Sp Sm + Sm Sp = 2(Sx Sx + Sy Sy) = 1/2 (Sigmax Sigmax + Sigmay Sigmay)
        self.add_multi_coupling(J*2.0, [('P0', [-1], 0), ('Sp', [0], 0), ('Sm', [1], 0), ('P0', [2], 0)], plus_hc=True)

        if model_params['bc_x'] == 'open':
            L = model_params['L']
            # If J is an array, I am not sure what J_boundary will do.
            J_boundary = model_params.get('J_boundary', J, 'real_or_array')
            Jz_boundary = model_params.get('Jz_boundary', Jz, 'real_or_array')
            Jzzp_boundary = model_params.get('Jzzp_boundary', Jzzp, 'real_or_array')
            self.add_multi_coupling_term(Jz_boundary*4.0, [0, 1, 2], ['Sz', 'Sz', 'P0'], ['Id', 'Id'])
            self.add_multi_coupling_term(Jz_boundary*4.0, [L-3, L-2, L-1], ['P0', 'Sz', 'Sz'], ['Id', 'Id'])
            self.add_multi_coupling_term(Jzzp_boundary*4.0, [0, 2, 3], ['Sz', 'Sz', 'P0'], ['Id', 'Id'])
            self.add_multi_coupling_term(Jzzp_boundary*4.0, [L-4, L-3, L-1], ['P0', 'Sz', 'Sz'], ['Id', 'Id'])
            self.add_multi_coupling_term(J_boundary*2.0, [0, 1, 2], ['Sp', 'Sm', 'P0'], ['Id', 'Id'], plus_hc=True)
            self.add_multi_coupling_term(J_boundary*2.0, [L-3, L-2, L-1], ['P0', 'Sp', 'Sm'], ['Id', 'Id'], plus_hc=True)


class GeneralizedPXXZPModel(CouplingMPOModel):
    r"""The PXXZP model on arbitrary lattices with arbitrary blockade radius.

    The Hamiltonian reads:

    .. math ::
        H = \mathtt{J} \sum_{<ij>} \prod_{k \in \mathcal{N}(i)} P_{k} \prod_{ell \in \mathcal{N}(j) \ i} P_{ell} J (Sp{i} Sm{j} + h.c.) + Jz(Z{i} Z{j})

    where the projectors are applied to the neighbors of the pair (i, j). 
    
    See documentation of GeneralizedPXPModel for details.

    Options
    -------
    .. cfg:config :: GeneralizedPXXZPModel
        :include: CouplingMPOModel

        conserve : 'best' | 'Sz' | None
            What should be conserved. See :class:`~tenpy.networks.Site.SpinHalfSite`.
        staggered : bool
            Do we generate the staggered Hamiltonian, where a minus sign is placed on
            all terms with the first non-projector on the right half of the lattice?
            Note that the staggering is slightly different than in PExpPChain.
        J, Jz, J_boundary, J_boundary_z : float | array
            Couplings as defined for the Hamiltonian above.
        neighbor_keys : str
            Keys for generating neighbors, used to define Rydberg blockade.
        interaction_keys : str
            Keys for generating pairs for interactions. This does not need to be the same as
            `neighbor_keys`, so the blockade radius can be different than the hopping radius.
        sum_over_lattice_sites : book
            If True, we add the terms via functions that sum over lattice sites. This is needed
            for infinite MPS. Only completed terms are added; partial terms on the boundary are
            omitted.
            If False, we add the terms for each site separately. This allows us to handle
            boundary couplings with partial terms.
    """

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'best', None)
        sort_charge = model_params.get('sort_charge', True, bool)
        if conserve == 'best':
            # This model has U(1) symmetry.
            conserve = 'Sz'
        s = SpinHalfSite(conserve=conserve)
        # P is defined as P0, the projector onto the state 0, i.e. the up spin
        return s

    def init_terms(self, model_params):
        J = model_params.get('J', 2.0, 'real_or_array')
        Jz = model_params.get('Jz', 0.0, 'real_or_array')
        sum_over_lattice_sites = model_params.get('sum_over_lattice_sites', True, bool)
        neighbor_keys = model_params.get('neighbor_keys', 'nearest_neighbors')
        neighbor_keys = neighbor_keys.split('-')
        interaction_keys = model_params.get('interaction_keys', 'nearest_neighbors')
        interaction_keys = interaction_keys.split('-')

        if sum_over_lattice_sites:
            # Build neighbor_dict and interaction_couplings based on keys for neighbors
            neighbor_dict = _build_neighbor_dict_via_couplings(neighbor_keys, self.lat)
            interaction_coupling = _neighbor_couplings_from_keys(interaction_keys, self.lat)
        else:
            assert model_params['bc_MPS'] != 'infinite', "For infinite MPS, we cannot enumerate pairs between sites."
            from ..algorithms.dmt_utils import generate_pairs, neighbors_from_pairs
            pairs = []
            for nk in neighbor_keys:
                pairs.extend(generate_pairs(self.lat, nk))
            neighbor_dict = neighbors_from_pairs(pairs)

            pairs = []
            for ik in interaction_keys:
                pairs.extend(generate_pairs(self.lat, ik))
            # Only get interactions with i < j to avoid double counting
            interaction_dict = neighbors_from_pairs(pairs, symmetric=False)

        if sum_over_lattice_sites:
            # We need to add terms using the function that sums over lattice sites.
            # We loop over all interaction pairs (i,j) and build a XY and ZZ term, with
            # projectors on all neighbors of (i,j).
            # NO BOUNDARIES WILL BE INCLUDED
            for (u1, u2, dx) in interaction_coupling:
                # Get neighbors of i and j, making copy of list
                projectors1 = [] + neighbor_dict[u1]
                projectors2 = [] + neighbor_dict[u2]

                # Remove j from neighbors of i                
                for k, (tag, disp, site) in enumerate(projectors1):
                    if tag == 'P0' and site == u2 and np.array_equal(disp, dx):
                        projectors1.pop(k)
                        break

                # Shift neighbors of j by dx
                for pj in range(len(projectors2)):
                    proj = projectors2[pj]
                    projectors2[pj] = (proj[0], proj[1] + dx, proj[2])

                # Remove i from neighbors of j
                for k, (tag, disp, site) in enumerate(projectors2):
                    if tag == 'P0' and site == u1 and np.array_equal(disp, np.array([0] * self.lat.dim)):
                        projectors2.pop(k)
                        break

                projectors = projectors1 + projectors2
                # Convert np.array (unhashable) into tuple (hashable)
                projectors = [(p[0], tuple(p[1]), p[2]) for p in projectors]
                # Remove duplicates, sites that are neighbors to both i and j
                projectors = list(set(projectors))
                # Convert tuples back to np.array
                projectors = [(p[0], np.array(p[1]), p[2]) for p in projectors]
                print(u1, u2, dx, projectors)

                # Add terms
                self.add_multi_coupling(2.0*J, projectors + [('Sp', np.array([0] * self.lat.dim), u1), ('Sm', dx, u2)], plus_hc=True)
                self.add_multi_coupling(4.0*Jz, projectors + [('Sz', np.array([0] * self.lat.dim), u1), ('Sz', dx, u2)], plus_hc=False)
        else:
            # Add each term separately
            # Get number of projectros of each interaction term
            len_terms = []
            for u1 in interaction_dict.keys():
                for u2 in interaction_dict[u1]:
                    len_terms.append(len(set(neighbor_dict[u1] + neighbor_dict[u2]) - {u1, u2}))
            # The max number of neighbors a site couples to defines the bulk coupling.
            # For any coupling to fewer sites, we use the boundary coupling strength.
            bulk = np.max(len_terms)

            # Couplings on the boundary
            J_boundary = model_params.get('J_boundary', J, 'real_or_array')
            Jz_boundary = model_params.get('Jz_boundary', Jz, 'real_or_array')
            
            #Do we want the staggered model, with half of the terms negated.
            staggered = model_params.get('staggered', False, bool)
            L = len(self.lat.mps_sites())

            # For each interaction, we need to build a list of operators.
            for u1 in interaction_dict.keys():
                for u2 in interaction_dict[u1]:
                    assert u1 < u2
                    op_inds = sorted(set(neighbor_dict[u1] + neighbor_dict[u2]) - {u1, u2})
                    ops = ['P0'] * len(op_inds)
                    bisect.insort(op_inds, u1)
                    bisect.insort(op_inds, u2)
                    u1_ind = op_inds.index(u1)
                    u2_ind = op_inds.index(u2)

                    XY_ops = ops.copy()
                    ZZ_ops = ops.copy()
                    XY_ops.insert(u1_ind, 'Sp')
                    XY_ops.insert(u2_ind, 'Sm')
                    ZZ_ops.insert(u1_ind, 'Sz')
                    ZZ_ops.insert(u2_ind, 'Sz')
                    
                    J_term = J if len(op_inds) == (bulk + 2) else J_boundary
                    Jz_term = Jz if len(op_inds) == (bulk + 2) else Jz_boundary
                    # Staggering based on u1, where the first non-projector is placed.
                    self.add_multi_coupling_term(2.0 * J_term * ((-1)**(u1 >= L // 2) if staggered else 1), op_inds, XY_ops, ['Id'] * (len(op_inds)-1), plus_hc=True)
                    self.add_multi_coupling_term(4.0 * Jz_term * ((-1)**(u1 >= L // 2) if staggered else 1), op_inds, ZZ_ops, ['Id'] * (len(op_inds)-1), plus_hc=False)


class PExpPChain(CouplingMPOModel):
    r"""An combination of NN PXP and PXXZP model with exponential interactions.

    The Hamiltonian reads:

    .. math ::
        H = \mathtt{J} \sum_{i} P_{i-1} X_i P_{i+1}
            + \sum_{i} \sum_{j > i+2} f(|j-i-2|) 2*\mathtt{J_xy} P_{i} S^+_{i+1} N_{i+2} ... N_{j-2} S^-_{j-1} P_{j} + h.c.
            + \sum_{i} \sum_{j > i+2} f(|j-i-2|) \mathtt{J_z} P_{i} Z_{i+1} N_{i+2} ... N_{j-2} Z_{j-1} P_{j} + h.c.

    The operator N is a projector onto the down spin, or state 1. P + N = Id.

    We DO NOT add boundary terms. In principle, we could add each term that leaves off a P on either the left
    or the right, as these terms would not violate the Rydberg Blockade or the U(1) charge (provided the PXP
    term is absent). However, this would require adding an exponentially decaying term for the left and right
    boundaries for both the XY and ZZ term.

    The P Sp N ... N Sm P + h.c. term allows for isolated domains of down spins to be mobile; 01100 -> 00110.
    However, the number of Rydberg blockade violations (Number of bonds on which sites to the left and right are up)
    is preserved by the Hamiltonian and thus by dynamics.

    The interaction strength f(|j-i-2|) is specified by a collection of exponentials, with both prefactors and
    decay constants. Note that this term generates all terms P X Y P, P X N Y P, P X N N Y P, etc., just with
    exponentially decaying (provided that the decay constants are positive) strengths. This then allows an arbitrarily
    sized domain of down spins (1s) to be mobile; 0 1^k 00 -> 00 1^k 0.

    Options
    -------
    .. cfg:config :: PExpPChain
        :include: CouplingMPOModel

        conserve : 'best' | 'parity' | None
            What should be conserved. See :class:`~tenpy.networks.Site.SpinHalfSite`.
        J, J_z, J_boundary J_boundary_z, lambdas, prefactors : float | array
            Couplings as defined for the Hamiltonian above.
    """

    default_lattice = Chain
    force_default_lattice = True  # we implicitly assume a 1D chain,
    # otherwise more P's need to be added

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'best', None)
        if conserve == 'best':
            # check how much we can conserve
            if not model_params.any_nonzero(['Jx'], "check Sz conservation"):
                conserve = 'Sz'
            else:
                conserve = 'parity'
            self.logger.info("%s: set conserve to %s", self.name, conserve)
        s = SpinHalfSite(conserve=conserve)
        if conserve != 'Sz':
            s.add_op('X', s.get_op('Sigmax'), hc='X')  # X is already defined under other name
        # P is defined as P0, the projector onto the state 0, i.e. the up spin
        return s

    def init_terms(self, model_params):
        #Do we want the staggered model, with half of the terms negated.
        staggered = model_params.get('staggered', False, bool)
        L = len(self.lat.mps_sites())
        sign = (-1 if staggered else 1)**(np.arange(L) >= L//2)

        # PXP Part
        Jx = model_params.get('Jx', 2.0, 'real_or_array')
        if np.any(np.asarray(Jx) != 0):
            # If staggered, we want a minus sign on all terms that END on the latter half of the chain.
            self.add_multi_coupling(Jx * sign[2:], [('P0', [-1], 0), ('X', [0], 0), ('P0', [1], 0)])
        
        # PXNYP part
        Jxy = model_params.get('Jxy', 2.0, 'real_or_array')
        Jz = model_params.get('Jz', 0.0, 'real_or_array')

        prefactors = model_params.get('prefactors', [0], 'real_or_array')
        lambdas = model_params.get('lambdas', [0], 'real_or_array')
        if np.isscalar(prefactors):
            prefactors = np.full(1, prefactors)
        if np.isscalar(lambdas):
            lambdas = np.full(1, lambdas)
        
        for lam, pre in zip(lambdas, prefactors):
            self.add_multi_exponentially_decaying_coupling(Jxy*2.0*pre * sign, lambda_=lam, ops_i=['P0', 'Sp'], ops_j=['Sm', 'P0'], 
                    subsites=None, subsites_start=None, op_string='P1', plus_hc=True)
            self.add_multi_exponentially_decaying_coupling(Jz*4.0*pre * sign, lambda_=lam, ops_i=['P0', 'Sz'], ops_j=['Sz', 'P0'], 
                    subsites=None, subsites_start=None, op_string='P1', plus_hc=False)


# Helper functions for generalized models
def _neighbor_couplings_from_keys(neighbor_keys, lat):
    neighbor_couplings = []
    for key in neighbor_keys:
        if key == 'NN':
            key = 'nearest_neighbors'
        elif key == 'nNN':
            key = 'next_nearest_neighbors'
        elif key == 'nnNN':
            key = 'next_next_nearest_neighbors'
        elif key == 'nnnNN':
            key = 'next_next_next_nearest_neighbors'
        elif key == 'nnnnNN':
            key = 'next_next_next_next_nearest_neighbors'
        neighbor_couplings.extend(lat.pairs[key])
    #u1, u2, dx 
    return neighbor_couplings
    
def _build_neighbor_dict_via_couplings(neighbor_keys, lat):
    neighbor_dict = defaultdict(list)
    neighbor_couplings = _neighbor_couplings_from_keys(neighbor_keys, lat)
    
    nc_count = 0

    # Same unit cell index, different unit cells
    # i.e. square or triangular lattice; A-A coupling in Honeycomb
    for (u1, u2, dx) in neighbor_couplings:
        if u1 == u2:
            neighbor_dict[u1].append(('P0', dx, u2))        # coupling to next unit cell in direction dx
            neighbor_dict[u1].append(('P0', -1*dx, u2))     # coupling to previous unit cell in direction dx
            nc_count += 1
            
    # Same unit cell, different unit cell index
    # i.e. A-B coupling within unit cell in Honeycomb
    for (u1, u2, dx) in neighbor_couplings:
        if u1 != u2 and np.all(dx == np.array([0] * len(dx))):
            neighbor_dict[u1].append(('P0', dx, u2))        # coupling within same unit cell, u1 -> u2
            neighbor_dict[u2].append(('P0', dx, u1))        # coupling within same unit cell, u2 -> u1
            nc_count += 1

    # Different unit cell, different unit cell index
    # i.e. A-B coupling between unit cells in Honeycomb
    for (u1, u2, dx) in neighbor_couplings:
        if u1 != u2 and not np.all(dx == np.array([0] * len(dx))):
            neighbor_dict[u1].append(('P0', dx, u2))        # coupling to next unit cell in direction dx, u1 -> u2
            neighbor_dict[u2].append(('P0', -1*dx, u1))     # coupling to previous unit cell in direction dx, u2 -> u1
            nc_count += 1

    assert nc_count == len(neighbor_couplings)
    assert lat.Lu == len(neighbor_dict)
    
    return neighbor_dict
