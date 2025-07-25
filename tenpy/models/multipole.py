"""Multiple conserving spin-S models.

Uniform lattice of spin-S sites, coupled by multiple conserving interactions
"""
# Copyright (C) TeNPy Developers, Apache license

import numpy as np
from scipy.special import comb

from ..linalg import np_conserved as npc
from ..networks.site import SpinSite
from .model import CouplingMPOModel
from .lattice import Chain, Square
from ..tools.params import asConfig

__all__ = ['MultipoleChain']

class MultipoleChain(CouplingMPOModel):
    r"""Spin-S sites coupled by interactions which preserve multipole symmetry

    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`SpinModel` below.

    Options
    -------
    .. cfg:config :: SpinModel
        :include: CouplingMPOModel

        S : {0.5, 1, 1.5, 2, ...}
            The 2S+1 local states range from m = -S, -S+1, ... +S.
        conserve : 'best' | 'Sz' | 'parity' | None
            What should be conserved. See :class:`~tenpy.networks.Site.SpinSite`.
            For ``'best'``, we check the parameters what can be preserved.
        sort_charge : bool
            Whether to sort by charges of physical legs. `True` by default.
        r  : list of lists
            For each multipole level, what range operators do we consider.
            level = -1 has operators Sp (and Sm from Hermitian conjugate) -> S[-1] = Sp
            r[i] says how to build S[i] from S[i-1]
                S[i] contains S[i-1](0) S[i-1](r)^\dag
        m  : which multipole operator is preserved by the hoppings
            m=0 -> charge
            m=1 -> dipole
            m=2 -> quadrupole
        J  : flat
            Strength of nearest neighbot ZZ interaction.

        Default parameters prepare Sp[0] Sm[1] Sm[2] Sp[3] + Sz[0] Sz[1]; summed over all possible sites
    """
    default_lattice = Chain
    force_default_lattice = True


    def init_sites(self, model_params):
        S = model_params.get('S', 0.5, 'real')
        conserve = model_params.get('conserve', 'best', str)
        if conserve == 'best':
            # check how much we can conserve
            if not model_params.any_nonzero([('Jx', 'Jy'), 'hx', 'hy', 'E'],
                                            "check Sz conservation"):
                conserve = 'Sz'
            elif not model_params.any_nonzero(['hx', 'hy'], "check parity conservation"):
                conserve = 'parity'
            else:
                conserve = None
            self.logger.info("%s: set conserve to %s", self.name, conserve)
        sort_charge = model_params.get('sort_charge', True, bool)
        site = SpinSite(S, conserve, sort_charge)
        
        # Build higher order raising and lowering operators
        # For spin S, we can have order 2S operators.
        # Spin-1/2 -> Sp, Sm
        # Spin-1   -> Sp2, Sm2, Sp, Sm
        # Spin-3/2 -> Sp3, Sm3, Sp2, Sm2, Sp, Sm
        # Sp, Sm are already added

        Sp = site.get_op('Sp')
        Sm = site.get_op('Sm')
        new_Sp = Sp.copy(deep=True)
        new_Sm = Sm.copy(deep=True)
        for i in range(2, int(np.round(2*S + 1))):
            new_Sp = npc.tensordot(new_Sp, Sp, axes=(['p*'], ['p']))
            new_Sm = npc.tensordot(new_Sm, Sm, axes=(['p*'], ['p'])) 
            site.add_op(f'Sp{i}', new_Sp, hc=f'Sm{i}')
            site.add_op(f'Sm{i}', new_Sm, hc=f'Sp{i}')
        return site

    def init_terms(self, model_params):
        # r - range of couplings
        # For each level of multipole, we want a list of ranges; these are used to recursively build the operator
        # at level n from those at level n-1, separated by distance r.
        # This starts with level 0; supposing r[0] = [i, j], the level 0 terms are S[0] = Sp_{x} Sm_{x+i} + Sp_{x} Sm_{x+j} + h.c.
        # Then the level n term S[n] is built from S[n-1] and r[n]; recursion starts with S[-1] = Sp

        # Default builds the dipole operator Sp_{x} Sm_{x+1} Sm_{x+2} Sp_{x+3}
        r = model_params.get('r', [[1],[2]], list)
        # m - which order multipole do the couplings conserve; m = 0 is charge, m=1 is dipole, m=2 is quadrupole
        m = model_params.get('m', 1, int)
        assert len(r) == m + 1, "Need to have a list of couplings for every multipole level, including 0"
        # Coupling strengths
        J = model_params.get('J', 1., 'real_or_array')      # Strength of multipole operators; assumed to be the same for all
        Jz = model_params.get('Jz', 1., 'real_or_array')    # Strength of ZZ NN coupling

        # Sp = Sx + i Sy, Sm = Sx - i Sy,  Sx = (Sp+Sm)/2, Sy = (Sp-Sm)/2i
        # Sx.Sx = 0.25 ( Sp.Sm + Sm.Sp + Sp.Sp + Sm.Sm )
        # Sy.Sy = 0.25 ( Sp.Sm + Sm.Sp - Sp.Sp - Sm.Sm )
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(Jz, u1, 'Sz', u2, 'Sz', dx)

        S_op = [np.array([1], dtype=int)]  # 1 means Sp
        for mi in range(m+1):
            # Build op at level mi
            ri = r[mi]  # How do we translate S_op
            new_S_op = []
            for s_op in S_op:
                ls = len(s_op)
                for rj in ri:
                    pad_s_op1 = np.zeros(rj+ls, dtype=int)
                    pad_s_op1[:ls] = np.array(s_op)
                    pad_s_op2 = np.zeros(rj+ls, dtype=int)
                    pad_s_op2[rj:] = np.array(s_op)

                    new_s_op = pad_s_op1 - pad_s_op2
                    new_S_op.append(new_s_op)
            S_op = new_S_op
            print(S_op)
                    
        for s_op in S_op:
            ops = []
            for ind, ss_op in enumerate(s_op):
                if ss_op == 0:      # Identity
                    continue
                elif ss_op == 1:    # Sp
                    op_name = 'Sp'
                elif ss_op == -1:   # Sm
                    op_name = 'Sm'
                elif ss_op > 1:
                    op_name = 'Sp' + str(ss_op)
                elif ss_op < -1:
                    op_name = 'Sm' + str(np.abs(ss_op))
                ops.append((op_name, [ind], 0))
            print(ops)
            self.add_multi_coupling(strength=J, ops=ops, plus_hc=True)
        # done
