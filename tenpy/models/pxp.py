"""Implementation of the PXP model and variants on a chain."""
# Copyright (C) TeNPy Developers, Apache license

import numpy as np

from ..networks.site import SpinHalfSite
from .lattice import Chain
from .model import CouplingMPOModel

__all__ = ['PXPChain', 'PXXZPChain', 'PExpPChain']


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
        return s

    def init_terms(self, model_params):
        J = model_params.get('J', 2.0, 'real_or_array')
        self.add_multi_coupling(J, [('P0', [-1], 0), ('X', [0], 0), ('P0', [1], 0)])

        if model_params['bc_x'] == 'open':
            L = model_params['L']
            # If J is an array, I am not sure what J_boundary will do.
            J_boundary = model_params.get('J_boundary', J, 'real_or_array')
            self.add_coupling_term(J_boundary, 0, 1, 'X', 'P0')
            self.add_coupling_term(J_boundary, L - 2, L - 1, 'P0', 'X')


class PXXZPChain(CouplingMPOModel):
    r"""The PXXZP model, i.e. a chain of Rydberg-blockaded atoms with a U(1) symmetric XXZ interaction.

    The Hamiltonian reads:

    .. math ::
        H =  \mathtt{J} \sum_{i} P_{i-1} (X_i X_{i+1} + Y_{i} Y_{i+1} + Z_{i} Z_{i+1}) P_{i+1}
            + \mathtt{J_z} \sum_{i} P_{i-1} Z_{i} Z_{i+1} P_{i+1}
            + \mathtt{J_boundary} (X_0 X_1 + Y_0 Y_1 ) P_2 
            + P_{L-3} (X_{L-2} X_{L-1} + Y_{L-2} Y_{L-1})
            + \mathtt{J_boundaryz} Z_0 Z_1 P_2 + P_{L-3} Z_{L-2} Z_{L-1}

    where we only add the boundary terms for open boundaries with `J_boundary` defaulting to `J`.
    `P` is the projector onto the up state of the site, which corresponds to the ground state
    of the atom.

    This model respects both the U(1) magnetization symmetry and the Rydberg blockade. In principle,
    this model can be realized in dipolar Rydberg simulators, where the native interaction is a dipolar
    XY flip-flop and Van der Wals ZZ; however, due to the long-range dipolar (alpha=3) tails of the interaction, it is not
    immediately clear how to get a NN model with an infinitely strong blockade. So this model is
    more of theoretical interest.

    Options
    -------
    .. cfg:config :: PXYPChain
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
            conserve = 'Sz'
        s = SpinHalfSite(conserve=conserve)
        # P is defined as P0, the projector onto the state 0, i.e. the up spin
        return s

    def init_terms(self, model_params):
        J = model_params.get('J', 2.0, 'real_or_array')
        Jz = model_params.get('Jz', 0.0, 'real_or_array')

        self.add_multi_coupling(Jz, [('P0', [-1], 0), ('Sigmaz', [0], 0), ('Sigmaz', [1], 0), ('P0', [2], 0)])
        # Sp = Sx + i Sy; Sx = 1/2 Sigma x
        # Sp Sm + Sm Sp = 2(Sx Sx + Sy Sy) = 1/2 (Sigmax Sigmax + Sigmay Sigmay)
        self.add_multi_coupling(J*2.0, [('P0', [-1], 0), ('Sp', [0], 0), ('Sm', [1], 0), ('P0', [2], 0)], plus_hc=True)

        if model_params['bc_x'] == 'open':
            L = model_params['L']
            # If J is an array, I am not sure what J_boundary will do.
            J_boundary = model_params.get('J_boundary', J, 'real_or_array')
            J_boundary_z = model_params.get('J_boundary_z', Jz, 'real_or_array')
            self.add_multi_coupling_term(Jz, [0, 1, 2], ['Sigmaz', 'Sigmaz', 'P0'])
            self.add_multi_coupling_term(Jz, [L-3, L-2, L-1], ['P0', 'Sigmaz', 'Sigmaz'])
            self.add_multi_coupling_term(J*2.0, [0, 1, 2], ['Sp', 'Sm', 'P0'], plus_hc=True)
            self.add_multi_coupling_term(J*2.0, [L-3, L-2, L-1], ['P0', 'Sp', 'Sm'], plus_hc=True)


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
            if not model_params.any_nonzero([J], "check Sz conservation"):
                conserve = 'Sz'
            else:
                conserve = 'parity'
            self.logger.info("%s: set conserve to %s", self.name, conserve)
        s = SpinHalfSite(conserve=conserve)
        # P is defined as P0, the projector onto the state 0, i.e. the up spin
        return s

    def init_terms(self, model_params):
        raise NotImplementedError("Not implemented yet. See comment in code.")
        """
        Implementing P X N^k Y P interactions requires generalization of the `add_exponentially_decaying_coupling`. \\
                                   N will be the `op_string` placed between operators `A` and `B`, but now, the end point operators act on TWO \\
                                   sites. This then generates all possible P X Y P, P X N Y P, etc. operators.")
        """
