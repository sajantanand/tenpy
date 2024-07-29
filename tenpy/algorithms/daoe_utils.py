import numpy as np
from copy import deepcopy

from ..networks.mpo import MPO
from ..networks.site import DoubledSite
from ..linalg import np_conserved as npc
from ..linalg.charges import LegCharge
import warnings

def weight_distribution_MPO(L, sites, max_weight=-1):
    """
    Given some operator represented as an MPS with sites such that the Identity
    operator is in index 0 (the site may permute this later), we want to measure the
    operator weight distribution.

    args:
        L: int
            length of produced MPO
        sites: list of doubled Sites
            Sites used to contruct MPO; tiled commensurately if fewer than L provided
        max_weight: int
            To what Pauli weight do we count? We count strings of length 0, ..., max_weight - 1, >= max_weight.
            So `max_weight=2` means separately count strings of weight 0, 1, and >= 2

    returns:
        mpo: MPO
            MPO with dangling wR leg on site L-1
    """

    if max_weight == -1:
        # Count weight 0 <-> L
        max_weight = L

    return DAOE_MPO(L, sites, 0, max_weight, danging_right=True)

def DAOE_MPO(L, sites, gamma, lstar, danging_right=False):
    """
    Build the MPO specified in https://arxiv.org/abs/2004.05177. This MPO represents the superoperator
    $$ \hat{S} = e^{-\gamma max[0, (\hat{l} - l_star)]}. $$
    When applied to an operator $O$ with length $l$, this superoperator applies $e^{-gamma (l - l_star)}$
    if $l - l_star > 0$ and 1 otherwise. This has the effect of reducing the amplitude of long strings.

    If we conserve 'Sz', then we need to define this in terms of the non-Hermitian doubled basis,
    {I, Z, S+, S-} for qubits. If we don't conserve anything, we can use the Hermitian doubled basis,
    {I, X, Y, Z} for qubits, or the non-Hermitian basis. We should define the DAOE superoperator in this
    basis, convert back to the standard |i><j| basis, and then rotate to the desired basis for evolution.
    
    args:
        L: int
            length of produced MPO
        sites: list of doubled Sites
            Sites used to contruct MPO; tiled commensurately if fewer than L provided
        gamma: float
            Strength of the dissipation; e^{-gamma}
        lstar: int
            At what string length do we start applying e^{-gamma}; all strings of length l > l_star are
            dissipated according to their excess length
        dangling_right: bool
            Do we have a dangling virtual bond at the end of the MPO? This is used to "post-select" on the
            desired length of operator string.

    returns:
        mpo: MPO
            MPO with dangling wR leg on site L-1
    """
    D = lstar + 1

    damp_matrix = np.zeros((D, D))
    for i in range(D-1):
        damp_matrix[i,i+1] = 1
    damp_matrix[D-1,D-1] = np.exp(-gamma)

    bulk_tensors = []
    bulk_tensors_npc = []
    for i in range(L):
        s = sites[i % len(sites)]
        
        # build dummy site
        if s.conserve == 'Sz':
            ds = DoubledSite(int(np.sqrt(s.dim)), conserve=s.conserve, hermitian=False, trivial=False)
        elif s.conserve == 'None':
            ds = DoubledSite(int(np.sqrt(s.dim)), conserve=s.conserve, hermitian=True, trivial=False)
        else:
            raise NotImplementedError(f"conservation{s.conserve} not yet implemented for DAOE.")
            
        leg_p = ds.leg
        leg_vL = LegCharge.from_trivial(D, ds.leg.chinfo, qconj=+1).sort()[1]
        inverted_perm = np.argsort(ds.perm)

        d = ds.dim
        bulk_tensor = np.zeros((D, D, d, d))
        bulk_tensor[:,:,0,0] = np.eye(D)
        for i in range(1, d):
            bulk_tensor[:,:,i,i] = damp_matrix

        bulk_tensors.append(bulk_tensor)
        bulk_tensor_npc = npc.Array.from_ndarray(bulk_tensor[:,:,inverted_perm[:,None],inverted_perm[None,:]],
                                                 [leg_vL, leg_vL.conj(), leg_p, leg_p.conj()], dtype=np.float64,
                                                 qtotal=None, labels=['wL','wR', 'p', 'p*'])

        # rotate back to |i><j| basis
        bulk_tensor_npc = npc.tensordot(npc.tensordot(ds.d2s, bulk_tensor_npc, axes=(['p*'], ['p'])), ds.s2d, axes=(['p*'], ['p'])).transpose(['wL','wR', 'p', 'p*'])
        bulk_tensors_npc.append(bulk_tensor_npc)

    bulk_tensors_npc[0] = bulk_tensors_npc[0][0,:,:,:].add_trivial_leg(axis=0, label='wL', qconj=+1)
    if not danging_right:
        bulk_tensors_npc[-1] = bulk_tensors_npc[-1][:,-1,:,:].add_trivial_leg(axis=-1, label='wR', qconj=-1)
    mpo = MPO(sites, bulk_tensors_npc, bc='finite', IdL=0, IdR =-1)

    return mpo

def weight_split_truncate(psi, max_weight, apply_naively=False, options={}):
    """
    We want to split a wavefunction represeting operators into two MPSs, one containing all strings
    with weight < max_weight and another with all remaining strings.
    """
    wd_MPO = weight_distribution_MPO(psi.L, psi.sites, max_weight=max_weight)
    leg_wL = wd_MPO.get_W[-1].get_leg('wR').conj()

    proj1 = np.ones(max_weight + 1)
    proj1[-1] = 0

    proj2 = np.zeros(max_weight + 1)
    proj2[-1] = 1

    phis = []
    for i, proj in zip([proj1, proj2]):
        phi = psi.copy()

        proj = npc.Array.from_ndarray(proj, [leg_wL], dtype=np.float64, qtotal=None, labels=['wL']).add_trivial_leg(axis=-1, label='wR', qconj=-1)
        proj_MPO = wd_MPO.copy()
        proj_MPO._W[-1] = npc.tensordot(proj_MPO._W[-1], proj, axes=(['wR'], ['wL']))
        print(phi.norm)
        if i == 0 or apply_naively:
            # For the MPS with short strings, apply the projection but don't do any truncation.
            phi = proj_MPO.apply_naively(phi)
            phi.canonical_form()
        else:
            # Truncate long strings
            phi = proj_MPO.apply(phi, options)
        print(phi.norm)
        phis.append(phi)
    return phis[0].add(phis[1], 1, 1), phis