import numpy as np
from copy import deepcopy

from ..models.lattice import Chain
from ..models.model import MPOModel, NearestNeighborModel
from ..networks.mps import MPSEnvironment, MPS
from ..networks.doubled_mps import DoubledMPS
from ..networks.site import DoubledSite
from ..linalg import np_conserved as npc
from ..algorithms.truncation import svd_theta, TruncationError
import warnings

def double_model(H_MPO, NN=False, doubled=False, conjugate=False, lat=None):
    if lat is None:
        lat = Chain
        
    if not doubled:
        doubled_MPO = H_MPO.make_doubled_MPO()
    else:
        doubled_MPO = deepcopy(H_MPO)
    
    if conjugate:
        doubled_MPO.conjugate_MPO(doubled_MPO.sites[0].Q.conj())
    
    doubled_lat = lat(H_MPO.L, doubled_MPO.sites[0]) # SAJANT - what if we have different types of sites in the lattice?
    # What if we don't want a chain lattice?
    
    doubled_model = MPOModel(doubled_lat, doubled_MPO)
    if NN:
        doubled_model = NearestNeighborModel.from_MPOModel(doubled_model)
        
    return doubled_model

# Bra for density matrix expectation value.
def trace_identity_DMPS(DMPS, traceful_id=None):
    d = DMPS.sites[0].dim
    # We are in the computational / bra-ket basis. Make doubled MPS with two physical legs
    # per site, using the identity
    I = np.eye(d).reshape(d, d, 1, 1)
    return DoubledMPS.from_Bflat(DMPS.sites,
                               [I] * DMPS.L,
                               SVs=None,
                               bc='finite',
                               dtype=None,
                               permute=True,
                               form='B', # Form doesn't matter since it's a product state?
                               legL=None)

# Bra for density matrix expectation value.
def trace_identity_MPS(DMPS, traceful_id=None):
    d = DMPS.sites[0].dim
    assert type(DMPS.sites[0]) is DoubledSite
    # In rotated, HOMT basis. Identity is vector vector with 1 at location specified by traceful_id
    I = np.zeros((d,1,1))
    I[DMPS.sites[0].traceful_ind,0,0] = 1.
    
    return MPS.from_Bflat(DMPS.sites,
                               [I] * DMPS.L,
                               SVs=None,
                               bc='finite',
                               dtype=None,
                               permute=True,
                               form='B', # Form doesn't matter since it's a product state?
                               legL=None)


def dmt_theta(dMPS, i, svd_trunc_par, dmt_par, trace_env, MPO_env, connected=True, move_right=True):
    """Performs Density Matrix Truncation (DMT) on an MPS representing a density matrix or operator.
    We truncate on the bond between site i to the left and i+1 to the right. This however requires
    the entire state, as we use non-local properties to do the truncation.
    
    The DMT algorithm was propsed in https://arxiv.org/abs/1707.01506 and extended to 2D and
    long range interactions in https://arxiv.org/abs/2312.XXXXX
    
    See documentation of svd_theta, which we follow.
    """
    if move_right:
        theta = dMPS.get_B(i, form='Th')
        U, s, VH, err, renormalization = svd_theta(theta.combine_legs(['vL', 'p']), trunc_par={'chi_max': 0})
        dMPS.set_B(i, U.split_legs(), form='A')
        dMPS.set_SR(i, s)
        dMPS.set_B(i+1, npc.tensordot(VH, dMPS.get_B(i+1, form='B'), axes=(['vR', 'vL'])), form='B')
    else:
        theta = dMPS.get_B(i+1, form='Th')
        U, s, VH, err, renormalization = svd_theta(theta.combine_legs(['p', 'vR']), trunc_par={'chi_max': 0})
        dMPS.set_B(i+1, VH.split_legs(), form='B')
        dMPS.set_SR(i, s)
        dMPS.set_B(i, npc.tensordot(dMPS.get_B(i, form='A'), U, axes=(['vR', 'vL'])), form='A')
    
    
    S = dMPS.get_SR(i) # singular values to the right of site i
    chi = len(S)
    # SAJANT - NEAREST NEIGHBOR FOR NOW

    # Make trace_env if none
    if trace_env is None:
        trace_env = MPSEnvironment(trace_identity_MPS(dMPS), dMPS)
    elif trace_env.ket is not dMPS:
        raise ValueError("Ket in 'trace_env' is not the current doubled MPS.")
    
    keep_L = 4
    keep_R = 4
    if keep_L >= chi or keep_R >= chi:
        # We cannot truncate, so return.
        # Nothing is done to the MPS.
        return err, renormalization
        
    # Define basis change matrices - Eqn. 16 of paper
    # Get env strictly to the left of site i and contract the A form site i tensor to it.
    QR_L = trace_env._contract_with_LP(dMPS.get_B(i, form='A'), i)
    QR_L = QR_L.squeeze() # Remove dummy leg associated with the trace state; remaining legs should be p, vR
    Q_L, R_L = npc.qr(QR_L.itranspose(['vR', 'p']),
                      mode='complete',
                      inner_labels=['vR*', 'vR'], 
                      #cutoff=1.e-12, # Need this to be none to have square Q
                      pos_diag_R=True,
                      inner_qconj=QR_L.get_leg('vR').conj().qconj)

    QR_R = trace_env._contract_with_RP(dMPS.get_B(i+1, form='B'), i+1)
    QR_R = QR_R.squeeze() # Remove dummy leg associated with the trace state; remaining legs should be p, vL
    Q_R, R_R = npc.qr(QR_R.itranspose(['vL', 'p']),
                      mode='complete',
                      inner_labels=['vL*', 'vL'], 
                      #cutoff=1.e-12, # Need this to be none to have square Q
                      pos_diag_R=True,
                      inner_qconj=QR_R.get_leg('vL').conj().qconj)
    # SAJANT - Do I need to worry about charge of Q and R? Q is chargless by default.

    
    # Build M matrix, Eqn. 15 of paper
    M = npc.tensordot(Q_L, Q_R.scale_axis(S, axis='vL'), axes=(['vR', 'vL'])).ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
    
    # Connected component
    if connected:
        orig_M = M
        M = orig_M - npc.outer(orig_M.take_slice([0], ['vR']), 
                               orig_M.take_slice([0], ['vL'])) / orig_M[0,0]
    
    # M_DR is lower right block of M
    M_DR = M.copy()
    M_DR.iproject([[False] * keep_L + [True] * (M_DR.get_leg('vL').ind_len - keep_L),
                      [False] * keep_R + [True] * (M_DR.get_leg('vR').ind_len - keep_R)],
                     ['vL', 'vR'])
    
    # Do SVD of M_prime block, truncating according to svd_trunc_par
    # We DO NOT normalize the SVs of M_DR before truncating, so the svd_trunc_par['svd_min'] is the
    # cutoff used to discard singular values. The error returned by svd_theta is the total weight of
    # discard SVs**2, divided by the original norm squared of all of the SVs.
    # This is the error in SVD truncation, given a non-normalized initial state.
    UM, sM, VM, err, renormalization = svd_theta(M_DR, svd_trunc_par, renormalize=False)
    # All we want to do here is reduce the rank of M_DR
    if svd_trunc_par.get('chi_max', 100) == 0:
        # Only keep the part necessary to preserve local properties; not sure when we would want to do this.
        sM *= 0
        
    # Lower right block; norm is `renormalization`
    M_DR_trunc = npc.tensordot(UM, VM.scale_axis(sM, axis='vL'), axes=(['vR', 'vL']))

    # SAJANT - Fix This; don't convert to numpy
    M_trunc = M.copy()
    M_trunc[keep_L:, keep_R:] = M_DR_trunc
    
    if connected:
        M_trunc = M_trunc + npc.outer(orig_M.take_slice([0], ['vR']), 
                               orig_M.take_slice([0], ['vL'])) / orig_M[0,0]
    
    # SAJANT - Set svd_min to 0 to make sure no SVs are dropped? Or do we need some cutoff to remove the
    # SVs corresponding to the rank we removed earlier from M_DR
    U, S, VH, err2, renormalization2 = svd_theta(M_trunc, {'chi_max': np.max([M_trunc.get_leg('vL').ind_len, 
                                                                              M_trunc.get_leg('vR').ind_len]),
                                                           'svd_min': 1.e-14})
    # M_trunc would have norm 1 if we did no truncation; so the new norm (given by `renormalization2`) is akin to
    # the error.
    
    if move_right:
        VH.iscale_axis(S, axis='vL')
        form_left = 'A'
        form_right = 'Th'
    else:
        U.iscale_axis(S, axis='vR')
        form_left = 'Th'
        form_right = 'B'
    new_A = npc.tensordot(npc.tensordot(dMPS.get_B(i, form='A'), Q_L.conj(), axes=(['vR', 'vR*'])), U, axes=(['vR', 'vL']))
    new_B = npc.tensordot(VH, npc.tensordot(Q_R.conj(), dMPS.get_B(i+1, form='B'), axes=(['vL*', 'vL'])),axes=(['vR', 'vL']))
    
    dMPS.set_SR(i, S)
    dMPS.set_B(i, new_A, form=form_left)
    dMPS.set_B(i+1, new_B, form=form_right)
    dMPS.test_sanity() # SAJANT - remove this for miniscule speed boost?
    return err + err2, renormalization2
