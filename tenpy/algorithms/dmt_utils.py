import numpy as np
from copy import deepcopy

from ..models.lattice import Chain, TrivialLattice
from ..models.model import MPOModel, NearestNeighborModel
from ..networks.mps import MPSEnvironment, MPS
from ..networks.doubled_mps import DoubledMPS
from ..networks.site import DoubledSite
from ..linalg import np_conserved as npc
from ..linalg.charges import LegPipe
from ..algorithms.truncation import svd_theta, TruncationError, _machine_prec_trunc_par
import warnings

def double_model(H_MPO, NN=False, doubled=False, conjugate=False):
    if not doubled:
        doubled_MPO = H_MPO.make_doubled_MPO()
    else:
        doubled_MPO = deepcopy(H_MPO)

    if conjugate:
        doubled_MPO.conjugate_MPO([s.Q.conj() for s in doubled_MPO.sites])

    #doubled_lat = lat(H_MPO.L, doubled_MPO.sites[0]) # SAJANT - what if we have different types of sites in the lattice?
    doubled_lat = TrivialLattice(doubled_MPO.sites) # Trivial lattice uses the sites of the doubled MPO.
    # What if we don't want a chain lattice?

    doubled_model = MPOModel(doubled_lat, doubled_MPO)
    if NN:
        doubled_model = NearestNeighborModel.from_MPOModel(doubled_model)

    return doubled_model

def generate_pairs(lat, key='nearest_neighbors'):
    idXs, idYs = [], []
    for dx1, dx2, bv in lat.pairs[key]:
        idX, idY = lat.possible_couplings(dx1,dx2,bv)[:2]
        idXs.append(idX)
        idYs.append(idY)
    pairs = [sorted((a,b)) for a,b in zip(np.concatenate(idXs), np.concatenate(idYs))]
    return pairs

def distribute_pairs(pairs, bi, symmetric=True):
    cut_pairs = [a for a in pairs if a[0] <= bi and a[1] > bi]
    left_points, right_points =[], []
    if symmetric:
        left_points = list(set([cp[0] for cp in cut_pairs]))
        right_points = list(set([cp[1] for cp in cut_pairs]))
    else:
        for cp in cut_pairs:
            if cp[0] not in left_points and cp[1] not in right_points:
                # add to preserve list
                if len(left_points) < len(right_points):
                    left_points.append(cp[0])
                else:
                    right_points.append(cp[1])
    return left_points, right_points

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
    # SAJANT - Generalize this to case where the Identity isn't a unit vector.
    I = np.zeros((d,1,1))
    I[DMPS.sites[0].traceful_ind,0,0] = 1 # When taking the trace, we ought to account for the factor of

    return MPS.from_Bflat(DMPS.sites,
                               [I] * DMPS.L,
                               SVs=None,
                               bc='finite',
                               dtype=None,
                               permute=True,
                               form='B', # Form doesn't matter since it's a product state?
                               legL=None)

def build_QR_matrices(dMPS, i, dmt_par, trace_env, MPO_envs):
    QR_Ls, QR_Rs = [], []
    keep_L, keep_R = 0, 0
    local_par = dmt_par.get('k_local_par', None)
    conjoined_par = dmt_par.get('conjoined_par', None)
    if local_par is not None:
        k_local = local_par.get('k_local', (1,1)) # How many sites to include on either side of cut
        start_L = np.max([i+1-np.max([k_local[0], 1]), 0]) # include endpoint
        end_R = np.min([i+1+np.max([k_local[1], 1]), dMPS.L]) # do not include endpoint

        # Accounts for non-uniform local Hilbert space
        keep_L += np.prod([1] + [dMPS.dim[k] for k in range(start_L, i+1)]) if k_local[0] > 0 else 1
        keep_R += np.prod([1] + [dMPS.dim[k] for k in range(i+1, end_R)]) if k_local[1] > 0 else 1

        # Define basis change matrices - Eqn. 16 of paper
        # Get env strictly to the left of site i and contract the A form site i tensor to it.

        QR_L = trace_env._contract_with_LP(dMPS.get_B(start_L, form='A'), start_L)
        for k, j in enumerate(range(start_L+1, i+1)):
            A = dMPS.get_B(j, form='A').replace_label('p', 'p1')
            QR_L = npc.tensordot(QR_L, A, axes=['vR', 'vL'])  # axes_p + (vR*, vR)
            QR_L = QR_L.combine_legs(['p', 'p1']).ireplace_label('(p.p1)', 'p')
        QR_L = QR_L.squeeze() # Remove dummy leg associated with the trace state; remaining legs should be p, vR
        if k_local[0] == 0:
            # SAJANT - Assumes that the first index is the identity; use traceful_ind instead
            p_index = QR_L.get_leg_index('p')
            if isinstance(QR_L.legs[p_index], LegPipe):
                QR_L.legs[p_index] = QR_L.legs[p_index].to_LegCharge()
            QR_L.iproject([True] + [False] * QR_L.shape[p_index], 'p')
        QR_Ls.append(QR_L)

        QR_R = trace_env._contract_with_RP(dMPS.get_B(end_R-1, form='B'), end_R-1)
        for k, j in enumerate(reversed(range(i+1, end_R-1))):
            B = dMPS.get_B(j, form='B').replace_label('p', 'p1')
            QR_R = npc.tensordot(B, QR_R, axes=['vR', 'vL'])  # axes_p + (vL, vL*)
            QR_R = QR_R.combine_legs(['p', 'p1']).ireplace_label('(p.p1)', 'p')
        QR_R = QR_R.squeeze() # Remove dummy leg associated with the trace state; remaining legs should be p, vL
        if k_local[1] == 0:
            # SAJANT - Assumes that the first index is the identity; use traceful_ind instead
            p_index = QR_R.get_leg_index('p')
            if isinstance(QR_R.legs[p_index], LegPipe):
                QR_R.legs[p_index] = QR_R.legs[p_index].to_LegCharge()
            QR_R.iproject([True] + [False] * QR_R.shape[p_index], 'p')
            #QR_R.iproject([True] + [False] * QR_R.shape[QR_R.get_leg_index('p')], 'p')
        QR_Rs.append(QR_R)

    if MPO_envs is not None:
        for Me in MPO_envs:
            # Don't store envs since we are about to change the bond?
            # Maybe store them and delete them later if necessary.
            QR_L = Me.get_LP(i+1, store=True).squeeze().replace_label('wR', 'p')
            QR_R = Me.get_RP(i, store=True).squeeze().replace_label('wL', 'p')
            if np.linalg.norm([d.imag for d in QR_L._data]) < dmt_par.get('imaginary_cutoff', 1.e-12): # Remove small imaginary part
                QR_L.iunary_blockwise(np.real)
            if np.linalg.norm([d.imag for d in QR_R._data]) < dmt_par.get('imaginary_cutoff', 1.e-12): # Remove small imaginary part
                QR_R.iunary_blockwise(np.real)
            QR_Ls.append(QR_L)
            QR_Rs.append(QR_R)
            keep_L += QR_L.shape[QR_L.get_leg_index('p')]
            keep_R += QR_R.shape[QR_R.get_leg_index('p')]

    if conjoined_par is not None:
        pairs = conjoined_par.get('pairs')
        symmetric = conjoined_par.get('symmetric', True)
        left_pairs, right_pairs = distribute_pairs(pairs, i, symmetric=symmetric)
        #print(left_pairs, right_pairs, i)
        keep_L += int(np.sum([dMPS.dim[k]-1 for k in left_pairs])) + 1
        keep_R += int(np.sum([dMPS.dim[k]-1 for k in right_pairs])) + 1
        # left
        # SAJANT - only need identity if we aren't using another method
        QR_L = trace_env.get_LP(i+1, store=False).replace_label('vR*', 'p') # Identity
        QR_Ls.append(QR_L)
        for lp in left_pairs:
            QR_L = trace_env._contract_with_LP(dMPS.get_B(lp, form='A'), lp)
            p_index = QR_L.get_leg_index('p')
            if isinstance(QR_L.legs[p_index], LegPipe):
                QR_L.legs[p_index] = QR_L.legs[p_index].to_LegCharge()
            QR_L.iproject([False] + [True] * QR_L.shape[p_index], 'p') # Remove the identity leg
            # The QR rank reduction takes care of this.
            for j in range(lp+1, i+1):
                TT = trace_env.bra.get_B(j)
                A = npc.tensordot(TT.conj(), dMPS.get_B(j, form='A'), axes=(['p*'], ['p'])) # vL*, vR*, vL, VR
                QR_L = npc.tensordot(QR_L, A, axes=(['vR*', 'vR'], (['vL*', 'vL'])))  # axes_p + (vR*, vR)
            QR_L = QR_L.squeeze()
            QR_Ls.append(QR_L)

        # right
        # SAJANT - only need identity if we aren't using another method
        QR_R = trace_env.get_RP(i, store=False).replace_label('vL*', 'p') # Identity
        QR_Rs.append(QR_R)
        for rp in right_pairs:
            QR_R = trace_env._contract_with_RP(dMPS.get_B(rp, form='B'), rp)
            p_index = QR_R.get_leg_index('p')
            if isinstance(QR_R.legs[p_index], LegPipe):
                QR_R.legs[p_index] = QR_R.legs[p_index].to_LegCharge()
            QR_R.iproject([False] + [True] * QR_R.shape[p_index], 'p') # Remove the identity leg
            # The QR rank reduction takes care of this.
            for j in reversed(range(i+1, rp)):
                TT = trace_env.bra.get_B(j)
                B = npc.tensordot(TT.conj(), dMPS.get_B(j, form='B'), axes=(['p*'], ['p'])) # vL*, vR*, vL, VR
                QR_R = npc.tensordot(B, QR_R, axes=(['vR*', 'vR'], (['vL*', 'vL'])))  # axes_p + (vR*, vR)
            QR_R = QR_R.squeeze()
            QR_Rs.append(QR_R)
        #print([q.shape for q in QR_Ls], [q.shape for q in QR_Rs])
    QR_L = npc.concatenate(QR_Ls, axis='p')
    QR_R = npc.concatenate(QR_Rs, axis='p')

    return QR_L, QR_R, keep_L, keep_R, trace_env, MPO_envs

def dmt_theta(dMPS, i, svd_trunc_par, dmt_par,
              trace_env, MPO_envs,
              svd_trunc_par_2=_machine_prec_trunc_par): #, move_right=True):
    """Performs Density Matrix Truncation (DMT) on an MPS representing a density matrix or operator.
    We truncate on the bond between site i to the left and i+1 to the right. This however requires
    the entire state, as we use non-local properties to do the truncation.

    The DMT algorithm was propsed in https://arxiv.org/abs/1707.01506 and extended to 2D and
    long range interactions in https://arxiv.org/abs/2312.XXXXX

    See documentation of svd_theta, which we follow.

    We assume that the orthogonality center is already on bond i of the MPS. So tensor i should be in left ('A')
    form while tensor i+1 should be in right ('B') form. TEBD moves the OC by applying gates, but if we want to
    truncate a dMPS using this function, we need to explicitly move the OC.
    """
    # Make trace_env if none
    # Used to speed up identity contractions
    if trace_env is None:
        trace_env = MPSEnvironment(trace_identity_MPS(dMPS), dMPS)
    elif trace_env.ket is not dMPS:
        raise ValueError("Ket in 'trace_env' is not the current doubled MPS.")

    # Check that the MPS has the proper form; need A to the left of (i,i+1) and B to the right
    assert dMPS.form[:i] == [(1.0, 0.0) for _ in range(i)], dMPS.form[:i]
    assert dMPS.form[i+2:] == [(0.0, 1.0) for _ in range(dMPS.L - i - 2)], dMPS.form[i+2:]

    # We want to remove the environments containing the bond i
    trace_env.del_RP(i)
    trace_env.del_LP(i+1)
    if MPO_envs is not None:
        for Me in MPO_envs:
            Me.del_RP(i)
            Me.del_LP(i+1)

    old_A = dMPS.get_B(i, form='A')
    old_B = dMPS.get_B(i+1, form='B')
    old_S = S = dMPS.get_SR(i) # singular values to the right of site i
    chi = len(S)
    if chi == 1:
        return TruncationError(), 1, trace_env, MPO_envs

    QR_L, QR_R, keep_L, keep_R, trace_env, MPO_envs = build_QR_matrices(dMPS, i, dmt_par, trace_env, MPO_envs)
    Q_L, R_L = npc.qr(QR_L.itranspose(['vR', 'p']),
                          mode='complete',
                          inner_labels=['vR*', 'vR'],
                          #cutoff=1.e-12, # Need this to be none to have square Q
                          pos_diag_R=True,
                          inner_qconj=QR_L.get_leg('vR').conj().qconj)
    Q_R, R_R = npc.qr(QR_R.itranspose(['vL', 'p']),
                          mode='complete',
                          inner_labels=['vL*', 'vL'],
                          #cutoff=1.e-12, # Need this to be none to have square Q
                          pos_diag_R=True,
                          inner_qconj=QR_R.get_leg('vL').conj().qconj)
    """
    If any of the diagonal elements of R_L/R are zero, we get the warning:
    WARNING : /global/common/software/m3859/tenpy_sajant/tenpy/linalg/np_conserved.py:3993: RuntimeWarning: invalid value encountered in true_divide
    phase = r_diag / np.abs(r_diag)
    """
    # SAJANT - Do I need to worry about charge of Q and R? Q is chargeless by default.

    # SAJANT - Do this without converting to numpy array for charge conservation; not sure how to get diagonals of the R
    # QL may be rank difficient. Let's project out rows (p) that are unneeded, based on the QR
    R_cutoff = dmt_par.get('R_cutoff', 1.e-14)
    R_truncate = dmt_par.get('R_truncate', True)
    projs_L = np.diag(R_L.to_ndarray()) > R_cutoff
    projs_R = np.diag(R_R.to_ndarray()) > R_cutoff
    if np.any(projs_L == False) and R_truncate:
        p_index = QR_L.get_leg_index('p')
        if isinstance(QR_L.legs[p_index], LegPipe):
            QR_L.legs[p_index] = QR_L.legs[p_index].to_LegCharge()
        QR_L.iproject(projs_L, 'p')
        keep_L = np.sum(projs_L)
        Q_L, R_L = npc.qr(QR_L.itranspose(['vR', 'p']),
                          mode='complete',
                          inner_labels=['vR*', 'vR'],
                          #cutoff=1.e-12, # Need this to be none to have square Q
                          pos_diag_R=True,
                          inner_qconj=QR_L.get_leg('vR').conj().qconj)

    if np.any(projs_R == False) and R_truncate:
        p_index = QR_R.get_leg_index('p')
        if isinstance(QR_R.legs[p_index], LegPipe):
            QR_R.legs[p_index] = QR_R.legs[p_index].to_LegCharge()
        QR_R.iproject(projs_R, 'p')
        keep_R = np.sum(projs_R)
        Q_R, R_R = npc.qr(QR_R.itranspose(['vL', 'p']),
                          mode='complete',
                          inner_labels=['vL*', 'vL'],
                          #cutoff=1.e-12, # Need this to be none to have square Q
                          pos_diag_R=True,
                          inner_qconj=QR_R.get_leg('vL').conj().qconj)

    # SAJANT - Should this be moved until after we remove redundant rows?
    if keep_L >= chi or keep_R >= chi:
        # We cannot truncate, so return.
        # Nothing is done to the MPS, except for moving the OC one site ot the left
        return TruncationError(), 1, trace_env, MPO_envs

    # Build M matrix, Eqn. 15 of paper
    M = npc.tensordot(Q_L, Q_R.scale_axis(S, axis='vL'), axes=(['vR', 'vL'])).ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])

    connected = dmt_par.get('connected', True)
    # Connected component
    if connected:
        orig_M = M
        if np.isclose(orig_M[0,0], 0.0):
            connected = False
        else:
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
    if svd_trunc_par.get('chi_max', None) == 0:
        # Only keep the part necessary to preserve local properties; not sure when we would want to do this.
        # So we kill the entire lower right block
        sM *= 0
    # SAJANT - do something with err so that truncation error is recorded?

    # Lower right block; norm is `renormalization`
    M_DR_trunc = npc.tensordot(UM, VM.scale_axis(sM, axis='vL'), axes=(['vR', 'vL']))

    M_trunc = M.copy()
    M_trunc[keep_L:, keep_R:] = M_DR_trunc

    if connected:
        M_trunc = M_trunc + npc.outer(orig_M.take_slice([0], ['vR']),
                               orig_M.take_slice([0], ['vL'])) / orig_M[0,0]

    # SAJANT - Set svd_min to 0 to make sure no SVs are dropped? Or do we need some cutoff to remove the
    # SVs corresponding to the rank we removed earlier from M_DR
    U, S, VH, err2, renormalization2 = svd_theta(M_trunc, svd_trunc_par_2, renormalize=True)
    err2 = TruncationError.from_norm(renormalization2)
    # M_trunc would have norm 1 if we did no truncation; so the new norm (given by `renormalization2`) is akin to
    # the error.

    # We want to remove the environments containing the bond i
    trace_env.del_RP(i)
    trace_env.del_LP(i+1)
    if MPO_envs is not None:
        for Me in MPO_envs:
            Me.del_RP(i)
            Me.del_LP(i+1)
    new_A = npc.tensordot(npc.tensordot(dMPS.get_B(i, form='A'), Q_L.conj(), axes=(['vR', 'vR*'])), U, axes=(['vR', 'vL']))
    new_B = npc.tensordot(VH, npc.tensordot(Q_R.conj(), dMPS.get_B(i+1, form='B'), axes=(['vL*', 'vL'])),axes=(['vR', 'vL']))
    dMPS.set_SR(i, S)
    dMPS.set_B(i, new_A, form='A')
    dMPS.set_B(i+1, new_B, form='B')
    #dMPS.test_sanity() # SAJANT - remove this for miniscule speed boost?

    return err + err2, renormalization2, trace_env, MPO_envs

"""
def dmt_theta_OLD(dMPS, i, svd_trunc_par, dmt_par, trace_env, MPO_envs, connected=True, move_right=True):
    # Make trace_env if none
    # Used to speed up identity contractions
    if trace_env is None:
        trace_env = MPSEnvironment(trace_identity_MPS(dMPS), dMPS)
    elif trace_env.ket is not dMPS:
        raise ValueError("Ket in 'trace_env' is not the current doubled MPS.")

    # Check that the MPS has the proper form; need A to the left of (i,i+1) and B to the right
    assert dMPS.form[:i] == [(1.0, 0.0) for _ in range(i)], [(1.0, 0.0) for _ in range(i)]
    assert dMPS.form[i+2:] == [(0.0, 1.0) for _ in range(dMPS.L - i - 2)], [(0.0, 1.0) for _ in range(dMPS.L - i - 2)]


    # SAJANT - Is this really needed?
    # Use QR instead of SVD
    if move_right:
        theta = dMPS.get_B(i, form='Th')
        U, s, VH, err, renormalization0 = svd_theta(theta.combine_legs(['vL', 'p']), trunc_par={'chi_max': 0}, renormalize=True)
        dMPS.set_B(i, U.split_legs(), form='A')
        dMPS.set_B(i+1, npc.tensordot(VH, dMPS.get_B(i+1, form='B'), axes=(['vR', 'vL'])), form='B')
        dMPS.set_SR(i, s)

        trace_env.del_RP(i)
        if MPO_envs is not None:
            for Me in MPO_envs:
                Me.del_RP(i)
    else:
        theta = dMPS.get_B(i+1, form='Th')
        U, s, VH, err, renormalization0 = svd_theta(theta.combine_legs(['p', 'vR']), trunc_par={'chi_max': 0}, renormalize=True)
        dMPS.set_B(i+1, VH.split_legs(), form='B')
        dMPS.set_B(i, npc.tensordot(dMPS.get_B(i, form='A'), U, axes=(['vR', 'vL'])), form='A')
        dMPS.set_SR(i, s)

        trace_env.del_LP(i+1)
        if MPO_envs is not None:
            for Me in MPO_envs:
                Me.del_LP(i+1)
    dMPS.norm *= renormalization0
    S = dMPS.get_SR(i) # singular values to the right of site i
    chi = len(S)

    QR_Ls, QR_Rs = [], []
    keep_L, keep_R = 0, 0
    local_par = dmt_par.get('k_local_par', None)
    conjoined_par = dmt_par.get('conjoined_par', None)
    if local_par is not None:
        k_local = local_par.get('k_local', (1,1)) # How many sites to include on either side of cut
        start_L = np.max([i+1-np.max([k_local[0], 1]), 0]) # include endpoint
        end_R = np.min([i+1+np.max([k_local[1], 1]), dMPS.L]) # do not include endpoint

        # Accounts for non-uniform local Hilbert space
        keep_L += np.prod([1] + [dMPS.dim[k] for k in range(start_L, i+1)]) if k_local[0] > 0 else 1
        keep_R += np.prod([1] + [dMPS.dim[k] for k in range(i+1, end_R)]) if k_local[1] > 0 else 1

        # Define basis change matrices - Eqn. 16 of paper
        # Get env strictly to the left of site i and contract the A form site i tensor to it.

        QR_L = trace_env._contract_with_LP(dMPS.get_B(start_L, form='A'), start_L)
        for k, j in enumerate(range(start_L+1, i+1)):
            A = dMPS.get_B(j, form='A').replace_label('p', 'p1')
            QR_L = npc.tensordot(QR_L, A, axes=['vR', 'vL'])  # axes_p + (vR*, vR)
            QR_L = QR_L.combine_legs(['p', 'p1']).ireplace_label('(p.p1)', 'p')
        QR_L = QR_L.squeeze() # Remove dummy leg associated with the trace state; remaining legs should be p, vR
        if k_local[0] == 0:
            # SAJANT - Assumes that the first index is the identity; use traceful_ind instead
            QR_L.iproject([True] + [False] * QR_L.shape[QR_L.get_leg_index('p')], 'p')
        QR_Ls.append(QR_L)

        QR_R = trace_env._contract_with_RP(dMPS.get_B(end_R-1, form='B'), end_R-1)
        for k, j in enumerate(reversed(range(i+1, end_R-1))):
            B = dMPS.get_B(j, form='B').replace_label('p', 'p1')
            QR_R = npc.tensordot(B, QR_R, axes=['vR', 'vL'])  # axes_p + (vL, vL*)
            QR_R = QR_R.combine_legs(['p', 'p1']).ireplace_label('(p.p1)', 'p')
        QR_R = QR_R.squeeze() # Remove dummy leg associated with the trace state; remaining legs should be p, vL
        if k_local[1] == 0:
            # SAJANT - Assumes that the first index is the identity; use traceful_ind instead
            QR_R.iproject([True] + [False] * QR_R.shape[QR_R.get_leg_index('p')], 'p')
        QR_Rs.append(QR_R)

    if MPO_envs is not None:
        for Me in MPO_envs:
            # Don't store envs since we are about to change the bond?
            # Maybe store them and delete them later if necessary.
            QR_L = Me.get_LP(i+1, store=True).squeeze().replace_label('wR', 'p')
            QR_R = Me.get_RP(i, store=True).squeeze().replace_label('wL', 'p')
            QR_Ls.append(QR_L)
            QR_Rs.append(QR_R)
            keep_L += QR_L.shape[QR_L.get_leg_index('p')]
            keep_R += QR_R.shape[QR_R.get_leg_index('p')]

    if conjoined_par is not None:
        pairs = conjoined_par.get('pairs')
        symmetric = conjoined_par.get('symmetric', True)
        left_pairs, right_pairs = distribute_pairs(pairs, i, symmetric=symmetric)
        print(left_pairs, right_pairs, i)
        keep_L += np.sum([dMPS.dim[k]-1 for k in left_pairs]) + 1
        keep_R += np.sum([dMPS.dim[k]-1 for k in right_pairs]) + 1

        # left
        # SAJANT - only need identity if we aren't using another method
        QR_L = trace_env.get_LP(i+1, store=False).replace_label('vR*', 'p') # Identity
        QR_Ls.append(QR_L)
        for lp in left_pairs:
            QR_L = trace_env._contract_with_LP(dMPS.get_B(lp, form='A'), lp)
            QR_L.iproject([False] + [True] * QR_L.shape[QR_L.get_leg_index('p')], 'p') # Remove the identity leg
            # The QR rank reduction takes care of this.
            for j in range(lp+1, i+1):
                TT = trace_env.bra.get_B(j)
                A = npc.tensordot(TT.conj(), dMPS.get_B(j, form='A'), axes=(['p*'], ['p'])) # vL*, vR*, vL, VR
                QR_L = npc.tensordot(QR_L, A, axes=(['vR*', 'vR'], (['vL*', 'vL'])))  # axes_p + (vR*, vR)
            QR_L = QR_L.squeeze()
            QR_Ls.append(QR_L)

        # right
        # SAJANT - only need identity if we aren't using another method
        QR_R = trace_env.get_RP(i, store=False).replace_label('vL*', 'p') # Identity
        QR_Rs.append(QR_R)
        for rp in right_pairs:
            QR_R = trace_env._contract_with_RP(dMPS.get_B(rp, form='B'), rp)
            QR_R.iproject([False] + [True] * QR_R.shape[QR_R.get_leg_index('p')], 'p') # Remove the identity leg
            # The QR rank reduction takes care of this.
            for j in reversed(range(i+1, rp)):
                TT = trace_env.bra.get_B(j)
                B = npc.tensordot(TT.conj(), dMPS.get_B(j, form='B'), axes=(['p*'], ['p'])) # vL*, vR*, vL, VR
                QR_R = npc.tensordot(B, QR_R, axes=(['vR*', 'vR'], (['vL*', 'vL'])))  # axes_p + (vR*, vR)
            QR_R = QR_R.squeeze()
            QR_Rs.append(QR_R)
        #print([q.shape for q in QR_Ls], [q.shape for q in QR_Rs])

    QR_L = npc.concatenate(QR_Ls, axis='p')
    QR_R = npc.concatenate(QR_Rs, axis='p')

    if keep_L >= chi or keep_R >= chi:
        # We cannot truncate, so return.
        # Nothing is done to the MPS, except for moving the OC one site ot the left
        return TruncationError(), 1, trace_env, MPO_envs

    Q_L, R_L = npc.qr(QR_L.itranspose(['vR', 'p']),
                          mode='complete',
                          inner_labels=['vR*', 'vR'],
                          #cutoff=1.e-12, # Need this to be none to have square Q
                          pos_diag_R=True,
                          inner_qconj=QR_L.get_leg('vR').conj().qconj)
    Q_R, R_R = npc.qr(QR_R.itranspose(['vL', 'p']),
                          mode='complete',
                          inner_labels=['vL*', 'vL'],
                          #cutoff=1.e-12, # Need this to be none to have square Q
                          pos_diag_R=True,
                          inner_qconj=QR_R.get_leg('vL').conj().qconj)
    # SAJANT - Do I need to worry about charge of Q and R? Q is chargeless by default.

    # SAJANT - Do this without converting to numpy array for charge conservation; not sure how to get diagonals of the R
    # QL may be rank difficient. Let's project out rows (p) that are unneeded, based on the QR
    projs_L = np.diag(R_L.to_ndarray()) > 1.e-14
    projs_R = np.diag(R_R.to_ndarray()) > 1.e-14
    if np.any(projs_L == False):
        QR_L.iproject(projs_L, 'p')
        keep_L = np.sum(projs_L)
        Q_L, R_L = npc.qr(QR_L.itranspose(['vR', 'p']),
                          mode='complete',
                          inner_labels=['vR*', 'vR'],
                          #cutoff=1.e-12, # Need this to be none to have square Q
                          pos_diag_R=True,
                          inner_qconj=QR_L.get_leg('vR').conj().qconj)

    if np.any(projs_R == False):
        QR_R.iproject(projs_R, 'p')
        keep_R = np.sum(projs_R)
        Q_R, R_R = npc.qr(QR_R.itranspose(['vL', 'p']),
                          mode='complete',
                          inner_labels=['vL*', 'vL'],
                          #cutoff=1.e-12, # Need this to be none to have square Q
                          pos_diag_R=True,
                          inner_qconj=QR_R.get_leg('vL').conj().qconj)

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

    M_trunc = M.copy()
    M_trunc[keep_L:, keep_R:] = M_DR_trunc

    if connected:
        M_trunc = M_trunc + npc.outer(orig_M.take_slice([0], ['vR']),
                               orig_M.take_slice([0], ['vL'])) / orig_M[0,0]

    # SAJANT - Set svd_min to 0 to make sure no SVs are dropped? Or do we need some cutoff to remove the
    # SVs corresponding to the rank we removed earlier from M_DR
    U, S, VH, err2, renormalization2 = svd_theta(M_trunc, {'chi_max': np.max([M_trunc.get_leg('vL').ind_len,
                                                                              M_trunc.get_leg('vR').ind_len]),
                                                           'svd_min': 1.e-14}, renormalize=True)
    dMPS.norm *= renormalization2
    err2 = TruncationError.from_norm(renormalization2 * renormalization0)
    # M_trunc would have norm 1 if we did no truncation; so the new norm (given by `renormalization2`) is akin to
    # the error.
    if move_right:
        VH.iscale_axis(S, axis='vL')
        form_left = 'A'
        form_right = 'Th'
        trace_env.del_RP(i)
        if MPO_envs is not None:
            for Me in MPO_envs:
                Me.del_RP(i)
                Me.del_LP(i+1)
    else:
        U.iscale_axis(S, axis='vR')
        form_left = 'Th'
        form_right = 'B'
        trace_env.del_LP(i+1)
        if MPO_envs is not None:
            for Me in MPO_envs:
                Me.del_LP(i+1)
                Me.del_RP(i)
    new_A = npc.tensordot(npc.tensordot(dMPS.get_B(i, form='A'), Q_L.conj(), axes=(['vR', 'vR*'])), U, axes=(['vR', 'vL']))
    new_B = npc.tensordot(VH, npc.tensordot(Q_R.conj(), dMPS.get_B(i+1, form='B'), axes=(['vL*', 'vL'])),axes=(['vR', 'vL']))
    dMPS.set_SR(i, S)
    dMPS.set_B(i, new_A, form=form_left)
    dMPS.set_B(i+1, new_B, form=form_right)
    dMPS.test_sanity() # SAJANT - remove this for miniscule speed boost?

    return err2, renormalization2 * renormalization0, trace_env, MPO_envs
"""
