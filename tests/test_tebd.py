"""A collection of tests to check the functionality of `tenpy.tebd`"""
from __future__ import division

import itertools as it
import tenpy.linalg.np_conserved as npc
import numpy as np
from tenpy.networks.mps import MPS
from tenpy.models.xxz_chain import XXZChain
import tenpy.algorithms.tebd as tebd
from tenpy.algorithms.exact_diag import ExactDiag


def check_tebd(bc_MPS='finite'):
    xxz_pars = dict(L=2, Jxx=1., Jz=3., hz = 0.,bc_MPS=bc_MPS)
    L = xxz_pars['L']
    M = XXZChain(xxz_pars)
    state = ([0, 1] * L)[:L]  # Neel
    psi = MPS.from_product_state(M.lat.mps_sites(), state, bc=bc_MPS)

    tebd_param = {
        'verbose': 2,
        'chi_max':100
        }

    engine = tebd.Engine(psi,M,tebd_param)
    engine.run_GS()

    if bc_MPS == 'finite':
        ED = ExactDiag(M)
        ED.build_full_H_from_mpo()
        ED.full_diagonalization()
        psi_ED = ED.groundstate()
        ov = npc.inner(psi_ED, ED.mps_to_full(psi), do_conj=True)
        print "compare with ED: overlap = ", abs(ov)**2
        assert (abs(abs(ov) - 1.) < 1.e-10)

    for i in range(13):
        engine.run()
    # TODO: time evolve mit TEBD

    # TODO: compare with known ground state (energy) / ED !


def test_tebd():
    for bc_MPS in ['finite']:
        yield check_tebd, bc_MPS


if __name__ == "__main__":
    for f_args in test_tebd():
        f = f_args[0]
        print "=" * 80
        print ' '.join([str(a) for a in f_args])
        print "=" * 80
        f(*f_args[1:])
