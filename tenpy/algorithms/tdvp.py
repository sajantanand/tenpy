"""Time Dependant Variational Principle (TDVP) with MPS (finite version only).

The TDVP MPS algorithm was first proposed by :cite:`haegeman2011`. However the stability of the
algorithm was later improved in :cite:`haegeman2016`, that we are following in this implementation.
The general idea of the algorithm is to project the quantum time evolution in the manyfold of MPS
with a given bond dimension. Compared to e.g. TEBD, the algorithm has several advantages:
e.g. it conserves the unitarity of the time evolution and the energy (for the single-site version),
and it is suitable for time evolution of Hamiltonian with arbitrary long range in the form of MPOs.
We have implemented:

1. The one-site formulation following the TDVP principle in :class:`SingleSiteTDVPEngine`,
   which **does not** allow for growth of the bond dimension.

2. The two-site algorithm in the :class:`TwoSiteTDVPEngine`, which does allow the bond
   dimension to grow - but requires truncation as in the TEBD case, and is no longer strictly TDVP,
   i.e. it does *not* strictly preserve the energy.

Much of the code is very similar to DMRG, and also based on the
:class:`~tenpy.algorithms.mps_common.Sweep` class.

.. versionchanged :: 0.10.0
    The interface changed compared to version 0.9.0:
    Just :class:`TDVPEngine` will result in a error.
    Use :class:`SingleSiteTDVPEngine` or :class:`TwoSiteTDVPEngine` instead.


.. todo ::
    extend code to infinite MPS

.. todo ::
    allow for increasing bond dimension in SingleSiteTDVPEngine, similar to DMRG Mixer
"""
# Copyright (C) TeNPy Developers, Apache license

from ..linalg.krylov_based import LanczosEvolution
from ..linalg.truncation import svd_theta, TruncationError, _machine_prec_trunc_par
from .mps_common import Sweep, ZeroSiteH, OneSiteH, TwoSiteH
from .algorithm import TimeEvolutionAlgorithm, TimeDependentHAlgorithm
from ..linalg import np_conserved as npc
from ..algorithms import dmt_utils as dmt
from ..tools.misc import consistency_check
from ..tools.params import asConfig

import numpy as np
import time
import warnings
import logging

logger = logging.getLogger(__name__)

__all__ = ['TDVPEngine', 'SingleSiteTDVPEngine', 'TwoSiteTDVPEngine',
           'TimeDependentSingleSiteTDVP', 'TimeDependentTwoSiteTDVP']


class TDVPEngine(TimeEvolutionAlgorithm, Sweep):
    """Time dependent variational principle algorithm for MPS.

    This class contains all methods that are generic between
    :class:`SingleSiteTDVPEngine` and :class:`TwoSiteTDVPEngine`.
    Use the latter two classes for actual TDVP runs.

    .. versionchanged :: 1.1
        Previously had separate `lanczos_options`, which have been renamed to `lanczos_params`
        for consistency with the Sweep class.

    Parameters
    ----------
    psi, model, options, **kwargs:
        Same as for :class:`~tenpy.algorithms.algorithm.Algorithm`.

    Options
    -------
    .. cfg:config :: TDVPEngine
        :include: TimeEvolutionAlgorithm, Sweep

        max_dt : float | None
            Threshold for raising errors on too large time steps. Default ``1.0``.
            See :meth:`~tenpy.tools.misc.consistency_check`.
            For large time steps, the projection to the MPS manifold that is the main building block
            of TDVP, can not be a good approximation anymore. We raise in that case.
            Can be downgraded to a warning by setting this option to ``None``.

    """
    EffectiveH = None

    def __init__(self, psi, model, options, **kwargs):
        if self.__class__.__name__ == 'TDVPEngine':
            msg = ("TDVP interface changed. \n"
                   "The new TDVPEngine has subclasses SingleSiteTDVPEngine"
                   " and TwoSiteTDVPEngine that you can use.\n"
                   )
            raise NameError(msg)
        if psi.bc != 'finite':
            raise NotImplementedError("Only finite TDVP is implemented")
        assert psi.bc == model.lat.bc_MPS
        options = asConfig(options, self.__class__.__name__)
        options.deprecated_alias("lanczos_options", "lanczos_params",
                                 "See also https://github.com/tenpy/tenpy/issues/459")
        super().__init__(psi, model, options, **kwargs)
        self.lanczos_params = self.options.subconfig('lanczos_params')
        self.Krylov_options = self.options.subconfig('Krylov_options')

    # run() from TimeEvolutionAlgorithm

    @property
    def lanczos_options(self):
        """Deprecated alias of :attr:`lanczos_params`."""
        warnings.warn("Accessing deprecated alias TDVPEngine.lanczos_options instead of lanczos_params",
                      FutureWarning, stacklevel=2)
        return self.lanczos_params

    def prepare_evolve(self, dt):
        """Expand the basis using Krylov or random vectors using the algorithm from `:cite:yang20202`.

        This action of this function is specified by the 'Krylov_options' field of the options passed when constructing the
        TDVP engine. Below, I list the possible keys of the 'Krylov_options' dictionary:

        1. Krylov_expansion_dim: how many additional vectors do we use to expand the basis; > 1 is sufficient for random extension.

        2. mpo: what MPO do we use for expanion? If none is specified, we use the Hamiltonian. If 'None' is specified, we do
            random extension. If a list is given, one applies multiple MPOs to get the next Krylov vector, e.g. with WII and a
            higher order time step.

        3. trunc_params: standard dictionary for truncation settings.
                chi_max: max number of states that are added on each site.
                svd_min: cutoff for kept sqrt(eigenvalues) of the RDM

        4. apply_mpo_options: how do we apply the MPO to the MPS; e.g. SVD, zip_up, variational and associated parameters.
        """
        Krylov_expansion_dim = self.Krylov_options.get('expansion_dim', 0)
        if Krylov_expansion_dim > 0:    # Do some basis expansion
            # Need to clear out left and right environments since the bond dimensions no longer match.
            # So we will need to recalculate the H envs for the next TDVP step
            # Do this before expanding the basis of psi to save RAM.
            self.env.clear()

            # If we do expansion and are using DMT, the trace and MPO envs are no longer valid as we have changed the state.
            # So we must remove these.
            trace_env = self.options.get('trace_env', None)
            MPO_envs = self.options.get('MPO_envs', None)
            # Remove any existing environments, since applying the MPO will mess them up.
            new_psi = self.psi.copy()
            if trace_env is not None:
                trace_env.ket = new_psi
            if MPO_envs is not None:
                for ME in MPO_envs:
                    ME.ket = new_psi

            logger.info(f"Original bond dimension: {self.psi.chi}.")
            # Get the MPO A that will be used to generate Krylov vectors; {A^k |psi>}
            # We might want to use the WII MPO or (1 - itH) rather than H
            Krylov_mpo = self.Krylov_options.get('mpo', self.model.H_MPO)
            Krylov_trunc_params = self.Krylov_options.subconfig('trunc_params', self.trunc_params)    # How do we truncate the RDMs when extending?
            if Krylov_mpo is None:  # Random expansion
                extension_err = self.psi.subspace_expansion(expand_into=[], trunc_par=Krylov_trunc_params)
            else:                   # Expansion by MPO application
                # Cast to list to allow for multiple mpos (i.e. W2 with order > 1)
                Krylov_mpo = [Krylov_mpo] if not isinstance(Krylov_mpo, list) else Krylov_mpo
                # First generate Krylov basis
                Krylov_apply_mpo_options = self.Krylov_options.subconfig('apply_mpo_options')
                Krylov_apply_mpo_options.update({'trace_env': trace_env,
                                                 'MPO_envs': MPO_envs})
                # Needs to contain 'compression_method' and options for doing the MPO application
                Krylov_extended_basis = []
                for i in range(Krylov_expansion_dim):
                    for krylov_mpo in Krylov_mpo:
                        krylov_mpo.apply(new_psi, Krylov_apply_mpo_options)
                    Krylov_extended_basis.append(new_psi.copy())
                extension_err = self.psi.subspace_expansion(expand_into=Krylov_extended_basis, trunc_par=Krylov_trunc_params)
                if 'trace_env' in Krylov_apply_mpo_options.keys():
                    del Krylov_apply_mpo_options['trace_env']
                if 'MPO_envs' in Krylov_apply_mpo_options.keys():
                    del Krylov_apply_mpo_options['MPO_envs']
            logger.info(f"Extended bond dimension: {self.psi.chi}.")

            if trace_env is not None:
                trace_env.clear()
                trace_env.ket = self.psi
                self.options['trace_env'] = trace_env
            if MPO_envs is not None:
                for ME in MPO_envs:
                    ME.clear()
                    ME.ket = self.psi
                self.options['MPO_envs'] = MPO_envs
        return

    def evolve(self, N_steps, dt):
        """Evolve by ``N_steps * dt``.

        Parameters
        ----------
        N_steps : int
            The number of steps to evolve.
        """
        consistency_check(dt, self.options, 'max_dt', 1.,
                          'dt > ``max_dt`` is unreasonably large for TDVP.',
                          compare=lambda dt, max_dt: abs(dt) <= max_dt)
        self.dt = dt
        trunc_err = TruncationError()
        for _ in range(N_steps):
            self.sweep()
            for eps in self.trunc_err_list:
                trunc_err += TruncationError(eps, 1 - 2 * eps)
        self.evolved_time = self.evolved_time + N_steps * self.dt
        return trunc_err


class TwoSiteTDVPEngine(TDVPEngine):
    """Engine for the two-site TDVP algorithm.

    Parameters
    ----------
    psi, model, options, **kwargs:
        Same as for :class:`~tenpy.algorithms.algorithm.Algorithm`.

    Options
    -------
    .. cfg:config :: TwoSiteTDVPEngine
        :include: TDVPEngine

    """
    EffectiveH = TwoSiteH

    def __init__(self, psi, model, options, **kwargs):
        super().__init__(psi, model, options, **kwargs)

    def get_sweep_schedule(self):
        """Slightly different sweep schedule than DMRG"""
        L = self.psi.L
        if self.finite:
            i0s = list(range(0, L - 2)) + list(range(L - 2, -1, -1))
            move_right = [True] * (L - 2) + [False] * (L - 2) + [None]
            update_LP_RP = [[True, False]] * (L - 2) + [[False, True]] * (L - 2) + [[False, False]]
        else:
            raise NotImplementedError("Only finite TDVP is implemented")
        return zip(i0s, move_right, update_LP_RP)

    def update_local(self, theta, **kwargs):
        i0 = self.i0
        L = self.psi.L

        dt = -0.5j * self.dt
        if i0 == L - 2:
            dt = 2. * dt  # instead of updating the last pair of sites twice, we double the time
        # update two-site wavefunction
        theta, N = LanczosEvolution(self.eff_H, theta, self.lanczos_params).run(dt)
        if npc.norm(theta.unary_blockwise(np.imag)) < self.imaginary_cutoff: # Remove small imaginary part
            # Needed for Lindblad evolution in Hermitian basis where density matrix / operator must be real
            theta.iunary_blockwise(np.real)
        if self.combine:
            theta.itranspose(['(vL.p0)', '(p1.vR)'])  # shouldn't do anything
        else:
            theta = theta.combine_legs([['vL', 'p0'], ['p1', 'vR']], new_axes=[0, 1],
                                       qconj=[+1, -1])
        qtotal_i0 = self.psi.get_B(i0, form=None).qtotal
        U, S, VH, err, renormalize = svd_theta(theta,
                                     self.trunc_params,
                                     qtotal_LR=[qtotal_i0, None],
                                     inner_labels=['vR', 'vL'])
        self.psi.norm *= renormalize

        B0 = U.split_legs(['(vL.p0)']).replace_label('p0', 'p')
        B1 = VH.split_legs(['(p1.vR)']).replace_label('p1', 'p')

        self.psi.set_B(i0, B0, form='A')  # left-canonical
        self.psi.set_B(i0 + 1, B1, form='B')  # right-canonical
        self.psi.set_SR(i0, S)
        update_data = {'err': err, 'N': N, 'U': U, 'VH': VH}
        # earlier update of environments, since they are needed for the one_site_update()
        super().update_env(**update_data)  # new environments, e.g. LP[i0+1] on right move.

        if self.move_right:
            # note that i0 == L-2 is left-moving
            self.one_site_update(i0 + 1, 0.5j * self.dt)
        elif (self.move_right is False):
            self.one_site_update(i0, 0.5j * self.dt)
        # for the last update of the sweep, where move_right is None, there is no one_site_update

        return update_data

    def update_env(self, **update_data):
        """Do nothing; super().update_env() is called explicitly in :meth:`update_local`."""
        pass

    def one_site_update(self, i, dt):
        H1 = OneSiteH(self.env, i, combine=False)
        theta = self.psi.get_theta(i, n=1, cutoff=self.S_inv_cutoff)
        theta = H1.combine_theta(theta)
        theta, _ = LanczosEvolution(H1, theta, self.lanczos_params).run(dt)
        if npc.norm(theta.unary_blockwise(np.imag)) < self.imaginary_cutoff: # Remove small imaginary part
            # Needed for Lindblad evolution in Hermitian basis where density matrix / operator must be real
            theta.iunary_blockwise(np.real)
        self.psi.set_B(i, theta.replace_label('p0', 'p'), form='Th')


class DMTTwoSiteTDVPEngine(TwoSiteTDVPEngine):
    """Engine for the two-site TDVP algorithm.

    Parameters
    ----------
    psi, model, options, **kwargs:
        Same as for :class:`~tenpy.algorithms.algorithm.Algorithm`.

    Options
    -------
    .. cfg:config :: TDVP
        :include: TimeEvolutionAlgorithm

        trunc_params : dict
            Truncation parameters as described in :func:`~tenpy.algorithms.truncation.truncate`
        lanczos_params : dict
            Lanczos params as described in :cfg:config:`Lanczos`.

    Attributes
    ----------
    options: dict
        Optional parameters.
    evolved_time : float | complex
        Indicating how long `psi` has been evolved, ``psi = exp(-i * evolved_time * H) psi(t=0)``.
    psi : :class:`~tenpy.networks.mps.MPS`
        The MPS, time evolved in-place.
    env : :class:`~tenpy.networks.mpo.MPOEnvironment`
        The environment, storing the `LP` and `RP` to avoid recalculations.
    lanczos_params : :class:`~tenpy.tools.params.Config`
        Params passed on to :class:`~tenpy.linalg.lanczos.LanczosEvolution`.
    """

    def update_local(self, theta, **kwargs):
        """
        The trace can change when doing TDVP since we evolve the entirety of the Theta matrix.
        In principle, we could evolve all of the Theta matrix except for the (0,0) entry so that the
        norm is guaranteed to be unchanged. I'm not sure how to do this in practice.
        """
        i0 = self.i0
        L = self.psi.L

        dt = self.dt
        if i0 == L - 2:
            dt = 2. * dt  # instead of updating the last pair of sites twice, we double the time
        # update two-site wavefunction
        theta, N = LanczosEvolution(self.eff_H, theta, self.lanczos_params).run(-0.5j * dt)
        if npc.norm(theta.unary_blockwise(np.imag)) < self.imaginary_cutoff: # Remove small imaginary part
            # Needed for Lindblad evolution in Hermitian basis where density matrix / operator must be real
            theta.iunary_blockwise(np.real)
        if self.combine:
            theta.itranspose(['(vL.p0)', '(p1.vR)'])  # shouldn't do anything
        else:
            theta = theta.combine_legs([['vL', 'p0'], ['p1', 'vR']], new_axes=[0, 1],
                                       qconj=[+1, -1])
        qtotal_i0 = self.psi.get_B(i0, form=None).qtotal
        svd_trunc_params_0 = self.options.get('svd_trunc_params_0', _machine_prec_trunc_par)
        #print(f"Bond {i0}:", dmt.trace_identity_MPS(self.psi).overlap(self.psi) * 2**(self.psi.L//2))
        U, S, VH, err, renormalize = svd_theta(theta,
                                     svd_trunc_params_0,
                                     qtotal_LR=[qtotal_i0, None],
                                     inner_labels=['vR', 'vL'])
        self.psi.norm *= renormalize
        B0 = U.split_legs(['(vL.p0)']).replace_label('p0', 'p')
        B1 = VH.split_legs(['(p1.vR)']).replace_label('p1', 'p')

        self.psi.set_B(i0, B0, form='A')  # left-canonical
        self.psi.set_B(i0 + 1, B1, form='B')  # right-canonical
        self.psi.set_SR(i0, S)
        #print(f"Bond {i0}:", dmt.trace_identity_MPS(self.psi).overlap(self.psi) * 2**(self.psi.L//2))
        dmt_params = self.options['dmt_params']
        trace_env = self.options.get('trace_env', None)
        MPO_envs = self.options.get('MPO_envs', None)
        timing = self.options.get('timing', False)
        svd_trunc_params_2 = self.options.get('svd_trunc_params_2', _machine_prec_trunc_par)

        trunc_err2, renormalize, trace_env, MPO_envs = dmt.dmt_theta(self.psi, i0, self.trunc_params, dmt_params, trace_env=trace_env, MPO_envs=MPO_envs, svd_trunc_params_2=svd_trunc_params_2, timing=timing)
        self.psi.norm *= renormalize

        # Need to keep track of the envs for use on future steps
        self.options['trace_env'] = trace_env
        self.options['MPO_envs'] = MPO_envs

        U = self.psi.get_B(i0, form='A').replace_label('p', 'p0').combine_legs(['vL', 'p0'])
        VH = self.psi.get_B(i0+1, form='B').replace_label('p', 'p1').combine_legs(['p1', 'vR'])
        update_data = {'err': err+trunc_err2, 'N': N, 'U': U, 'VH': VH}
        # earlier update of environments, since they are needed for the one_site_update()
        super().update_env(**update_data)  # new environments, e.g. LP[i0+1] on right move.
        #print(f"Bond {i0}:", dmt.trace_identity_MPS(self.psi).overlap(self.psi) * 2**(self.psi.L//2))

        #print(f"Bond {i0}")
        #self.env.clear() # TODO: we don't want to clear everything!


        #print(self.psi.form)
        if self.move_right:
            # note that i0 == L-2 is left-moving
            self.one_site_update(i0 + 1, 0.5j * self.dt)
        elif (self.move_right is False):
            self.one_site_update(i0, 0.5j * self.dt)
        # for the last update of the sweep, where move_right is None, there is no one_site_update
        #print(f"Bond {i0}:", dmt.trace_identity_MPS(self.psi).overlap(self.psi) * 2**(self.psi.L//2))
        #print(self.psi.form)
        #print(self.psi.chi)
        return update_data

    def one_site_update(self, i, dt):
        H1 = OneSiteH(self.env, i, combine=False)
        theta = self.psi.get_theta(i, n=1, cutoff=self.S_inv_cutoff)
        theta = H1.combine_theta(theta)
        #print(npc.norm(theta))
        theta, _ = LanczosEvolution(H1, theta, self.lanczos_params).run(dt)
        #print(npc.norm(theta))
        if npc.norm(theta.unary_blockwise(np.imag)) < self.imaginary_cutoff: # Remove small imaginary part
            # Needed for Lindblad evolution in Hermitian basis where density matrix / operator must be real
            theta.iunary_blockwise(np.real)
        self.psi.set_B(i, theta.replace_label('p0', 'p'), form='Th')

class SingleSiteTDVPEngine(TDVPEngine):
    """Engine for the single-site TDVP algorithm.

    Parameters
    ----------
    psi, model, options, **kwargs:
        Same as for :class:`~tenpy.algorithms.algorithm.Algorithm`.

    Options
    -------
    .. cfg:config :: SingleSiteTDVPEngine
        :include: TDVPEngine

    """
    EffectiveH = OneSiteH

    def get_sweep_schedule(self):
        """slightly different sweep schedule than DMRG"""
        L = self.psi.L
        if self.finite:
            i0s = list(range(0, L - 1)) + list(range(L - 1, -1, -1))
            move_right = [True] * (L - 1) + [False] * (L - 1) + [None]
            update_LP_RP = [[True, False]] * (L - 1) + [[False, True]] * (L - 1) + [[False, False]]
        else:
            raise NotImplementedError("Only finite TDVP is implemented")
        return zip(i0s, move_right, update_LP_RP)

    def update_local(self, theta, **kwargs):
        i0 = self.i0
        L = self.psi.L

        dt = -0.5j * self.dt
        if i0 == L - 1:
            dt = 2. * dt  # instead of updating the last site twice, we double the time

        # update one-site wavefunction
        theta, N = LanczosEvolution(self.eff_H, theta, self.lanczos_params).run(dt)
        if npc.norm(theta.unary_blockwise(np.imag)) < self.imaginary_cutoff: # Remove small imaginary part
            # Needed for Lindblad evolution in Hermitian basis where density matrix / operator must be real
            theta.iunary_blockwise(np.real)
        if self.move_right:
            self.right_moving_update(i0, theta)
        else:
            # note: left_moving_update() also covers the "non-moving" case move_right=None
            # of the last update in a sweep
            self.left_moving_update(i0, theta)
        return {}  # no truncation error in single-site TDVP!

    def right_moving_update(self, i0, theta):
        if self.combine:
            theta.itranspose(['(vL.p0)', 'vR'])
        else:
            theta = theta.combine_legs(['vL', 'p0'], qconj=+1, new_axes=0)
        U, S, VH = npc.svd(theta, qtotal_LR=[theta.qtotal, None], inner_labels=['vR', 'vL'])
        # no truncation
        A0 = U.split_legs(['(vL.p0)']).replace_label('p0', 'p')
        self.psi.set_B(i0, A0, form='A')  # left-canonical
        self.psi.set_SR(i0, S)

        if True:  # note that i0 == L - 1 is left moving, so we always do a zero-site update
            super().update_env(U=U)
            theta = VH.scale_axis(S, 'vL')
            theta, H0 = self.zero_site_update(i0 + 1, theta, 0.5j * self.dt)
            next_B = self.psi.get_B(i0 + 1, form='B')
            next_th = npc.tensordot(theta, next_B, axes=['vR', 'vL'])
            self.psi.set_B(i0 + 1, next_th, form='Th')  # used and updated for next i0

    def left_moving_update(self, i0, theta):
        if self.combine:
            theta.itranspose(['vL', '(p0.vR)'])
        else:
            theta = theta.combine_legs(['p0', 'vR'], qconj=-1, new_axes=1)
        U, S, VH = npc.svd(theta, qtotal_LR=[None, theta.qtotal], inner_labels=['vR', 'vL'])
        if i0 == 0:
            assert U.shape == (1, 1)
            VH *= U[0, 0]  # just a global phase, but better keep it!
        B1 = VH.split_legs(['(p0.vR)']).replace_label('p0', 'p')
        self.psi.set_B(i0, B1, form='B')  # right-canonical
        self.psi.set_SL(i0, S)

        if i0 != 0:  # left-moving, but not the last site of the update
            super().update_env(VH=VH)  # note: no update needed if i0=0!
            theta = U.iscale_axis(S, 'vR')
            theta, H0 = self.zero_site_update(i0, theta, 0.5j * self.dt)
            next_A = self.psi.get_B(i0 - 1, form='A')
            next_th = npc.tensordot(next_A, theta, axes=['vR', 'vL'])
            self.psi.set_B(i0 - 1, next_th, form='Th')  # used and updated for next i0
            # note: this zero-site update can change the singular values on the bond left of i0.
            # however, we *don't* save them in psi: it turns out that the right singular
            # values for correct expectation values/entropies are the ones set before the if above.
            # (Believe me - I had that coded up and spent days looking for the bug...)

    def update_env(self, **update_data):
        """Do nothing; super().update_env() is called explicitly in :meth:`update_local`."""
        pass

    def zero_site_update(self, i, theta, dt):
        """Zero-site update on the left of site `i`."""
        H0 = ZeroSiteH(self.env, i)
        theta, _ = LanczosEvolution(H0, theta, self.lanczos_params).run(dt)
        return theta, H0

    def post_update_local(self, **update_data):
        self.trunc_err_list.append(0.)  # avoid error in return of sweep()


class TimeDependentSingleSiteTDVP(TimeDependentHAlgorithm,SingleSiteTDVPEngine):
    """Variant of :class:`SingleSiteTDVPEngine` that can handle time-dependent Hamiltonians.

    See details in :class:`~tenpy.algorithms.algorithm.TimeDependentHAlgorithm` as well.
    """
    def reinit_model(self):
        # recreate model
        TimeDependentHAlgorithm.reinit_model(self)
        # and reinitialize environment accordingly
        self.init_env(self.model)


class TimeDependentTwoSiteTDVP(TimeDependentHAlgorithm,TwoSiteTDVPEngine):
    """Variant of :class:`TwoSiteTDVPEngine` that can handle time-dependent Hamiltonians.

    See details in :class:`~tenpy.algorithms.algorithm.TimeDependentHAlgorithm` as well.
    """

    def reinit_model(self):
        TimeDependentSingleSiteTDVP.reinit_model(self)
