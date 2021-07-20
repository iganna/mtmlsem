"""
Function to optimize mtmlSEM model
"""

import numpy as np
import scipy.stats as st

from semopy import Model
from dataset import *


class OptBayes:

    t_lat, t_obs, t_ord = ['latent', 'observed', 'ordinal']
    def __init__(self, relations, data: Data, var_ord=None):
        """

        :param sem:
        :param data:
        :param var_ord:
        """
        # Get relationships between variables
        # relations = sem.inspect()
        v_endo = relations['lval'][relations['op'] == '~']
        v_exo = relations['rval'][relations['op'] == '~']
        estims = relations['Estimate'][relations['op'] == '~']
        v_endo_uniq = v_endo.unique().tolist()

        # Split variables
        vars = list(set(v_exo) | set(v_endo))

        self.v = dict()
        self.v[self.t_lat] = [v for v in vars
                              if v not in data.d_all.columns]
        self.v[self.t_obs] = [v for v in vars
                              if v in data.d_all.columns]

        self.n = data.n_samples

        self.data = dict()
        self.data[self.t_lat] = np.random.rand(self.n, len(self.v[self.t_lat]))
        self.data[self.t_obs] = data.d_all.loc[:, self.v[self.t_obs]].to_numpy()

        if var_ord is not None:
            self.v[self.t_ord] = list(var_ord)
            self.data[self.t_ord] = np.random.rand(self.n, len(self.v[self.t_ord]))
        else:
            self.v[self.t_ord] = var_ord




        self.endo = []
        self.exo = []
        self.params = []
        self.reff = []  # random effect
        self.priors = []

        for v in v_endo_uniq:
            self.endo += [self.find_type_and_id(v)]
            exo_tmp = v_exo[v_endo == v].tolist()
            priors_tmp = estims[v_endo == v].tolist()
            self.exo += [[self.find_type_and_id(u) for u in exo_tmp]]
            # self.params += [[0] * len(self.exo[-1])]
            self.params += [np.random.rand(len(self.exo[-1]))]
            self.priors += [np.array(priors_tmp)]
            self.reff += [[]]

        self.mcmc = []
        # # add fake equations for exogenous latent variables - it needs covariance
        # for i_lat in range(len(self.v[self.t_lat])):
        #     if i_lat in []:
        #         continue



        self.n_eq = len(self.endo)  # number of equations

        # alpha and beta params for error terms
        self.e_ab_prior = [(3, 10)] * self.n_eq
        self.e = [0.1] * self.n_eq

        # # pre-calculation of covariance matrix between parameters in each equation
        # self.phi = []
        # for ieq in range(self.n_eq):


        # get help rapameters for Structural and Measurement parts
        self.create_sem_parts()


    def find_type_and_id(self, v):
        for t in self.v.keys():
            if v in self.v[t]:
                return (t, self.v[t].index(v))

        raise ValueError('Undefined variable or type')


    def optimize(self, n_mcmc = 1000):

        for _ in range(n_mcmc):

            self.update_latent()

            # for ismpl in range(self.n):
            #     self.update_ordinal(ismpl)

            for ieq in range(self.n_eq):
                self.update_params(ieq)

            self.mcmc += [[item for sublist in self.params for item in sublist]]
            # self.update_reff()

        # self.mcmc = np.array(self.mcmc)

        n_burnin = round(n_mcmc/10)
        mcmc_params = np.array(self.mcmc[n_burnin:n_mcmc]).mean(axis=0)
        id = 0
        for ieq in range(self.n_eq):
            for i in range(len(self.params[ieq])):
                self.params[ieq][i] = mcmc_params[id]
                id += 1


    def create_sem_parts(self):
        """
        This function creates measurement and structural parts
        :return:
        """
        # Structural part
        # names of latent variables do NOT make sense, because it is
        self.n_lat = len(self.v[self.t_lat])
        # get equations corresponding to latent variables
        ieq_lat = [[i for i, endo in enumerate(self.endo)
                    if endo[0] == self.t_lat and endo[1] == i_lat]
                   for i_lat in range(self.n_lat)]
        for ieq in ieq_lat:
            if len(ieq) > 1:
                raise ValueError('Something is going wrong in update_latent')
        ieq_lat = [ieq[0] if len(ieq) > 0 else -1
                   for ieq in ieq_lat]

        self.ieq_lat = ieq_lat

        # Measurement part
        # get indexes of equations, where endogenous variables are not latent
        self.ieq_mp = [i for i, endo in enumerate(self.endo)
                  if endo[0] is not self.t_lat]
        self.n_mp = len(self.ieq_mp)



    def update_latent(self):

        # ======================================================
        # Structural part
        #
        # get B matrix
        # and get (Пg -> mx_pg)  for all samples (n x n_eq_lat)
        # ------------------------------------------------------
        mx_beta = np.zeros((self.n_lat, self.n_lat))
        mx_pg = np.zeros((self.n_lat, self.n))
        psi1 = np.zeros(self.n_lat)

        for ilat, ieq in enumerate(self.ieq_lat):
            if ieq == -1:
                psi1[ilat] = 1
                continue

            psi1[ilat] = self.e[ieq]
            for iparam, exo in enumerate(self.exo[ieq]):
                t, i = exo
                if t is self.t_lat:
                    mx_beta[ilat, i] = self.params[ieq][iparam]
                else:
                    mx_pg[ilat, :] += self.params[ieq][iparam] * self.data[t][:, i]

        # get C matrix
        mx_c = np.linalg.inv(np.identity(self.n_lat) - mx_beta)

        # ======================================================
        # Measurement part
        #
        # get Lambda matrix
        # get (p-Ky) -> mx_pky  for all samples (n x n_eq_phen)
        # ------------------------------------------------------
        mx_lambda = np.zeros((self.n_mp, self.n_lat))
        mx_pky = np.zeros((self.n_mp, self.n))
        psi2 = np.zeros(self.n_mp)

        for imp, ieq in enumerate(self.ieq_mp):
            psi2[imp] = self.e[ieq]

            endo = self.endo[ieq]
            mx_pky[imp, :] = self.data[endo[0]][:, endo[1]]
            # REMOVE RANDOM EFFECT
            for iparam, exo in enumerate(self.exo[ieq]):
                t, i = exo
                if t is self.t_lat:
                    mx_lambda[imp, i] = self.params[ieq][iparam]
                else:
                    # MINUS: it is correct!
                    mx_pky[imp, :] -= self.params[ieq][iparam] * self.data[t][:, i]

        mx_lambda_inv = np.linalg.pinv(mx_lambda)


        # from structural part
        mu1 = mx_c @ mx_pg
        sigma1 = mx_c @ np.diag(psi1) @ mx_c.transpose()
        sigma1_inv = np.linalg.inv(sigma1)

        # from measurement part
        mu2 = mx_lambda_inv @ mx_pky
        sigma2_inv = mx_lambda.transpose() @ np.diag(1/psi2) @ mx_lambda

        sigma = np.linalg.inv(sigma1_inv + sigma2_inv)
        mu = sigma @ (sigma1_inv @ mu1 + sigma2_inv @ mu2)


        for i in range(self.n):
            self.data[self.t_lat][i, :] = \
                st.multivariate_normal.rvs(mean=mu[:, i], cov=sigma)


    def update_ordinal(self, ismpl):
        pass


    def update_params(self, ieq):
        """
        Update one line from the
        :param ieq:
        :return:
        """

        # ---- Pre calculation ----
        # Priors for errors
        a0, b0 = self.e_ab_prior[ieq]
        endo = self.endo[ieq]
        exo = self.exo[ieq]
        params = self.params[ieq]
        reff = self.reff[ieq]  # random effect
        priors = self.priors[ieq]

        # Values for influencing variables
        x = np.array([self.data[t][:, i] for t, i in exo])
        # Values for the dependent variable
        z = self.data[endo[0]][:, endo[1]]

        xx = x @ x.transpose()
        zx = z @ x.transpose()

        phi_inv = xx + np.identity(len(x))
        phi = np.linalg.inv(phi_inv)

        post = phi.dot(zx + priors)  # porsterior mean

        a = a0 + self.n/2
        b = b0 + 1/2 * (- post @ phi_inv @ post +
                        z @ z +
                        priors @ priors)

        # update error terms
        # b parameter the scale parameter, carefully checked !!!
        self.e[ieq] = st.invgamma.rvs(a=a, scale=b)
        # update params
        if(len(post) == 1):
            self.params[ieq] = [st.multivariate_normal.rvs(mean=post,
                                                          cov=phi * self.e[ieq])]
        else:
            self.params[ieq] = st.multivariate_normal.rvs(mean=post,
                                                          cov=phi*self.e[ieq])


    def update_reff(self, ieff):
        """

        :param ieff:
        :return:
        """
        pass

    def inspect(self):
        relations = DataFrame(columns=['lval', 'rval', 'Estimate'])

        id = 0
        for ieq in range(self.n_eq):
            endo = self.endo[ieq]


            for ipar in range(len(self.params[ieq])):
                exo = self.exo[ieq][ipar]
                relations.loc[id] = [self.v[endo[0]][endo[1]],
                                     self.v[exo[0]][exo[1]],
                                     self.params[ieq][ipar]]
                id += 1

        return relations


