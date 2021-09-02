"""
Function to optimize mtmlSEM model
"""

import numpy as np
import scipy.stats as st

from semopy import Model
from .dataset import *


class OptBayes:

    t_lat, t_obs, t_ord, t_m, t_nan = ['_latent', '_observed', '_ordinal',
                                       '_mean', '_nan']
    def __init__(self, relations,
                 data: Data,
                 random_effects=None,
                 var_ord=None):
        """

        :param relations:
        :param data:
        :param random_effects:
        :param var_ord:
        """
        # Get relationships between variables
        # relations = sem.inspect()
        v_endo_all = relations['lval'][relations['op'] == '~']
        v_exo_all = relations['rval'][relations['op'] == '~']

        # Stlit exogenous variables to really observed and random effects
        # observed
        if random_effects is None:
            random_effects = []
        self.random_effects = random_effects

        i_exo = [i for i, v in enumerate(v_exo_all) if v not in random_effects]
        v_exo = v_exo_all[i_exo]
        v_endo = v_endo_all[i_exo]
        v_endo_uniq = v_endo.unique().tolist()
        estims = relations['Estimate'][relations['op'] == '~'][i_exo]

        # random effects
        i_reff = [i for i, v in enumerate(v_exo_all) if v in random_effects]
        v_exo_reff = v_exo_all[i_reff]
        v_endo_reff = v_endo_all[i_reff]


        # Fins types of variables: latent/observed
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

        # # Additional tips with missing data
        # self.data[self.t_nan] = 1 - np.isnan(self.data[self.t_obs])

        # add random effects to data
        for v_reff in random_effects:
            self.data[v_reff] = data.r_eff[v_reff].z.transpose()
        self.data[self.t_m] = np.ones((self.n, 1))


        # TODO
        # if var_ord is not None:
        #     self.v[self.t_ord] = list(var_ord)
        #     self.data[self.t_ord] = np.random.rand(self.n, len(self.v[self.t_ord]))
        # else:
        #     self.v[self.t_ord] = var_ord


        self.endo = []
        self.exo = []
        self.params = []
        self.priors = []
        self.p_cov = []
        self.p_cov_inv = []

        for v in v_endo_uniq:
            self.endo += [self.find_type_and_id(v)]
            exo_tmp = v_exo[v_endo == v].tolist()
            priors_tmp = estims[v_endo == v].tolist()
            self.exo += [[self.find_type_and_id(u) for u in exo_tmp]]
            self.params += [np.random.rand(len(self.exo[-1]))]
            self.priors += [np.array(priors_tmp)]

            # # add means
            # if self.endo[-1][0] is not self.t_lat:
            #     # add mean
            #     self.exo[-1] += [(self.t_m, 0)]
            #     self.params[-1] = np.concatenate((self.params[-1], [0]))
            #     self.priors[-1] = np.concatenate((self.priors[-1], [0]))


            self.p_cov += [np.identity(len(self.params[-1]))]


            # construct random effects for each endogenous variable
            r_eq = v_exo_reff[v_endo_reff == v].tolist()
            if len(r_eq) == 0:
                self.p_cov_inv += [np.linalg.inv(self.p_cov[-1])]
                continue

            for r in r_eq:
                r_exo = [(r, i) for i in range(self.data[r].shape[1])]
                self.exo[-1] += r_exo

                r_cov = data.r_eff[r].cov_mx
                tmp_mx = np.zeros((self.p_cov[-1].shape[0], r_cov.shape[0]))
                self.p_cov[-1] = np.block([[self.p_cov[-1], tmp_mx],
                                           [tmp_mx.transpose(), r_cov]])

                self.params[-1] = np.concatenate((self.params[-1],
                                                  [0] * len(r_exo)))
                self.priors[-1] = np.concatenate((self.priors[-1],
                                                  [0] * len(r_exo)))
                # self.priors[-1] = np.concatenate((self.priors[-1],
                #                                   [0, 2, 200]))

            self.p_cov_inv += [np.linalg.inv(self.p_cov[-1])]


            # r_params = []
            # r_types = []
            # r_z = np.empty((0, self.n))
            # r_t = np.empty((0, 0))
            # r_cov = np.empty((0, 0))
            # r_err = []
            #
            # for rv in v_reff_tmp:
            #     r_types += [(rv, value) for value in data.r_eff[rv].u_values]
            #     r_params += [0] * data.r_eff[rv].n_values
            #     r_err += [1]
            #     r_z = np.concatenate((r_z, data.r_eff[rv].z), axis=0)
            #
            #     r_cov_add = data.r_eff[rv].cov_mx
            #     tmp_mx = np.zeros((r_cov.shape[0], r_cov_add.shape[0]))
            #     r_cov = np.block([[r_cov, tmp_mx],
            #                       [tmp_mx.transpose(), r_cov_add]])
            #
            #     r_t_tmp = np.ones((1, data.r_eff[rv].n_values))
            #     tmp_mx1 = np.zeros((r_t.shape[0], data.r_eff[rv].n_values))
            #     tmp_mx2 = np.zeros((1, r_t.shape[1]))
            #
            #     r_t = np.block([[r_t, tmp_mx1],
            #                     [tmp_mx2, r_t_tmp]])
            #
            # self.reff += [dict(params=r_params,
            #                    types=r_types,
            #                    z=r_z,
            #                    z_inv=np.linalg.pinv(r_z),
            #                    cov=r_cov,
            #                    err=r_err,
            #                    t=r_t,
            #                    cov_inv=np.linalg.inv(r_cov),
            #                    z_cov=r_z @ r_z.transpose())]

        self.n_eq = len(self.endo)  # number of equations

        # alpha and beta params for error terms
        self.e_ab_prior = [(3, 10)] * self.n_eq
        self.e = [0.1] * self.n_eq

        # # pre-calculation of covariance matrix between parameters in each equation
        # self.phi = []
        # for ieq in range(self.n_eq):

        # get help rapameters for Structural and Measurement parts
        self.create_sem_parts()

        self.mcmc = []


    def find_type_and_id(self, v):
        for t in self.v.keys():
            if v in self.v[t]:
                return (t, self.v[t].index(v))

        raise ValueError('Undefined variable or type')


    def calc_reff(self, ieq):
        return self.reff[ieq]['params'] @ self.reff[ieq]['z']


    def optimize(self, n_mcmc = 1000):

        # print(f'n lat {self.n_lat}')
        for i_mcmc in range(n_mcmc):
            # print(i_mcmc)

            if self.n_lat > 0:
                self.update_latent()

            # for ismpl in range(self.n):
            #     self.update_ordinal(ismpl)

            for ieq in range(self.n_eq):
                self.update_params(ieq)


            # print(self.e)


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

            # # ADD RANDOM EFFECT to mx_pg
            # if len(self.reff[ieq]) > 0:
            #     mx_pg[ilat, :] += self.calc_reff(ieq)

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

            for iparam, exo in enumerate(self.exo[ieq]):
                t, i = exo
                if t is self.t_lat:
                    mx_lambda[imp, i] = self.params[ieq][iparam]
                else:
                    # MINUS: it is correct!
                    mx_pky[imp, :] -= self.params[ieq][iparam] * self.data[t][:, i]

            # # REMOVE RANDOM EFFECT from mx_pky
            # if len(self.reff[ieq]) > 0:
            #     mx_pky[imp, :] -= self.calc_reff(ieq)

        # Inverse matrix for Lambda
        mx_lambda_inv = np.linalg.pinv(mx_lambda)

        # # missing data
        # mx_pg[np.isnan(mx_pg)] = np.mean(mx_pg[np.isnan(mx_pg)==False])

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
        priors = self.priors[ieq]
        p_cov_inv = self.p_cov_inv[ieq]

        # TODO make it faster
        # Values for influencing variables
        x = np.array([self.data[t][:, i] for t, i in exo])
        # x_miss = 1 - np.isnan(x)
        # x[np.isnan(x)] = 0

        # Values for the dependent variable
        z = 0 + self.data[endo[0]][:, endo[1]]
        # z_miss = 1 - np.isnan(z)
        # z[np.isnan(z)] = 0


        xx = (x @ x.transpose()) #/  (x_miss @ x_miss.transpose()) * self.n
        zx = (z @ x.transpose()) #/ (z_miss @ x_miss.transpose()) * self.n


        phi_inv = xx + p_cov_inv #+ np.identity(len(x))
        phi = np.linalg.inv(phi_inv)

        post = phi.dot(zx + p_cov_inv @ priors)  # porsterior mean

        a = a0 + self.n/2
        b = b0 + 1/2 * (- post @ phi_inv @ post +
                        z @ z +
                        priors @ p_cov_inv @ priors)

        # update error terms
        # b parameter is the scale parameter, carefully checked !!!
        self.e[ieq] = st.invgamma.rvs(a=a, scale=b)
        # update params
        if(len(post) == 1):
            self.params[ieq] = [st.multivariate_normal.rvs(mean=post,
                                                          cov=phi * self.e[ieq])]
        else:
            self.params[ieq] = st.multivariate_normal.rvs(mean=post,
                                                          cov=phi*self.e[ieq])


    # def update_reff(self, ieq):
    #     """
    #     Upda random effects for each equation separately
    #     :param ieq:
    #     :return:
    #     """
    #
    #     # get observed part
    #     endo = self.endo[ieq]
    #     y = 0 + self.data[endo[0]][:, endo[1]]
    #     for iparam, exo in enumerate(self.exo[ieq]):
    #         t, i = exo
    #         # MINUS: it is correct!
    #         y -= self.params[ieq][iparam] * self.data[t][:, i]
    #
    #
    #     mu1 = y @ self.reff[ieq]['z_inv']
    #     sigma1_inv = self.reff[ieq]['z_cov'] # /
    #
    #     mu2 = mu1 * 0
    #     # TODO
    #     sigma2_inv = self.reff[ieq]['cov_inv']
    #
    #     sigma = np.linalg.inv(sigma1_inv + sigma2_inv)
    #     mu = sigma @ (sigma1_inv @ mu1 + sigma2_inv @ mu2)
    #
    #     self.reff[ieq]['params'] = \
    #         st.multivariate_normal.rvs(mean=mu, cov=sigma * self.e[ieq])
    #
    #     print(self.reff[ieq]['params'])

    def inspect(self):
        relations = DataFrame(columns=['lval', 'rval', 'Estimate', 'op'])

        id = 0
        for ieq in range(self.n_eq):
            endo = self.endo[ieq]


            for ipar in range(len(self.params[ieq])):
                exo = self.exo[ieq][ipar]
                if exo[0] in self.random_effects:
                    relations.loc[id] = [self.v[endo[0]][endo[1]],
                                         f'{exo[0]}:{exo[1]}',
                                         self.params[ieq][ipar], 'reff']
                elif exo[0] is self.t_m:
                    relations.loc[id] = [self.v[endo[0]][endo[1]],
                                         'mean',
                                         self.params[ieq][ipar], '~']

                else:
                    relations.loc[id] = [self.v[endo[0]][endo[1]],
                                         self.v[exo[0]][exo[1]],
                                         self.params[ieq][ipar], '~']

                id += 1

        return relations


    def calc_pvals(self):
        """
        Calculate p-values after Bayes
        :return:
        """
        #TODO: Georgy/Anna
        pass