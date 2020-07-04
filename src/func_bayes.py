"""
This module contains functions to perform Bayesian estimation of parameter in SEM - MCMC
"""

__author__ = "Anna Igolkina"
__license__ = "MIT"
__maintainer__ = "Anna Igolkina"
__email__ = "igolkinaanna11@gmail.com"

import numpy as np
from collections import Iterable
from itertools import product
from scipy import stats as st

from semopy import Model
import pandas as pd
pd.options.mode.chained_assignment = None


class SEMmx:
    BETA = 'Beta'
    PI = 'Pi'
    LAMBDA = 'Lambda'
    KAPPA = 'Kappa'
    PHI_Y = 'Phi_y'
    THETA_DELTA = 'Theta_delta'
    THETA_EPS = 'Theta_eps'
    SIGMA_LAT = 'Sigma_lat'


class ModelBay():
    def __init__(self, mod, data, param_prior, var_ordinal, kappa_flag=False):

        model = Model(mod)
        self.kappa_flag = kappa_flag
        # -------------------------------------------
        # Get data profiles "d_" means "data", 'v_' means variable
        # -------------------------------------------

        v_all = set(param_prior.loc[:, 0]) | set(param_prior.loc[:,1])
        prior_tmp = [[param_prior.loc[i, 0], param_prior.loc[i, 1], param_prior.loc[i, 2]]
                           for i in range(param_prior.shape[0])]

        self.v_lat = model.vars['Latents']
        self.v_ord = [v for v in model.vars['Indicators'] if v in var_ordinal]
        self.v_u = [v for v in model.vars['Indicators'] if v not in var_ordinal]

        self.v_snp = list(v_all - set(self.v_lat + self.v_ord + self.v_u))
        self.v_dim2 = self.v_lat + self.v_snp
        self.v_phen = self.v_u + self.v_ord

        self.d_snp = data[self.v_snp]
        self.d_u = data[self.v_u]  # normally distributed
        self.d_ord = data[self.v_ord]  # ordinal variables

        # Normalise data and get boundaries for ordinal variables and SNPs
        self.prepare_data()
        # self.fix_data_nans()

        # -------------------------------------------
        # Counts
        # -------------------------------------------
        self.n_obs = data.shape[0]  # Number of observatioons (samples)
        self.n_lat = len(self.v_lat)
        self.n_snp = len(self.v_snp)
        self.n_ord = len(self.v_ord)
        self.n_z = len(self.v_ord)
        self.n_dim2 = self.n_lat + self.n_snp  # Second dimention
        self.n_phen = len(self.v_phen)

        # -------------------------------------------
        # Latent constructs and metrices
        # -------------------------------------------
        self.d_z = np.zeros((self.n_obs, self.n_ord))
        self.d_y = np.zeros((self.n_obs, self.n_snp))
        self.d_lat = np.zeros((self.n_obs, self.n_lat))

        self.mx = dict()
        self.mx[SEMmx.BETA] = np.zeros((self.n_lat, self.n_lat))
        self.mx[SEMmx.PI] = np.zeros((self.n_lat, self.n_snp))
        self.mx[SEMmx.LAMBDA] = np.zeros((self.n_phen, self.n_lat))
        self.mx[SEMmx.KAPPA] = np.zeros((self.n_phen, self.n_snp))
        self.mx[SEMmx.THETA_DELTA] = np.zeros((self.n_lat, self.n_lat))
        self.mx[SEMmx.THETA_EPS] = np.zeros((self.n_phen, self.n_phen))
        self.mx[SEMmx.PHI_Y] = np.diag(np.ones(self.n_snp))



        # -------------------------------------------
        # Get parameters: initial values and annotation
        # -------------------------------------------


        self.param_pos, self.param_val, self.param_prior = self.define_params(prior_tmp, model)
        self.fill_mx()
        self.param_val *= (np.random.random(len(self.param_val)) + 0.5)

        self.mcmc = [np.array(self.param_val)]


        # -------------------------------------------
        self.coefs_spart = self.get_coefs_spart()
        self.coefs_mpart = self.get_coefs_mpart()


        # Proportion of samples of the same level
        # Cummulative fractions to calculate CDFs
        self.z_cumm_fract = self.get_z_cumm_fract()
        self.z_counts, self.z_values = self.get_z_counts()
        self.z_pers = self.get_z_persentilles()

        self.y_counts, self.y_values = self.get_y_counts()
        self.y_pers = self.get_y_persentilles()


        # -------------------------------------------
        # Parameters for prior distributions
        # -------------------------------------------
        # Phi matrices ~ Inverse Wishart:
        self.p_phi_y_df, self.p_phi_y_cov_inv = self.get_params_phi_y()



        # Theta matrices ~ Inverse Gamma
        self.p_theta_delta_alpha, self.p_theta_delta_beta = \
            self.get_params_theta_delta()
        self.p_theta_eps_alpha, self.p_theta_eps_beta = \
            self.get_params_theta_eps()

        # Parameters for normal distribution of path coefficients in
        # Structural part

        self.p_spart_mean, self.p_spart_cov_inv = self.get_params_spart()
        self.p_spart_loc = [cov_inv @ mean
                            for mean, cov_inv in zip(self.p_spart_mean,
                                                     self.p_spart_cov_inv)]
        self.p_spart_qform = [mean.T @ loc
                              for mean, loc in zip(self.p_spart_mean,
                                                   self.p_spart_loc)]

        # Measurement part
        self.p_mpart_mean, self.p_mpart_cov_inv = self.get_params_mpart()
        self.p_mpart_loc = [cov_inv @ mean
                            for mean, cov_inv in zip(self.p_mpart_mean,
                                                     self.p_mpart_cov_inv)]
        self.p_mpart_qform = [mean.T @ loc
                              for mean, loc in zip(self.p_mpart_mean,
                                                   self.p_mpart_loc)]

        # -------------------------------------------
        # Values for prediction
        # -------------------------------------------
        self.u_slope = []
        self.u_intercept = []

        self.v_slope = []
        self.v_intercept = []

        self.regress_flag = True

    def fill_mx(self, params=None):
        """
        Set parameter values into matrices
        :param params:
        :return:
        """
        if params is None:
            params = self.param_val

        for mx in [SEMmx.BETA, SEMmx.LAMBDA,
                   SEMmx.THETA_DELTA, SEMmx.THETA_EPS,
                   SEMmx.PI, SEMmx.KAPPA]:
            for pos1, pos2, idx in self.param_pos[mx]:
                self.mx[mx][pos1, pos2] = params[idx]


    def define_params(self, prior_tmp, model):
        """
        Set initial parameters (positions) and prior values
        :param prior_tmp:
        :return:
        """
        param_pos_beta = []
        param_pos_lambda = []
        param_pos_pi = []
        param_pos_kappa = []
        param_val = []
        idx_param = 0


        # Set parameters for "ones" in Lambda
        mx_lambda = model.mx_lambda
        model_phen = model.vars['Indicators']
        for i in range(self.n_phen):
            if sum(mx_lambda[i, :] == 1) > 0:
                i2 = int(np.where(mx_lambda[i, :] == 1)[0])
                i1 = self.v_phen.index(model_phen[i])

                param_pos_lambda += [[i1, i2, idx_param]]
                param_val += [1]
                idx_param += 1



        # Beta and Lambda, Pi and Kappa

        for v1, v2, val in prior_tmp:
            if v1 in self.v_lat and v2 in self.v_lat:  # structural part Beta
                i1 = self.v_lat.index(v1)
                i2 = self.v_lat.index(v2)
                param_pos_beta += [[i1, i2, idx_param]]
                param_val += [val]
                idx_param += 1
            elif v1 in self.v_lat and v2 in self.v_snp:
                i1 = self.v_lat.index(v1)
                i2 = self.v_snp.index(v2)
                param_pos_pi += [[i1, i2, idx_param]]
                param_val += [val]
                idx_param += 1
            elif v1 in self.v_phen and v2 in self.v_lat:
                i1 = self.v_phen.index(v1)
                i2 = self.v_lat.index(v2)

                f = False
                for j1, j2, _ in param_pos_lambda:
                    if j1 == i1 and j2 == i2:
                        f = True
                        break
                if f:
                    continue

                param_pos_lambda += [[i1, i2, idx_param]]
                param_val += [val]
                idx_param += 1

            elif v1 in self.v_phen and v2 in self.v_snp and self.kappa_flag:
                i1 = self.v_phen.index(v1)
                i2 = self.v_snp.index(v2)
                param_pos_kappa += [[i1, i2, idx_param]]
                param_val += [val]
                idx_param += 1


        # Add zero-beta
        beta = model.mx_beta
        for i, j in product(range(self.n_lat), repeat=2):
            if beta[i, j] == 0:
                continue

            beta_flag = False
#             print(param_pos_beta)
            if len(param_pos_beta) > 0:
                for i1, j1, _ in param_pos_beta:
                    if i1 == i and j1 == j:
                        beta_flag = True
                        break
                if beta_flag:
                    continue

            param_pos_beta += [[i, j, idx_param]]
            param_val += [0]
            idx_param += 1




        # Theta_delta and Theta_epsilon
        param_pos_theta_delta = [[i, i, i + idx_param] for i in range(self.n_lat)]  # Spart
        idx_param += self.n_lat
        param_val += [0.5] * self.n_lat

        param_pos_theta_epsilon = [[i, i, i + idx_param] for i in range(self.n_phen)]  # Mpart
        idx_param += self.n_phen
        param_val += [0.5] * self.n_phen

        # --------------------------------------------------------------------------------
        param_pos = dict()
        param_pos[SEMmx.BETA] = param_pos_beta
        param_pos[SEMmx.LAMBDA] = param_pos_lambda
        param_pos[SEMmx.PI] = param_pos_pi
        param_pos[SEMmx.KAPPA] = param_pos_kappa
        param_pos[SEMmx.THETA_DELTA] = param_pos_theta_delta
        param_pos[SEMmx.THETA_EPS] = param_pos_theta_epsilon
        return param_pos, np.array(param_val), np.array(param_val)


    # def fix_data_nans(self):
    #     """
    #     Replace nan-values to zeroes by the mean
    #     :return:
    #     """
    #     self.snp_mean = []
    #     for v in list(self.d_snp):
    #         idx = np.isnan(self.d_snp[v])
    #         self.snp_mean += list([np.mean(self.d_snp[v])])
    #         self.d_snp[v][idx] = np.mean(self.d_snp[v])

    def prepare_test_data_new(self, t_data):

        t_u = t_data[self.v_u]
        t_ord = t_data[self.v_ord]
        t_snp = t_data[self.v_snp]

        #
        # for i, u in enumerate(self.v_u):
        #     t_u.loc[:, u] = t_u.loc[:, u] * self.u_std[i] + self.
        #     tmp = t_u[u] / self.u_std[i]
        #     t_u.loc[:, u] = tmp

        # for i in range(self.n_snp):
        #     t_snp.ix[t_snp[:,i].isna(),i]

        t_snp = t_snp.fillna(0)

        return t_u, t_ord, t_snp


    def prepare_test_data(self, t_data):
        # TODO
        """
        Normalise the data and get boundaries for SNPs and ordinal variables
        :return:
        """
        # For normally distributed phenotypes - standardisation

        t_u = t_data[self.v_u]
        t_ord = t_data[self.v_ord]
        t_snp = t_data[self.v_snp]


        # Prepare mormally distributed
        for i, u in enumerate(self.v_u):
            tmp = t_u[u] - self.u_mean[i]
            t_u.loc[:, u] = tmp
            tmp = t_u[u] / self.u_std[i]
            t_u.loc[:, u] = tmp

        # Prepare nans in snps
        for i, snp in enumerate(self.v_snp):
            idx = np.isnan(t_snp[snp])
            t_snp.loc[:,snp][idx] = self.snp_mean[i]

            cnts = self.y_counts[i]
            cnts = cnts / sum(cnts)
            cnts = np.cumsum(cnts)
            frac = [cnts[0]/2]
            for j in range(len(cnts)-1):
                frac += [(cnts[j] + cnts[j+1])/2]
            vals_new = st.norm.ppf(frac)
            vals = self.y_values[1][i]
            for j, v in enumerate(vals):
                t_snp[snp][t_snp[snp] == v] = vals_new[j]


        # DO NOT NEED TO MAKE PREPARATION FOR PHENOTYPES,
        # they are not used in calculations


        return t_u, t_ord, t_snp


    def prepare_data(self):
        """
        Normalise the data and get boundaries for SNPs and ordinal variables
        :return:
        """
        # For normally distributed phenotypes - standardisation

        self.u_mean = []
        self.u_std = []

        for u in self.v_u:
            self.u_mean += list([np.mean(self.d_u[u])])
            tmp = self.d_u[u] - np.mean(self.d_u[u])
            self.d_u.loc[:, u] = tmp

            self.u_std += list([np.std(self.d_u[u])])
            tmp = self.d_u[u] / np.std(self.d_u[u])
            self.d_u.loc[:, u] = tmp

        # For ordinal variables - define boundaries
        bound_ord = []
        variants_ord = []
        for v in self.v_ord:
            variants, cnts = np.unique(self.d_ord[v], return_counts=True)
            variants = list(variants)
            persentil = np.cumsum(cnts / sum(cnts))
            bound_ord += [st.norm.ppf([0] + list(persentil))]
            variants_ord += [variants]

        # For snps - define boundaries
        bound_snp = []
        variants_snp = []
        for v in self.v_snp:
            variants, cnts = np.unique(self.d_snp[v][self.d_snp[v].notnull()], return_counts=True)
            variants = list(variants)
            persentil = np.cumsum(cnts / sum(cnts))
            bound_snp += [st.norm.ppf([0] + list(persentil))]
            variants_snp += [variants]

        return bound_ord, variants_ord, bound_snp, variants_snp

    @property
    def d_omega(self):
        return self.d_eta

    @property
    def d_spart(self):
        return np.concatenate((self.d_eta, self.d_y), axis=1)

    @property
    def d_mpart(self):
        return np.concatenate((self.d_eta, self.d_y), axis=1)

    @property
    def d_x(self):
        return np.concatenate((self.d_u, self.d_z), axis=1)

    @property
    def d_x_obs(self):
        return np.concatenate((self.d_u, self.d_ord), axis=1)

    @property
    def mx_spart(self):
        return np.concatenate((self.mx[SEMmx.BETA], self.mx[SEMmx.PI]), axis=1)

    @property
    def mx_mpart(self):
        return np.concatenate((self.mx[SEMmx.LAMBDA], self.mx[SEMmx.KAPPA]), axis=1)

    def optimise(self):

        n_iter = 5000
        print(n_iter)
        for _ in range(n_iter):

            # --------------------------------------------------------
            # Initial values for z-boundaries
            # --------------------------------------------------------
            # self.z_bounds = self.calc_z_bounds()

            # --------------------------------------------------------
            # Sample values for latent variables
            # --------------------------------------------------------

            self.d_z = self.gibbs_z_new()
            self.d_y = self.gibbs_y_new()
            self.d_eta = self.gibbs_omega()

            # --------------------------------------------------------
            # Sample covariance matrices Phi_xi and Phi_y
            # --------------------------------------------------------

            # --------------------------------------------------------
            # Samples all Parameters (errors first)
            # --------------------------------------------------------
            # Structural Part
            self.gibbs_spart()
            # Measurement Part
            self.gibbs_mpart()


            # --------------------------------------------------------
            # Remember values of parameters after each iteration
            # --------------------------------------------------------

            self.mcmc = np.append(self.mcmc, [self.param_val], axis=0)
            print(self.mcmc.shape)

        return self.mcmc

    def snp_effects(self):
        """
        This function return coefficients at SNPS
        :return:
        """
        t_snp = pd.DataFrame(data=np.eye(self.n_snp),
                           columns=self.v_snp)

        n_obs = t_snp.shape[0]

        d_omega = np.full((n_obs, self.n_lat), np.nan)

        d_spart = np.concatenate((d_omega, t_snp), axis=1)
        m_spart = self.mx_spart

        flag_continue = True
        while flag_continue == True:
            flag_continue = False
            for i in range(self.n_lat):
                omega_tmp = np.full(n_obs, 0)
                flag_done = True
                for j, path_coef in enumerate(m_spart[i]):
                    if path_coef == 0:
                        continue
                    if np.isnan(d_spart[0, j]):
                        flag_continue = True
                        flag_done = False
                        break
                    omega_tmp = omega_tmp + path_coef * d_spart[:, j]
                if flag_done:
                    d_spart[:, i] = omega_tmp

        m_mpart = self.mx_mpart
        d_phen = d_spart @ m_mpart.T
        values = d_phen

        return values


    def predict_new(self, t_data):
        """

        :param mode_new:
        :return:
        """
        t_u, t_ord, t_snp = self.prepare_test_data_new(t_data)


        n_obs = t_snp.shape[0]


        d_omega = np.full((n_obs, self.n_lat), np.nan)

        d_spart = np.concatenate((d_omega, t_snp), axis=1)
        m_spart = self.mx_spart

        flag_continue = True
        while flag_continue == True:
            flag_continue = False
            for i in range(self.n_lat):
                omega_tmp = np.full(n_obs, 0)
                flag_done = True
                for j, path_coef in enumerate(m_spart[i]):
                    if path_coef == 0:
                        continue
                    if np.isnan(d_spart[0, j]):
                        flag_continue = True
                        flag_done = False
                        break
                    omega_tmp = omega_tmp + path_coef * d_spart[:, j]
                if flag_done:
                    d_spart[:, i] = omega_tmp

        m_mpart = self.mx_mpart
        d_phen = d_spart @ m_mpart.T
        values = d_phen

        # np.savetxt('../../data/min/mcmc/' + 'values.txt', values)

        d_x = np.concatenate((t_u, t_ord), axis=1)

        # for i in range(self.v_u):
        #     t_u[:,]



        #
        # # Get ordinal values for phenotypes
        # z_bounds = self.calc_z_bounds()
        # _, z_vals = self.get_z_counts()
        # for z_range, z_val, i in \
        #         zip(z_bounds, z_vals, range(self.n_ord, self.n_phen)):
        #     for j in range(n_obs):
        #         for k, z in enumerate(z_range):
        #             if values[i][j] <= z:
        #                 values[i][j] = z_val[k]
        #                 break

        # # Normalisation
        # for i in range(self.n_phen - self.n_obs):
        #     d_x.loc[:, i] = d_x.loc[:, i] * self.u_std[i] + self.u_mean[i]
        #     values[:, i] = values[:, i] * self.u_std[i] + self.u_mean[i]


        values = values.T
        d_x = d_x.T
        corr_all = []
        for v1, v2, name in zip(values, d_x, self.v_phen):
            idx = (np.isnan(v2) == False)
            if name not in self.v_ord:
                corrd = st.pearsonr(v1[idx], v2[idx])
                res = corrd[0]

                if self.regress_flag:
                    slope, intercept, r_value, p_value, std_err = st.linregress(v1[idx], v2[idx])
                    self.u_slope += [slope]
                    self.u_intercept += [intercept]

                idx_tmp = self.v_phen.index(name)
                v1[idx] = v1[idx] * self.u_slope[idx_tmp] + self.u_intercept[idx_tmp]
            else:
                # res = sum(v1 == v2) / len(v1)
                corrd = st.pearsonr(v1[idx], v2[idx])
                res = corrd[0]

                if self.regress_flag:
                    slope, intercept, r_value, p_value, std_err = st.linregress(v1[idx], v2[idx])
                    self.u_slope += [slope]
                    self.u_intercept += [intercept]

                idx_tmp = self.v_phen.index(name)
                v1[idx] = v1[idx] * self.u_slope[idx_tmp] + self.u_intercept[idx_tmp]

            print(name, res)
            corr_all += [res]

        self.regress_flag = False

        # np.savetxt('/Users/anna/OneDrive/polytech/gene_phen/data/mcmc/7.txt', list(zip(values[-1], d_x[-1])))

        # Do something with ordinary variables

        print('\n')
        return values.T, d_x.T



    def load_prior_params(self, param_prior):
        self.param_prior = param_prior



    def get_params_phi_y(self):
        """
        Get prior parameters for latent variables correspond to genetic:
        df and scale
        :return:
        """
        if self.param_prior is None:
            return 5, np.identity(self.n_snp)

        m_phi = np.diag(np.zeros(self.n_snp))

        r = len(self.param_pos[SEMmx.PI]) + len(self.param_pos[SEMmx.KAPPA])
        # r = len([param_id for mx_type, _, _, param_id in self.param_pos
        #          if mx_type in {SEMmx.PI, SEMmx.KAPPA}])
        pho = r + 4

        m_r_inv = m_phi * (pho - self.n_snp - 1)
        return pho, m_r_inv

    def get_params_theta_delta(self):
        """
        Get patameters for InvGamma distribution
        :return: alpha, beta
        """
        if self.param_prior is None:
            return np.ones(self.n_lat) * 9, np.ones(self.n_lat) * 4

        m_theta = np.diag(self.mx[SEMmx.THETA_DELTA])
        alpha = np.ones(self.n_lat) * 3
        beta = (alpha - 1) * m_theta
        return alpha, beta


    def get_params_theta_eps(self):
        """
        Get patameters for InvGamma distribution
        :return:
        """
        if self.param_prior is None:
            return np.ones(self.n_phen) * 9, np.ones(self.n_phen) * 4

        m_theta = np.diag(self.mx[SEMmx.THETA_EPS])
        alpha = np.ones(self.n_phen) * 3
        beta = (alpha - 1) * m_theta
        return alpha, beta


    def get_params_spart(self):
        """
        Get parameters of Normal Distribution for path coefficients in
        the Structural part
        :return: mean value and INVERSE covariance matrix
        """
        res_mean = []
        res_invcov = []
        for i in range(self.n_lat):
            n_terms = len(self.coefs_spart[i])

            # For informative
            if self.param_prior is None:
                res_mean += [np.ones(n_terms) * 0.8]
            else:
                res_mean += [np.array([self.param_prior[param_id]
                             for _, param_id in self.coefs_spart[i]])]

            res_invcov += [np.identity(n_terms)]


            # # For non-informative
            # res += (np.ones(n_terms) * 0.8,
            #         np.zeros(n_terms))

            # self. = np.linalg.inv(self.p_spart_cov)

        return res_mean, res_invcov

    def get_params_mpart(self):
        """
        Get parameters of Normal Distribution for factor loadings
        in the Measurement part
        :return: mean value and INVERSE covariance matrix
        """
        res_mean = []
        res_invcov = []
        for i in range(self.n_phen):
            n_terms = len(self.coefs_mpart[i])

            # For informative
            if self.param_prior is None:
                res_mean += [np.ones(n_terms) * 0.8]
            else:
                res_mean += [np.array([self.param_prior[param_id]
                             for _, param_id in self.coefs_mpart[i]])]

            res_invcov += [np.identity(n_terms)]

            # # For non-informative
            # res += (np.ones(n_terms) * 0.8,
            #         np.zeros(n_terms))
        return res_mean, res_invcov

    def get_z_cumm_fract(self):
        """
        Returm cummulative fractions
        :return:
        # """
        z_cumm_fract = []
        for i, v in enumerate(self.v_ord):
            v_sort = np.array(self.d_ord[v])
            v_sort.sort()
            unique, counts = np.unique(v_sort, return_counts=True)
            cnts = counts / sum(counts)
            cumm_counts = np.cumsum(cnts)

            z_cumm_fract += [cumm_counts]

        # here is an example
        return z_cumm_fract

    def get_z_counts(self):
        """
        Returm cummulative fractions
        :return:
        # """
        z_counts = []
        z_values = []
        for i, v in enumerate(self.v_ord):
            v_sort = np.array(self.d_ord[v])
            v_sort = [v for v in v_sort if not np.isnan(v)]
            v_sort.sort()
            unique, counts = np.unique(v_sort, return_counts=True)
            z_counts += [counts]
            z_values += [unique]

        # here is an example
        return z_counts, z_values


    def get_z_persentilles(self):
        """

        :return:
        """
        counts = self.z_counts
        persentilles = []
        for i in range(len(counts)):
            cnts = counts[i]
            cumm_counts = np.cumsum(cnts)
            cumm_counts = cumm_counts / sum(cnts)
            prs = st.norm.ppf([0] + list(cumm_counts))
            persentilles += [prs]

        return persentilles

    def get_y_counts(self):
        """
        Returm cummulative fractions
        :return:
        # """
        y_counts = []
        y_values = []
        for i, v in enumerate(self.v_snp):
            v_sort = np.array(self.d_snp[v])
            v_sort = [v for v in v_sort if not np.isnan(v)]
            v_sort.sort()
            unique, counts = np.unique(v_sort, return_counts=True)
            y_counts += [counts]
            y_values += [unique]

        # here is an example
        return y_counts, y_values

    def get_coefs_spart(self):
        """

        :return:
        """
        coefs = []
        for irow in range(self.n_lat):
            coefs_row = []
            for pos1, pos2, param_id in self.param_pos[SEMmx.BETA]:
                if pos1 != irow:
                    continue
                coefs_row += [(pos2, param_id)]

            for pos1, pos2, param_id in self.param_pos[SEMmx.PI]:
                if pos1 != irow:
                    continue
                coefs_row += [(pos2 + self.n_lat, param_id)]

            coefs += [coefs_row]

        return coefs

    def get_coefs_mpart(self):
        """

        :return:
        """
        coefs = []
        for irow in range(self.n_phen):
            coefs_row = []
            for pos1, pos2, param_id in self.param_pos[SEMmx.LAMBDA]:
                if pos1 != irow:
                    continue
                coefs_row += [(pos2, param_id)]

            for pos1, pos2, param_id in self.param_pos[SEMmx.KAPPA]:
                if pos1 != irow:
                    continue
                coefs_row += [(pos2+self.n_lat, param_id)]
            coefs += [coefs_row]

        return coefs
    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------


    def calc_z_bounds(self):
        """
        This function calculates alpha-values for boundaries
        :return:
        """
        if self.n_z == 0:
            return []
        percentiles = []
        for fractions in self.z_cumm_fract:
            percentiles += [st.norm.ppf(fractions, scale=1)]

        return percentiles


    def gibbs_z_new(self):
        """

        :return:
        """
        d_z = np.zeros((self.n_obs, self.n_z))
        if self.n_z == 0:
            return d_z

        counts = self.z_counts
        values = self.z_values
        persentilles = self.z_pers

        for i in range(self.n_z):

            cnt = counts[i]
            prs = persentilles[i]
            vls = values[i]
            for j in range(len(cnt)):
                tmp = list(st.truncnorm.rvs(a=prs[j], b=prs[j + 1], size=cnt[j]))
                d_z[self.d_ord.iloc[:, i] == vls[j], i] = tmp

            d_z[self.d_ord.iloc[:, i].isna(), i] = st.norm.rvs(size=self.d_ord.iloc[:, i].isna().sum())

#             print(d_z[:, i].mean(), d_z[:, i].std())

        return d_z




    def get_y_persentilles(self):
        """

        :return:
        """
        counts = self.y_counts
        persentilles = []
        for i in range(len(counts)):
            cnts = counts[i]
            cumm_counts = np.cumsum(cnts)
            cumm_counts = cumm_counts / sum(cnts)
            prs = st.norm.ppf([0] + list(cumm_counts))
            persentilles += [prs]

        return persentilles

    def gibbs_y_new(self):

        d_y = np.zeros((self.n_obs, self.n_snp))
        if self.n_snp == 0:
            return d_y

        counts = self.y_counts
        values = self.y_values
        persentilles = self.y_pers

        for i in range(self.n_snp):

            cnt = counts[i]
            prs = persentilles[i]
            vls = values[i]
            for j in range(len(cnt)):
                tmp = list(st.truncnorm.rvs(a=prs[j], b=prs[j + 1], size=cnt[j]))
                d_y[self.d_snp.iloc[:, i] == vls[j], i] = tmp

            d_y[self.d_snp.iloc[:, i].isna(), i] = st.norm.rvs(size=self.d_snp.iloc[:, i].isna().sum())

            # print(d_y[:, i].mean(), d_y[:, i].std())

        return d_y


    def gibbs_omega(self):
        """
        Sampling Omega -- latent variables
        result: new sample Omega"""
        d_omega = np.zeros((self.n_obs, self.n_lat))
        if self.n_lat == 0:
            d_eta = d_omega[:, 0:self.n_lat]
            return d_eta



        # GOOD
        m_inv_theta_eps = np.linalg.pinv(self.mx[SEMmx.THETA_EPS])

        m_inv_sigma_omega = np.linalg.pinv(self.get_sigma_lat())



        m_lambda = self.mx[SEMmx.LAMBDA]
        m_kappa = self.mx[SEMmx.KAPPA]
        m_inv_q = m_lambda.T @ m_inv_theta_eps @ m_lambda + m_inv_sigma_omega
        m_q = np.linalg.pinv(m_inv_q)

        for i in range(self.n_obs):
            x = self.d_x[i, :]  # Do not need to transpose
            y = self.d_y[i, :]  # Do not need to transpose
            q = m_lambda.T @ m_inv_theta_eps @ (x - m_kappa @ y)
            d_omega[i, :] = st.multivariate_normal.rvs(mean=m_q @ q,
                                                       cov=m_q)

        d_eta = d_omega[:, 0:self.n_lat]
        return d_eta

    def gibbs_spart(self):
        """
         Sampling covariance matrixes Theta_delta, B, Pi, Gamma
          :return matrix Theta_delta,
                  parameter of Gamma distribution of Theta_delta:
                   p_theta_delta_alpha,
                  matrixes B, Pi, Gamma,
                  parameters of Normal distribution of matrix(B, Pi, Gamma):
                  p_b_pi_gamma_means, p_b_pi_gamma_covs """

        n_obs = self.n_obs
        d_spart = self.d_spart
        d_eta = self.d_eta

        # if self.n_lat == 0:
        #     return

        # Sampling Theta and (Beta, Gamma, Pi) by rows
        for irow in range(self.n_lat):
            pos_of_coef = [pos2 for pos2, param_id in self.coefs_spart[irow]]
            id_of_params = [param_id for pos2, param_id in
                            self.coefs_spart[irow]]
            d_tmp = d_spart[:, pos_of_coef]
            # Calculate auxiliary variables
            a_cov_inv = self.p_spart_cov_inv[irow] + d_tmp.T @ d_tmp
            a_cov = np.linalg.inv(a_cov_inv)
            a_mean = a_cov @ (self.p_spart_loc[irow] +
                              d_tmp.T @ d_eta[:, irow].T)

            # Calculate new parameters of InvGamma dna InvWishart
            p_alpha = self.p_theta_delta_alpha[irow] + n_obs / 2
            p_beta = self.p_theta_delta_beta[irow] + 1 / 2 * \
                     (d_eta[:, irow].T @ d_eta[:, irow] -
                      a_mean.T @ a_cov_inv @ a_mean +
                      self.p_spart_qform[irow])

            value_of_theta = st.invgamma.rvs(a=p_alpha,
                                             scale=p_beta)

            # OLD was commented
            if value_of_theta > 0.1:
                value_of_theta = 0.1


            value_of_coef = \
                st.multivariate_normal.rvs(mean=a_mean,
                                           cov=a_cov*value_of_theta)

            if not isinstance(value_of_coef, Iterable):
                value_of_coef = [value_of_coef]

            # -------------------------------
            # Set new parameters values
            # -------------------------------
            for pos, val in zip(pos_of_coef, value_of_coef):
                if pos < self.n_lat:
                    self.mx[SEMmx.BETA][irow, pos] = val
                else:
                    self.mx[SEMmx.PI][irow, pos-self.n_lat] = val

            self.mx[SEMmx.THETA_DELTA][irow, irow] = value_of_theta

            for pos1, pos2, param_id in self.param_pos[SEMmx.THETA_DELTA]:
                if pos1 == irow:
                    self.param_val[param_id] = value_of_theta

            for param_id, value in zip(id_of_params, value_of_coef):
                self.param_val[param_id] = value

    def gibbs_mpart(self):
        """

        :return:
        """

        n_obs = self.n_obs
        d_mpart = self.d_mpart
        d_x = self.d_x
        irow = 0

        # Sampling Theta_eps and (Lambda, Kappa) by rows
        for irow in range(self.n_phen):
            pos_of_coef = [pos2 for pos2, param_id in self.coefs_mpart[irow]]
            id_of_params = [param_id for pos2, param_id
                            in self.coefs_mpart[irow]]
            d_tmp = d_mpart[:, pos_of_coef]
            # Calculate auxiliary variables
            a_cov_inv = self.p_mpart_cov_inv[irow] + d_tmp.T @ d_tmp
            a_cov = np.linalg.inv(a_cov_inv)
            a_mean = a_cov @ (self.p_mpart_loc[irow] + d_tmp.T @ d_x[:, irow])

            # Calculate new parameters of InvGamma dna InvWishart
            p_alpha = self.p_theta_eps_alpha[irow] + n_obs / 2
            p_beta = self.p_theta_eps_beta[irow] + 1 / 2 * \
                     (d_x[:, irow].T @ d_x[:, irow] -
                      a_mean.T @ a_cov_inv @ a_mean +
                      self.p_mpart_qform[irow])

            value_of_theta = st.invgamma.rvs(a=p_alpha,
                                            scale=p_beta)

            # Old was commented
            # if value_of_theta > 0.05:
            #     value_of_theta = 0.05


            value_of_coef = \
                st.multivariate_normal.rvs(mean=a_mean,
                                           cov=a_cov*value_of_theta,
                                           size=1)

            if not isinstance(value_of_coef, Iterable):
                value_of_coef = [value_of_coef]

            # -------------------------------
            # Set new parameters values
            # -------------------------------
            for pos, val in zip(pos_of_coef, value_of_coef):
                if pos < self.n_lat:
                    self.mx[SEMmx.LAMBDA][irow, pos] = val
                else:
                    self.mx[SEMmx.KAPPA][irow, pos-self.n_lat] = val

            self.mx[SEMmx.THETA_EPS][irow, irow] = value_of_theta



            for pos1, pos2, param_id in self.param_pos[SEMmx.THETA_EPS]:
                if pos1 == irow:
                    self.param_val[param_id] = value_of_theta
            for param_id, value in zip(id_of_params, value_of_coef):
                self.param_val[param_id] = value


    def get_sigma_lat(self):
        """
        This function calculates the covariance matrix for latent variables
        :return:
        """


        m_beta = self.mx[SEMmx.BETA]
        m_pi = self.mx[SEMmx.PI]

        m_phi_y = self.mx[SEMmx.PHI_Y]
        m_theta_delta = self.mx[SEMmx.THETA_DELTA]

        m_c = np.linalg.pinv(np.identity(m_beta.shape[0]) - m_beta)

        return m_c @ (m_pi @ m_phi_y @ m_pi.T +
                      m_theta_delta) @ m_c.T

    def get_param_names(self):
        """
        This function returns names of all params
        :return:
        """

        variable_order = []
        for i in range(len(self.param_val)):
            for pos1, pos2, param_id in self.param_pos[SEMmx.BETA]:
                if param_id != i:
                    continue
                variable_order += [[self.v_lat[pos1], self.v_lat[pos2]]]
                break


            for pos1, pos2, param_id in self.param_pos[SEMmx.LAMBDA]:
                if param_id != i:
                    continue
                variable_order += [[self.v_phen[pos1], self.v_lat[pos2]]]
                break

            for pos1, pos2, param_id in self.param_pos[SEMmx.PI]:
                if param_id != i:
                    continue
                variable_order += [[self.v_lat[pos1], self.v_snp[pos2]]]
                break


            for pos1, pos2, param_id in self.param_pos[SEMmx.KAPPA]:
                if param_id != i:
                    continue
                variable_order += [[self.v_phen[pos1], self.v_snp[pos2]]]
                break

            for pos1, pos2, param_id in self.param_pos[SEMmx.THETA_DELTA]:
                if param_id != i:
                    continue
                variable_order += [[self.v_lat[pos1], self.v_lat[pos2]]]
                break

            for pos1, pos2, param_id in self.param_pos[SEMmx.THETA_EPS]:
                if param_id != i:
                    continue
                variable_order += [[self.v_phen[pos1], self.v_phen[pos2]]]
                break

        return variable_order