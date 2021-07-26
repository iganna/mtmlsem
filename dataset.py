"""
Module to work with the data and to create cross-validation set
"""

import copy
import warnings
import numpy as np
from pandas import read_csv, DataFrame, concat, Series
from semopy.utils import calc_reduced_ml


from math import ceil


from func_util import *

class PhenType:
    """
    Different types of variables
    """
    norm = 'norm'
    ord = 'ordinal'
    freq = 'frequency'


class Data:

    possible_phen_types = [PhenType.norm, PhenType.ord, PhenType.freq]
    kinship_var_name = 'kinship'

    def __init__(self,
                 d_snps: DataFrame = None,
                 d_phens: DataFrame = None,
                 d_phen_types=None,
                 s_reffs=None,
                 cov_reffs=None,
                 estim_kinship=False,
                 std_flag=True,
                 ord_vars_flag=True,
                 ord_vars_thresh=3,
                 echo=False,
                 show_warning=True):
        """

        :param d_snps:
        :param d_phens:
        :param d_phen_types:
        :param std_flag: to standardize data or not
        :param r_effs: dictionary with random effects
        :param ord_vars_flag:
        :param ord_vars_thresh:
        :param s_nan:
        :param sep:
        :param echo:
        """

        self.echo = echo
        # --------------------------------------------------
        # Set SNPs and phenotypes (including metadata)
        self.d_phens = self.set_phens(d_phens=d_phens)
        self.d_snps = self.set_snps(d_snps=d_snps)

        # Get correspondence between phens and snps:
        # gen common samples
        self.samples = intersect(self.d_snps.index, self.d_phens.index)
        self.snps = list(self.d_snps.columns)
        self.phens = list(self.d_phens.columns)
        self.n_samples = len(self.samples)
        self.n_snps = len(self.snps)
        self.n_phens = len(self.phens)

        if echo:
            if len(self.d_snps.index) > len(self.samples):
                print(f'Some samples with SNPs were omitted: '
                      f'{diff(self.d_snps.index, self.samples)}')
            if len(self.d_phens.index) > len(self.samples):
                print(f'Some samples with phenotypes were omitted: '
                      f'{diff(self.d_phens.index, self.samples)}')
            print(f'Number of samples: {self.n_samples}',
                  f'Number of SNPs: {self.n_snps}')

        self.d_snps = self.d_snps.loc[self.samples]
        self.d_phens = self.d_phens.loc[self.samples]

        # Check names of variables: they must start with letters
        check_names(self.snps)
        check_names(self.phens)

        # do not name any variable with the default kinship variable name
        if self.kinship_var_name is self.snps + self.phens:
            raise ValueError(f'No variables can be named with {self.kinship_var_name}')


        # --------------------------------------------------
        # Set random effects
        # Random effect can be set by the following ways:
        # (0) estim_kinship: estimate kinship
        # (1) s_reffs: names of variables - then identity matrix
        # (2) cov_reffs: dictionary with keys - names of variables, items - covariance matrices
        self.r_eff = dict()

        # (0) Estimate Kinship
        if estim_kinship:
            mx_kinship = self.estim_kinship()
            # # add dummy variable into data
            self.d_phens[self.kinship_var_name] = range(self.n_samples)
            self.r_eff.update({self.kinship_var_name:
                                   REff(self.d_phens[self.kinship_var_name],
                                        mx_kinship, show_warning=False and show_warning)})
        #
        # (1)
        if s_reffs is not None:
            for s in s_reffs:
                if s not in self.d_all.columns:
                    raise ValueError(f'random effect {s} is not in the data')
            self.r_eff.update({s: REff(self.d_all[s], show_warning=show_warning)
                               for s in s_reffs})

        # (2)
        if cov_reffs is not None:
            for s in cov_reffs.keys():
                if s not in self.d_all.columns:
                    raise ValueError(f'random effect {s} is not in the data')
            tmp = {s: REff(self.d_all[s], mx, show_warning=show_warning)
                   for s, mx in cov_reffs.items()}
            self.r_eff.update(tmp)

        # --------------------------------------------------
        # Assess types of variables
        # do not standardize variables, which are in random effects
        # Types of phenotypic variables
        self.d_phen_types = self.set_phen_types(d_phen_types=d_phen_types,
                                                ord_vars_flag=ord_vars_flag,
                                                ord_vars_thresh=ord_vars_thresh)

        # Standardization of phenotypes
        # Z-score for phenotypes
        self.std(std_flag=std_flag)

    @property
    def d_all(self):
        return concat([self.d_phens, self.d_snps], axis=1)

    def subdata(self,smpl_ids=None, smpl_names=None):
        """
        Get Data object for the subset of samples.
        User can provide either names or IDs or samples.
        Only one can be provided
        :param smpl_names:
        :param smpl_ids:
        :return:
        """
        if (smpl_ids is None) and (smpl_names is None):
            raise ValueError('Please, provide samples for subdataset')
        if (smpl_ids is not None) and (smpl_names is not None):
            raise ValueError('Please, do not provide both '
                             'smpl_names and smpl_ids')

        if smpl_names is not None:
            if len(diff(smpl_names, self.samples)) > 0:
                raise ValueError(f'Not all of the samples are in the dataset: '
                                 f'{diff(smpl_names, self.samples)}')

            smpl_ids = [self.samples.index(s) for s in smpl_names]


        if not all([i in range(self.n_samples) for i in smpl_ids]):
            raise ValueError('Indexes of samples is our of range')


        raw_phens = self.d_phens.iloc[smpl_ids]

        raw_phens = raw_phens.divide(1 / self.s_phens, axis='columns')
        raw_phens = raw_phens.add(self.m_phens, axis='columns')

        # Create data object
        # the problem can appear, if a subset contains not all of the values
        # of a random effect variable
        data_sub = Data(d_snps=self.d_snps.iloc[smpl_ids],
                        d_phens=raw_phens,
                        d_phen_types=self.d_phen_types,
                        estim_kinship=False,
                        show_warning=False)
        # random variables
        data_sub.r_eff.update({s: v.get_subset(smpl_ids)
                               for s, v in self.r_eff.items()})

        return data_sub

    def std(self, m=None, s=None, std_flag=True):
        """
        Standardization of thr data, z-score
        :param m: mean values for thr standardization
        :param s: standard deviation for the standardization
        :param std_flag: False: if you want to return to initial values of phenotypes
        :return:
        """
        # try:
        #     # Return the dataset to the initial form
        #     # if it was already loaded and standardized
        #     self.d_phens = self.d_phens.divide(1/self.s_phens, axis='columns')
        #     self.d_phens = self.d_phens.add(self.m_phens, axis='columns')
        # except:
        #     pass


        # TODO do not normalise ordinal variables
        if m is None:
            m = self.d_phens.mean()
            # m[[tmp not in self.d_phen_types[PhenType.norm] for tmp in m.index]] = 0

        if s is None:
            s = self.d_phens.std()
            # s[[tmp not in self.d_phen_types[PhenType.norm] for tmp in s.index]] = 1

        if not std_flag:
            m = m * 0
            s = s * 0 + 1

        # m[3] = 0
        # s[3] = 1


        self.m_phens = m
        self.s_phens = s

        self.d_phens = self.d_phens.add(-m, axis='columns')
        self.d_phens = self.d_phens.divide(s, axis='columns')

        return m, s

    def estim_kinship(self):
        """
        Estimate Kinship as in rrBLUP - Georgy
        :return: kinship
        """
        # TODO
        # return np.identity(self.n_samples)
        if len(self.d_snps.columns) < 2:
            raise ValueError('Kinship matrix cannot be calculated')
        return self.d_snps.transpose().cov().to_numpy()


    # ---------------------------------------------
    # Set functions with checks

    def set_snps(self,
                 d_snps=None,
                 file_snps=None, sep='\t'):
        """
        Set SNPs
        :param d_snps:
        :param file_snps:
        :param sep:
        :return:
        """
        if (file_snps is None) and (d_snps is None):
            # raise ValueError('Please, provide SNPs')
            if self.echo:
                print('SNPs are not provided')
            self.d_snps = DataFrame(index=self.d_phens.index)
        elif d_snps is None:
            check_file(file_snps)
            self.d_snps = read_csv(file_snps, sep=sep, index_col=0)
        else:
            self.d_snps = d_snps
        return self.d_snps

    def set_phens(self,
                  d_phens=None,
                  file_phens=None, sep='\t'):
        """
        Set phenotypes
        :param d_phens:
        :param file_phens:
        :param sep:
        :return:
        """
        if (file_phens is None) and (d_phens is None):
            raise ValueError('Please, provide phenotypes')
        if d_phens is None:
            check_file(file_phens)
            self.d_phens = read_csv(file_phens, sep=sep, index_col=0)
        else:
            self.d_phens = d_phens
        return self.d_phens


    def set_phen_types(self,
                       d_phen_types=None,
                       file_phen_types=None,
                       ord_vars_flag=True,
                       ord_vars_thresh=3,
                       echo=False):
        """

        :param d_phen_types:
        :param file_phen_types:
        :param ord_vars_flag:
        :param ord_vars_thresh:
        :return:
        """

        def check_types(d_phen_types, possible_phen_types):
            for k in d_phen_types:
                if k not in possible_phen_types:
                    raise ValueError(f'Undefined type for phenotypes provided {k}')

        if (d_phen_types is None) and (file_phen_types is not None):
            self.d_phen_types = get_groups(file_phen_types)
            check_types(self.d_phen_types, self.possible_phen_types)
        if (d_phen_types is not None):
            self.d_phen_types = d_phen_types
            check_types(self.d_phen_types, self.possible_phen_types)
        if d_phen_types is None:
            self.d_phen_types = dict()

        if ord_vars_flag:
            phens_anon = diff(self.phens, sum(self.d_phen_types.values(), []))
            phens_ord = [p for p in phens_anon
                         if self.d_phens[p].nunique() <= ord_vars_thresh]
            if echo:
                print(f'Ordinal phenotypes will be found automatically '
                      f'as having max {ord_vars_thresh} unique values:\n'
                      f'{phens_ord}')

            if PhenType.ord in self.d_phen_types:
                self.d_phen_types[PhenType.ord] += phens_ord
            else:
                self.d_phen_types[PhenType.ord] = phens_ord

        # Consider all remaining phenotypes as normally distributed
        phens_anon = diff(self.phens, sum(self.d_phen_types.values(), []))
        if PhenType.norm in self.d_phen_types:
            self.d_phen_types[PhenType.norm] += phens_anon
        else:
            self.d_phen_types[PhenType.norm] = phens_anon

        return self.d_phen_types


class CVset:
    def __init__(self, dataset: Data,
                 n_cv=10,
                 params_tune=None,
                 rnd_seed=239):
        if n_cv < 2:
            raise ValueError('Cross validation requires 2 folds minimum')

        np.random.seed(rnd_seed)
        n_samples = dataset.n_samples
        idx = np.random.permutation(list(range(n_cv)) * ceil(n_samples/n_cv))
        idx = idx[0:n_samples]
        self.test = []
        self.train = []
        for i in range(n_cv):
            idx_test = list(np.where(idx == i)[0])
            idx_train = [i for i in range(n_samples) if i not in idx_test]

            self.test += [dataset.subdata(smpl_ids=idx_test)]
            self.train += [dataset.subdata(smpl_ids=idx_train)]


class REff:
    """
    This class contain:
    name of random effect variable
    variable
    unique values of variable
    z-matrix of loagings
    covariance matrix
    object does not know anything about the names of samples: slice only by index
    """
    def __init__(self,
                 variable,
                 covariance=None,
                 show_warning=True):

        if isinstance(variable, DataFrame):
            self.name = variable.columns.to_list()
            if len(self.name) > 0:
                raise ValueError(f'Random effect should be provided '
                                 f'for only one variable. You set {self.name}')
            variable = variable.iloc[:, 1]

        if not isinstance(variable, Series):
            raise ValueError('Incorrect type of variable for random effect.')

        self.var = variable

        if covariance is None:
            self.cov_mx = np.identity(self.n_values)
        else:
            #TODO
            if show_warning:
                warnings.warn(f'\nAchtung! the covariance matrix was provided \n'
                              f'for values in the variable {self.var.name} \n'
                              f'in the following order: \n {self.u_values}')

            self.cov_mx = covariance


        # Check symmetric
        if self.cov_mx.shape[0] != self.cov_mx.shape[1]:
            raise ValueError(f'Covariance matrix for random effect {self.name}'
                             f'is not square')

        if not is_symmetric(self.cov_mx):
            raise ValueError(f'Covariance matrix for random effect {self.name}'
                             f'is not symmetric')

        if self.cov_mx.shape[0] != self.n_values:
            raise ValueError(f'Covariance matrix for random effect {self.name}'
                             f'does not match the number of unique values')

        # Create loadings
        self.z = np.zeros((self.n_values, self.n_samples))
        for i, v in enumerate(self.u_values):
            cond = self.var == v
            self.z[i, cond.to_list()] = 1

    @property
    def n_values(self):
        return len(self.u_values)

    @property
    def n_samples(self):
        return len(self.var)

    @property
    def u_values(self):
        tmp = self.var.unique().tolist()
        tmp.sort()
        return tmp

    def get_subset(self, smpl_ids):
        """
        Get cory of this random effect, but for lower number of samples
        :param smpl_ids: indexes of samples to remain
        :return: new random effect
        """

        reff = REff(self.var, self.cov_mx, show_warning=False)
        reff.var = reff.var.iloc[smpl_ids]
        reff.z = reff.z[:, smpl_ids]

        idx_remain = [i for i, v in enumerate(self.u_values)
                      if v in reff.u_values]
        reff.cov_mx = reff.cov_mx[:, idx_remain][idx_remain, :]
        reff.z = reff.z[idx_remain, :]

        if not is_symmetric(self.cov_mx):
            raise ValueError(f'During subset, the covariance matrix '
                             f'for random effect'
                             f'is not symmetric')

        return reff



