"""
Module to work with the data and to create cross-validation set
"""

import numpy as np
from pandas import read_csv, DataFrame, concat
import warnings

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

    def __init__(self,
                 d_snps: DataFrame = None,
                 d_phens: DataFrame = None,
                 d_phen_types=None,
                 d_meta: DataFrame = None,
                 d_kinship: DataFrame = None,
                 file_snps=None,
                 file_phens=None,
                 file_phen_types=None,
                 file_meta=None,
                 file_kinship=None,
                 std_flag=True,
                 kinship_flag=True,
                 ord_vars_flag=True,
                 ord_vars_thresh=3,
                 s_nan='NA',
                 sep='\t',
                 echo=False):

        self.echo = echo
        # If both data and file are provided, file will not be read

        self.d_snps = self.set_snps(d_snps=d_snps, file_snps=file_snps, sep=sep)
        self.d_phens = self.set_phens(d_phens=d_phens, file_phens=file_phens, sep=sep)

        # Get correspondence
        self.samples = intersect(self.d_snps.index, self.d_phens.index)
        self.snps = list(self.d_snps.columns)
        self.phens = list(self.d_phens.columns)
        self.n_samples = len(self.samples)
        self.n_snps = len(self.snps)
        self.n_phens = len(self.phens)

        self.d_snps = self.d_snps.loc[self.samples]
        self.d_phens = self.d_phens.loc[self.samples]

        # Set group phenotype for semopy ModelEffects
        self.d_group = DataFrame(data=list(self.d_phens.index),
                                 index=self.d_phens.index,
                                 columns=['group'])


        # Types of phenotypic variables
        self.d_phen_types = self.set_phen_types(d_phen_types=d_phen_types,
                                                file_phen_types=file_phen_types,
                                                ord_vars_flag=ord_vars_flag,
                                                ord_vars_thresh=ord_vars_thresh)

        # Get Kinship
        self.d_kinship = self.set_kinship(d_kinship=d_kinship,
                                          file_kinship=file_kinship,
                                          kinship_flag=kinship_flag,
                                          sep=sep)
        # Get metadata
        self.d_meta = self.set_meta(d_meta=d_meta,
                                    file_meta=file_meta,
                                    sep=sep)

        # Z-score for phenotypes
        self.std(std_flag=std_flag)
        # Covariance matrix
        self.cov = self.estim_cov()

        if echo:
            if len(self.d_snps.index) > len(self.samples):
                print(f'Some samples with SNPs were omitted: '
                      f'{diff(self.d_snps.index, self.samples)}')
            if len(self.d_phens.index) > len(self.samples):
                print(f'Some samples with phenotypes were omitted: '
                      f'{diff(self.d_phens.index, self.samples)}')
            print(f'Number of samples: {self.n_samples}',
                  f'Number of SNPs: {self.n_snps}')

    @property
    def d_all(self):
        return concat([self.d_phens, self.d_snps, self.d_group], axis=1)

    def subdata(self, smpl_names=None, smpl_ids=None):
        """
        Get Data object for the subset of samples.
        User can provide either names or IDs or samples.
        If both are provided, only names are used
        :param smpl_names:
        :param smpl_ids:
        :return:
        """
        if (smpl_ids is None) and (smpl_names is None):
            raise ValueError('Please, provide samples for subdataset')
        if smpl_names is not None:
            if smpl_ids is not None:
                print('Please, do not provide both smpl_names and smpl_ids: '
                      'only smpl_names was used')
            if len(diff(smpl_names, self.samples)) > 0:
                raise ValueError(f'Not all of the samples are in the dataset: '
                                 f'{diff(smpl_names, self.samples)}')

            smpl_ids = [self.samples.index(s) for s in smpl_names]


        if not all([i in range(self.n_samples) for i in smpl_ids]):
            raise ValueError('Indexes of samples is our of range')


        raw_phens = self.d_phens.iloc[smpl_ids]

        raw_phens = raw_phens.divide(1 / self.s_phens, axis='columns')
        raw_phens = raw_phens.add(self.m_phens, axis='columns')

        data_sub = Data(d_snps=self.d_snps.iloc[smpl_ids],
                        d_phens=raw_phens,
                        d_kinship=self.d_kinship.iloc[smpl_ids, smpl_ids],
                        d_phen_types=self.d_phen_types)

        return data_sub

    def std(self, m=None, s=None, std_flag=True):
        """
        Standardization of thr data, z-score
        :param m: mean values for thr standardization
        :param s: standard deviation for the standardization
        :param std_flag: False: if you want to return to initial values of phenotypes
        :return:
        """
        try:
            # Return the dataset to the initial form
            # if it was already loaded and standardized
            self.d_phens = self.d_phens.divide(1/self.s_phens, axis='columns')
            self.d_phens = self.d_phens.add(self.m_phens, axis='columns')
        except:
            pass

        if not std_flag:
            self.m_phens = 1
            self.s_phens = 1
            return m, s

        # TODO do not normalise ordinal variables
        if m is None:
            m = self.d_phens.mean()
        if s is None:
            s = self.d_phens.std()

        self.m_phens = m
        self.s_phens = s

        self.d_phens = self.d_phens.add(-m, axis='columns')
        self.d_phens = self.d_phens.divide(s, axis='columns')

        return m, s

    def estim_kinship(self):
        """
        Estimate Kinship as in rrBLUP
        :return: kinship
        """
        # TODO
        # return np.identity(self.n_samples)
        return self.d_snps.transpose().cov()

    def estim_cov(self):
        """
        Estimate the covariance matrix between:
        phens vs. phens
        phens vs. snps
        :return:
        """
        # TODO
        return np.random.rand(self.n_phens, self.n_phens + self.n_samples)



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
            raise ValueError('Please, provide SNPs')
        if d_snps is None:
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

    def set_kinship(self,
                    d_kinship=None,
                    file_kinship=None,
                    kinship_flag=True,
                    sep='\t'):
        """
        Estimate or set Kinship matrix
        :param d_kinship:
        :param file_kinship:
        :param kinship_flag:
        :param sep:
        :return:
        """
        if not kinship_flag:
            self.d_kinship = None
            return None


        if d_kinship is not None:
            self.d_kinship = d_kinship
        elif file_kinship is not None:
            check_file(file_kinship)
            self.d_kinship = read_csv(file_kinship, sep=sep, index_col=0)
        else:
            self.d_kinship = self.estim_kinship()


        return self.d_kinship

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

    def set_meta(self, d_meta=None, file_meta=None, sep='\t'):
        # TODO
        # if (d_meta is None) and (file_meta is not None):
        #     check_file(file_meta)
        #     d_meta = read_csv(file_meta, sep=sep, index_col=0)

        return d_meta


class CVset:
    def __init__(self, dataset: Data,
                 n_cv=10,
                 params_tune=None,
                 rnd_seed=239):

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





