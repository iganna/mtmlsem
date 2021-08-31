#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Helper module for performing MTSL GWAS inside of MTMLSEM model.

Note that this has little to do with MTMLSEM, it merely fits the classical LMM
model of the kind:
    Y = X B + E + U,
where Y and X are deterministic data matrices, B is a matrix of regression
coefficients, E and U are matrices random matrices with U being the random
effect matrix, that takes genetic kinship between individuals into an account.
"""
from itertools import combinations
import pandas as pd
import numpy as np
from utils import translate_names


def gwas(Model, y: list[str], phenos, genes, desc='', init_args=None,
         fit_args=None, dropna=True):
    """
    Multi-trait single-locus GWAS via linear (possibly mixed) model.

    Parameters
    ----------
    Model : class
        semopy class.
    y : list[str]
        List of phenotype names.
    phenos : pd.DataFrame
        Phenotypes + possibly other variables.
    genes : pd.DataFrame
        Genotypes/SNPs.
    desc : str, optional
        Extra model description. The default is ''.
    init_args : dict, optional
        Extra arguments for Model constructor. The default is None.
    fit_args : dict, optional
        Extra arguments for Model fit method (e.g., k). The default is None.
    dropna : bool, optional
        If True, then NaN rows are dropped for each gene test. The default is
        True.

    Returns
    -------
    pd.DataFrame
        GWAS results.

    """
    
    if init_args is None:
        init_args = dict()
    if fit_args is None:
        fit_args = dict()
    res = list()
    desc= desc + '\n{} ~ snp'.format(', '.join(y))
    for a, b in combinations(y, 2):
        desc += f'\n{a} ~~ {b}'
    m = Model(desc, **init_args)
    phenos = phenos.copy()
    for name, gene in genes.iteritems():
        chr, pos = translate_names(name)
        phenos['snp'] = gene.values
        if dropna:
            data = phenos.dropna()
        try:
            r = m.fit(data, clean_slate=True, **fit_args)
            if type(r) is not tuple:
                succ = r.success
                fun = r.fun
            else:
                succ = r[0].success & r[1].success
                fun = r[1].fun
        except np.linalg.LinAlgError:
            succ = False
        if  not succ:
            t = [name, chr, pos, float('nan')] + [1.0] * len(y)
            t += [float('nan')] * len(y)
            res.append(t)
        else:
            ins = m.inspect()
            ins = ins[(ins['rval'] == 'snp') & (ins['op'] == '~')]
            pvals = list()
            ests = list()
            for _, row in ins.iterrows():
                pvals.append(row['p-value'])
                ests.append(row['Estimate'])
            res.append([name, chr, pos, fun] + pvals + ests)
    cols = ['SNP', 'chr', 'pos'] + [f'{p}_p-value' for p in y] + [f'{p}_b'
                                                                  for p in y]
    return pd.DataFrame(res, columns=cols)



