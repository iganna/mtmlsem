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
from itertools import combinations, product
from tqdm import tqdm
import pandas as pd
import numpy as np
from utils import translate_names, unique_mapping


def gwas_lmm(Model, y: list[str], phenos, genes, desc='', init_args=None,
             fit_args=None, dropna=True, verbose=True):
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
    it = genes.iteritems()
    if verbose:
        it = tqdm(list(it))
    for name, gene in it:
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
    cols = ['SNP', 'chr', 'pos', 'lf'] + [f'{p}_p-value' for p in y] +\
           [f'{p}_b' for p in y]
    return pd.DataFrame(res, columns=cols)


def gwas_w(lt):
    gs, lt = lt
    mod, y, phenos, genes, desc, init_args, fit_args = lt
    return gwas_lmm(mod, y, phenos, genes[gs], desc, init_args, fit_args,
                    verbose=False)

def gwas(Model, y: list[str], phenos, genes, desc='', init_args=None, 
         fit_args=None, num_processes=-1, chunk_size=1000, verbose=True):
    """
     Multi-trait single-locus GWAS with multiprocessing support.

    Parameters
    ----------
    Model : class
        semopy class.
    y : list[str]
        List of phenotype names. If string is passed, then it is automatically
        converted to list that contains a single element.
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
    num_processes : int, optional
        Number of processes to run. If -1, then it is selected to number of
        avaialable CPU cores minus 1. "None" is the same as 1. The default is
        -1.
    chunk_size : int, optional
        Number of SNPs to be sent onto a single process. The default is 1000.
    verbose : bool, optional
        If False, then no progress bar will be printed. The default is True.

    Returns
    -------
    pd.DataFrame
        GWAS results.

    """
    from multiprocessing import Pool, cpu_count
    from tqdm import tqdm

    if type(y) is str:
        y = [y]
    if num_processes == -1:
        num_processes = cpu_count() - 1
    if num_processes in (None, 0, 1):
        return gwas_lmm(Model, y, phenos, genes, desc, init_args, fit_args,
                        verbose=verbose)
    # We rule out duplicate SNPs to ease the computational burden:
    unique = unique_mapping(genes)
    genes = genes[list(unique.keys())]

    result = None
    lt = list(genes.columns)
    chunk_size = min(chunk_size, len(lt) // num_processes )
    lt2 = [lt[i:i+chunk_size] for i in range(0, len(lt), chunk_size)]
    lt = [(Model, y, phenos, genes, desc, init_args, fit_args)]
    prod = product(lt2, lt)
    with Pool(num_processes) as p:
        it = p.imap(gwas_w, prod)
        if verbose:
            it = tqdm(it, total=len(lt2))
        for t in it:
            if result is None:
                result = t
            else:
                result = pd.concat([result, t])

    # Now, we add back removed duplicate SNPs
    dups = list()
    for _, row in result.iterrows():
        for dup in unique[row['SNP']]:
            row = row.copy()
            row['SNP'] = dup
            dups.append(row)
    for dup in dups:
        result = result.append(dup)       
    result = result.sort_values(['chr', 'pos'])
    result.index = list(range(1, len(result) + 1))
    return result
            
