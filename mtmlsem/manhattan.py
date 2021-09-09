#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manhattan plot for visualizing GWAS experiments.
"""
import datatable as dt
import numpy as np
from adjustText import adjust_text
from matplotlib import pyplot as plt
from statsmodels.stats.multitest import multipletests


def manhattan(tests, name: str, alpha=0.05, mp_test='fdr_bh',
              label_snps=True, clipping=16, chr_shift=0,
              show_num_accepts=False):
    """
    Draw manhattan plot via Matplotlib.pyplot library.

    Note that it won't automatically save the figure to a file. One should
    run plt.save("filename.format") afterwards.
    Parameters
    ----------
    tests : str or pandas.DataFrame
        Either a filename with gwas results or the gwas results in the format
        of pandas DataFrame.
    name : str
        Name of the phenotype. Note that the DataFrame should have columns
        name_p-value.
    alpha : float, optional
        P-value cutoff value. The default is 0.05.
    mp_test : str, optional
        Name of multipletest method. If None, then exactly alpha is used. The
        default is 'fdr_bh'.
    label_snps : bool, optional
        If True, the labels for selected SNPs are shown. The default is True.
    clipping : float, optional
        Can be None or float. If float, then its used as a ceiling for the
        plot. Also, p-values are clipped by this ceiling. The default is 16.
    chr_shift : int, optional
        Distance between chromosomes. The default is 0.
    show_num_accepts : bool, optional
        If True, then title also has a number of chosen SNPs. The default is
        False.

    Returns
    -------
    Selected SNPs.

    """
    
    if type(tests) is str:
        tests = dt.fread(tests).to_pandas()
    tests = tests.sort_values(['chr', 'pos'])
    pvals = sorted(tests[f'{name}_p-value'])[::-1]
    if mp_test:
        mp = multipletests(pvals, alpha, mp_test)[0]
        for n in range(len(mp)):
            if mp[n]:
                break
        if mp[n]:
            alpha = pvals[n] - (pvals[n] - pvals[n - 1]) / 2
        else:
            alpha = float('inf')
    alpha = -np.log(alpha) / np.log(10)
    pvals = -np.log(tests[f'{name}_p-value']) / np.log(10)
    if clipping is not None:
        pvals = np.clip(pvals, 0, clipping)
    first_chr = tests['chr'].iloc[0]
    last_chr = tests['chr'].iloc[-1]
    ticks = []
    shift = 0
    tmp = []
    pos = tests['pos'].values
    texts = list()
    for c in range(first_chr, last_chr + 1):
        a = list(tests['chr']).index(c)
        try:
            b = list(tests['chr']).index(c + 1)
        except ValueError:
            b = len(tests['chr'])
        x = list(range(a + shift, b + shift))
        y = pvals[a:b]
        ps = pos[a:b]
        for i, pval in enumerate(y):
            if label_snps and pval >= alpha:
                name = f'{c}_{ps[i]}'
                txt = plt.text(x[i], pval + np.random.random() / 50, name,
                                fontsize="xx-small")
                texts.append(txt)
        plt.scatter(x, y, 1)
        ticks.append(a + (b - a) // 2 + shift)
        for i in range(a, b):
            t = tests.index[i]
            if t in tmp:
                plt.plot(i + shift, pvals[i], 'kx')
        shift += chr_shift
    if label_snps:
        adjust_text(texts, 
                    autoalign='y', ha='left',
                    lim=250,
                    only_move={'text':'y'},
                    force_text=(0,0.7),
                    arrowprops=dict(arrowstyle="-",color='k', lw=0.5,
                                    alpha=0.6))
    if np.isfinite(alpha):
        plt.plot([0, i + shift], [alpha, alpha], 'k--')
    if clipping is not None:
        plt.ylim([0, clipping])
    plt.xticks(ticks, list(range(first_chr, last_chr + 1)))
    plt.margins(x=0, y=0)
    plt.xlabel('chromosome')
    plt.ylabel('-log10(p-value)')
    t = pvals > alpha
    if show_num_accepts:
        c = sum(t)
        plt.title(name + f' [{c}]')
    else:
        plt.title(name)
    return tests.iloc[np.where(t)[0]]