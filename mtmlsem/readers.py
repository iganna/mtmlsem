#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple readers for phenotype and genotype files."""
import datatable as dt
import pandas as pd
import numpy as np



def read_genotypes(file: str, fmt='auto'):
    """
    Read genotype file.

    This function tries to guess the correct format of the genotype
    automatically. 
    Parameters
    ----------
    file : str
        Filename.
    fmt : str, optional
        File format, can be 'sam' or 'tabular'. If auto, the function
        attempts to guess the file format from the filename. 'sam' allows for
        any samtools-compatible format. The default is 'auto'.

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame with columns as SNPs in the 0-1-2 format, and
        index as indiviuduals/genotype IDs.

    """
    
    if fmt == 'tabular':
        return read_tabular(file)
    elif fmt == 'sam':
        return read_variant(file)
    elif fmt == 'auto':
        if file.endswith(('.gz', '.bcf', '.vcf', '.bam')):
            return read_variant(file)
        return read_tabular(file)
    else:
        raise NotImplementedError(f"Unkown format {fmt}.")


def read_phenotypes(file: str):
    """
    Read phenotype file.

    If first column is unnamed, then the resulting dataframe assumes it as an
    index column.
    Parameters
    ----------
    file : str
        Filename.

    Returns
    -------
    df : pd.DataFrame
        Pandas DataFrame.

    """
    
    df = dt.fread(file).to_pandas()
    if df.columns[0] in ('C0', 'C1'):
        df = df.set_index(df.columns[0])
    return df


def read_variant(file: str):
    from pysam import VariantFile
    vf = VariantFile(file)
    d = dict()
    for rec in vf:
        snp = f's_{rec.chrom}_{rec.pos}'
        gts = list()
        for _, info in rec.samples.items():
            try:
                gts.append(sum(info['GT']))
            except TypeError:
                gts.append(float('nan'))
        d[snp] = gts
    df = pd.DataFrame.from_dict(d, dtype=float)
    df.index = list(rec.samples.keys())
    return df


def read_tabular(file: str):
    df = dt.fread(file).to_pandas()
    cols = list(df.columns)
    cols[0] = 'indv'
    df.columns = cols
    df = df.set_index(df.columns[0])
    try:
        unqs = np.unique(df).astype(float)
        if (not np.any(np.isnan(unqs))) and (unqs.max() > 1) and\
            np.all((unqs[unqs < 0] == -1)):
            df = df.replace(-1, float('nan'))
        return df.astype(float)
    except (ValueError, TypeError):
        raise ValueError(f"Couldn't convert {file} items to float values."
                         " Make sure that there are no illegal and non-numeric"
                         " characters. SNP tabular should start with a column"
                         " of individuals/genotypying IDs that correspond to"
                         " individuals in phenotypes file, followed by columns"
                         " of genotype/SNP values.")
