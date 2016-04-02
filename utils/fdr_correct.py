from __future__ import division
import numpy as np


def fdr2(pvals, q = 0.05):
    """
    compute FDR-threshold for vector of pvalues
    based of off fdr2.m
    """
    q = 0.05
    cV = 1

    #part 1 : fdr threshold
    #sort pvals
    oidx = np.argsort(pvals)
    pvals = np.sort(pvals)

    #number of obervation
    V = pvals.size

    #order (indices) in the same size as pvalues
    idx = (np.arange(0,V))

    # line to be used as cutoff
    thrline = idx * q / (V * cV)

    # find the largest pval, still under the line
    try:
        thr = np.max(pvals[pvals<=thrline])
    except ValueError:
        thr = []

    #deal with the case where all the other points under the line are equal to zero and other points are above the line
    if thr == 0:
        thr = np.max(thrline[pvals<=thrline])

    #case when it does not cross
    if not(thr):
        thr = 0

    #part 2: fdr corrected (doesn't match matlab)
    #p-corrected
    #pcor = pvals * V * cV /idx

    #sort back to original order and output
    #oidxR = np.sort(oidx)
    #pcor = pcor[oidxR]

    return thr
