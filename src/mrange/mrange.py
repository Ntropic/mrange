# -*- coding: utf-8 -*-
from numba import njit

@njit(fastmath=True, nogil=True)# , cache = True)
def mrange(n_s):
    # mrange is used to replace nested range loops
    # Input: n_s -> Tuple of integers
    # Output: Generator outputting len(n_s) values with every call, generating every combination of values in the intervals [0,1,...,n_s[i]-1]
    # Use:
    # for a, b, ..., m in mrange((n_a, n_b, ..., n_m)):
    #     ...
    # Replaces:
    # for a in range(n_a):
    #     for b in range(n_b):
    #        ...
    #            for m in range(n_m):
    #                ...
    # By Michael Schilling, with the help of Francesco Preti
    if len(n_s) > 1:
        for i in range(n_s[0]):
            for x in mrange(n_s[1:]):
                yield i, *x
    else:
        for i in range(n_s[0]):
            yield i,