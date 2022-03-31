# -*- coding: utf-8 -*-
from numba import njit
from numpy import prod, mod, zeros_like

do_cache = False

@njit(fastmath=True, nogil=True , cache = do_cache)
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
        sub_n_s = n_s[1:]
        for i in range(n_s[0]):
            for x in mrange(sub_n_s):
                yield i, *x
    else:
        for i in range(n_s[0]):
            yield i,

@njit(fastmath=True, nogil=True , cache = do_cache)
def next_in_mrange(A, n_s, n_1):
    i = n_1
    keep_going = 1
    while keep_going:
        A[i] = mod((A[i]+1), n_s[i])
        if not A[i] == 0:
            keep_going = 0
        i += -1
    return A
@njit(fastmath=True, nogil=True , cache = do_cache) #
def mrange_array(n_s):
    prod_n_s_1 = prod(n_s)-1
    n = len(n_s)
    n_1 = n-1
    A = zeros_like(n_s)
    yield A
    how_high = 0
    while how_high < prod_n_s_1:
        A = next_in_mrange(A, n_s, n_1)
        how_high += 1
        yield A