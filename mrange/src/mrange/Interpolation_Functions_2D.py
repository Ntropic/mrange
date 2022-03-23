# -*- coding: utf-8 -*-
from numpy import empty, dot, exp, mod, floor, abs, sign
from numba import njit, types, complex128, float64, int32, int64

from .UI_Prepare import *

##### Indexes ##########################################################################################################
@njit(fastmath=True, nogil=True, cache=True)
def Index_Norm_2D(i0, j0, i_max, j_max): # Add in i_max and j_max for boundary cases
    # Find square index and coefficient
    li = len(i0)
    cr = empty(li, dtype='i8') #int64
    dr = empty(li, dtype='i8')
    ic = empty(li, dtype='i8')
    for i in range(li):
        cr[i] = int(round((i0[i]+j0[i])/2))    # Coordinates in rotated and stretched reference frame
        dr[i] = int(round((i0[i]-j0[i])/2))    # Coordinates in rotated and stretched reference frame
    ia, ib = cr+dr, cr-dr # Indexes
    for i in range(li):
        ic[i] = int(floor(ia[i])/2)
    ca = (i0 - ia)
    cb = (j0 - ib)
    return ia, ib, ic, ca, cb

def Make_U_Partial(U_1, U_0):
    s = U_1.shape[0]
    D0 = 1j*logmU(dot(U_1,U_0))
    E0, V0 = eigh(D0)
    return E0, V0

##### Unitaries ########################################################################################################
@njit(fastmath=True, nogil=True, cache=True)
def expmH_2D(s, L, EL, C, ER, R, i_a, i_b, i_c, c_a, c_b):
    A = empty((s, s), dtype='c16') #complex128)
    A2 = empty((s, s), dtype='c16') #complex128)
    sb = int((sign(c_b)-1)/2)
    sa = int((sign(c_a)-1)/2)
    i_r = i_b + sb
    i_l = i_a + sa
    ind = -2*sb-sa
    for i in range(s):
        A[i, :]  = exp(-1j * ER[i_r, i_a, i] * abs(c_b)) * R[i_r, i_a, i, :]
        A2[i, :] = exp(-1j * EL[i_b, i_l, i] * abs(c_a)) * C[i_b, i_c, ind, i, :]
    return dot(L[i_b, i_l, ...], dot(A2, A))

@njit(fastmath=True, nogil=True, cache=True)
def Interpolate_2D_Pulse(s, L, EL, C, ER, R, i_a, i_b, i_c, c_a, c_b):
    U = expmH_2D(s, L, EL, C, ER, R, i_a[0], i_b[0], i_c[0], c_a[0], c_b[0])
    for i_ai, i_bi, i_ci, c_ai, c_bi in zip(i_a[1:], i_b[1:], i_c[1:], c_a[1:], c_b[1:]):
        U = dot(expmH_2D(s, L, EL, C, ER, R, i_ai, i_bi, i_ci, c_ai, c_bi), U)
    return U

@njit(fastmath=True, nogil=True, cache=True)
def dAB_expmH_2D(s, L, EL, C, ER, R, i_a, i_b, i_c, c_a, c_b, das, dbs):
    A = empty((s, s), dtype='c16') #complex128)
    A2 = empty((s, s), dtype='c16') #complex128)
    dA = empty((s, s), dtype='c16') #complex128)
    dA2 = empty((s, s), dtype='c16') #complex128)
    sb = int((sign(c_b)-1)/2)
    sa = int((sign(c_a)-1)/2)
    i_r = i_b + sb
    i_l = i_a + sa
    ind = -2*sb-sa
    for i in range(s):
        a = exp(-1j * ER[i_r, i_a, i] * abs(c_b)) * R[i_r, i_a, i, :]
        b = exp(-1j * EL[i_b, i_l, i] * abs(c_a)) * C[i_b, i_c, ind, i, :]
        A[i, :]  = a
        dA[i, :] = -1j * dbs * ER[i_r, i_a, i] * a
        A2[i, :] = b
        dA2[i, :] = -1j * das * EL[i_b, i_l, i] * b
    return dot(L[i_b, i_l, ...], dot(dA2, A)), dot(L[i_b, i_l, ...], dot(A2, dA))
@njit(fastmath=True, nogil=True, cache=True)
def AB_dAB_expmH_2D(s, L, EL, C, ER, R, i_a, i_b, i_c, c_a, c_b, das, dbs):
    A = empty((s, s), dtype='c16') #complex128)
    A2 = empty((s, s), dtype='c16') #complex128)
    dA = empty((s, s), dtype='c16') #complex128)
    dA2 = empty((s, s), dtype='c16') #complex128)
    sb = int((sign(c_b)-1)/2)
    sa = int((sign(c_a)-1)/2)
    i_r = i_b + sb
    i_l = i_a + sa
    ind = -2*sb-sa
    for i in range(s):
        a = exp(-1j * ER[i_r, i_a, i] * abs(c_b)) * R[i_r, i_a, i, :]
        b = exp(-1j * EL[i_b, i_l, i] * abs(c_a)) * C[i_b, i_c, ind, i, :]
        A[i, :]  = a
        dA[i, :] = -1j * dbs * ER[i_r, i_a, i] * a
        A2[i, :] = b
        dA2[i, :] = -1j * das * EL[i_b, i_l, i] * b
    return dot(L[i_b, i_l, ...], dot(A2, A)), dot(L[i_b, i_l, ...], dot(dA2, A)), dot(L[i_b, i_l, ...], dot(A2, dA))
@njit(fastmath=True, nogil=True, cache=True)
def AB_dA_expmH_2D(s, L, EL, C, ER, R, i_a, i_b, i_c, c_a, c_b, das, dbs):
    A = empty((s, s), dtype='c16') #complex128)
    A2 = empty((s, s), dtype='c16') #complex128)
    dA2 = empty((s, s), dtype='c16') #complex128)
    sb = int((sign(c_b)-1)/2)
    sa = int((sign(c_a)-1)/2)
    i_r = i_b + sb
    i_l = i_a + sa
    ind = -2*sb-sa
    for i in range(s):
        a = exp(-1j * ER[i_r, i_a, i] * abs(c_b)) * R[i_r, i_a, i, :]
        b = exp(-1j * EL[i_b, i_l, i] * abs(c_a)) * C[i_b, i_c, ind, i, :]
        A[i, :]  = a
        A2[i, :] = b
        dA2[i, :] = -1j * das * EL[i_b, i_l, i] * b
    return dot(L[i_b, i_l, ...], dot(A2, A)), dot(L[i_b, i_l, ...], dot(dA2, A))
@njit(fastmath=True, nogil=True, cache=True)
def AB_dB_expmH_2D(s, L, EL, C, ER, R, i_a, i_b, i_c, c_a, c_b, das, dbs):
    A = empty((s, s), dtype='c16') #complex128)
    A2 = empty((s, s), dtype='c16') #complex128)
    dA = empty((s, s), dtype='c16') #complex128)
    sb = int((sign(c_b)-1)/2)
    sa = int((sign(c_a)-1)/2)
    i_r = i_b + sb
    i_l = i_a + sa
    ind = -2*sb-sa
    for i in range(s):
        a = exp(-1j * ER[i_r, i_a, i] * abs(c_b)) * R[i_r, i_a, i, :]
        b = exp(-1j * EL[i_b, i_l, i] * abs(c_a)) * C[i_b, i_c, ind, i, :]
        A[i, :]  = a
        dA[i, :] = -1j * dbs * ER[i_r, i_a, i] * a
        A2[i, :] = b
    return dot(L[i_b, i_l, ...], dot(A2, A)), dot(L[i_b, i_l, ...], dot(A2, dA))


## Wavefunctions #######################################################################################################
@njit(fastmath=True, nogil=True, cache=True)
def expmH_2D_wf(L, EL, C, ER, R, i_a, i_b, i_c, c_a, c_b, wf):
    sb = int((sign(c_b) - 1) / 2)
    sa = int((sign(c_a) - 1) / 2)
    i_r = i_b + sb
    i_l = i_a + sa
    ind = -2 * sb - sa
    wf = dot(R[i_r, i_a, ...], wf)
    wf = vecxvec(ER[i_r, i_a, :], wf, abs(c_b))
    wf = dot(C[i_b, i_c, ind, ...], wf)
    wf = vecxvec(EL[i_b, i_l, :], wf, abs(c_a))
    return dot(L[i_b, i_l, ...], wf)

@njit(fastmath=True, nogil=True, cache=True)
def Interpolate_2D_Pulse_wf(L, EL, C, ER, R, i_a, i_b, i_c, c_a, c_b, wf):
    U = expmH_2D_wf(L, EL, C, ER, R, i_a[0], i_b[0], i_c[0], c_a[0], c_b[0], wf)
    for i_ai, i_bi, i_ci, c_ai, c_bi in zip(i_a[1:], i_b[1:], i_c[1:], c_a[1:], c_b[1:]):
        U = dot(expmH_2D_wf(L, EL, C, ER, R, i_ai, i_bi, i_ci, c_ai, c_bi), U, wf)
    return U

@njit(fastmath=True, nogil=True, cache=True)
def dAB_expmH_2D_wf(EA, EB, VRL, VLL, VB, i_a, i_b, c_a, c_b, das, dbs, wf):
    sb = int((sign(c_b) - 1) / 2)
    sa = int((sign(c_a) - 1) / 2)
    i_r = i_b + sb
    i_l = i_a + sa
    ind = -2 * sb - sa

    wf = dot(R[i_r, i_a, ...], wf)
    wf2 = dbs * dvecxvec(ER[i_r, i_a, :], wf, abs(c_b))
    wf = vecxvec(ER[i_r, i_a, :], wf, abs(c_b))
    wf = dot(C[i_b, i_c, ind, ...], wf)
    wf2 = dot(C[i_b, i_c, ind, ...], wf2)
    wf2 = vecxvec(EL[i_b, i_l, :], wf2, abs(c_a))
    wf = das*dvecxvec(EL[i_b, i_l, :], wf, abs(c_a))
    return dot(L[i_b, i_l, ...], wf), dot(L[i_b, i_l, ...], wf2)

@njit(fastmath=True, nogil=True, cache=True)
def AB_dAB_expmH_2D_wf(EA, EB, VRL, VLL, VB, i_a, i_b, c_a, c_b, das, dbs, wf):
    sb = int((sign(c_b) - 1) / 2)
    sa = int((sign(c_a) - 1) / 2)
    i_r = i_b + sb
    i_l = i_a + sa
    ind = -2 * sb - sa

    wf = dot(R[i_r, i_a, ...], wf)
    wf2 = dbs * dvecxvec(ER[i_r, i_a, :], wf, abs(c_b))
    wf = vecxvec(ER[i_r, i_a, :], wf, abs(c_b))
    wf = dot(C[i_b, i_c, ind, ...], wf)
    wf3[:] = wf # Copy to new variable
    wf2 = dot(C[i_b, i_c, ind, ...], wf2)
    wf2 = vecxvec(EL[i_b, i_l, :], wf2, abs(c_a))
    wf3 = vecxvec(EL[i_b, i_l, :], wf3, abs(c_a))
    wf = das*dvecxvec(EL[i_b, i_l, :], wf, abs(c_a))
    return dot(L[i_b, i_l, ...], wf3), dot(L[i_b, i_l, ...], wf), dot(L[i_b, i_l, ...], wf2)

@njit(fastmath=True, nogil=True, cache=True)
def AB_dA_expmH_2D_wf(EA, EB, VRL, VLL, VB, i_a, i_b, c_a, c_b, das, dbs, wf):
    sb = int((sign(c_b) - 1) / 2)
    sa = int((sign(c_a) - 1) / 2)
    i_r = i_b + sb
    i_l = i_a + sa
    ind = -2 * sb - sa

    wf = dot(R[i_r, i_a, ...], wf)
    wf = vecxvec(ER[i_r, i_a, :], wf, abs(c_b))
    wf = dot(C[i_b, i_c, ind, ...], wf)
    wf3[:] = wf # Copy to new variable
    wf3 = vecxvec(EL[i_b, i_l, :], wf3, abs(c_a))
    wf = das*dvecxvec(EL[i_b, i_l, :], wf, abs(c_a))
    return dot(L[i_b, i_l, ...], wf3), dot(L[i_b, i_l, ...], wf)

@njit(fastmath=True, nogil=True, cache=True)
def AB_dB_expmH_2D_wf(EA, EB, VRL, VLL, VB, i_a, i_b, c_a, c_b, das, dbs, wf):
    sb = int((sign(c_b) - 1) / 2)
    sa = int((sign(c_a) - 1) / 2)
    i_r = i_b + sb
    i_l = i_a + sa
    ind = -2 * sb - sa

    wf = dot(R[i_r, i_a, ...], wf)
    wf2 = dbs * dvecxvec(ER[i_r, i_a, :], wf, abs(c_b))
    wf = vecxvec(ER[i_r, i_a, :], wf, abs(c_b))
    wf = dot(C[i_b, i_c, ind, ...], wf)
    wf2 = dot(C[i_b, i_c, ind, ...], wf2)
    wf2 = vecxvec(EL[i_b, i_l, :], wf2, abs(c_a))
    wf = vecxvec(EL[i_b, i_l, :], wf, abs(c_a))
    return dot(L[i_b, i_l, ...], wf), dot(L[i_b, i_l, ...], wf2)