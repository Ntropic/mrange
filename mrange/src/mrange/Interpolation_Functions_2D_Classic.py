# -*- coding: utf-8 -*-
from numpy import empty, dot, exp
from numba import njit, types, complex128, float64, int32

from .UI_Prepare import *

##### Indexes ##########################################################################################################
@njit(fastmath=True, nogil=True, cache=True)
def Index_Norm(i0, i_max):
    j = empty(len(i0), int32)
    for n, i in enumerate(i0):
        j[n] = int(floor(i))
        if i > i_max - 1:
            j[n] = i_max - 1
        elif i < 0:
            j[n] = 0
    c = i0 - j
    return j, c

#@njit(fastmath=True, nogil=True, cache=True)
def Make_U_Partial(U_1, U_0):
    s = U_1.shape[0]
    D0 = 1j*logmU(dot(U_1,Dag(U_0)))
    E0, V0 = eigh(D0)
    return E0, V0

##### Unitaries ########################################################################################################
@njit(fastmath=True, nogil=True, cache=True)
def expmH_Lower(s, EA, EB, VRL, VLL, VB, i_a, i_b, c_a, c_b):
    A = empty((s, s), complex128)
    A2 = empty((s, s), complex128)
    for i in range(s):
        A[i, :]  = exp(-1j * EA[i_b, i_a, i] * c_a) * VRL[i_b, i_a, i, :]
        A2[i, :] = exp(-1j * EB[i_b, i_a, i] * c_b) * VLL[i_b, i_a, i, :]
    return dot(VB[i_b, i_a, ...], dot(A2, A))
@njit(fastmath=True, nogil=True, cache=True)
def expmH_Upper(s, EC, ED, VRU, VLU, VD, i_a, i_b, c_a, c_b):
    A = empty((s, s), complex128)
    A2 = empty((s, s), complex128)
    for i in range(s):
        A[i, :] = exp(-1j * EC[i_b, i_a, i] * c_b) * VRU[i_b, i_a, i, :]
        A2[i, :] = exp(-1j * ED[i_b, i_a, i] * c_a) * VLU[i_b, i_a, i, :]
    return dot(VD[i_b, i_a, ...], dot(A2, A))
@njit(fastmath=True, nogil=True, cache=True)
def Interpolate_2D_Pulse(s, EA, EB, EC, ED, VRL, VLL, VRU, VLU, VB, VD, i_a, i_b, c_a, c_b):
    if c_a[0] >= c_b[0]:
        U = expmH_Lower(s, EA, EB, VRL, VLL, VB, i_a[0], i_b[0], c_a[0], c_b[0])
    else:
        U = expmH_Upper(s, EC, ED, VRU, VLU, VD, i_a[0], i_b[0], c_a[0], c_b[0])
    for i_ai, i_bi, c_ai, c_bi in zip(i_a[1:], i_b[1:], c_a[1:], c_b[1:]):
        if c_ai >= c_bi:
            U = dot(expmH_Lower(s, EA, EB, VRL, VLL, VB, i_ai, i_bi, c_ai, c_bi), U)
        else:
            U = dot(expmH_Upper(s, EC, ED, VRU, VLU, VD, i_ai, i_bi, c_ai, c_bi), U)
    return U

@njit(fastmath=True, nogil=True, cache=True)
def dAB_expmH_Lower(s, EA, EB, VRL, VLL, VB, i_a, i_b, c_a, c_b, das, dbs):
    A = empty((s, s), complex128)
    A2 = empty((s, s), complex128)
    dA = empty((s, s), complex128)
    dA2 = empty((s, s), complex128)
    for i in range(s):
        A[i, :] = exp(-1j * EA[i_b, i_a, i] * c_a) * VRL[i_b, i_a, i, :]
        dA[i, :] = -1j * das * EA[i_b, i_a, i] * exp(-1j * EA[i_b, i_a, i] * c_a) * VRL[i_b, i_a, i, :]
        A2[i, :] = exp(-1j * EB[i_b, i_a, i] * c_b) * VLL[i_b, i_a, i, :]
        dA2[i, :] = -1j * dbs * EB[i_b, i_a, i] * exp(
            -1j * EB[i_b, i_a, i] * c_b) * VLL[i_b, i_a, i, :]
    return dot(VB[i_b, i_a, ...], dot(A2, dA)), dot(VB[i_b, i_a, ...], dot(dA2, A))
@njit(fastmath=True, nogil=True, cache=True)
def dAB_expmH_Upper(s, EC, ED, VRU, VLU, VD, i_a, i_b, c_a, c_b, das, dbs):
    A = empty((s, s), complex128)
    A2 = empty((s, s), complex128)
    dA = empty((s, s), complex128)
    dA2 = empty((s, s), complex128)
    for i in range(s):
        A[i, :] = exp(-1j * EC[i_b, i_a, i] * c_b) * VRU[i_b, i_a, i, :]
        dA[i, :] = -1j * dbs * EC[i_b, i_a, i] * exp(-1j * EC[i_b, i_a, i] * c_b) * VRU[i_b, i_a, i, :]
        A2[i, :] = exp(-1j * ED[i_b, i_a, i] * c_a) * VLU[i_b, i_a, i, :]
        dA2[i, :] = -1j * das * ED[i_b, i_a, i] * exp(-1j * ED[i_b, i_a, i] * c_a) * VLU[i_b, i_a, i, :]
    return dot(VD[i_b, i_a, ...], dot(dA2, A)), dot(VD[i_b, i_a, ...], dot(A2, dA))

@njit(fastmath=True, nogil=True, cache=True)
def AB_dAB_expmH_Lower(s, EA, EB, VRL, VLL, VB, i_a, i_b, c_a, c_b, das, dbs):
    A = empty((s, s), complex128)
    A2 = empty((s, s), complex128)
    dA = empty((s, s), complex128)
    dA2 = empty((s, s), complex128)
    for i in range(s):
        A[i, :] = exp(-1j * EA[i_b, i_a, i] * c_a) * VRL[i_b, i_a, i, :]
        dA[i, :] = -1j * das * EA[i_b, i_a, i] * exp(-1j * EA[i_b, i_a, i] * c_a) * VRL[i_b, i_a, i, :]
        A2[i, :] = exp(-1j * EB[i_b, i_a, i] * c_b) * VLL[i_b, i_a, i, :]
        dA2[i, :] = -1j * dbs * EB[i_b, i_a, i] * exp(-1j * EB[i_b, i_a, i] * c_b) * VLL[i_b, i_a, i, :]
    return dot(VB[i_b, i_a, ...], dot(A2, A)), dot(VB[i_b, i_a, ...], dot(A2, dA)), dot(VB[i_b, i_a, ...], dot(dA2, A))
@njit(fastmath=True, nogil=True, cache=True)
def AB_dAB_expmH_Upper(s, EC, ED, VRU, VLU, VD, i_a, i_b, c_a, c_b, das, dbs):
    A = empty((s, s), complex128)
    A2 = empty((s, s), complex128)
    dA = empty((s, s), complex128)
    dA2 = empty((s, s), complex128)
    for i in range(s):
        A[i, :] = exp(-1j * EC[i_b, i_a, i] * c_b) * VRU[i_b, i_a, i, :]
        dA[i, :] = -1j * dbs * EC[i_b, i_a, i] * exp(-1j * EC[i_b, i_a, i] * c_b) * VRU[i_b, i_a, i, :]
        A2[i, :] = exp(-1j * ED[i_b, i_a, i] * c_a) * VLU[i_b, i_a, i, :]
        dA2[i, :] = -1j * das * ED[i_b, i_a, i] * exp(-1j * ED[i_b, i_a, i] * c_a) * VLU[i_b, i_a, i, :]
    return dot(VD[i_b, i_a, ...], dot(A2, A)), dot(VD[i_b, i_a, ...], dot(dA2, A)), dot(VD[i_b, i_a, ...], dot(A2, dA))

@njit(fastmath=True, nogil=True, cache=True)
def AB_dA_expmH_Lower(s, EA, EB, VRL, VLL, VB, i_a, i_b, c_a, c_b, das):
    A = empty((s, s), complex128)
    A2 = empty((s, s), complex128)
    dA = empty((s, s), complex128)
    for i in range(s):
        A[i, :] = exp(-1j * EA[i_b, i_a, i] * c_a) * VRL[i_b, i_a, i, :]
        dA[i, :] = -1j * das * EA[i_b, i_a, i] * exp(-1j * EA[i_b, i_a, i] * c_a) * VRL[i_b, i_a, i, :]
        A2[i, :] = exp(-1j * EB[i_b, i_a, i] * c_b) * VLL[i_b, i_a, i, :]
    return dot(VB[i_b, i_a, ...], dot(A2, A)), dot(VB[i_b, i_a, ...], dot(A2, dA))
@njit(fastmath=True, nogil=True, cache=True)
def AB_dA_expmH_Upper(s, EC, ED, VRU, VLU, VD, i_a, i_b, c_a, c_b, das):
    A = empty((s, s), complex128)
    A2 = empty((s, s), complex128)
    dA2 = empty((s, s), complex128)
    for i in range(s):
        A[i, :] = exp(-1j * EC[i_b, i_a, i] * c_b) * VRU[i_b, i_a, i, :]
        A2[i, :] = exp(-1j * ED[i_b, i_a, i] * c_a) * VLU[i_b, i_a, i, :]
        dA2[i, :] = -1j * das * ED[i_b, i_a, i] * exp(-1j * ED[i_b, i_a, i] * c_a) * VLU[i_b, i_a, i, :]
    return dot(VD[i_b, i_a, ...], dot(A2, A)), dot(VD[i_b, i_a, ...], dot(dA2, A))


## Wavefunctions #######################################################################################################
@njit(fastmath=True, nogil=True, cache=True)
def expmH_Lower_wf(EA, EB, VRL, VLL, VB, i_a, i_b, c_a, c_b, wf):
    wf = dot(VRL[i_b, i_a, ...], wf)
    wf = vecxvec(EA[i_b, i_a, :], wf, c_a)
    wf = dot(VLL[i_b, i_a, ...], wf)
    wf = vecxvec(EB[i_b, i_a, :], wf, c_b)
    return dot(VB[i_b, i_a, ...], wf)
@njit(fastmath=True, nogil=True, cache=True)
def expmH_Upper_wf(EC, ED, VRU, VLU, VD, i_a, i_b, c_a, c_b, wf):
    wf = dot(VRU[i_b, i_a, ...], wf)
    wf = vecxvec(EC[i_b, i_a, :], wf, c_b)
    wf = dot(VLU[i_b, i_a, ...], wf)
    wf = vecxvec(ED[i_b, i_a, :], wf, c_a)
    return dot(VD[i_b, i_a, ...], wf)
@njit(fastmath=True, nogil=True, cache=True)
def Interpolate_2D_Pulse_wf(s, EA, EB, EC, ED, VRL, VLL, VRU, VLU, VB, VD, i_a, i_b, c_a, c_b, wf):
    if c_a[0] >= c_b[0]:
        U = expmH_Lower_wf(s, EA, EB, VRL, VLL, VB, i_a[0], i_b[0], c_a[0], c_b[0], wf)
    else:
        U = expmH_Upper_wf(s, EC, ED, VRU, VLU, VD, i_a[0], i_b[0], c_a[0], c_b[0], wf)
    for i_ai, i_bi, c_ai, c_bi in zip(i_a[1:], i_b[1:], c_a[1:], c_b[1:]):
        if c_ai >= c_bi:
            U = dot(expmH_Lower_wf(s, EA, EB, VRL, VLL, VB, i_ai, i_bi, c_ai, c_bi), U, wf)
        else:
            U = dot(expmH_Upper_wf(s, EC, ED, VRU, VLU, VD, i_ai, i_bi, c_ai, c_bi), U, wf)
    return U

@njit(fastmath=True, nogil=True, cache=True)
def dAB_expmH_Lower_wf(EA, EB, VRL, VLL, VB, i_a, i_b, c_a, c_b, das, dbs, wf):
    wf = dot(VRL[i_b, i_a, ...], wf)
    wf2 = das*dvecxvec(EA[i_b, i_a, :], wf, c_a)
    wf = vecxvec(EA[i_b, i_a, :], wf, c_a)
    wf = dot(VLL[i_b, i_a, ...], wf)
    wf = dbs*dvecxvec(EB[i_b, i_a, :], wf, c_b)
    wf2 = dot(VLL[i_b, i_a, ...], wf2)
    wf2 = vecxvec(EB[i_b, i_a, :], wf2, c_b)
    return dot(VB[i_b, i_a, ...], wf2), dot(VB[i_b, i_a, ...], wf)
@njit(fastmath=True, nogil=True, cache=True)
def dAB_expmH_Upper_wf(EC, ED, VRU, VLU, VD, i_a, i_b, c_a, c_b, das, dbs, wf):
    wf = dot(VRU[i_b, i_a, ...], wf)
    wf2 = dbs*dvecxvec(EC[i_b, i_a, :], wf, c_b)
    wf = vecxvec(EC[i_b, i_a, :], wf, c_b)
    wf = dot(VLU[i_b, i_a, ...], wf)
    wf = das*dvecxvec(ED[i_b, i_a, :], wf, c_a)
    wf2 = dot(VLU[i_b, i_a, ...], wf2)
    wf2 = vecxvec(ED[i_b, i_a, :], wf2, c_a)
    return dot(VD[i_b, i_a, ...], wf), dot(VD[i_b, i_a, ...], wf2)

@njit(fastmath=True, nogil=True, cache=True)
def AB_dAB_expmH_Lower_wf(EA, EB, VRL, VLL, VB, i_a, i_b, c_a, c_b, das, dbs, wf):
    wf = dot(VRL[i_b, i_a, ...], wf)
    wf2 = das*dvecxvec(EA[i_b, i_a, :], wf, c_a)
    wf = vecxvec(EA[i_b, i_a, :], wf, c_a)
    wf = dot(VLL[i_b, i_a, ...], wf)
    wf3[:] = wf
    wf = dbs*dvecxvec(EB[i_b, i_a, :], wf, c_b)
    wf2 = dot(VLL[i_b, i_a, ...], wf2)
    wf2 = vecxvec(EB[i_b, i_a, :], wf2, c_b)
    wf3 = vecxvec(EB[i_b, i_a, :], wf3, c_b)
    return dot(VB[i_b, i_a, ...], wf3), dot(VB[i_b, i_a, ...], wf2), dot(VB[i_b, i_a, ...], wf)
@njit(fastmath=True, nogil=True, cache=True)
def AB_dAB_expmH_Upper_wf(EC, ED, VRU, VLU, VD, i_a, i_b, c_a, c_b, das, dbs, wf):
    wf = dot(VRU[i_b, i_a, ...], wf)
    wf2 = dbs*dvecxvec(EC[i_b, i_a, :], wf, c_b)
    wf = vecxvec(EC[i_b, i_a, :], wf, c_b)
    wf = dot(VLU[i_b, i_a, ...], wf)
    wf3[:] = wf
    wf = das*dvecxvec(ED[i_b, i_a, :], wf, c_a)
    wf2 = dot(VLU[i_b, i_a, ...], wf2)
    wf2 = vecxvec(ED[i_b, i_a, :], wf2, c_a)
    wf3 = vecxvec(ED[i_b, i_a, :], wf3, c_a)
    return dot(VD[i_b, i_a, ...], wf3), dot(VD[i_b, i_a, ...], wf), dot(VD[i_b, i_a, ...], wf2)

@njit(fastmath=True, nogil=True, cache=True)
def AB_dA_expmH_Lower_wf(EA, EB, VRL, VLL, VB, i_a, i_b, c_a, c_b, das, wf):
    wf = dot(VRL[i_b, i_a, ...], wf)
    wf2 = das*dvecxvec(EA[i_b, i_a, :], wf, c_a)
    wf = vecxvec(EA[i_b, i_a, :], wf, c_a)
    wf = dot(VLL[i_b, i_a, ...], wf)
    wf2 = dot(VLL[i_b, i_a, ...], wf2)
    wf2 = vecxvec(EB[i_b, i_a, :], wf2, c_b)
    wf = vecxvec(EB[i_b, i_a, :], wf, c_b)
    return dot(VB[i_b, i_a, ...], wf), dot(VB[i_b, i_a, ...], wf2)
@njit(fastmath=True, nogil=True, cache=True)
def AB_dA_expmH_Upper_wf( EC, ED, VRU, VLU, VD, i_a, i_b, c_a, c_b, das, wf):
    wf = dot(VRU[i_b, i_a, ...], wf)
    wf = vecxvec(EC[i_b, i_a, :], wf, c_b)
    wf = dot(VLU[i_b, i_a, ...], wf)
    wf2 = das*dvecxvec(ED[i_b, i_a, :], wf, c_a)
    wf = vecxvec(ED[i_b, i_a, :], wf, c_a)
    return dot(VD[i_b, i_a, ...], wf), dot(VD[i_b, i_a, ...], wf2)