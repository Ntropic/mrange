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
def Interpolate_1D_Step(s, L, E, R, i_a, c_a):
    A = empty((s, s), complex128)
    for i in range(s):
        A[i, :] = exp(-1j * E[i_a, i] * c_a) * R[i_a, i, ...]
    return dot(L[i_a, ...], A)
@njit(fastmath=True, nogil=True, cache=True)
def Interpolate_1D_Pulse(s, L, E, R, i_a, c_a):
    A = empty((s, s), complex128)
    B = empty((s, s), complex128)
    i_as = i_a[0]
    for i in range(s):
        A[i, :] = exp(-1j * E[i_as, i] * c_a[0]) * R[i_as, i, ...]
    B = dot(L[i_as, ...], A)
    for i_ai, c_ai in zip(i_a[1:], c_a[1:]):
        B = dot(R[i_ai, ...], B)
        for i in range(s):
            A[i, :] = exp(-1j * E[i_ai, i] * c_ai) * B[i, ...]
        B = dot(L[i_ai, ...], A)
    return B
@njit(fastmath=True, nogil=True, cache=True)
def dA_Interpolate_1D_Step(s, L, E, R, i_a, c_a, das):
    dA = empty((s, s), complex128)
    for i in range(s):
        dC = -1j * das * E[i_a, i] * exp(-1j * E[i_a, i] * c_a)
        dA[i, :] = dC * A[i, :]
    return dot(L[i_a, ...], dA)
@njit(fastmath=True, nogil=True, cache=True)
def A_dA_Interpolate_1D_Step(s, L, E, R, i_a, c_a, das):
    A = empty((s, s), complex128)
    dA = empty((s, s), complex128)
    for i in range(s):
        C = exp(-1j * E[i_a, i] * c_a[0])
        dC = -1j * das * E[i_a, i]
        A[i, :] = C * R[i_a, i, ...]
        dA[i, :] = dC * A[i, :]
    return dot(L[i_a, ...], A), dot(L[i_a, ...], dA)

## Wavefunctions #######################################################################################################
@njit(fastmath=True, nogil=True, cache=True)
def Interpolate_1D_wf(L, E, R, i_a, c_a):
    wf = dot(L[i_a, ...], wf)
    wf = vecxvec(E[i_a, ...], wf, c_a)
    return dot(R[i_a, ...], wf)
@njit(fastmath=True, nogil=True, cache=True)
def Interpolate_1D_Pulse_wf(L, E, R, i_a, c_a):
    i_s = i_a[0]
    wf = dot(L[i_as, ...], wf)
    wf = vecxvec(E[i_as, ...], wf, c_a[0])
    wf = dot(R[i_as, ...], wf)
    for i_ai, c_ai in zip(i_a[1:], c_a[1:]):
        wf = dot(L[i_ai, ...], wf)
        wf = vecxvec(E[i_ai, ...], wf, c_ai)
        wf = dot(R[i_ai, ...], wf)
    return wf
@njit(fastmath=True, nogil=True, cache=True)
def dA_Interpolate_1D_wf(L, E, R, i_a, c_a, das):
    wf = dot(L[i_a, ...], wf)
    wf = das*dvecxvec(E[i_a, ...], wf, c_a)
    return dot(R[i_a, ...], wf)
@njit(fastmath=True, nogil=True, cache=True)
def A_dA_Interpolate_1D_wf(L, E, R, i_a, c_a, das):
    wf = dot(L[i_a, ...], wf)
    wf = das * dvecxvec(E[i_a, ...], wf, c_a)
    wf2 = das*dvecxvec(E[i_a, ...], wf, c_a)
    return dot(R[i_a, ...], wf), dot(R[i_a, ...], wf2)