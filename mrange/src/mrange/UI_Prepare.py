# -*- coding: utf-8 -*-
from numpy import empty, dot, exp, log, floor, ceil, sum, transpose, conj, diag, array
from numpy.linalg import eigh
from numba import njit, types, complex128, float64, int32
from scipy.linalg import schur

##### Preparation Scripts ##############################################################################################

@njit(fastmath=True, nogil=True, cache=True)
def Dag(A):
    return transpose(conj(A))

@njit(fastmath=True, nogil=True, cache=True)
def Commutator(A,B):
    return dot(A,B)-dot(B,A)

@njit(fastmath=True, nogil=True, cache=True)
def vm_exp_mul(E, V, dt = 1.0): # expm(diag(vector))*matrix  multiplication via exp(vector)*matrix
    s = E.size
    A = empty((s, s), complex128)
    for i in range(s):
        A[i,:] = exp(-1j * E[i] * dt)*Dag(V[:,i])
    return A
@njit(fastmath=True, nogil=True, cache=True)
def expmH_from_Eig(E, V, dt = 1.0):
    U = dot(V, vm_exp_mul(E, V, dt))
    return U
@njit(fastmath=True, nogil=True, cache=True)
def expmH(H, dt = 1.0):
    E, V = eigh(H)
    return expmH_from_Eig( E, V, dt)
@njit(fastmath=True, nogil=True, cache=True)
def expmH_Multi(H, dt = array([1.0,0.5])):
    E, V = eigh(H)
    return expmH_from_Eig( E, V, dt[0]), expmH_from_Eig( E, V, dt[1])    # (expmH_from_Eig( E, V, dti) for dti in dt) #-> does not work with numba nopython mode

@njit(fastmath=True, nogil=True, cache=True)
def vm_log_mul(E, V):
    s = E.size
    A = empty((s, s), complex128)
    for i in range(s):
        A[i,:] = log(E[i])*Dag(V[:,i])
    return A

#@njit(fastmath=True, nogil=True, cache=True)
#def unitary_eig(U): # alternative to eig returning unitary matrices V #Doesn't necessarily work for equal real parts of eigenvalues
#    _, V= eigh(U+Dag(U))
#    s = V.shape[0]
#    E = empty(s, complex128)
#    W = Dag(V) @ U
#    for i in range(s):
#        E[i] = dot(W[i,:], V[:,i])
#    #E = einsum('ij,jk,ki -> i', Dag(V), U, V) # Only jax not numba compatible
#    return E, V
def unitary_eig(A): # alternative to np.eig returning unitary matrices V
    Emat, V = schur(A, output='complex')
    return diag(Emat), V
#@njit(fastmath=True, nogil=True, cache=True)
def logmU(U):
    E, V = unitary_eig(U)
    return dot(V, vm_log_mul(E, V))

##### Wavefunction Exponential #########################################################################################

@njit(fastmath=True, nogil=True, cache=True)
def vecxvec(E, wf, alpha):  # Many times faster than multiply
    s = E.size
    for i in range(s):
        wf[i] = exp(-1j * E[i] * alpha) * wf[i]
    return wf
@njit(fastmath=True, nogil=True, cache=True)
def dvecxvec(E, wf, alpha):  # Many times faster than multiply
    s = E.size
    for i in range(s):
        wf[i] = -1j * E[i] * exp(-1j * E[i] * alpha) * wf[i]
    return wf

##### Binning Scripts ##################################################################################################

@njit(fastmath=True, nogil=True, cache=True)
def How_many_Bins_1D(H0, H1_max, max_I):
    d = H0.shape[0]
    ComCom = Commutator(H1_max, Commutator(H1_max, H0))
    E, V = eigh(ComCom)
    factor = 1/(d+1)/max_I/4608*sum(E**2)
    n_condition = ceil(factor**(1/4))
    return int(n_condition)
@njit(fastmath=True, nogil=True, cache=True)
def How_large_Error_1D(H0, H1_max, n_bins):   #Estimate the size of the maximum error (typically off by a factor of 2)
    d = H0.shape[0]
    ComCom = Commutator(H1_max,Commutator(H1_max,H0))
    E, V = eigh(ComCom)
    error = 1/(d+1)/n_bins**4/4608*sum(E**2)
    return error

@njit(fastmath=True, nogil=True, cache=True)
def How_many_Bins_2D(H0, H1_max, H2_max, max_I): # Improve this function!
    d = H0.shape[0]
    Com = Commutator(H1_max,H2_max)/8
    Com2 = Commutator(H1_max,Commutator(H0, H1_max))/48  # Other terms are assumed smaller but should be added here at a later point
    E, V = eigh(Com)
    E2, V2 = eigh(Com2)
    if max(E**2) > max(E2**2):
        factor = 1/(d+1)/max_I/2*sum(E**2)
        a_bins = ceil(factor ** (1 / 4))
        b_bins = a_bins
    else:
        factor = 1/(d+1)/max_I/2*sum(E**2)
        a_bins = ceil(factor ** (1 / 4))
        b_bins = 1
    return int(a_bins), int(b_bins)
@njit(fastmath=True, nogil=True, cache=True)
def How_large_Error_2D(H0, H1_max, H2_max, a_bins, b_bins):
    d = H0.shape[0]
    Com = Commutator(H1_max,H2_max)/8/(a_bins*b_bins)+Commutator(H1_max,Commutator(H0, H1_max))/48/(a_bins**2) # Other terms are assumed smaller but should be added here at a later point
    E, V = eigh(Com)
    error = 1/(d+1)/2*sum(E**2)
    return error

