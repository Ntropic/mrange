# -*- coding: utf-8 -*-
from numpy import array, linspace, empty, dot, exp
from numpy.linalg import eigh

from .UI_Prepare import *
from .Interpolation_Functions_1D import *

##### UI in 1D #########################################################################################################

# Unitary Interpolation
class UI_1D(object):
    def __init__(self, H0, H1, min_max_a=array([0, 1]), a_bins=1):
        self.H0 = H0
        self.H1 = H1
        s = H0.shape[0]
        self.s = s
        self.min_a = min_max_a[0]
        self.max_a = min_max_a[1]
        self.a_bins = a_bins

        alphas = linspace(self.min_a, self.max_a, self.a_bins + 1)
        self.das = a_bins / (min_max_a[1] - min_max_a[0])
        self.i_a = array([0], dtype='i4')
        self.c_a = array([0.25], dtype='f8')
        self.E = empty((a_bins, s), dtype='f8')
        self.L, self.R = [empty((a_bins, s, s), dtype='c16') for i in range(2)]
        U = empty((a_bins + 1, s, s), dtype='c16')
        for a in range(a_bins + 1):
            U[a, ...] = expmH(H0 + alphas[a] * H1)
        for i in range(0, a_bins):
            U_0 = U[i, ...]
            U_1 = U[i + 1, ...]
            E0, V0 = Make_U_Partial(U_1, U_0)
            B0 = dot(Dag(V0), U_0)
            self.E[i, ...] = E0
            self.L[i, ...] = V0
            self.R[i, ...] = B0


    def Indexes_and_Constants(self, alpha):
        a_rel_index = (alpha - self.min_a) * self.das
        self.i_a, self.c_a = Index_Norm(a_rel_index, self.a_bins)

    def Interpolate(self, alpha):  # (Es, Vs, Bs, min_alpha, max_alpha, alpha):
        self.Indexes_and_Constants(alpha)
        return Interpolate_1D_Step(self.s, self.L, self.E, self.R, self.i_a[0], self.c_a[0])
    def Pulse_Interpolate(self, alphas):
        self.Indexes_and_Constants(alphas)
        return Interpolate_1D_Pulse(self.s, self.L, self.E, self.R, self.i_a, self.c_a)
    def dA_Interpolate(self, alpha):  # Differential dU/dalpha
        self.Indexes_and_Constants(alpha)
        return dA_Interpolate_1D_Pulse(self.s, self.L, self.E, self.R, self.i_a[0], self.c_a[0], self.das)
    def A_dA_Interpolate(self, alpha):  # Differential dU/dalpha
        self.Indexes_and_Constants(alpha)
        return A_dA_Interpolate_1D_Pulse(self.s, self.L, self.E, self.R, self.i_a[0], self.c_a[0], self.das)

    def Interpolate_wf(self, alpha, wf):  # (Es, Vs, Bs, min_alpha, max_alpha, alpha):
        self.Indexes_and_Constants(alpha)
        return Interpolate_1D_wf(self.L, self.E, self.R, self.i_a[0], self.c_a[0])
    def Pulse_Interpolate_wf(self, alphas, wf):  # Differential dU/dalpha
        self.Indexes_and_Constants(alphas)
        return Interpolate_1D_Pulse_wf(self.L, self.E, self.R, self.i_a, self.c_a)
    def dA_Interpolate_wf(self, alpha, wf):  # Differential dU/dalpha
        self.Indexes_and_Constants(alpha)
        return dA_Interpolate_1D_wf(self.L, self.E, self.R, self.i_a[0], self.c_a[0], self.das)
    def A_dA_Interpolate_wf(self, alpha, wf):  # Differential dU/dalpha
        self.Indexes_and_Constants(alpha)
        return dA_Interpolate_1D_wf(self.L, self.E, self.R, self.i_a[0], self.c_a[0], self.das)

