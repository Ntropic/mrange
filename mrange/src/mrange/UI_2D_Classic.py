# -*- coding: utf-8 -*-
from numpy import array, linspace, empty, dot, exp
from numpy.linalg import eigh

from .UI_Prepare import *
from .Interpolation_Functions_2D_Classic import *

##### UI in 2D #########################################################################################################

# Unitary Interpolation
class UI_2D_Classic(object):
    def __init__(self, H0, H1, H2, min_max_a, min_max_b, a_bins, b_bins):
        self.H0 = H0
        self.H1 = H1
        self.H2 = H2
        s = H0.shape[0]
        self.s = s
        self.min_a = min_max_a[0]
        self.max_a = min_max_a[1]
        self.min_b = min_max_b[0]
        self.max_b = min_max_b[1]
        self.a_bins = a_bins
        self.b_bins = b_bins

        alphas = linspace(self.min_a, self.max_a, self.a_bins + 1)
        betas = linspace(self.min_b, self.max_b, self.b_bins + 1)
        self.das = a_bins / (min_max_a[1] - min_max_a[0])
        self.dbs = b_bins / (min_max_b[1] - min_max_b[0])
        self.i_a, self.i_b = [array([0], dtype='i4') for i in range(2)]
        self.c_a, self.c_b = [array([0.25], dtype='f8') for i in range(2)]

        U = empty((b_bins + 1, a_bins + 1, s, s), dtype='c16')
        self.VB, self.VD, VA, VC = [empty((b_bins, a_bins, s, s), dtype='c16') for i in range(4)]
        self.EA, self.EB, self.EC, self.ED = [empty((b_bins, a_bins, s), dtype='f8') for i in range(4)]
        self.VLL, self.VLU, self.VRL, self.VRU = [empty((b_bins, a_bins, s, s), dtype='c16') for i in range(4)]
        # Generate Sample Hamiltonians
        for a in range(a_bins + 1):
            for b in range(b_bins + 1):
                U[b, a, ...] = expmH(H0 + alphas[a] * H1 + betas[b] * H2)
        for a in range(a_bins):
            for b in range(b_bins):
                U00 = U[b, a, ...]
                U01 = U[b, a + 1, ...]
                U10 = U[b + 1, a, ...]
                U11 = U[b + 1, a + 1, ...]
                self.EA[b, a, ...], VA[b, a, ...] = Make_U_Partial(U01, U00)
                self.EB[b, a, ...], self.VB[b, a, ...] = Make_U_Partial(U11, U01)
                self.EC[b, a, ...], VC[b, a, ...] = Make_U_Partial(U10, U00)
                self.ED[b, a, ...], self.VD[b, a, ...] = Make_U_Partial(U11, U10)
                self.VLL[b, a, ...] = dot(Dag(self.VB[b, a, ...]), VA[b, a, ...])
                self.VLU[b, a, ...] = dot(Dag(self.VD[b, a, ...]), VC[b, a, ...])
                self.VRL[b, a, ...] = dot(Dag(VA[b, a, ...]), U00)
                self.VRU[b, a, ...] = dot(Dag(VC[b, a, ...]), U00)

    def Indexes_and_Constants(self, alpha, beta):
        a_rel_index = (alpha - self.min_a) * self.das
        self.i_a, self.c_a = Index_Norm(a_rel_index, self.a_bins)
        b_rel_index = (beta - self.min_b) * self.dbs
        self.i_b, self.c_b = Index_Norm(b_rel_index, self.b_bins)

    def Interpolate(self, alpha, beta):
        self.Indexes_and_Constants(alpha, beta)
        if self.c_a[0] >= self.c_b[0]:
            return expmH_Lower(self.s, self.EA, self.EB, self.VRL, self.VLL, self.VB, self.i_a[0], self.i_b[0], self.c_a[0], self.c_b[0])
        else:
            return expmH_Upper(self.s, self.EC, self.ED, self.VRU, self.VLU, self.VD, self.i_a[0], self.i_b[0], self.c_a[0], self.c_b[0])
    def Pulse_Interpolate(self, alphas, betas):
        self.Indexes_and_Constants(alphas, betas)
        return Interpolate_2D_Pulse(self.s, self.EA, self.EB, self.EC, self.ED, self.VRL, self.VLL, self.VRU, self.VLU, self.VB, self.VD, self.i_a, self.i_b, self.c_a, self.c_b)
    def dAB_Interpolate(self, alpha, beta):
        self.Indexes_and_Constants(alpha, beta)
        if self.c_a[0] >= self.c_b[0]:
            return dAB_expmH_Lower(self.s, self.EA, self.EB, self.VRL, self.VLL, self.VB, self.i_a[0], self.i_b[0], self.c_a[0], self.c_b[0], self.das, self.dbs)
        else:
            return dAB_expmH_Upper(self.s, self.EC, self.ED, self.VRU, self.VLU, self.VD, self.i_a[0], self.i_b[0], self.c_a[0], self.c_b[0], self.das, self.dbs)
    def AB_dAB_Interpolate(self, alpha, beta):
        self.Indexes_and_Constants(alpha, beta)
        if self.c_a[0] >= self.c_b[0]:
            return AB_dAB_expmH_Lower(self.s, self.EA, self.EB, self.VRL, self.VLL, self.VB, self.i_a[0], self.i_b[0], self.c_a[0], self.c_b[0], self.das, self.dbs)
        else:
            return AB_dAB_expmH_Upper(self.s, self.EC, self.ED, self.VRU, self.VLU, self.VD, self.i_a[0], self.i_b[0], self.c_a[0], self.c_b[0], self.das, self.dbs)
    def AB_dA_Interpolate(self, alpha, beta):
        self.Indexes_and_Constants(alpha, beta)
        if self.c_a[0] >= self.c_b[0]:
            return AB_dA_expmH_Lower(self.s, self.EA, self.EB, self.VRL, self.VLL, self.VB, self.i_a[0], self.i_b[0], self.c_a[0], self.c_b[0], self.das)
        else:
            return AB_dA_expmH_Upper(self.s, self.EC, self.ED, self.VRU, self.VLU, self.VD, self.i_a[0], self.i_b[0], self.c_a[0], self.c_b[0], self.das)

    def Interpolate_wf(self, alpha, beta, wf):
        self.Indexes_and_Constants(alpha, beta)
        if self.c_a[0] >= self.c_b[0]:
            return expmH_Lower_wf(self.EA, self.EB, self.VRL, self.VLL, self.VB, self.i_a[0], self.i_b[0], self.c_a[0], self.c_b[0], wf)
        else:
            return expmH_Upper_wf(self.EC, self.ED, self.VRU, self.VLU, self.VD, self.i_a[0], self.i_b[0], self.c_a[0], self.c_b[0], wf)
    def Pulse_Interpolate_wf(self, alphas, betas, wf):
        self.Indexes_and_Constants(alphas, betas)
        return Interpolate_2D_Pulse_wf(self.EA, self.EB, self.EC, self.ED, self.VRL, self.VLL, self.VRU, self.VLU, self.VB, self.VD, self.i_a[0], self.i_b[0], self.c_a[0], self.c_b[0], self.das, self.dbs, wf)
    def dAB_Interpolate_wf(self, alpha, beta, wf):
        self.Indexes_and_Constants(alpha, beta)
        if self.c_a[0] >= self.c_b[0]:
            return dAB_expmH_Lower_wf(self.EA, self.EB, self.VRL, self.VLL, self.VB, self.i_a[0], self.i_b[0], self.c_a[0], self.c_b[0], self.das, self.dbs, wf)
        else:
            return dAB_expmH_Upper_wf(self.EC, self.ED, self.VRU, self.VLU, self.VD, self.i_a[0], self.i_b[0], self.c_a[0], self.c_b[0], self.das, self.dbs, wf)
    def AB_dAB_Interpolate_wf(self, alpha, beta, wf):
        self.Indexes_and_Constants(alpha, beta)
        if self.c_a[0] >= self.c_b[0]:
            return AB_dAB_expmH_Lower_wf(self.EA, self.EB, self.VRL, self.VLL, self.VB, self.i_a[0], self.i_b[0], self.c_a[0], self.c_b[0], self.das, self.dbs, wf)
        else:
            return AB_dAB_expmH_Upper_wf(self.EC, self.ED, self.VRU, self.VLU, self.VD, self.i_a[0], self.i_b[0], self.c_a[0], self.c_b[0], self.das, self.dbs, wf)
    def AB_dA_Interpolate_wf(self, alpha, beta, wf):
        self.Indexes_and_Constants(alpha, beta)
        if self.c_a[0] >= self.c_b[0]:
            return AB_dA_expmH_Lower_wf(self.EA, self.EB, self.VRL, self.VLL, self.VB, self.i_a[0], self.i_b[0], self.c_a[0], self.c_b[0], self.das, wf)
        else:
            return AB_dA_expmH_Upper_wf(self.EC, self.ED, self.VRU, self.VLU, self.VD, self.i_a[0], self.i_b[0], self.c_a[0], self.c_b[0], self.das, wf)


