# -*- coding: utf-8 -*-
from numpy import array, linspace, empty, dot, exp, ceil
from numpy.linalg import eigh

from .UI_Prepare import *
from .Interpolation_Functions_2D_Sym import *

##### UI in 2D #########################################################################################################

# Unitary Interpolation
class UI_2D_Sym(object):
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
        self.i_a, self.i_b, self.i_c = [array([0], dtype='i4') for i in range(3)]
        self.c_a, self.c_b = [array([0.25], dtype='f8') for i in range(2)]

        U, V = [empty((b_bins + 1, a_bins + 1, s, s), dtype='c16') for i in range(2)]
        V = empty((b_bins + 1, a_bins + 1, s, s), dtype='c16')
        self.L, self.R, = [empty((b_bins+1, a_bins, s, s), dtype='c16') for i in range(2)]
        T = empty((b_bins, a_bins+1, s, s), dtype='c16')
        self.EL, self.ER = [empty((b_bins+1, a_bins, s), dtype='f8') for i in range(2)]
        self.EC = empty((b_bins, a_bins+1, s), dtype='f8')

        ca = int(ceil((a_bins+1)/2))
        self.CL, self.CR = [empty((b_bins+1, ca, 4, s, s), dtype='c16') for i in range(2)]

        # Generate Sample Hamiltonians
        for a in range(a_bins + 1):
            for b in range(b_bins + 1):
                U[b, a, ...], V[b, a, ...] = expmH_Multi(H0 + alphas[a] * H1 + betas[b] * H2,array([ 1.0, 0.5]))
        for a in range(a_bins):
            for b in range(b_bins+1):
                V00 = V[b, a, ...]; V01 = V[b, a + 1, ...]
                if mod(a+b,2) == 0:
                    self.EL[b, a, ...], self.L[b, a, ...] = Make_U_Partial( V01, Dag(V00) )
                    self.ER[b, a, ...], S = Make_U_Partial( Dag(V00), V01 )
                else:
                    self.EL[b, a, ...], self.L[b, a, ...] = Make_U_Partial(V00, Dag(V01) )
                    self.ER[b, a, ...], S = Make_U_Partial(Dag(V01), V00 )
                self.R[b, a, ...] = Dag(S)
        for a in range(a_bins+1):
            for b in range(b_bins):
                if mod(a + b, 2) == 0:
                    self.EC[b, a, ...], T[b, a, ...] = Make_U_Partial_3(Dag(V[b, a, ...]), U[b + 1, a, ...])
                else:
                    self.EC[b, a, ...], T[b, a, ...] = Make_U_Partial_3(Dag(V[b + 1, a, ...]), U[b, a, ...])
        # Define C_L and C_R
        for a2 in range(ca):
            for b in range(b_bins + 1):
                a = int(a2 * 2 + mod(b,2))
                if a <a_bins+1:
                    if (a > 0) and (a < a_bins):
                        i_list = [0, 1]
                    elif a == 0:
                        i_list = [0]
                    else:
                        i_list = [1]
                    if (b > 0) and (b < b_bins):
                        j_list = [0, 1]
                    elif b == 0:
                        j_list = [0]
                    else:
                        j_list = [1]
                    for i in i_list:
                        for j in j_list:
                            k = i+2*j
                            self.CL[b, a2, k, ...] = dot(Dag(self.L[b, a - i, ...]), dot( V[b, a, ...], T[b - j, a, ...]))
                            self.CR[b, a2, k, ...] = dot(Dag(T[b - j, a, ...]), dot( V[b, a, ...], Dag(self.R[b, a - i, ...])))

    def Indexes_and_Constants(self, alpha, beta):
        a_rel_index = (alpha - self.min_a) * self.das
        b_rel_index = (beta - self.min_b) * self.dbs
        self.i_a, self.i_b, self.i_c, self.c_a, self.c_b, = Index_Norm_2D(a_rel_index, b_rel_index, self.a_bins, self.b_bins)

    def Interpolate(self, alpha, beta):
        self.Indexes_and_Constants(alpha, beta)
        return expmH_2D(self.s, self.L, self.EL, self.CL, self.EC, self.CR, self.ER, self.R, self.i_a[0], self.i_b[0], self.i_c[0], self.c_a[0], self.c_b[0] )
    def Pulse_Interpolate(self, alphas, betas):
        self.Indexes_and_Constants(alphas, betas)
        return Interpolate_2D_Pulse(self.s, self.L, self.EL, self.C, self.ER, self.R, self.i_a, self.i_b, self.i_c, self.c_a, self.c_b)

    def dAB_Interpolate(self, alpha, beta):
        self.Indexes_and_Constants(alpha, beta)
        return dAB_expmH_2D(self.s, self.L, self.EL, self.C, self.ER, self.R, self.i_a[0], self.i_b[0], self.i_c[0], self.c_a[0], self.c_b[0], self.das, self.dbs)
    def AB_dAB_Interpolate(self, alpha, beta):
        self.Indexes_and_Constants(alpha, beta)
        return AB_dAB_expmH_2D(self.s, self.L, self.EL, self.C, self.ER, self.R, self.i_a[0], self.i_b[0], self.i_c[0], self.c_a[0], self.c_b[0], self.das, self.dbs)
    def AB_dA_Interpolate(self, alpha, beta):
        self.Indexes_and_Constants(alpha, beta)
        return AB_dA_expmH_2D(self.s, self.L, self.EL, self.C, self.ER, self.R, self.i_a[0], self.i_b[0], self.i_c[0],self.c_a[0], self.c_b[0], self.das, self.dbs)
    def AB_dB_Interpolate(self, alpha, beta):
        self.Indexes_and_Constants(alpha, beta)
        return AB_dB_expmH_2D(self.s, self.L, self.EL, self.C, self.ER, self.R, self.i_a[0], self.i_b[0], self.i_c[0],self.c_a[0], self.c_b[0], self.das, self.dbs)


    def Interpolate_wf(self, alpha, beta, wf):
        self.Indexes_and_Constants(alpha, beta)
        return expmH_2D_wf(self.L, self.EL, self.C, self.ER, self.R, self.i_a[0], self.i_b[0], self.i_c[0],self.c_a[0], self.c_b[0], wf)
    def Pulse_Interpolate_wf(self, alphas, betas, wf):
        self.Indexes_and_Constants(alphas, betas)
        return Interpolate_2D_Pulse_wf(self.L, self.EL, self.C, self.ER, self.R, self.i_a[0], self.i_b[0], self.i_c[0], self.c_a[0], self.c_b[0], wf)
    def dAB_Interpolate_wf(self, alpha, beta, wf):
        self.Indexes_and_Constants(alpha, beta)
        return dAB_expmH_2D_wf(self.EA, self.EB, self.VRL, self.VLL, self.VB, self.i_a[0], self.i_b[0], self.i_c[0], self.c_a[0], self.c_b[0], self.das, self.dbs, wf)

    def AB_dAB_Interpolate_wf(self, alpha, beta, wf):
        self.Indexes_and_Constants(alpha, beta)
        return AB_dAB_expmH_2D_wf(self.EA, self.EB, self.VRL, self.VLL, self.VB, self.i_a[0], self.i_b[0], self.i_c[0],self.c_a[0], self.c_b[0], self.das, self.dbs, wf)
    def AB_dA_Interpolate_wf(self, alpha, beta, wf):
        self.Indexes_and_Constants(alpha, beta)
        return AB_dA_expmH_2D_wf(self.EA, self.EB, self.VRL, self.VLL, self.VB, self.i_a[0], self.i_b[0], self.i_c[0], self.c_a[0], self.c_b[0], self.das, self.dbs, wf)
    def AB_dB_Interpolate_wf(self, alpha, beta, wf):
        self.Indexes_and_Constants(alpha, beta)
        return AB_dB_expmH_2D_wf(self.EA, self.EB, self.VRL, self.VLL, self.VB, self.i_a[0], self.i_b[0], self.i_c[0], self.c_a[0], self.c_b[0], self.das, self.dbs, wf)


