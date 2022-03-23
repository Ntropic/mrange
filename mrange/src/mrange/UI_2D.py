# -*- coding: utf-8 -*-
from numpy import array, linspace, empty, dot, exp, ceil, any
from numpy.linalg import eigh

from .UI_Prepare import *
from .Interpolation_Functions_2D import *

##### UI in 2D #########################################################################################################

# Unitary Interpolation
class UI_2D(object):
    def __init__(self, H0, H1, H2, min_max_a, min_max_b, a_bins, b_bins, do_boundary_checks = 0):
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
        self.do_boundary_checks = do_boundary_checks

        alphas = linspace(self.min_a, self.max_a, self.a_bins + 1)
        betas = linspace(self.min_b, self.max_b, self.b_bins + 1)
        self.das = a_bins / (min_max_a[1] - min_max_a[0])
        self.dbs = b_bins / (min_max_b[1] - min_max_b[0])
        self.i_a, self.i_b, self.i_c = [array([0], dtype='i4') for i in range(3)]
        self.c_a, self.c_b = [array([0.25], dtype='f8') for i in range(2)]

        U = empty((b_bins + 1, a_bins + 1, s, s), dtype='c16')
        self.L = empty((b_bins+1, a_bins, s, s), dtype='c16')
        self.R = empty((b_bins, a_bins+1, s, s), dtype='c16')
        self.EL = empty((b_bins+1, a_bins, s), dtype='f8')
        self.ER = empty((b_bins, a_bins+1, s), dtype='f8')

        ca = int(ceil((a_bins+1)/2))
        self.C = empty((b_bins+1, ca, 4, s, s), dtype='c16')

        # Generate Sample Hamiltonians
        for a in range(a_bins + 1):
            for b in range(b_bins + 1):
                U[b, a, ...] = expmH(H0 + alphas[a] * H1 + betas[b] * H2)
        for a in range(a_bins):
            for b in range(b_bins+1):
                U00 = U[b, a, ...]; U01 = U[b, a + 1, ...]
                if mod(a+b,2) == 0:
                    self.EL[b, a, ...], self.L[b, a, ...] = Make_U_Partial(U01, Dag(U00))
                else:
                    self.EL[b, a, ...], self.L[b, a, ...] = Make_U_Partial(U00, Dag(U01))
        for a in range(a_bins + 1):
            for b in range(b_bins):
                U00 = U[b, a, ...];
                U10 = U[b + 1, a, ...]
                if mod(a + b, 2) == 0:
                    self.ER[b, a, ...], T = Make_U_Partial(Dag(U00), U10)
                else:
                    self.ER[b, a, ...], T = Make_U_Partial(Dag(U10), U00)
                self.R[b, a, ...] = Dag(T)
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
                            self.C[b, a2, i + j*2, ...] = dot(Dag(self.L[b, a-i, ...]), dot( U[b, a, ...], Dag(self.R[b-j, a, ...]) ) )

    def Indexes_and_Constants(self, alpha, beta):
        a_rel_index = (alpha - self.min_a) * self.das
        b_rel_index = (beta - self.min_b) * self.dbs
        if self.do_boundary_checks == 1:
            if any(alpha >= self.max_a):
                print('Warning: Alpha too large! ' + str(max(alpha)) + ' > ' + str(self.max_a))
                print(alpha)
            if any(alpha < self.min_a):
                print('Warning: Alpha too small! ' + str(min(alpha)) + ' < ' + str(self.min_a))
            if any(beta >= self.max_b):
                print('Warning: Beta too large! ' + str(max(beta)) + ' > ' + str(self.max_b))
            if any(beta < self.min_b):
                print('Warning: Beta too small! ' + str(min(beta)) + ' < ' + str(self.min_b))
        self.i_a, self.i_b, self.i_c, self.c_a, self.c_b, = Index_Norm_2D(a_rel_index, b_rel_index, self.a_bins, self.b_bins)

    def Interpolate(self, alpha, beta):
        self.Indexes_and_Constants(alpha, beta)
        return expmH_2D(self.s, self.L, self.EL, self.C, self.ER, self.R, self.i_a[0], self.i_b[0], self.i_c[0], self.c_a[0], self.c_b[0] )
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


