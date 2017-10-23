#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#@Filename : rank1_inc_svd
#@Date : 2015-10-20-21-15
#@Project: ISVD
#@AUTHOR : grl

'''

See Matthew Brand, "Fast low-rank modifications of the thin singular value decomposition".

'''
import numpy as np
import math


# ===============================================================================
# inc_SVD_update
#
# The initial the right space basis to V'=IdentityMatrix size K
# Initially thin QR factorization of A=QR  U=Q. The SVD of R produces the singular
# values of A. (C. Baker 2010)
# A=[a1;a2;a3;a4;..an] Q=[q1;q2;q3;q4;...qn] both mx1 matrix
# Rank 1 modificaiton - Given column vectors a,b  and the known SVD
#
#   X = U*S*V'    X is p x q      update it to be the SVD of
#
#   X + AB' = Up*Sp*Vp'
#       X is pxq  A is pxc  B is qxc A & B have rank c
#       U,V,A,B are tall thin matrices.
#
#                   | S  0 |
#   X + AB' = [U A] |      | [V B]'
#                   | 0  I |
#   OR
#   P is orthogonal basis of column of the column-sapce (I-UU')A, the component of A
#   that is orthogonal to U. Ra=P'(I-UU')A
#   cols(P) = rows(Ra) = rank((I-UU')A <=c and may be zero.
#   [U A] = [U P][IU' A; 0 Ra] Similarly - QRb = (I-VV')B
#
#   X + AB' = [U P]K[V Q]'
#
#       |S 0|   |U'A||V'B|'
#   K = |   | + |   ||   |
#       |0 0|   |Ra ||Rb |
#
#    |U'A||V'B|'   |m||n|'
#    |   ||   | =  | || |
#    |Ra ||Rb |    |p||q|
#
#   K is usually small, highly structured and sparse. Diagonalizing K as
#   U^'KV^=S^  gives rotations U^ and V^ of the expanded subsapces [U P] and [V Q]
#
#   X+AB' = ([U P]U^)S^([V Q]V^)'
# To reduce the complexity extend the decomposition: r is rank
#    Upxr*U^rxr*S^rxr*V^'rxr*V'qxr
#  Only update the smaller interior matrices U^, V^
#
# The subspace rotations involved may not preserve orthogonality due
# to numerical round-off errors. Over time U is no longer a basis matrix.
# So this needs checking and recalcualting. Something done outside this fuctions.
# See Matthew Brand "incremental singular value decomposition of uncertain data missing values" - 2002
# See Matthew Brand, "Fast low-rank modifications of the thin singular value decomposition".
# To start the procedure you need the QR factorization of first k columns of A (input data)
# The right space basis V' = I
# This implementation assumes the update to X is a new column so n=0.
# Which differs from general solutions.
# This method does not recalculate orthogonality. After multiple updates multiplication may
# reduce orthogonality of U' via numerical error. Recalculating of orthogonality is needed
# outside of this function.
# ===============================================================================
def inc_SVD_update(U, S, V, frame):
    # convert to numpy matrix
    U = np.mat(U)
    S = np.mat(S)
    V = np.mat(V)
    frame = np.mat(frame)

    # Gram-Schmidt orthogonalization row version
    # To get orthonormal vectors- subtract from the second vector its projection onto first
    # vector and normalizing both vectors. This extends to multiple vectors.
    m = U.T * frame  # m = Q'a set of weights that project a orthogonally onto subspace U
    p = frame - U * m  # p = a - Qr residual vectors
    pnorm = math.sqrt(np.sum(np.multiply(p, p)))  # pnorm  = ||p||

    # The resulting p and pnorm form the reduced QR factorization of A
    # P is an orthogonal  basis of the column-space
    if (pnorm > 1e-6):
        P = p / pnorm
    else:
        P = np.zeros(p.shape)

    #     | diag(s) m |
    # K = |
    #     | 0   ||p|| |
    K = np.asarray([[S.item(0), m.item(0)], [0, pnorm]])

    # Gu St Gv are of size (k+1) x (k+1) If only first k singular vectors are needed
    # Gu is U rotation and Gv is V rotation.
    Gu, St, Gvt = np.linalg.svd(K)
    Gv = Gvt.T
    St = np.diag(St)

    # U is size of image (LxWxD)x1 --- V grows
    # just get the first singular value i.e. Sigma[0][0]
    Sp = St[0][0]
    Up = U * Gu[0][0] + P * Gu[1][0]
    # element-wise multiply of V[0][0] and adding V[1][0] to the end of the matrix. V is a Nx1 matrix
    Vp = V * Gv[0][0]
    Vp = np.concatenate((Vp, np.mat(Gv[1][0])), axis=0)

    return Up, Sp, Vp


# ===============================================================================
# inc_SVD_Downdate
#
# At each iteration the rank of the SVD is increased by one.
# To keep the solution rank-K the singular vectors corresponding to the smallest
# singular value iof S are dropped at each iteration. However, each iteration will
# still add a new row to V corresponding to the new column. But we cannot just drop
# the old row since we need to maintain orthogonality. Instead DOWNDATE.
# Downdating the ith column only reqiures knowing the ith row of V
#
#
#       |diag(S) 0|   |U'A||V'B|'
#   K = |         | + |   ||   |
#       |0       0|   |Ra ||Rb |
# Setting  y =0 (first column of X) effectively downdates the SVD by zeroing the column selected by b =1
#
#        | S  0 | {     | Sn | | n           |'}
#    K = |      | { I - |    | |             | }
#        | 0  0 | {     | 0  | | sqrt(1-n'n) | }

# ===============================================================================
def inc_SVD_Downdate(U, S, V):
    U = np.mat(U)
    S = np.mat(S)
    V = np.mat(V)

    _, Ncols = U.shape
    N = max(V.shape)

    ## Downdating is on the V matrix here Brand
    # n = V'b   q = b-Vn Rb = ||q|| Q = Rbinv*q
    # in this case b is 1
    n = V[0, :]
    n = n.T
    q = -V * n  # q = b-Vn
    # b is only 1 for the first row all others are zero
    q[0] = q[0] + 1

    rho = math.sqrt((1 - n.T * n))

    if (rho > 1e-8):
        Q = q / rho
    else:
        Q = np.zeros(q.shape)

    #        | S  0 | {     | Sn | | n           |'}
    #    K = |      | { I - |    | |             | }
    #        | 0  0 | {     | 0  | | sqrt(1-n'n) | }
    k00 = S - S * n * n.T
    k01 = -rho * S * n
    K = [[k00.item(0), k01.item(0)], [0, 0]]

    Gu, St, Gvt = np.linalg.svd(K)
    Gv = Gvt.T
    St = np.diag(St)  # turn singular values into diagonal matrix

    Sp = St[0][0]
    Up = U * Gu[0][0]
    Vtmp = V * Gv[0][0] + Q * Gv[1][0]

    Vp = Vtmp[0:N - 2, :]  # 1 to Nth element

    return Up, Sp, Vp
