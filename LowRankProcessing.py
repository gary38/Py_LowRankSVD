#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#@Filename : LowRankProcessing
#@Date : 2015-10-20-17-41
#@Project: ISVD
#@AUTHOR : grl

import sys
import os
import cv2
import numpy as np
import src.rank1_inc_svd as rk1
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import logging


# ===========================================================================
# ISVD
#
# Incremental SVD class.
#
# inputs:
#   data_dir - directory of contiguous video frames numbered accordingly frame1, frame2 ...
#               either .jpg or .bmp
#   gray - if set to 1 will convert frames to grayscale
#   orthog_loss_thrshld - ratio of background frame values that cause a re-orthogonilzing of U.
#   batch_size - number of frames before downdating occurs.
#
# outputs:
#   none
#
# simply writes three images per frame to plt figure.
# ===========================================================================
class ISVD(object):
    def __init__(self,data_dir,gray,orthog_loss_thrshld,batch_size):

        # Then set up a handler for logging:
        handler = logging.FileHandler('log_isvd.log', mode='w', encoding=None, delay=False)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        # Add handler to modules, so we see what's going on inside:
        self.logger = logging.getLogger("ISVD")
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)

        self.data_dir=data_dir
        self.src_img = None
        self.low_rank_img = None
        self.sparse_img = None
        self.gray=gray
        self.orthog_loss_thrshld=orthog_loss_thrshld
        self.batch_size=batch_size

        # ===========================================================================
        # setup the figure and axes to plot results
        # ===========================================================================
        self.fig = plt.figure('Incremental Low Rank SVD Applied to Video Sequence', figsize=(12, 6))
        self.fig.suptitle('figures in order left to right  Source, Background,   Foreground Normalized', fontsize=14, fontweight='bold')
        self.fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        self.ax_src_frame = self.fig.add_subplot(1, 3, 1)
        self.ax_low_rank = self.fig.add_subplot(1, 3, 2)
        self.ax_sparse = self.fig.add_subplot(1, 3, 3)
        self.fig.subplots_adjust(wspace=0.02, hspace=0.02)

        # Get Frame size and list of file names
        n_rows, n_cols, n_dims, self.imgs = self.data_props(data_dir)
        self.n_rows = n_rows; self.n_cols = n_cols; self.n_dims = n_dims

        # ===========================================================================
        # clear all formatting
        # ===========================================================================
        self.set_axes_defaults(self.ax_src_frame,n_rows, n_cols)
        self.set_axes_defaults(self.ax_low_rank,n_rows, n_cols)
        self.set_axes_defaults(self.ax_sparse,n_rows, n_cols)


        plt.pause(.001)
        plt.draw()

    # ===========================================================================
    # clear all formatting
    # ===========================================================================
    def set_axes_defaults(self,ax,n_rows,n_cols):
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # =====================================================
        # set axis limits
        # =====================================================
        ax.set_xlim([0, n_cols])
        ax.set_ylim([0, n_rows])



    # ===============================================================================
    # data frame properties and list of frames
    # ===============================================================================
    def data_props(self, data_dir ):

        # most test data is .jpg files. If none then look for .bmp
        imgs = [fn for fn in os.listdir(data_dir) if fn.endswith('.jpg')]
        if len(imgs) == 0:
            imgs = [fn for fn in os.listdir(data_dir) if fn.endswith('.bmp')]

        # if there is no test data
        if len(imgs) == 0:
            print('no data')
            exit()

        init_file = data_dir + imgs[0];

        sample_frame = cv2.imread(init_file)

        if self.gray == 1:
            sample_frame = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2GRAY);

        n_rows, n_cols, n_dims = sample_frame.shape;

        return n_rows, n_cols, n_dims, imgs

    # ===========================================================================
    # read and re-shape frame into single column
    #
    # ===========================================================================
    def get_frame(self, file_name ):

        # convert pixel values to between 0-1
        if self.gray == 1:
            frame = cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2GRAY).astype(float) / 255.0
        else:
            frame = cv2.imread(self.data_dir+file_name).astype(float) / 255.0

        # openCV returns images as BGR not RGB
        blue, green, red = cv2.split(frame)  # get b,g,r


        n_rows, n_cols, n_dims = frame.shape
        if n_rows!= self.n_rows or n_cols!= self.n_cols or n_dims!= self.n_dims:
            print('Image size must not change from frame to frame')
            exit()
        rgb_img = np.concatenate((red.flatten(order='F'), green.flatten(order='F'), blue.flatten(order='F')))
        return np.reshape(rgb_img, (n_rows * n_cols * n_dims, 1))  # switch it to r g b



    # ===========================================================================
    # incremental SVD
    # ===========================================================================
    def inc_svd(self):

        # ------------------
        # --- Initialization
        # ------------------
        # read first frame
        D = self.get_frame(self.imgs[0])

        # [Q,R] = qr(A), where A is m-by-n, produces an m-by-n upper triangular matrix R and an m-by-m unitary matrix Q so that A = Q*R.
        # [Q,R] = qr(A,0) produces the economy-size decomposition. If m > n, only the first n columns of Q and the first n rows of R
        #  are computed. If m<=n, this is the same as [Q,R] = qr(A).
        #  Q denotes orthogonal and unitary matrices and the letter R denotes right, or upper, triangular matrices.
        # MATLAB -> [Q,R,P]=qr(a,0) Python -> Q,R = np.linalg.qr(a)
        U, Sigma = np.linalg.qr(D)
        V = 1

        low_rank_matrix = U * Sigma;  # low_rank_matrix is SVD of current frame

        # Frame loop - skip first frame since is used for initialization above
        for k,file_name in enumerate(self.imgs[1:]):

            # read current frame
            frame = self.get_frame(file_name)

            ## Perform rank 1 incremental SVD -------------------------------------
            U, Sigma, V = rk1.inc_SVD_update(U, Sigma, V, frame);

            # save previous low rank matrix
            previous_low_rank_matrix= low_rank_matrix

            # For each frame construct low rank from orthogonal matrices and sigma
            low_rank_matrix = U * Sigma * V[-1, :].T  # L is SVD of current frame


            # -- As explained in M. Brand Incremental Singular Value Decomposition of uncertain data with missing data - 2002
            # Use a modified Gram-Schmidt re-orthonormalization procedure whenever loss of orthogonality is detected.
            low_rank_dist = np.sum(np.abs(low_rank_matrix[:] - previous_low_rank_matrix[:])) / (self.n_rows * self.n_cols)

            if ((k > 2)):
                low_rank_ratio = low_rank_dist /prev_low_rank_dist
            else:
                low_rank_ratio = 1
                prev_low_rank_dist=low_rank_dist

            v_row_cnt, _ = V.shape

            # M. Brand when using incSVD loss of orthogonality happens. used modified Gram-Schmidt
            # re-orthonormalization procedure. 2002
            if ((low_rank_ratio > self.orthog_loss_thrshld) and (v_row_cnt >= self.batch_size)):
                U, Sigma = np.linalg.qr(frame)
                V = np.mat(1);
                v_row_cnt = 1;


            # If ortogonality was just calculated via QR then do not downdate.
            # -- Downdate - M. Brand - Fast low-rank downdate the SVD by zeroing a column
            # SVD downdating (or decremental SVD), composing little subspaces into a higher
            # dimensional one or splitting an existing subspace into little ones.
            # Each iteration will add a new row to V corresponding to the new column. Obvisou solution is remove
            # the old row of R and updating with new values. Problem is the R must be orthogonal. Solution is
            # downdate the SVD which removes a column from it. M Brand 2006
            if (v_row_cnt >= self.batch_size):
                U, Sigma, V = rk1.inc_SVD_Downdate(U, Sigma, V);


            _, self.sparse_img = self.update_axes_img(self.normalize(frame - low_rank_matrix), self.sparse_img, self.ax_sparse)
            _, self.src_img = self.update_axes_img(self.normalize(frame), self.src_img, self.ax_src_frame)
            _, self.low_rank_img = self.update_axes_img(self.normalize(low_rank_matrix), self.low_rank_img, self.ax_low_rank)

            # Need to pause to see frames.
            plt.pause(.0001)
            plt.draw()

        return

    # ===========================================================================
    # updateAxesImg
    # ===========================================================================
    def update_axes_img(self, frame, img, ax):

        ax.cla()
        img = ax.imshow(np.flipud(frame), origin='lower')
        return img, frame


    # ===========================================================================
    # normalize
    #
    # This normalizes the input frame and also reshapes for openCV.
    # ===========================================================================
    def normalize(self, x, vmin=None, vmax=None):
        y = np.asarray(x)

        if vmax is None:
            vmax = np.max(y[:])
        if vmin is None:
            vmin = np.min(y[:]);

        if self.gray:
            y[0] = vmin;
            y[-1] = vmax;

        y = (y - vmin).astype(float) / (vmax - vmin)

        r = np.reshape(y[0:self.n_rows * self.n_cols], (self.n_rows, self.n_cols), order='F');
        g = np.reshape(y[self.n_rows * self.n_cols:2 * self.n_rows * self.n_cols], (self.n_rows, self.n_cols), order='F');
        b = np.reshape(y[2 * self.n_rows * self.n_cols:], (self.n_rows, self.n_cols), order='F')

        y = cv2.merge([r, g, b])

        return y

    # =========================================================================


    # ===========================================================================
    # maskFrame
    #
    # Threshold to black and white
    # ===========================================================================
    def mask_frame(self, x, nRows, nCols, thres=0.1):
        y = np.absolute(np.asarray(x))

        r = np.reshape(y[0: nRows * nCols], (nRows, nCols), order='F');
        g = np.reshape(y[nRows * nCols:2 * nRows * nCols], (nRows, nCols), order='F');
        b = np.reshape(y[2 * nRows * nCols:], (nRows, nCols), order='F')

        y = cv2.merge([r, g, b])

        ygray = np.dot(y[..., :3], [0.299, 0.587, 0.144])
        ygray[ygray < thres] = 0.0000
        ygray[ygray > thres - 0.00001] = 1.0

        return ygray

    # =========================================================================


    # ===========================================================================
    # toGray
    #
    # This normalizes the input frame and also reshapes for openCV.
    # ===========================================================================
    def to_gray(self, x, vmin=None, vmax=None, gray=0, scale=1.0):
        y = np.absolute(np.asarray(x))  # x #np.mat(x)

        r = np.reshape(y[0:self.n_rows * self.n_cols], (self.n_rows, self.n_cols), order='F');
        g = np.reshape(y[self.n_rows * self.n_cols:2 * self.n_rows * self.n_cols], (self.n_rows, self.n_cols), order='F');
        b = np.reshape(y[2 * self.n_rows * self.n_cols:], (self.n_rows, self.n_cols), order='F')

        y = cv2.merge([r, g, b])

        ygray = np.dot(y[..., :3], [0.299, 0.587, 0.144])
        gFlat = ygray.flatten()
        gFlat = gFlat * scale

        return gFlat

    # =========================================================================



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data',
        dest='data',
        nargs='+',
        help='data: directory containing data'
    )

    parser.add_argument(
        '--batch',
        dest='batch',
        nargs='+',
        help='batch size: size V grows to before downdating'
    )

    parser.add_argument(
        '--thsld',
        dest='thsld',
        nargs='+',
        help='Threshold for re-orthogonalizing'
    )

    parser.add_argument(
        '--gray',
        dest='gray',
        nargs='+',
        help='gray: 1 or 0'
    )
    parsed = parser.parse_args()

    isvd = ISVD(parsed.data[0],parsed.gray[0],float(parsed.thsld[0]),int(parsed.batch[0]))
    sys.exit(isvd.inc_svd())
