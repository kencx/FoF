#!/usr/bin/env python3

import pickle
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s - %(message)s")
logging.getLogger('matplotlib.font_manager').disabled = True


def plot_clusters(clusters, flagging=True):
    # plt.rc('font', family='serif', size='12')

    clusters = [c for c in clusters if c.flag_poor == False]
    coords = np.array([[c.ra, c.dec] for c in clusters])
    clusters = np.array(clusters)
    z_arr = np.array([c.z for c in clusters])        

    bins = np.arange(0.5,2.53,0.00666)
    digitized = np.digitize(z_arr, bins)

    for i in range(0,len(bins)):
        binned_data = clusters[np.where(digitized==i)]
        binned_data = sorted(binned_data, key=lambda x: x.ra)
        logging.info(f'Number of clusters in bin {i}: {len(binned_data)}')

        if len(binned_data): # plot clusters for checking
            logging.info(f'Plotting bin {i}/{len(bins)}. Clusters with binned redshift: {bins[i]}')

            fig, ax = plt.subplots(figsize=(10,8))
            ax.minorticks_on()
            # ax.tick_params(top=True, right=True, labeltop=True, labelright=True, which='both', direction='inout')
            ax.tick_params(top=True, right=True, which='both', direction='inout')
            plt.title(f'z = {bins[i]:.2f}')

            for i, center in enumerate(binned_data):
                plt.scatter(center.galaxies[:,0], center.galaxies[:,1], s=5)
                plt.scatter(center.ra, center.dec, s=10)
                circ = plt.Circle((center.ra, center.dec), radius=0.14169, color='r', fill=False, ls='--', alpha=0.7)
                ax.add_patch(circ)
                plt.axis([149.4, 150.8, 1.6, 2.8])
                logging.info(f'{i}. Plotting ' + center.__str__())
            plt.tight_layout()
            plt.show()

            if flagging:
                print('Scan the plotted clusters for flagging. Clusters that are poor, merging or on the outer edges should be flagged.')
                flagged = input('Choose the clusters to flag: ') # format: 1,2,4
                
                if flagged != '-':
                    idx_list = list(map(int, flagged.strip().split(',')))
                    flagged_clusters = np.array(binned_data)[idx_list]

                    # if clusters are deemed poor, self.flag_poor = True
                    for c in flagged_clusters:
                        c.flag_poor = True

                    print([c.flag_poor for c in binned_data])

    if flagging:
        with open(fname+'cleaned_candidates_flagged.dat', 'wb') as f:
            pickle.dump(clusters, f)


def lowess(x, y, f=2. / 3., iter=3):
    """
    THIS FUNCTION IS FROM agramfort @ https://gist.github.com/agramfort/850437

    lowess(x, y, f=2./3., iter=3) -> yest
    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.
    The smoothing span is given by f. A larger value for f will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations.
    """
    n = len(x)
    r = int(np.ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)

    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2

    return yest