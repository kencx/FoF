import pickle
import numpy as np
import matplotlib.pyplot as plt

from kneed import KneeLocator
import astropy.units as u
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics

from analysis.methods import mean_separation, linear_to_angular_dist, redshift_to_velocity


def db_scan(data, richness):

    # split data into redshift bins of v <= 2000km/s (z = 0.006671; use upper bound of 0.01)
    z_arr = data[:,2]
    bins = np.arange(min(z_arr), max(z_arr), 0.00667)
    digitized = np.digitize(z_arr, bins) 

    clusters = []
    noise = []

    # perform DBSCAN on each redshift bin
    for i in range(1, len(bins)+1):
        X_bin = data[np.where(digitized==i)]
        X_bin_coords = X_bin[:,:2]
        max_radius = linear_to_angular_dist(1.5*u.Mpc/u.littleh, np.mean(X_bin[:,2])).value

        # determine eps of bin - elbow/knee method based on DBSCAN paper [Ester, M. et al., (1996, August)]
        neighbors = NearestNeighbors(n_neighbors=richness)
        neighbors_fit = neighbors.fit(X_bin_coords)

        dist, idx = neighbors_fit.kneighbors(X_bin_coords) # dist to NN for each data point
        dist = np.sort(dist, axis=0) # sort by dist to NN for each point
        # mean_dist = np.mean(dist, axis=1)
        mean_dist = dist[:,2]
        kneedle = KneeLocator(np.arange(0,len(dist)), mean_dist, curve='convex') # locate convex elbow of curve
        elbow = kneedle.elbow
        eps = mean_dist[elbow]

        # plt.plot(mean_dist)
        # plt.axvline(kneedle.elbow)
        # plt.show()

        db = DBSCAN(eps=eps, min_samples=richness, algorithm='kd_tree').fit(X_bin_coords) # eps: local mean sep, min_samples: N >= 12

        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # op = OPTICS(min_samples=richness, max_eps=max_radius)
        # op.fit(X_bin_coords)

        # reachability = op.reachability_[op.ordering_]
        # labels = op.labels_[op.ordering_]

        clusters.append([l for l in labels if l != -1])
        noise.append([l for l in labels if l == -1])

        # uniq_clusters = len(set(labels))
        # plt.axis([min(data[:,0]), max(data[:,0]), min(data[:,1]), max(data[:,1])])
        # for i in range(1, uniq_clusters):
        #     Xk = X_bin_coords[labels == i]
        #     plt.scatter(Xk[:,0], Xk[:,1], alpha=0.3, marker='.')
        # plt.show()

    # flatten binned arrays
    unbinned_clusters = np.concatenate(clusters, axis=0)
    noise = np.concatenate(noise, axis=0)

    n_clusters = len(unbinned_clusters)
    n_noise = len(noise)

    print(f'Number of clusters: {n_clusters}')
    print(f'Number of noise points: {n_noise}')

    return clusters



if __name__ == "__main__":

    ##############
    # PARAMETERS #
    ##############
    min_redshift = 0.5
    max_redshift = 2.5

    # max_velocity = 2000
    # linking_length_factor = 0.4
    # virial_radius = 1.5*u.Mpc/u.littleh

    richness = 25
    D = 2

    fname = f'analysis\\derived_datasets\\R{richness}_D{D}_vel\\'
    plot = False

    main_arr = np.loadtxt('analysis\\derived_datasets\\COSMOS_galaxy_data.csv', delimiter=',') 
    luminous_arr = np.loadtxt('analysis\\derived_datasets\\luminous_galaxy_velocity.csv', delimiter=',')

    main_arr = main_arr[(main_arr[:,2] >= min_redshift) & (main_arr[:,2] <= max_redshift + 0.02)]
    luminous_arr = luminous_arr[(luminous_arr[:,2] >= min_redshift) & (luminous_arr[:,2] <= max_redshift)]
    print(f'Number of galaxies: {len(main_arr)}, Number of candidates: {len(luminous_arr)}')

    db_scan(main_arr, richness)
    # print(f'{len(candidates)} candidate clusters found.')

    # # pickle candidates list
    # with open(fname+'DBSCAN_candidates.dat', 'wb') as f:
    #     pickle.dump(candidates, f)