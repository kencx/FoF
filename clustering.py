import pickle
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kneed import KneeLocator
import astropy.units as u
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics

from analysis.methods import mean_separation, linear_to_angular_dist, redshift_to_velocity
from cluster import Cluster
from params import *

def db_scan(data, richness):

    # split data into redshift bins of v <= 2000km/s (z = 0.006671; use upper bound of 0.01)
    z_arr = data[:,2]
    bins = np.arange(min(z_arr), max(z_arr), 0.01)
    digitized = np.digitize(z_arr, bins) 

    clusters = []
    cores = []
    all_labels = []

    # perform DBSCAN on each redshift bin
    for i in range(1, len(bins)+1):
        X_bin = data[np.where(digitized==i)]
        X_bin_coords = X_bin[:,:2]
        # max_radius = linear_to_angular_dist(1.5*u.Mpc/u.littleh, np.mean(X_bin[:,2])).value

        if len(X_bin_coords) >= richness:
        # determine eps of bin - elbow/knee method based on DBSCAN paper [Ester, M. et al., (1996, August)]
            neighbors = NearestNeighbors(n_neighbors=richness)
            neighbors_fit = neighbors.fit(X_bin_coords)

            dist, idx = neighbors_fit.kneighbors(X_bin_coords) # dist to NN for each data point
            dist = np.sort(dist, axis=0) # sort by dist to NN
            mean_dist = np.mean(dist, axis=1)
            # mean_dist = dist[:,-1]
            kneedle = KneeLocator(np.arange(0,len(dist)), mean_dist, curve='convex') # locate convex elbow of curve
            elbow = kneedle.elbow
            if elbow:
                eps = mean_dist[elbow]

        # plt.plot(mean_dist)
        # plt.axvline(kneedle.elbow)
        # plt.show()

            db = DBSCAN(eps=eps, min_samples=richness, algorithm='kd_tree').fit(X_bin_coords) # eps: local mean sep, min_samples: richness

            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_

        # op = OPTICS(min_samples=richness, max_eps=max_radius)
        # op.fit(X_bin_coords)

        # reachability = op.reachability_[op.ordering_]
        # labels = op.labels_[op.ordering_]

            bin_cluster = []
            bin_core = []
            bin_labels = []

            cl = [l for l in labels if l != -1] # cluster labels for each bin
            if len(cl):
                core = X_bin[core_samples_mask] # core samples for each bin
                uniq_cl = set(cl) # unique cluster labels
                for k in uniq_cl:
                    uniq_cluster = X_bin[k == labels] # unique cluster
                    bin_cluster.append(uniq_cluster)
                    bin_core.append(core)
        
            clusters.append(bin_cluster)
            cores.append(bin_core)
            all_labels.append(labels)
        
        else:
            all_labels.append([-1]*len(X_bin_coords))

    unbinned_clusters = [i for sublist in clusters for i in sublist] # flatten binned arrays
    print(f'Number of clusters: {len(unbinned_clusters)}')
    # all_labels = [i for sublist in all_labels for i in sublist] # flatten binned arrays
    
    clusters = [c for c in clusters if len(c) > 0]
    cores = [c for c in cores if len(c) > 0]

    final_labels = []
    for i in range(len(all_labels)):
        size = len(all_labels[i])
        c = [i]*size
        final_labels.append(c)
    final_labels = [i for sublist in final_labels for i in sublist] # flatten cluster arrays



    return clusters, cores, final_labels


if __name__ == "__main__":

    conn = sqlite3.connect('processing\\datasets\\galaxy_clusters.db')

    cosmic_web_df = pd.read_sql_query('''
    SELECT ra, dec, redshift, cluster_id
    FROM cosmic_web_bcg
    WHERE Ngal>=? AND redshift>=0.5
    ORDER BY redshift
    ''', conn, params=(richness,))
    cosmic_web_arr = cosmic_web_df.values

    cosmic_web_members = pd.read_sql_query('''
    SELECT ra, dec, redshift, cluster_id
    FROM cosmic_web_members
    WHERE redshift >= 0.5
    ORDER BY redshift
    ''', conn)
    cosmic_web_members_arr = cosmic_web_members.values
    conn.close()
 

    # extract members of interest
    members_arr = []
    cluster_id = cosmic_web_arr[:,-1]
    for i in cluster_id:
        members = cosmic_web_members_arr[cosmic_web_members_arr[:,-1] == i]
        members_arr.append(members)
    members_arr = np.concatenate(members_arr, axis=0)


    # create true labels for existing catalog - cosmic web and members
    total_cosmic = np.concatenate((cosmic_web_arr, members_arr), axis=0)
    total_cosmic = total_cosmic[np.argsort(total_cosmic[:,-1])] # sort by cluster_id
    cluster_id = (list(set(total_cosmic[:,-1])))
    labels = np.arange(1, len(cluster_id)+1)

    # replace cluster_id by labels from 1 to n
    for idx in labels:
        c_id = cluster_id[idx-1]
        total_cosmic[total_cosmic[:,-1] == c_id, -1] = idx

    labels_true = total_cosmic[:,-1]

    # conduct dbscan on catalog data
    clusters, cores, pred_labels = db_scan(total_cosmic, richness)


    # plot clusters
    # for i, c in enumerate(clusters):
    #     for j, k in enumerate(c):
    #         plt.scatter(k[:,0], k[:,1], alpha=0.3, s=5)
    #         core = cores[i][j]
    #         plt.scatter(core[:,0], core[:,1], s=10)
    #     plt.axis([149.4,150.8,1.6,2.8])
    #     plt.show()


    # create list of clusters
    # dbs_clusters = []

    # for i,c in enumerate(clusters):
    #     for j,k in enumerate(c):
    #         core = cores[i][j]
            
    #         center = [np.mean(core[:,0]), np.mean(core[:,1]), np.mean(core[:,2]), None, None, None, None]
    #         galaxies = np.concatenate((core, k), axis=0)
    #         d = Cluster(center, galaxies)
    #         dbs_clusters.append(d)
    
    # pickle candidates list
    # with open(fname+'DBSCAN_candidates.dat', 'wb') as f:
    #     pickle.dump(dbs_clusters, f)


    # perform metric test for true labels and labels: Homogeneity, completeness, v-measure, adjusted rand index, AMI
    print(f'Homogeneity: {metrics.homogeneity_score(labels_true, pred_labels):.3f}')
    print(f'Completeness: {metrics.completeness_score(labels_true, pred_labels):.3f}')
    print(f'V-Measure: {metrics.v_measure_score(labels_true, pred_labels):.3f}')
    print(f'Adjusted Rand Index: {metrics.adjusted_rand_score(labels_true, pred_labels):.3f}')
    print(f'Mutual Adjusted Information: {metrics.adjusted_mutual_info_score(labels_true, pred_labels):.3f}')
    print(f'FMI Score: {metrics.fowlkes_mallows_score(labels_true, pred_labels):.3f}')
