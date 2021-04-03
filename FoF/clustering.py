import sqlite3
import pandas as pd

from sklearn.cluster import DBSCAN, OPTICS
from sklearn import metrics
# from kneed import KneeLocator
# from sklearn.neighbors import NearestNeighbors

from analysis.methods import mean_separation, linear_to_angular_dist, redshift_to_velocity
from cluster import Cluster
from params import *

def optics(data, richness, redshift):

    # max_radius = linear_to_angular_dist(1.5*u.Mpc/u.littleh, redshift).value
    max_radius = linear_to_angular_dist(1.5*u.Mpc/u.littleh, redshift).to('rad')
    max_radius = (max_radius*cosmo.comoving_transverse_distance(redshift)).to(u.Mpc, u.dimensionless_angles()) # convert to comoving distance
    max_radius = linear_to_angular_dist(max_radius, redshift).value # convert comoving distance to angular separation

    if len(data) >= richness: # don't run if less points

        op = OPTICS(min_samples=richness, max_eps=max_radius, metric='haversine')
        op.fit(data)

        labels = op.labels_[op.ordering_]
        return labels

    else:
        return [None]


if __name__ == "__main__":

    galaxies = np.loadtxt('FoF\\analysis\\derived_datasets\\COSMOS_galaxy_data.csv', delimiter=',') 
    galaxies = galaxies[(galaxies[:,2]<=1.2) & (galaxies[:,6]>=9.6)]
    galaxies = galaxies[:,:3] # ra,dec,z

    conn = sqlite3.connect('FoF\\processing\\datasets\\galaxy_clusters.db')

    cosmic_web_df = pd.read_sql_query('''
    SELECT ra, dec, redshift, cluster_id
    FROM cosmic_web_bcg
    WHERE Ngal>=? AND redshift >=0.5
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

    def bin_data(data):

        X_arr = []
        # split data into redshift bins of v <= 2000km/s (z = 0.006671; use upper bound of 0.01)
        z_arr = data[:,2]
        bins = np.arange(min(z_arr), max(z_arr), 0.01)
        digitized = np.digitize(z_arr, bins) 

        for i in range(1, len(bins)+1):
            X_bin = data[np.where(digitized==i)]
            X_arr.append(X_bin)

        return X_arr
    
    
    def tests(labels_true, pred_labels):
        # perform metric test for true labels and labels: Homogeneity, completeness, v-measure, adjusted rand index, AMI
        # print(f'Homogeneity: {metrics.homogeneity_score(labels_true, pred_labels):.3f}')
        # print(f'Completeness: {metrics.completeness_score(labels_true, pred_labels):.3f}')
        # print(f'V-Measure: {metrics.v_measure_score(labels_true, pred_labels):.3f}')
        print(f'Adjusted Rand Index: {metrics.adjusted_rand_score(labels_true, pred_labels):.3f}')
        print(f'Mutual Adjusted Information: {metrics.adjusted_mutual_info_score(labels_true, pred_labels):.3f}')
        print(f'FMI Score: {metrics.fowlkes_mallows_score(labels_true, pred_labels):.3f}')


    def element_replacement(data):
        # find all common elements
        elem, idx = np.unique(data, return_index=True)

        # sort element by index
        elem = np.array([i for _,i in sorted(zip(idx,elem))])

        # replace each common element with number from 0
        for i,e in enumerate(elem):
            data[data == e] = i
        return data

    # bin data
    binned_true = bin_data(total_cosmic)
    binned_X = bin_data(galaxies)

    # perform optics on each bin
    for i, b in enumerate(binned_X):
        b_coords = b[:,:2]
        b_z = np.mean(b[:,2])

        labels_true = binned_true[i][:,-1]
        labels_true = element_replacement(labels_true)
        labels_pred = optics(b[:,:2], richness, b_z)

        if len(labels_pred) > 1:
            # tests(labels_true, labels_pred) # perform tests
            print(metrics.silhouette_score(b_coords, labels_pred, metric='haversine'))


            # check plots
            # X_bin = b[:,:2]
            # # plt.scatter(X_bin[:,0], X_bin[:,1], s=5, color='b', alpha=0.5)

            # colors = ['g.', 'r.', 'b.', 'y.', 'c.']
            # for klass, color in zip(range(0, 5), colors):
            #     Xk = X_bin[labels_pred == klass]
            #     plt.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
            # plt.plot(X_bin[labels_pred == -1, 0], X_bin[labels_pred == -1, 1], 'k+', alpha=0.1)
            # plt.show()


    # create list of clusters
    # clusters = []

    # for i,c in enumerate(clusters):
    #     for j,k in enumerate(c):
    #         core = cores[i][j]
            
    #         center = [np.mean(core[:,0]), np.mean(core[:,1]), np.mean(core[:,2]), None, None, None, None]
    #         galaxies = np.concatenate((core, k), axis=0)
    #         d = Cluster(center, galaxies)
    #         clusters.append(d)
    
    # pickle candidates list
    # with open(fname+'OPTICS_candidates.dat', 'wb') as f:
    #     pickle.dump(clusters, f)


    