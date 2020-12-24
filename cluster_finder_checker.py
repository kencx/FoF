import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from fof_kdtree import cluster_search, linear_to_angular_dist


# perform FoF on sample
# create sample candidate table
# import catalogue table


def compare_clusters(candidates, catalogue):

    count = 0
    for c in catalogue:
        
        # search a radius of 1Mpc and within redshift constraint
        redshift_constraint = 0.05
        redshift_idx = candidates[(candidates[2] >= c[2] - redshift_constraint) & (candidates[2] <= c[2] + redshift_constraint)]
        redshift_cutoff = candidates[redshift_cutoff]

        candidate_tree = cKDTree(redshift_cutoff) # create binary search tree for candidate sample
        centers_idx = candidate_tree.query_ball_point(c[:2], linear_to_angular_dist(1, c[2]))
        centers = redshift_cutoff[centers_idx]
        if len(centers) >= 1:
            count += len(centers) # add to count if constraints satisfied
    
    return (count/len(catalogue))*100 # good algorithm if >80%


def richness_plot(candidates, catalogue):

    # plot the candidate to catalogue richness
    fig = plt.figure()
    candidate_richness = 0
    catalogue_richness = 0
    plt.scatter(candidate_richness, catalogue_richness)
    plt.show() # good algorithm if follows a x=y trend

