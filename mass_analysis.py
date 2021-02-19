import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

from analysis.mass_estimator import virial_mass_estimator, projected_mass_estimator
from cluster import CleanedCluster


def find_masses(data, estimator): # adapt this into find_quantities, include total_luminosity (or sum of top 4 luminosities)

    new_cs = []
    for i, c in enumerate(data):
        if estimator == 'virial':
            mass, vel_disp, virial_radius = virial_mass_estimator(c)

            luminosity = c.galaxies[:,5]
            lum4 = sum(-np.sort(-luminosity, axis=0)[:4]) # sort lum in desc order
            new_c = CleanedCluster(c.center_attributes, c.galaxies, mass, vel_disp, virial_radius, lum4)

        if estimator == 'projected':
            mass = projected_mass_estimator(c)
            new_c = CleanedCluster(c.center_attributes, c.galaxies, mass, None, None, None)

        assert new_c.gal_id == c.gal_id, 'Different cluster has been added'
        new_cs.append(new_c)
    return new_cs


def compare_masses(projected, virial):
    virial = [c.cluster_mass.value for c in virial]
    projected = [c.cluster_mass.value for c in projected]

    # virial = virial[virial < 1e17]
    # projected = projected[projected < 1e17]

    fig, ax = plt.subplots()
    ax.scatter(x=np.log10(virial), y=np.log10(projected), s=10, alpha=0.75)
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])]

    m, c, r, _, _, = linregress(np.log10(virial), np.log10(projected))
    print(m,c,r)

    ax.plot(lims, lims, 'k--', alpha=0.75) # plot x = y
    X = np.linspace(np.min(ax.get_xlim()), np.max(ax.get_xlim())) 

    ax.plot(X, m*X+c, 'r--') # plot experimental relationship
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('virial masses')
    ax.set_ylabel('projected masses')
    
    plt.show()


def plot_masses(clusters, x, y): # adapt this into plot_quantities, with y vs x
    
    z_arr = np.array([c.z for c in clusters])
    masses = np.array([c.cluster_mass.value for c in clusters])

    fig, ax = plt.subplots(figsize=(10,8))
    p = ax.scatter(z_arr, np.log10(masses), s=10, alpha=0.75)

    ax.set_title('Mass against redshift')
    ax.set_xlabel('Redshift')
    ax.set_ylabel('Cluster Mass (log(M/h^-1 Msun))')
    ax.ticklabel_format(style='sci', axis='y')
    ax.yaxis.major.formatter._useMathText = True

    # x = bcg_arr[:,2]
    # a,b,r,_,_ = linregress(x, masses)
    # ax.plot(x,a*x+b,'k--',alpha=0.75)
    # print(r)

    ax.set_xlim(0.5,2.5)
    # ax.set_ylim(0,6e15)
    plt.show()


if __name__ == "__main__":

    richness = 25
    D = 2
    fname = f'analysis\\derived_datasets\\R{richness}_D{D}_vel\\'

    with open(fname+'cleaned_candidates.dat', 'rb') as f:
        cleaned_candidates = pickle.load(f)
    
    virial_clusters = find_masses(cleaned_candidates, 'virial')
    projected_clusters = find_masses(cleaned_candidates, 'projected')

    with open(fname+'clusters.dat', 'wb') as f: # with quantities
        pickle.dump(virial_clusters, f)

    # with open(fname+'clusters.dat', 'rb') as f:
    #     virial_clusters = pickle.load(f)

    compare_masses(projected_clusters, virial_clusters)
    plot_masses(virial_clusters, None, None)
