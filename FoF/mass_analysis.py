#!/usr/bin/env python3

from params import *
from cluster import CleanedCluster
from analysis.mass_estimator import virial_mass_estimator, projected_mass_estimator

def find_masses(data, estimator): # adapt this into find_quantities, include total_luminosity (or sum of top 4 luminosities)

    new_cs = []
    for i, c in enumerate(data):
        if estimator == 'virial':
            mass, mass_err, vel_disp, vel_disp_err, projected_radius = virial_mass_estimator(c)

            absmag = c.galaxies[:,5]
            total_absmag = -2.5*np.log10(sum(10**(-0.4*absmag))) # total absolute magnitude
            new_c = CleanedCluster(c.center_attributes, c.galaxies, mass, mass_err, vel_disp, vel_disp_err, projected_radius, total_absmag)

        if estimator == 'projected':
            mass, mass_err = projected_mass_estimator(c)
            new_c = CleanedCluster(c.center_attributes, c.galaxies, mass, mass_err, None, None, None, None)

        assert new_c.gal_id == c.gal_id, 'Different cluster has been added'
        new_cs.append(new_c)
    return new_cs