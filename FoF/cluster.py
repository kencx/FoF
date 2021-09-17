#!/usr/bin/env python3

import numpy as np
import astropy.units as u


class Cluster:

    # initialize with __slots__
    __slots__ = [
        "ra",
        "dec",
        "z",
        "z_unc",
        "bcg_absMag",
        "gal_id",
        "N",
        "galaxies",
        "D",
        "flag_poor",
        "properties",
        "_richness",
        "_center_attributes",
    ]

    def __init__(self, center, galaxies):

        # center properties
        self.ra = center[0]
        self.dec = center[1]
        self.z = center[2]
        self.z_unc = [
            center[3],
            center[4],
        ]  # z_lower, z_upper # make this into one array next time
        self.bcg_absMag = center[5]  # absolute magnitude
        self.gal_id = center[6]  # ID
        self.N = center[-1]  # N(0.5), number count
        self.galaxies = galaxies  # ndarray
        self.D = 0  # overdensity
        self.flag_poor = False  # for manual flagging
        if len(center) > 8:
            self.properties = center[7:-1]  # other optional properties

    @property
    def richness(self):
        return len(self.galaxies)

    @property
    def center_attributes(self):  # easy conversion of attributes to array
        center = np.array(
            [
                self.ra,
                self.dec,
                self.z,
                self.z_unc[0],
                self.z_unc[1],
                self.bcg_absMag,
                self.gal_id,
                self.N,
            ]
        )
        return center

    def remove_overlap(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        # compare N(0.5) of two clusters and removes the common galaxies from smaller cluster. if both are same N(0.5), remove common galaxies from cluster with less bright center galaxy
        if self.N > other.N:
            self_galaxies = self.galaxies
            other_galaxies = np.array([])
        elif self.N < other.N:
            self_galaxies = np.array([])
            other_galaxies = other.galaxies
        elif abs(self.bcg_absMag) > abs(other.bcg_absMag):
            self_galaxies = self.galaxies
            other_galaxies = np.array([])
        else:
            self_galaxies = np.array([])
            other_galaxies = other.galaxies
        return self_galaxies, other_galaxies

    def __str__(self):
        return f"Cluster at ({self.ra:.2f},{self.dec:.2f},{self.z:.2f}) with {self.richness} galaxies"


class CleanedCluster(Cluster):
    # __slots__ = ['cluster_mass', 'vel_disp', 'projected_radius']

    def __init__(
        self,
        center,
        galaxies,
        cluster_mass,
        mass_err,
        vel_disp,
        vel_disp_err,
        projected_radius,
        absmag,
    ):
        super().__init__(center, galaxies)

        self.cluster_mass = cluster_mass  # h^-1 M_sun
        self.mass_err = mass_err
        self.vel_disp = vel_disp  # (km/s)^2
        self.vel_disp_err = vel_disp_err
        self.projected_radius = projected_radius  # h^-1 Mpc
        self.total_absmag = absmag

    def r200(self, cosmo, delta):  # h^-1 Mpc
        rho = cosmo.critical_density(self.z)
        radius = (
            ((3 / 4) * (self.cluster_mass) / (delta * np.pi * rho)) ** (1 / 3)
        ).to(u.Mpc, u.with_H0(cosmo.H0))
        return (radius * ((cosmo.H0 / 100).value) / u.littleh).to(
            u.Mpc / u.littleh
        )  # littleh scaling

    @property
    def properties(self):
        return np.array(
            [
                self.ra,
                self.dec,
                self.z,
                self.cluster_mass.value,
                self.vel_disp.value,
                self.projected_radius.value,
                self.total_absmag,
                self.richness,
                self.D,
                self.flag_poor,
                self.gal_id,
            ]
        )
