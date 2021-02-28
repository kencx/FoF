import numpy as np


class Cluster:
    
    # initialize with __slots__
    __slots__ = ['ra', 'dec', 'z', 'z_unc', 'bcg_absMag', 'bcg_logSM', 'bcg_logSFR', 'gal_id', 'bcg_logLR', 'N', 'galaxies', 'D','flag_poor', '_richness', '_center_attributes']
    
    def __init__(self, center, galaxies):

        # center properties
        self.ra = center[0]
        self.dec = center[1]
        self.z = center[2]
        self.z_unc = [center[3], center[4]] # z_lower, z_upper
        self.bcg_absMag = center[5]
        self.bcg_logSM = center[6]
        self.bcg_logSFR = center[7]
        self.gal_id = center[8]
        self.bcg_logLR = center[9]
        self.N = center[-1]
        self.galaxies = galaxies # np array
        self.D = 0
        self.flag_poor = False


    @property
    def richness(self):
        return len(self.galaxies)


    @property
    def center_attributes(self): # easy conversion of attributes to array
        center = np.array([self.ra, self.dec, self.z, self.z_unc[0], self.z_unc[1], self.bcg_absMag, self.bcg_logSM, self.bcg_logSFR, self.gal_id, self.bcg_logLR, self.N])
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


    def export(self): # export to dataframe
        pass


    def __str__(self):
        return 'Cluster at ({self.ra},{self.dec},{self.z}) with {self.richness} galaxies'.format(self=self)



class CleanedCluster(Cluster):
    # __slots__ = ['cluster_mass', 'vel_disp', 'virial_radius']

    def __init__(self, center, galaxies, cluster_mass, vel_disp, virial_radius, lum4):
        super().__init__(center, galaxies)

        self.cluster_mass = cluster_mass  #(before and after interloper removal)
        self.vel_disp = vel_disp
        self.virial_radius = virial_radius
        self.total_luminosity = lum4
        # self.edge_flag = 0