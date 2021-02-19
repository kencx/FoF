import numpy as np


class Cluster:
    
    # initialize with __slots__
    __slots__ = ['ra', 'dec', 'z', 'bcg_brightness', 'gal_id', 'bcg_luminosity', 'N', 'galaxies', 'D', '_richness', '_center_attributes']
    
    def __init__(self, center, galaxies):

        # center properties
        self.ra = center[0]
        self.dec = center[1]
        self.z = center[2]
        self.bcg_brightness = center[3]
        self.gal_id = center[4]
        self.bcg_luminosity = center[5]
        self.N = center[-1]
        self.galaxies = galaxies # np array
        self.D = 0

    @property
    def richness(self):
        return len(self.galaxies)


    @property
    def center_attributes(self): # easy conversion of attributes to array
        center = np.array([self.ra, self.dec, self.z, self.bcg_brightness, self.gal_id, self.bcg_luminosity, self.N])
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
        elif abs(self.bcg_brightness) > abs(other.bcg_brightness):
            self_galaxies = self.galaxies
            other_galaxies = np.array([])
        else:
            self_galaxies = np.array([])
            other_galaxies = other.galaxies
        return self_galaxies, other_galaxies


    # def merge(self, other):
    #     if not isinstance(other, type(self)):
    #         return NotImplemented
    #     pass # compare N(0.5) of two clusters, keep larger cluster with merged points and deletes smaller one. if both have same N(0.5), compare brightness instead

    #     combined = np.concatenate((self.galaxies, other.galaxies), axis=0) # combine the galaxy arrays
    #     arr, uniq_count = np.unique(combined, axis=0, return_counts=True)
    #     combined = arr[uniq_count==1] # ensure all points are unique

    #     if self.N > other.N:
    #         self.galaxies = combined
    #         other.galaxies = np.array([])
    #     elif self.N < other.N:
    #         self.galaxies = np.array([])
    #         other.galaxies = combined
    #     elif abs(self.bcg_brightness) > abs(other.bcg_brightness):
    #         self.galaxies = combined
    #         other.galaxies = np.array([]) 
    #     else:
    #         self.galaxies = np.array([])
    #         other.galaxies = combined


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