import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad_vec

from astropy import units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord
from astropy.cosmology import LambdaCDM

from astropy.modeling.models import NFW
from astropy.stats import bootstrap
from astropy.utils import NumpyRNGContext

from processing.methods import split_df_into_groups
from analysis.methods import redshift_to_velocity

cosmo = LambdaCDM(H0=70*u.km/u.Mpc/u.s, Om0=0.3, Ode0=0.7) # define cosmology


# calculate velocity dispersion for an array of velocities
def velocity_dispersion(v_arr, group_size):
    return np.sum(v_arr**2)/(group_size-1)

# evaluate projected virial radius
def projected_radius(group_size, separation):
    return group_size*(group_size-1)/separation

# evaluate cluster mass
def cluster_mass(cluster_vel_disp, projected_radius):
    mass = (3/2)*np.pi*(cluster_vel_disp*projected_radius)/const.G
    return mass.to(u.M_sun/u.littleh)


def virial_mass_estimator(cluster): # INCLUDE MASS WEIGHTING
    ''' 
    Calculates virial masses of galaxy clusters with the virial mass formula

    Parameters
    ----------
    cluster: Cluster object
    
    Returns
    -------
    mass: float
        Mass of cluster in M_sun

    cluster_vel_disp: float
        Estimated velocity dispersion of cluster in (km/s)**2

    cluster_radius: float
        Estimated radius of cluster in Mpc
    '''
    member_galaxies = cluster.galaxies

    # velocity dispersion
    cluster_size = len(member_galaxies)
    z_arr = member_galaxies[:,2]

    average_redshift = np.mean(z_arr)
    cluster_vel = (redshift_to_velocity(z_arr) - redshift_to_velocity(average_redshift))/(1+average_redshift)
    cluster_vel_disp = velocity_dispersion(cluster_vel, cluster_size)

    # galaxy separations
    c = SkyCoord(ra=member_galaxies[:,0]*u.degree, dec=member_galaxies[:,1]*u.degree) # array of coordinates
    
    pairs_idx = np.asarray(list((i,j) for ((i,_), (j,_)) in itertools.combinations(enumerate(c), 2))) # index of galaxies in combinations
    photoz_pairs = np.take(z_arr, pairs_idx) # photoz values of galaxies from the indices
    d_A = cosmo.angular_diameter_distance(z=average_redshift) # angular diameter distance

    pairs = c[pairs_idx]
    total_separation = 0

    projected_sep = pairs[:,0].separation(pairs[:,1]) # projected separation (in deg)
    actual_sep = (projected_sep*d_A).to(u.Mpc, u.dimensionless_angles()) # convert projected separation to Mpc
    actual_sep = ((actual_sep*((cosmo.H0/100).value))/u.littleh).to(u.Mpc/u.littleh) # include littleh scaling
    total_separation = sum(1/actual_sep)
    
    cluster_radius = projected_radius(cluster_size, total_separation)
    mass = cluster_mass(cluster_vel_disp, cluster_radius)

    return mass, cluster_vel_disp, cluster_radius # M_sun/littleh, km^2/s^2, Mpc/littleh


def projected_mass_estimator(cluster):

    member_galaxies = cluster.galaxies
    N = len(member_galaxies)
    average_redshift = np.mean(member_galaxies[:,2])
    cluster_velocity = (redshift_to_velocity(member_galaxies[:,2]) - redshift_to_velocity(average_redshift))/(1+average_redshift) # in km/s

    c = SkyCoord(ra=member_galaxies[:,0]*u.degree, dec=member_galaxies[:,1]*u.degree)
    centroid = SkyCoord(ra=np.mean(member_galaxies[:,0])*u.degree, dec=np.mean(member_galaxies[:,1])*u.degree)
    # center = SkyCoord(ra=cluster_center[:,0]*u.degree, dec=cluster_center[:,1]*u.degree)

    cluster_separation = centroid.separation(c)
    d_A = cosmo.angular_diameter_distance(z=average_redshift)
    actual_separation = (cluster_separation*d_A).to(u.Mpc, u.dimensionless_angles())
    actual_separation = (actual_separation*((cosmo.H0/100).value)/u.littleh).to(u.Mpc/u.littleh) # include littleh scaling

    sumproduct = np.sum(actual_separation*cluster_velocity**2)
    projected_mass = (32/np.pi)*(1/const.G)*(1/(N-1.5))*sumproduct

    return projected_mass.to(u.M_sun/u.littleh)


def mass_correction(mass, r_200, bcg_arr):
    mass = mass*u.M_sun
    c_200 = 5
    r_200 = r_200*u.Mpc
    redshift = bcg_arr[:,2]

    nfw = NFW(mass=mass, concentration=c_200, redshift=redshift, cosmo=cosmo)
    n_200 = nfw(r_200)

    f = nfw.A_NFW(c_200)
    delta_c = (200/3)*(c_200**3)/f
    critical_density = cosmo.critical_density(z=bcg_arr[:,2])
    # rho_200 = (delta_c*critical_density)/(c_200*(1+c_200)**2)

    # integral = (4*np.pi*critical_density*delta_c)*((1/(c_200+1)) + np.log(c_200+1) - 1)*(r_200/c_200)**3

    def integrand(x):
        x = (x*u.Mpc).to(u.kpc)
        r_s = nfw.r_s
        return (4*np.pi*delta_c*critical_density*r_s**3*(x/(x+r_s)**2)).to(u.M_sun/u.Mpc).value

    integral, err = quad_vec(integrand, 0, r_200.value)
    integral = integral*u.M_sun
    density_term = n_200/integral
    dispersion = 1/3
    C = ((4*np.pi*r_200**3)*dispersion*density_term*mass).to(u.M_sun)
    return mass - C


def uncertainty(cluster_members):
    bootfunc = lambda x: virial_mass_estimator(x)
    with NumpyRNGContext(1):
        bootresult = bootstrap(cluster_members, bootnum=50) # returns 50 samples of 1 cluster
    return bootresult



if __name__ == "__main__":
    pass
    # fname = 'derived_datasets\\R25_D4_0.02_1.5r\\'
    # bcg_df = pd.read_csv(fname+'filtered_bcg.csv')
    # member_df = pd.read_csv(fname+'filtered_members.csv')

    # bcg_arr = bcg_df.sort_values('cluster_id').values
    # masses = np.loadtxt(fname+'virial_masses.txt')

    # arr, group_n = split_df_into_groups(member_df, 'cluster_id', -1)
    # bootvalues = np.zeros((len(group_n),50))

    # for i, g in enumerate(group_n):
    #     cluster = arr[arr[:,-1]==g]
    #     # center = bcg_arr[bcg_arr[:,-1]==g]
    #     # if estimator == 'virial':
    #     bootvalues[i] = uncertainty(cluster[:,:3])
    #     if i == 0:
    #         break

    # plt.figure()
    # plt.hist(bootvalues[0], bins=30)
    # plt.show()
    
    

# -----------------
    # corrected_mass = mass_correction(masses, 2, bcg_arr).value
    # np.savetxt('corrected_masses.txt', corrected_mass)

    # plot mass against corrected mass difference
    # plt.figure(figsize=(12,8))
    # plt.scatter(bcg_df['redshift'], np.log10(corrected_mass), s=8, color='k', alpha=0.5)
    # plt.scatter(bcg_df['redshift'], np.log10(masses), s=8, color='red', alpha=0.5)
    # plt.axis()
    # plt.scatter(masses, (masses-corrected_mass)/masses, s=8, alpha=0.75)
    # plt.show()




    