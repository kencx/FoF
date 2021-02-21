# Friends of Friends Algorithm

This project uses the Friends-of-friends (FoF) algorithm to search for galaxy clusters within the [COSMOS survey](https://cosmos.astro.caltech.edu/). 

## About
- Separations and nearest neighbour lookups are sped up with scipy's KD-tree implementation.
- Uses 3-sigma method to remove interlopers.
- Calculates mass, velocity dispersion and projected radius with the virial mass estimator.
- Performs catalog matching with other notable COSMOS galaxy cluster catalogs.
    - [Z. L. Wen & J. L. Han, 2011](https://iopscience.iop.org/article/10.1088/0004-637X/734/1/68)
    - [Ilona K. Söchting et al., 2012](https://academic.oup.com/mnras/article/423/3/2436/2460403)
- Parameters chosen are empirical
- Very much a work in progress

## Parameters
- max_velocity: Maximum relative velocity of member galaxies to cluster center.
- linking_length_factor: Constant of proportionality in determining linking length, default is 0.2 for dark matter halos.
- virial_radius: Maximum separation of member galaxies from to cluster center.
- richness, R: Minimum number of galaxies to form a cluster
- overdensity, D: Cluster density comparison with background density

## Future plans
- Analyze dynamical properties of clusters
- Implement unsupervised clustering algorithms (DBSCAN, OPTICS)

