# Friends of Friends Algorithm

Disclaimer: WORK IN PROGRESS!!

This project uses the Friends-of-friends (FoF) algorithm to search for high redshift (0.5 < z < 2) galaxy clusters within the [COSMOS survey](https://cosmos.astro.caltech.edu/).

This code base was developed for and used in my undergraduate thesis:
> Detection and Analysis of High Redshift Galaxy Clusters
>
> Completed: May 2021

## About
- Separations and nearest neighbour lookups are sped up with GrisPy bubble neighbors search.
- Uses 3-sigma method to remove interlopers.
- Calculates mass, velocity dispersion and projected radius with the virial mass estimator.
- Performs catalog matching with other notable COSMOS galaxy cluster catalogs.
    - [F. Bellagamba et al., 2011](https://academic.oup.com/mnras/article/413/2/1145/1067808?login=true)
    - [George et al., 2011](https://iopscience.iop.org/article/10.1088/0004-637X/742/2/125)
    - [Z. L. Wen & J. L. Han, 2011](https://iopscience.iop.org/article/10.1088/0004-637X/734/1/68)
    - [Ilona K. Söchting et al., 2012](https://academic.oup.com/mnras/article/423/3/2436/2460403)
- Parameters chosen are empirically determined

## Dependencies
- Python 3.6 or later
- Popular STEM libraries including: Numpy, Scipy
- Data Processing libraries: Pandas, Matplotlib
- Astropy
- [GrisPy](https://github.com/mchalela/GriSPy)

## Usage
1. Pull this repository with
```
git pull https://github.com/kennethcheo/FoF.git
```
2. Obtain your dataset in your desired format (FITS, csv etc.)
    - This project mainly uses Pandas' dataframes to hold data.
    - Refer to `data_processing.py` for conversion of FITS to DF

3. Ensure galaxy photometric or spectroscopic dataset with the relevant properties: RA, DEC, z, Absolute Magnitude, Galaxy ID

4. Run pipeline with `pipeline.py` by commenting out relevant parts (or using it as a guide)

5. Remember to change the default parameters in `params.py`

6. Run dataset through the pipeline and save them to the appropriate location
    - Large datasets (> 1 million points) might take awhile.

## Parameters
- `max_velocity`: Maximum relative velocity of member galaxies to cluster center.
- `linking_length_factor`: Constant of proportionality in determining linking length, default is 0.2 for dark matter halos.
- `virial_radius`: Maximum separation of member galaxies from to cluster center.
- richness, `R`: Minimum number of galaxies to form a cluster.
- overdensity, `D`: Cluster density comparison with background density.
