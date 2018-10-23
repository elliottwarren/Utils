"""
Forward operator constants

Separated from the functions by Elliott on Fri 28th Oct 2016
"""

from numpy import pi

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# AEROSOL CONSTANTS

# critical value of relative humidity to start swelling the aerosol
RH_crit = 38.0 # no longer used

# Parameters from Pete Clarks (2008) visibility paper:
rho_a = 1.7e3  # density of aerosol (for ammonium sulphate) [kg m-3]
r0_clark = 1.6e-7  # radius of a 'standard' aerosol particle [m]
r0_haywood = 1.1e-7 # update from Haywood
# r0_urban = 1.87e-7 # paper 2 (accum range of 80 - 1000 nm)
r0_urban = 1.59e-7 # paper 2 (accum range of 80 - 800 nm)
# r0_urban = 1.48e-7 # paper 2 (accum range of 80 - 700 nm)

# p_aer = 1./6 # (p) power used to represent the variation in aerosol
# particle size with mixing ratio
# (p) power used to represent the variation in aerosol particle size
# with mixing ratio UMDP 26 page 26 equation 39
p_aer = 1./6

# (B) Activation parameter (for ammonium sulphate)
B_activation_clark = 0.5
B_activation_haywood = 0.14

# Update from Haywood et.al. (2008) using flight data from around UK
# this is the set of parameters used in the UM v8.1 (see UMDP 26)
# to compute the aerosol number N_aer in code below
# N0 of 6824 cm-3 taken as total from clearflo winter, for particles between 0.02 and 0.7 microns.
# N0 of 4461 cm-3 taken as total from clearflo winter, for particles between 0.04 and 0.7 microns,
#   the range was interpreted as such from figure 13 of Harrison et al., 2012 MR and is the currently the default
# N0_aer = 8.0e9
N0_aer = 4.461e9 # 0.04 - 0.07 um # paper 1 clearFlo winter accum range
# N0_aer_urban = 1.31182e9 # paper 2 (accum range of 80 - 1000 nm) # test
N0_aer_urban = 1.31104e9 # paper 2 (accum range of 80 - 800 nm)
# N0_aer = 6.824e9 #4.0e9 #8.0e9 # (N0) standard number density of aerosol [m-3] 0.02 - 0.7 um # ClearfLo winter
m0_aer = 1.8958e-8  # (m0) standard mass mixing ratio of the aerosol [kg kg-1]
m0_aer_urban = 1.8958e-8 # paper 2 pm10 obs average at NK 01/01/2010 - 31/07/2016 [kg kg-1] # correct units
# m0_aer_urban = 24.0e-9 # paper 2 pm10 obs average at NK 01/01/2010 - 31/07/2016 [kg m-3]


# For use in  eqns. 17-18 in Clark et.al. (2008):
eta = 0.75
# eta = 1.0 # test
Q_ext_aer = 2.0 # geometric scattering

LidarRatio = {"Snow": 10.0,
              "Ice": 30.0,
              "Water": 18.8,
              "Rain": 5.0,
              "Aerosol": 60.0,
              "Molecular": 8.0 * pi / 3.0
              }

# OTHER CONSTANTS

model_resolution = {'UKV': '1p5km',
                    '333m': '0p3km',
                    '100m': '0p1km'}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SETUP

# sites with height pairs
# height above sea level (LUMA metadata)- height above ground (DTM) = height above surface
# surface height taken from DTM (Grimmond and ...) - Digital Terrain Model (surface only, no buildings)

#site_bsc = {'CL31-A_BSC_KSS45W': 64.3, 'CL31-B_BSC_RGS': 28.1 - 19.4, 'CL31-C_BSC_MR': 32.0 - 27.5,
#            'CL31-D_BSC_NK': 27.0 - 23.2}
#site_bsc = {'CL31-A_BSC_IMU': 72.8, 'CL31-B_BSC_RGS': 28.1 - 19.4, 'CL31-C_BSC_MR': 32.0 - 27.5,
#            'CL31-D_BSC_NK': 27.0 - 23.2}

site_bsc = {'CL31-A_IMU': 72.8, 'CL31-B_RGS': 28.1 - 19.4, 'CL31-C_MR': 32.0 - 27.5,
            'CL31-D_NK': 27.0 - 23.2, 'CL31-E_NK': 27.0 - 23.2,'CL31-A_KSS45W': 64.3}

#site_bsc = {'CL31-B_RGS': 28.1 - 19.4, 'CL31-C_MR': 32.0 - 27.5,
#            'CL31-D_NK': 27.0 - 23.2, 'CL31-A_KSS45W': 64.3}

site_rh = {'WXT_KSSW': 50.3, 'Davis_BCT': 106.4, 'Davis_BFCL': 6.8, 'Davis_BGH': 27.3, 'Davis_IMU': 72.8, 'Davis_IML': 67.9}
site_aer = {'PM10_RGS': 23.0 - 19.4, 'PM10_MR': 32.0 - 27.5, 'PM10_NK': 26.0 - 23.2}

# colours for plotting
site_bsc_colours = {'IMU': 'b', 'RGS': 'y', 'MR': 'r', 'NK': 'm', 'SW': 'k', 'KSSW': 'c', 'KSS45W': 'g'}
# ax.set_prop_cycle(cycler('color', ['b', 'c', 'm', 'y', 'k']))

# forward model version
aerFO_version = 'v0.2.1'