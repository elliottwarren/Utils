"""
Set of functions for the evaluation of the forward modelled backscatter, with respect to observations
Created by Charlton-Perez et al., 2015
Based on visibility parameterisation in Clark et al., 2008
Modified further by Elliott Warren using suggestions from Haywood et al., 2008, based on flight observations.

Currently, just capable of producing clear sky backscatter (no particles reach critical radius
during hygroscopic growth)
"""

import numpy as np
import iris
import cartopy.crs as ccrs
import os
from dateutil import tz

import datetime as dt
import ellUtils as eu
import ceilUtils as ceil
from forward_operator import FOconstants as FOcon

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. FORWARD OPERATOR

# Attenuated backscatter using aerosol extinction coeff and transmission
# Main one to call!
def forward_operator(aer_mod, rh_frac, r_v, mod_rho, z_mod,  ceil_lam, version, mod_time,
                 r0 = FOcon.r0_haywood, p = FOcon.p_aer,
                 N0=FOcon.N0_aer, m0 = FOcon.m0_aer, eta = FOcon.eta,
                 **kwargs):
    """
    Process the modelled data to get the attenuated backscatter and aerosol extinction

    :param aer_mod:
    :param rh_mod:
    :param z_mod: original 1D array of heights from the model
    :param mod_time: (array of datetimes) datetimes for each model time step
    :return: bsc_mod
    """

    # Redefine several aerFO constants for the urban case
    N0 = FOcon.N0_aer_urban
    m0 = FOcon.m0_aer_urban
    r0 = FOcon.r0_urban

    # create alpha and beta coefficients for aerosol
    FO_dict = calc_ext_coeff(aer_mod, rh_frac, r_v, mod_rho, z_mod, r0, p, N0, m0, eta, ceil_lam,
                             version, mod_time, **kwargs)

    mod_alpha = FO_dict['alpha_a']
    mod_bscUnnAtt = FO_dict['unnatenuated_backscatter']

    # NN = 28800 # number of enteries in time
    dz = np.zeros_like(z_mod)
    dz[0] = z_mod[0]
    dz[1:len(z_mod)] = z_mod[1:len(z_mod)] - z_mod[0:len(z_mod)-1]

    # integrated alpha and transmission for each height
    int_mod_alpha, mod_transm = compute_transmission(mod_alpha, dz)

    # derive modelled attenuated backscatter
    bsc_mod = mod_transm * mod_bscUnnAtt

    FO_dict['bsc_attenuated'] = bsc_mod
    FO_dict['transmission'] = mod_transm

    return FO_dict

# Calculate the extinction coefficient
def calc_ext_coeff(q_aer, rh_frac, r_v, mod_rho, z_mod, r0, p, N0, m0, eta, ceil_lam, version,
                   mod_time, **kwargs):

    """
    Compute extinction coefficient (aerosol extinction + water vapour extinction)

    :param q_aer: aerosol mass mizing ratio [micrograms kg-1]
    :param rh_frac: relative humidity [ratio/dimensionless]
    :param r0:
    :param B:
    :param mod_time: (array of datetimes) datetimes for each timestep
    :return: alpha_a: total extinction coefficient
    :return: beta_a: UNattenuated backscatter

    Most of the function calculates the AEROSOL extinction coefficient and relevent variables (particle extnction
    efficiency). Last part also calculates water vapour extinction coefficient (same as absorption coefficient
    as water vapour scattering is negligable, given the tiny size of vapour gas (think its ~1e-28)

    """

    def read_obs_aer(q_aer, mod_time, dN_key):
        """
        Read in observed total number for the accum range.
        :param q_aer:
        :param mod_time:
        :param dN_key: the key in dN for which data to extract
        """

        filedir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/' + \
                  'data/npy/number_distribution/'

        filepath = filedir + 'accum_Ntot_Dv_NK_APS_SMPS_' + mod_time[0].strftime('%Y') + '.npy'
        dN = np.load(filepath).flat[0]
        # ['Ntot_fine', 'Dn_fine', 'Ntot', 'time', 'Ntot_accum', 'Dn_accum']

        t_idx = np.array([eu.nearest(dN['time'], t)[1] for t in mod_time]) # time index
        t_diff = np.array([eu.nearest(dN['time'], t)[2] for t in mod_time]) # dt.timedelta() showing how close time is

        # pull out data
        dN = {key: dN[key][t_idx] for key in dN.iterkeys()}

        # overwrite t_idx locations where t_diff is too high with nans
        # only keep t_idx values where the difference is below 1 hour
        bad = np.array([abs(i.days * 86400 + i.seconds) > 60 * 60 for i in t_diff])

        for key in dN.iterkeys():
            dN[key][bad] = np.nan

        # As dN is only surface N values, repeat these values to all heights so the shape of obs_out == q_aer
        obs_out = np.transpose(np.tile(dN[dN_key], (q_aer.shape[1], 1)))

        return obs_out

    def calc_r_m_original(r_d, rh_frac, B=FOcon.B_activation_haywood):

        """
        Original method to calculate swollen radii size for the FO in version 0.1 of the aerFO
        :param r_d:
        :param rh_frac:
        :param B: RH activation parameter
        :return:
        """

        # convert units to percentage
        RH_perc = rh_frac * 100.0

        # rm is the mean volume radius. Eqn. 12 in Clark et.al. (2008)
        # "When no activated particles are present an analytic solution for rm" is
        RH_crit_perc = FOcon.RH_crit
        # mask over values less than critical
        RH_ge_RHcrit = np.ma.masked_less(RH_perc, RH_crit_perc)

        # calculate wet mean radius
        # eq 12 - calc rm for RH greater than critical
        r_m = np.ma.ones(rh_frac.shape) - (B / np.ma.log(rh_frac))
        r_m2 = np.ma.power(r_m, 1. / 3.)
        r_m = np.ma.array(r_d) * r_m2

        # set rm as 0 where RH is less than crit
        r_m = np.ma.MaskedArray.filled(r_m, [0.0])
        where_lt_crit = np.where(np.logical_or(RH_perc.data < RH_crit_perc, r_m == 0.0))
        # refill them with r_d
        r_m[where_lt_crit] = r_d[where_lt_crit]

        return r_m

    def get_S_climatology(mod_time, rh_frac, ceil_lam):

        """
        Create the S array from the climatology (month, RH_fraction) given the month and RH
        :param mod_time:
        :param rh_frac:
        :param ceil_lam (int): ceilometer wavelength [nm]
        :return: S (time, height):
        """

        # 1. Read in the data
        filename = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/common_data/Mie/' + \
                   'S_climatology_NK_SMPS_APS_' + str(ceil_lam) + 'nm.npy'

        data = np.load(filename).flat[0]
        S_clim = data['S_climatology']
        S_RH_frac = data['RH_frac']

        # 2. Create S array given the time and RH

        # get height range from rh_frac
        height_idx_range = rh_frac.shape[1]

        # find S array
        S = np.empty(rh_frac.shape)
        S[:] = np.nan
        for t, time_t in enumerate(mod_time):  # time
            # get month idx (e.g. idx for 5th month = 4)
            month_idx = time_t.month - 1
            for h in range(height_idx_range):  # height

                # find RH idx for this month, and put the element into the S array
                _, rh_idx, _ = eu.nearest(S_RH_frac, rh_frac[t, h])
                S[t, h] = S_clim[month_idx, rh_idx]

        return S

    def get_S_hourly_timeseries(mod_time, ceil_lam):

        """
        Create the S array from the climatology (month, RH_fraction) given the month and RH
        :param mod_time:
        :param rh_frac:
        :param ceil_lam (int): ceilometer wavelength [nm]
        :return: S (time, height):
        """

        # year from mod_time
        year = mod_time.strftime('%Y')

        # 1. Read in the data
        filename = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/npy/S_timeseries/' + \
                   'NK_SMPS_APS_PM10_withSoot_'+year+'_'+str(ceil_lam)+'nm.npy'

        data = np.load(filename).flat[0]
        S_climatology = data['S']
        S_time = data['met']['time']

        # 2. Create S array given the time and RH


        # Extract and store S
        S = np.empty(rh_frac.shape)
        S[:] = np.nan
        for t, time_t in enumerate(mod_time):  # time
            # get time index
            _, S_t_idx, _ = eu.nearest(S_time, time_t)
            # extract S for this time step
            S[t, :] = S_climatology[S_t_idx]

        return S

    # ---------------------------

    # Compute the aerosol number density N_aer. Eqn. 3 in Clark et.al. (2008) and Eqn 39 in UMDP 26 Large-scale precip.
    q_aer_kg_kg = q_aer * 1.0e-9  # convert micrograms kg-1 to kg/kg


    # Number concentration [m-3]
    if 'obs_N' not in kwargs.keys():
        # calculate from MURK
        N_aer = N0 * np.power((q_aer_kg_kg / m0), 1.0 - (3.0 * p))
    else:
        if kwargs['obs_N'] == True:
            # Read in N_aer from observations
            N_aer_cm3 = read_obs_aer(q_aer, mod_time, 'Ntot_accum') # [cm-3]
            N_aer = N_aer_cm3 *1e6 # convert to [m-3]


    # Dry mean radius of bulk aerosol (this radius is the same as the volume mean radius)
    if 'obs_r' not in kwargs.keys():
        r_d = r0 * np.power((q_aer_kg_kg / m0), p)
    else:
        if kwargs['obs_r'] == True:
            # mean radius (by volume, and it is not the geometric radius)
            D_d = read_obs_aer(q_aer, mod_time, 'Dv_accum') # Diameter [nm]
            r_d = D_d/2.0 * 1e-9 # radius [m]

    # Geometric mean radius of bulk aerosol [meters]
    #   derived from a linear fit between observed r_d (volume mean) and r_g (Pearson r = 0.65, p=0.0)
    #   used purely for the f_RH LUT in calc_Q_ext_wet()
    # r_g = (0.24621593450654974 * r_d) + 0.03258363072889052 # 80 - 700 nm paper 2
    r_g = (0.122 * r_d) + 4.59e-8  # 80 - 800 nm

    # calculate Q_ext (wet particle extinction efficiency)
    if version <= 1.0:
        # MURK = fixed 3 aerosol types (amm. nit.; amm. sulph.; OC)
        # Q_ext,dry function of dry size only
        # f(RH) function of RH only
        Q_ext, Q_ext_dry_matrix, f_RH_matrix = calc_Q_ext_wet_v1p0(ceil_lam, r_d, rh_frac)
        print 'using old method (v1.0) for Q_ext'
    elif version > 1.0:
        # MURK, monthly varying based on (amm. nit.; amm. sulph.; OC; BC; sea salt)
        # Q_ext,dry function of dry size, month
        # f(RH) function of RH, geometric mean of dry particle distribution, month
        Q_ext, Q_ext_dry_matrix, f_RH_matrix = calc_Q_ext_wet(ceil_lam, r_d, r_g, rh_frac, mod_time)

    # Calculate extinction coefficient
    # eqns. 17-18 in Clark et.al. (2008)
    if version == 0.1:
        # v0.1 original aerFO version - now outdated
        # calculate swollen radii and extinction coefficient using it
        r_m = calc_r_m_original(r_d, rh_frac)
        aer_ext_coeff = (eta * FOcon.Q_ext_aer) * np.pi * N_aer * np.power(r_m, 2)
        print 'Using old version 0.1 approach to swell particles'

    # v0.2 - use dry radii and an extinction enhancement factor, to include the effect of hygroscopic growth on optical
    #   properties
    elif version >= 0.2:
        # aer_ext_coeff = (eta * Q_ext) * np.pi * N_aer * np.power(r_d, 2) # when optical properties were not calc for distributions
        aer_ext_coeff = Q_ext * np.pi * N_aer * np.power(r_d, 2)

    # calculate the water vapour extinction coefficient
    # T = 16.85 degC, q = 0.01 kg kg-1; p = 1100 hPa
    # wv_ext_coeff = mass_abs * mod_rho * mod_r_v

    if ceil_lam == 905:
        # mass absorption of water vapour [m2 kg-1] for water vapour extinction coefficient
        # script to calculate mass aborption = htfrtc_optprop_gas_plot_elliott.py
        # gaussian weighted average (mean = 905, FWHM = 4) = 0.016709242714125036 # (current) should be used for CL31 (kotthaus et al., 2016)
        # gaussian weighted average (mean = 905, FWHM = 8) = 0.024222946249630242 # (test) test sensitivity to FWHM
        # gaussian weighted average (mean = 900, FWHM = 4) = 0.037273493204864103 # (highest wv abs for a central wavelength between 895 - 915)
        wv_ext_coeff = 0.016709242714125036 * mod_rho * r_v
    else:
        raise ValueError('ceilometer wavelength != 905 nm, need to calculate a new gaussian average to \n'
                         'calculate water vapour extinction coefficient for this new wavelength!')

    # total extinction coefficient
    alpha_a = aer_ext_coeff + wv_ext_coeff

    # Get lidar ratio (S)
    if version <= 1.0:
        # Constant lidar ratio = 60 sr (continental aerosol; Warren et al. 2018)
        S = FOcon.LidarRatio['Aerosol']

    elif version >= 1.1:
        if 'use_S_hourly' not in kwargs:
            # Read in from look up table (monthly varying, for either an urban or rural site)
            # hold it constant for now, until the climatology has been made
            S = get_S_climatology(mod_time, rh_frac, ceil_lam)

        else:
            if kwargs['use_S_hourly'] == True:
                # use the hourly timeseries of S, estimated using aerosol mass and
                #  number distribution data from SMPS and APS
                print 'use_S_hourly is active!: S taken from observations, not parameterised!'
                S = get_S_hourly_timeseries(mod_time, ceil_lam)
            else:
                raise ValueError('use_S_hourly kwarg is present by not set to True!')

    # Calculate backscatter using a constant lidar ratio
    # ratio between PARTICLE extinction and backscatter coefficient (not total extinction!).
    beta_a = aer_ext_coeff / S

    # store all elements into a dictionary for output and diagnostics
    FO_dict = {'unnatenuated_backscatter': beta_a,
                'alpha_a': alpha_a,
                'aer_ext_coeff': aer_ext_coeff,
                'wv_ext_coeff': wv_ext_coeff,
                'r_d': r_d,
                'r_g':r_g,
                'N': N_aer,
                'Q_ext': Q_ext,
                'Q_ext_dry': Q_ext_dry_matrix,
                'f_RH': f_RH_matrix,
                'S': S}

    return FO_dict

def forward_operator_from_obs(day, ceil_lam, version, r0 = FOcon.r0_haywood, p = FOcon.p_aer,
                 N0=FOcon.N0_aer, m0 = FOcon.m0_aer, eta = FOcon.eta, aer_modes=['accum'],
                 **kwargs):
    """
    Process the modelled data to get the attenuated backscatter and aerosol extinction

    :param aer_mod:
    :param rh_mod:
    :param z_mod: original 1D array of heights from the model
    :return: bsc_mod
    :param aer_modes: (list; default=['accum']) Aerosol modes to calculate extinction coefficients for (list can contain: 'fine, 'accum', 'coarse')

    kwargs
    :param fullForecast: (bool; default: False) Should the entire forecast be processed or just the main day?
    :param allvars: (bool; default: False) return all variables that were calculated in estimating the attenuated bacskcatter?
    :param obs_rh: (bool; default: False)
    :param obs_N_accum: (bool; default: False) Use observed accumulation number conc. from observations instead of estimating from aerosol mass
    :param obs_r_accum: (bool; default: False) Use observed accumulation r from observations instead of estimating from aerosol mass
    :param water_vapour_absorption_from_obs: (bool; default: False) calculate the water vapour absorption from observations
                instead of the NWP model output.

    :param version: (float scaler; default: 1.1) aerFO version number:
                0.1 - original aerFO using constant Q = 2, and swelling the radii for calculating the extinction coeff.
                    No aerosol species differentiation, taken straight from Chalrton-Perez et al. 2016.
                0.2 - developed aerFO using Q = Q_ext,dry * f(RH): fixed aerosol proportions of amm. sulphate,
                    amm. nitrate and organic carbon (aged fossil fuel OC) from Haywood et al. 2008 flights.
                1.0 - same as v0.2 and the version used in paper 1.
    (Current)   1.1 - Q_ext,dry and f(RH) now monthly varying and include black carbon and salt form observations.
                    Separate default versions of Q_ext,dry, f(RH) and S for urban (based on North Kensington (NK),
                    London) and rural (based on Chilbolton (CH), UK). f(RH) now varies with radius size too. geometric
                    standard deviation for CH and NK derived from observations to help calculated Q_ext,dry and f(RH).
    """

    def create_heights_and_times(day):

        """
        Create heights and times that match the hourly UKV extract as UKV data is not used
        in forward_operator_from_obs()
        :param day: datetime object for current day
        :return:height: np.array of heights that matches UKV data
        :return time: np.array of datetimes with hourly resolution to match what would have been read in from the UKV
        """

        # heights taken from the UKV
        height = np.array([  5.00000000e+00,   2.16666641e+01,   4.50000000e+01,
         7.50000000e+01,   1.11666679e+02,   1.55000000e+02,
         2.05000000e+02,   2.61666687e+02,   3.25000000e+02,
         3.95000000e+02,   4.71666809e+02,   5.55000000e+02,
         6.45000000e+02,   7.41666809e+02,   8.45000000e+02,
         9.55000000e+02,   1.07166675e+03,   1.19500000e+03,
         1.32500000e+03,   1.46166675e+03,   1.60500000e+03,
         1.75500000e+03,   1.91166675e+03,   2.07500000e+03,
         2.24500049e+03,   2.42166675e+03,   2.60500000e+03,
         2.79500000e+03,   2.99166675e+03,   3.19500000e+03,
         3.40500000e+03,   3.62166675e+03,   3.84500000e+03,
         4.07500000e+03,   4.31166797e+03,   4.55500000e+03,
         4.80500000e+03,   5.06166797e+03,   5.32500000e+03,
         5.59500000e+03,   5.87166797e+03,   6.15500781e+03,
         6.44514795e+03,   6.74249219e+03,   7.04781592e+03,
         7.36235986e+03,   7.68791992e+03,   8.02692822e+03,
         8.38258008e+03,   8.75891602e+03,   9.16094434e+03,
         9.59475977e+03,   1.00676680e+04,   1.05883076e+04,
         1.11667959e+04,   1.18148682e+04,   1.25460244e+04,
         1.33756758e+04,   1.43213203e+04,   1.54027041e+04,
         1.66419844e+04,   1.80639082e+04,   1.96960273e+04,
         2.15688516e+04,   2.37160645e+04,   2.61747168e+04,
         2.89854609e+04,   3.21927324e+04,   3.58450039e+04,
         4.00000000e+04])

        # match resolution of typically extracts UKV data (hourly)
        time = eu.date_range(day, day+dt.timedelta(hours=24), 60, 'minutes')

        return height, time

    # Redefine several aerFO constants for the urban case
    N0 = FOcon.N0_aer_urban
    m0 = FOcon.m0_aer_urban
    r0 = FOcon.r0_urban

    # create hourly time array and height for the day, that would match what the UKV would be
    z, time = create_heights_and_times(day)

    # read in all the necessary data
    wxt_obs = read_wxt_obs(day, time, z)
    rh_frac = wxt_obs['RH_frac']
    r_v = wxt_obs['r_v']
    rho = wxt_obs['air_density']

    # create alpha and beta coefficients for aerosol
    FO_dict = calc_att_backscatter_from_obs(rh_frac, r_v, rho, z, r0, p, N0, m0, eta, ceil_lam,
                                 version, time, aer_modes, **kwargs)

    mod_alpha = FO_dict['alpha_a'] # extinction
    mod_bscUnnAtt = FO_dict['unnatenuated_backscatter'] # backscatter (unattenuated)

    # /delta z to help compute AOD and transmission
    dz = np.zeros_like(z)
    dz[0] = z[0]
    dz[1:len(z)] = z[1:len(z)] - z[0:len(z)-1]

    # integrated alpha and transmission for each height
    int_mod_alpha, mod_transm = compute_transmission(mod_alpha, dz)

    # derive modelled attenuated backscatter
    bsc_mod = mod_transm * mod_bscUnnAtt
    FO_dict['backscatter'] = bsc_mod
    FO_dict['transmission'] = mod_transm
    FO_dict['level_height'] = z
    FO_dict['time'] = time

    # update FO_dict with earlier derived obs
    FO_dict.update(wxt_obs)

    return FO_dict

def calc_ext_coeff_from_obs(rh_frac, r_v, rho, z, r0, p, N0, m0, eta, ceil_lam, version,
                   time, aer_mode, **kwargs):

    """
    Compute extinction coefficient (aerosol extinction + water vapour extinction)

    :param q_aer: aerosol mass mizing ratio [micrograms kg-1]
    :param rh_frac: relative humidity [ratio/dimensionless]
    :param time: (array of datetimes) datetimes for each timestep
    :return: alpha_a: total extinction coefficient
    :return: beta_a: UNattenuated backscatter

    Most of the function calculates the AEROSOL extinction coefficient and relevent variables (particle extnction
    efficiency). Last part also calculates water vapour extinction coefficient (same as absorption coefficient
    as water vapour scattering is negligable, given the tiny size of vapour gas (think its ~1e-28)

    """

    def read_obs_aer(rh_frac, time, dN_key):
        """
        Read in observed total number for the accum range.
        :param rh_frac:
        :param mod_time:
        :param dN_key: the key in dN for which data to extract
        """

        filedir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/' + \
                  'data/npy/number_distribution/'

        filepath = filedir + 'NK_APS_SMPS_Ntot_Dv_fine_acc_coarse_' + time[0].strftime('%Y') + '_80-800nm.npy'
        data_in = np.load(filepath).flat[0]
        # delete accum range variable
        if 'accum_range' in data_in:
            del data_in['accum_range']
        # ['Ntot_fine', 'Dn_fine', 'Ntot', 'time', 'Ntot_accum', 'Dn_accum']

        t_idx = np.array([eu.nearest(data_in['time'], t)[1] for t in time]) # time index
        t_diff = np.array([eu.nearest(data_in['time'], t)[2] for t in time]) # dt.timedelta() showing how close time is

        # pull out data
        var = data_in[dN_key][t_idx]
        # a = {key: item[t_idx] for key, item in dN.iteritems()} # all data

        # overwrite t_idx locations where t_diff is too high with nans
        # only keep t_idx values where the difference is below 1 hour
        bad = np.array([abs(i.days * 86400 + i.seconds) > 60 * 60 for i in t_diff])

        # for key in dN.iterkeys(): # all data
        #     dN[key][bad] = np.nan

        var[bad] = np.nan

        # As dN is only surface N values, repeat these values to all heights so the shape of obs_out == rh_frac
        obs_out = np.transpose(np.tile(var, (rh_frac.shape[1], 1)))

        return obs_out

    def read_pm10_obs(time):

        pm10_obs = {}
        from_zone = tz.gettz('GMT')
        to_zone = tz.gettz('UTC')

        dir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/data/DEFRA/'
        aer_fname = dir + 'PM10_Hr_NK_DEFRA_AURN_01012014-30052018.csv'

        raw_aer = np.genfromtxt(aer_fname, delimiter=',', skip_header=5, dtype="|S20")

        # sort out times as they are in two columns
        rawtime = [i[0] + ' ' + i[1].replace('24:00:00', '00:00:00') for i in raw_aer]
        time_endHr = np.array([dt.datetime.strptime(i, '%d/%m/%Y %H:%M:%S') for i in rawtime])
        # convert from GMT to UTC and remove the timezone afterwards
        time_endHr = np.array([i.replace(tzinfo=from_zone) for i in time_endHr])  # label time as 'GMT'
        pro_time = np.array([i.astimezone(to_zone) for i in time_endHr])  # find time as 'UTC'
        pro_time = np.array([i.replace(tzinfo=None) for i in pro_time])  # remove 'UTC' timezone identifier

        # extract obs and time together as a dictionary entry for the site.
        data_obs = {'pm_10': np.array([np.nan if i == 'No data' else i for i in raw_aer[:, 2]], dtype=float),
                    'time': pro_time}

        # match the time sample resolution of mod_data

        # find nearest time in rh time
        # pull out ALL the nearest time idxs and differences
        t_idx = np.array([eu.nearest(data_obs['time'], t)[1] for t in time])
        t_diff = np.array([eu.nearest(data_obs['time'], t)[2] for t in time])

        # extract ALL nearest hours data, regardless of time difference
        pm10_obs['pm_10'] = data_obs['pm_10'][t_idx]
        pm10_obs['time'] = [data_obs['time'][i] for i in t_idx]

        # overwrite t_idx locations where t_diff is too high with nans
        # only keep t_idx values where the difference is below 5 minutes
        bad = np.array([abs(i.days * 86400 + i.seconds) > 10 * 60 for i in t_diff])
        pm10_obs['pm_10'][bad] = np.nan

        return pm10_obs

    def calc_r_m_original(r_d, rh_frac, B=FOcon.B_activation_haywood):

        """
        Original method to calculate swollen radii size for the FO in version 0.1 of the aerFO
        :param r_d:
        :param rh_frac:
        :param B: RH activation parameter
        :return:
        """

        # convert units to percentage
        RH_perc = rh_frac * 100.0

        # rm is the mean volume radius. Eqn. 12 in Clark et.al. (2008)
        # "When no activated particles are present an analytic solution for rm" is
        RH_crit_perc = FOcon.RH_crit
        # mask over values less than critical
        RH_ge_RHcrit = np.ma.masked_less(RH_perc, RH_crit_perc)

        # calculate wet mean radius
        # eq 12 - calc rm for RH greater than critical
        r_m = np.ma.ones(rh_frac.shape) - (B / np.ma.log(rh_frac))
        r_m2 = np.ma.power(r_m, 1. / 3.)
        r_m = np.ma.array(r_d) * r_m2

        # set rm as 0 where RH is less than crit
        r_m = np.ma.MaskedArray.filled(r_m, [0.0])
        where_lt_crit = np.where(np.logical_or(RH_perc.data < RH_crit_perc, r_m == 0.0))
        # refill them with r_d
        r_m[where_lt_crit] = r_d[where_lt_crit]

        return r_m

    def get_S_climatology(time, rh_frac, ceil_lam):

        """
        Create the S array from the climatology (month, RH_fraction) given the month and RH
        :param time:
        :param rh_frac:
        :param ceil_lam (int): ceilometer wavelength [nm]
        :return: S (time, height):
        """

        # 1. Read in the data
        filename = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/common_data/Mie/' + \
                   'S_climatology_NK_SMPS_APS_' + str(ceil_lam) + 'nm.npy'

        data = np.load(filename).flat[0]
        S_clim = data['S_climatology']
        S_RH_frac = data['RH_frac']

        # 2. Create S array given the time and RH

        # get height range from rh_frac
        height_idx_range = rh_frac.shape[1]

        # find S array
        S = np.empty(rh_frac.shape)
        S[:] = np.nan
        for t, time_t in enumerate(time):  # time
            # get month idx (e.g. idx for 5th month = 4)
            month_idx = time_t.month - 1
            for h in range(height_idx_range):  # height

                # find RH idx for this month, and put the element into the S array
                _, rh_idx, _ = eu.nearest(S_RH_frac, rh_frac[t, h])
                S[t, h] = S_clim[month_idx, rh_idx]

        return S

    # ---------------------------

    if 'pm10_as_aerosol_input' in kwargs:
        # convert pm10 from [microgram_aer m-3] to [kg_aer kg_air] and name it q_aer_kg_kg, which murk is usually
        #    called. This will pass it through the functions as if it was murk aerosol.
        # then estimate N and r from it
        pm10 = read_pm10_obs(time)
        q_aer_kg_kg = pm10['pm_10'][:, None] / rho * 1.0e-9
        N_aer = N0 * np.power((q_aer_kg_kg / m0), 1.0 - (3.0 * p))
        r_d = r0 * np.power((q_aer_kg_kg / m0), p)

    else:
        # Number concentration [m-3]
        # Read in N_aer from observations and time match to existing time resolution
        if 'obs_N' in kwargs.keys():
            N_aer_cm3 = read_obs_aer(rh_frac, time, 'Ntot_'+aer_mode) # [cm-3]
            N_aer = N_aer_cm3 *1e6 # convert to [m-3]

        # Dry mean radius of bulk aerosol (this radius is the same as the volume mean radius)
        # mean radius (by volume, and it is not the geometric radius)
        if 'obs_r' in kwargs.keys():
            D_d = read_obs_aer(rh_frac, time, 'Dv_'+aer_mode) # Diameter [nm]
            r_d = D_d/2.0 * 1e-9 # radius [m]

    # Geometric mean radius of bulk aerosol [meters]
    #   derived from a linear fit between observed r_d (volume mean) and r_g
    #   used purely for the f_RH LUT in calc_Q_ext_wet()
    if aer_mode == 'accum':
        # (Pearson r = 0.65, p > 0.0)
        # r_g = (0.24621593450654974 * r_d) + 0.03258363072889052 # 80 - 700 nm paper 2
        r_g = (0.122 * r_d) + 4.59e-8 # 80 - 800 nm
        print 'geometric radius calculated for accum mode aerosol'
    elif aer_mode == 'fine':
        # (Pearson r = 0.83, p > 0.0)
        # r_g = (1.05 * r_d) - 0.011 # 80 - 700 nm paper 2
        r_g = (1.0666 * r_d) - 1.10e-8 # 80 - 800 nm
        print 'geometric radius calculated for fine mode aerosol'
    elif aer_mode == 'coarse':
        # (Pearson r = 0.57, p > 0.0 where coarse maximum = 10.0 microns)
        # r_g = (0.1063 * r_d) + 0.4039 # 70 - 700 nm paper 2
        # alternative where coarse maximum is APS maximum bin # 80 - 700 nm paper 2
        # (Pearson r = 0.58, p > 0.0)
        # r_g = (0.0765 * r_d) + 0.4632
        r_g = (0.1211 * r_d) + 3.946e-7 # 80 - 800 nm
        print 'geometric radius calculated for coarse mode aerosol'

    if ('hourly_Q_extdry' in kwargs) & ('hourly_fRH' in kwargs):
        # read in hourly Q_ext,dry and f(RH) to make Q_ext
        Q_ext, Q_ext_dry_matrix, f_RH_matrix = hourly_obs_Q_ext_wet(ceil_lam, r_d, r_g, rh_frac, time)

    else:
        # calculate Q_ext (wet particle extinction efficiency)
        # if doing multiple modes, there would need to be one of these for each, i.e. Q_ext_accum, Q_ext_fine...
        if version <= 1.0:
            # MURK = fixed 3 aerosol types (amm. nit.; amm. sulph.; OC)
            # Q_ext,dry function of dry size only
            # f(RH) function of RH only
            Q_ext, Q_ext_dry_matrix, f_RH_matrix = calc_Q_ext_wet_v1p0(ceil_lam, r_d, rh_frac)
            print 'using old method (v1.0) for Q_ext'
        elif version > 1.0:
            # MURK, monthly varying based on (amm. nit.; amm. sulph.; OC; BC; sea salt)
            # Q_ext,dry function of dry size, month
            # f(RH) function of RH, geometric mean of dry particle distribution, month
            Q_ext, Q_ext_dry_matrix, f_RH_matrix = calc_Q_ext_wet(ceil_lam, r_d, r_g, rh_frac, time)

    # Calculate extinction coefficient
    # eqns. 17-18 in Clark et.al. (2008)
    if version == 0.1:
        # v0.1 original aerFO version - now outdated
        # calculate swollen radii and extinction coefficient using it
        r_m = calc_r_m_original(r_d, rh_frac)
        aer_ext_coeff = (eta * FOcon.Q_ext_aer) * np.pi * N_aer * np.power(r_m, 2)
        print 'Using old version 0.1 approach to swell particles'

    # v0.2 - use dry radii and an extinction enhancement factor, to include the effect of hygroscopic growth on optical
    #   properties
    # If multiple modes are used (e.g. fine, accum, coarse), aer_ext_coeff needs to be calculated for each
    #   i.e. aer_ext_coeff_accum, then summed at the end to make one aer_ext_coeff, with one S used to get the backscatter
    elif version >= 0.2:
        aer_ext_coeff = eta * Q_ext * np.pi * N_aer * np.power(r_d, 2)

    # store all elements into a dictionary for output and diagnostics
    ext_coeff_dict = {'aer_ext_coeff': aer_ext_coeff,
                'r_d': r_d,
                'r_g': r_g,
                'N': N_aer,
                'Q_ext': Q_ext,
                'Q_ext_dry': Q_ext_dry_matrix,
                'f_RH': f_RH_matrix}

    return ext_coeff_dict

def calc_att_backscatter_from_obs(rh_frac, r_v, rho, z, r0, p, N0, m0, eta, ceil_lam, version,
                   time, aer_modes, **kwargs):

    """
    Compute extinction coefficient (aerosol extinction + water vapour extinction)

    :param q_aer: aerosol mass mizing ratio [micrograms kg-1]
    :param rh_frac: relative humidity [ratio/dimensionless]
    :param time: (array of datetimes) datetimes for each timestep
    :return: alpha_a: total extinction coefficient
    :return: beta_a: UNattenuated backscatter

    Most of the function calculates the AEROSOL extinction coefficient and relevent variables (particle extnction
    efficiency). Last part also calculates water vapour extinction coefficient (same as absorption coefficient
    as water vapour scattering is negligable, given the tiny size of vapour gas (think its ~1e-28)

    """

    def read_obs_aer(rh_frac, time, dN_key):
        """
        Read in observed total number for the accum range.
        :param rh_frac:
        :param mod_time:
        :param dN_key: the key in dN for which data to extract
        """

        filedir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/' + \
                  'data/npy/number_distribution/'

        filepath = filedir + 'NK_APS_SMPS_Ntot_Dv_fine_acc_coarse_' + time[0].strftime('%Y') + '_80-700nm.npy'
        data_in = np.load(filepath).flat[0]
        # delete accum range variable
        if 'accum_range' in data_in:
            del data_in['accum_range']
        # ['Ntot_fine', 'Dn_fine', 'Ntot', 'time', 'Ntot_accum', 'Dn_accum']

        t_idx = np.array([eu.nearest(data_in['time'], t)[1] for t in time]) # time index
        t_diff = np.array([eu.nearest(data_in['time'], t)[2] for t in time]) # dt.timedelta() showing how close time is

        # pull out data
        var = data_in[dN_key][t_idx]
        # a = {key: item[t_idx] for key, item in dN.iteritems()} # all data

        # overwrite t_idx locations where t_diff is too high with nans
        # only keep t_idx values where the difference is below 1 hour
        bad = np.array([abs(i.days * 86400 + i.seconds) > 60 * 60 for i in t_diff])

        # for key in dN.iterkeys(): # all data
        #     dN[key][bad] = np.nan

        var[bad] = np.nan

        # As dN is only surface N values, repeat these values to all heights so the shape of obs_out == rh_frac
        obs_out = np.transpose(np.tile(var, (rh_frac.shape[1], 1)))

        return obs_out

    def calc_r_m_original(r_d, rh_frac, B=FOcon.B_activation_haywood):

        """
        Original method to calculate swollen radii size for the FO in version 0.1 of the aerFO
        :param r_d:
        :param rh_frac:
        :param B: RH activation parameter
        :return:
        """

        # convert units to percentage
        RH_perc = rh_frac * 100.0

        # rm is the mean volume radius. Eqn. 12 in Clark et.al. (2008)
        # "When no activated particles are present an analytic solution for rm" is
        RH_crit_perc = FOcon.RH_crit
        # mask over values less than critical
        RH_ge_RHcrit = np.ma.masked_less(RH_perc, RH_crit_perc)

        # calculate wet mean radius
        # eq 12 - calc rm for RH greater than critical
        r_m = np.ma.ones(rh_frac.shape) - (B / np.ma.log(rh_frac))
        r_m2 = np.ma.power(r_m, 1. / 3.)
        r_m = np.ma.array(r_d) * r_m2

        # set rm as 0 where RH is less than crit
        r_m = np.ma.MaskedArray.filled(r_m, [0.0])
        where_lt_crit = np.where(np.logical_or(RH_perc.data < RH_crit_perc, r_m == 0.0))
        # refill them with r_d
        r_m[where_lt_crit] = r_d[where_lt_crit]

        return r_m

    def get_S_climatology(time, rh_frac, ceil_lam):

        """
        Create the S array from the climatology (month, RH_fraction) given the month and RH
        :param time:
        :param rh_frac:
        :param ceil_lam (int): ceilometer wavelength [nm]
        :return: S (time, height):
        """

        # 1. Read in the data
        filename = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/common_data/Mie/' + \
                   'S_climatology_NK_SMPS_APS_' + str(ceil_lam) + 'nm.npy'

        data = np.load(filename).flat[0]
        S_clim = data['S_climatology']
        S_RH_frac = data['RH_frac']

        # 2. Create S array given the time and RH

        # get height range from rh_frac
        height_idx_range = rh_frac.shape[1]

        # find S array
        S = np.empty(rh_frac.shape)
        S[:] = np.nan
        for t, time_t in enumerate(time):  # time
            # get month idx (e.g. idx for 5th month = 4)
            month_idx = time_t.month - 1
            for h in range(height_idx_range):  # height

                # find RH idx for this month, and put the element into the S array
                _, rh_idx, _ = eu.nearest(S_RH_frac, rh_frac[t, h])
                S[t, h] = S_clim[month_idx, rh_idx]

        return S

    def extract_S_hourly(FO_dict, time, ceil_lam):

        """
        Read in and extract S calculated for that hour from calc_lidar_ratio_general.py on the linux system
        :return: S .shape(time, height): lidar ratio
        """

        # set up S array
        S = np.empty(FO_dict[aer_mode_i]['r_d'].shape)
        S[:] = np.nan

        # Read in the appropriate yearly file data
        Sfilename = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/data/npy/' \
                    'S_timeseries/NK_SMPS_APS_PM10_withSoot_' + time[0].strftime('%Y') + '_' + str(
            ceil_lam) + 'nm.npy'
        data = np.load(Sfilename).flat[0]
        S_time = data['met']['time']
        S_timeseries = data['optics']['S']

        # fill S array
        for t, time_t in enumerate(time):  # time
            _, t_idx, diff = eu.nearest(S_time, time_t)
            # if the difference is less than an hour, extract the value (so discard differences exactly equal to 1 hour)
            if diff.total_seconds() < 60 * 60:
                S[t, :] = S_timeseries[t_idx]

        return S


    # ---------------------------

    # prepare the FO_dict to store all the outputted variables together
    FO_dict={}

    # calculate extinction coefficients for each aerosol mode separately
    for aer_mode_i in aer_modes:
        FO_dict[aer_mode_i] = calc_ext_coeff_from_obs(rh_frac, r_v, rho, z, r0, p, N0, m0, eta, ceil_lam, version,
                                time, aer_mode_i, **kwargs)

    # calculate the water vapour extinction coefficient
    # T = 16.85 degC, q = 0.01 kg kg-1; p = 1100 hPa
    # wv_ext_coeff = mass_abs * mod_rho * mod_r_v

    if ceil_lam == 905:
        # mass absorption of water vapour [m2 kg-1] for water vapour extinction coefficient
        # script to calculate mass aborption = htfrtc_optprop_gas_plot_elliott.py
        # gaussian weighted average (mean = 905, FWHM = 4) = 0.016709242714125036 # (current) should be used for CL31 (kotthaus et al., 2016)
        # gaussian weighted average (mean = 905, FWHM = 8) = 0.024222946249630242 # (test) test sensitivity to FWHM
        # gaussian weighted average (mean = 900, FWHM = 4) = 0.037273493204864103 # (highest wv abs for a central wavelength between 895 - 915)
        wv_ext_coeff = 0.016709242714125036 * rho * r_v
    else:
        raise ValueError('ceilometer wavelength != 905 nm, need to calculate a new gaussian average to \n'
                         'calculate water vapour extinction coefficient for this new wavelength!')

    # # total extinction coefficient
    # alpha_a = aer_ext_coeff + wv_ext_coeff

    # total aerosol extinction coefficient
    aer_ext_coeff_tot = np.sum(np.array([FO_dict[key]['aer_ext_coeff'] for key in aer_modes]), axis=0)
    # total extinction
    alpha_a = aer_ext_coeff_tot + wv_ext_coeff

    # Get lidar ratio (S)
    if 'use_S_hourly' in kwargs:
        # use the calculated timeseries of S
        S = extract_S_hourly(FO_dict, time, ceil_lam)
    elif 'use_S_constant' in kwargs:
        print 'using a constant S of ' + str(kwargs['use_S_constant'])
        S = kwargs['use_S_constant']
    elif 'cheat_S_param' in kwargs:
        print 'using the cheat parameterisation of S!'
        S = (-0.02214879231039488 * (rh_frac*100.0)) + 54.800201651558169

    else:
        # Use parameterised S instead -> either constant or S(RH)
        if version <= 1.0:
            # Constant lidar ratio = 60 sr (continental aerosol; Warren et al. 2018)
            S = FOcon.LidarRatio['Aerosol']

        elif version >= 1.1:
            # Read in from look up table (monthly varying, for either an urban or rural site)
            # hold it constant for now, until the climatology has been made
            S = get_S_climatology(time, rh_frac, ceil_lam)

    # Calculate backscatter using a constant lidar ratio
    # ratio between PARTICLE extinction and backscatter coefficient (not total extinction!).
    beta_a = aer_ext_coeff_tot / S

    # store all elements into a dictionary for output and diagnostics
    export_vars = {'unnatenuated_backscatter': beta_a,
                'alpha_a': alpha_a,
                'aer_ext_coeff_tot': aer_ext_coeff_tot,
                'wv_ext_coeff': wv_ext_coeff,
                'S': S}

    FO_dict.update(export_vars)

    # FO_dict = {'beta_a': beta_a,
    #             'alpha_a': alpha_a,
    #             'aer_ext_coeff_tot': aer_ext_coeff_tot,
    #             'wv_ext_coeff': wv_ext_coeff,
    #             'r_d': r_d,
    #             'r_g':r_g,
    #             'N': N_aer,
    #             'Q_ext': Q_ext,
    #             'Q_ext_dry': Q_ext_dry_matrix,
    #             'f_RH': f_RH_matrix,
    #             'S': S}

    return FO_dict

# Transmission
def compute_transmission(alpha_extinction, dz):
    import numpy as np

    ss = np.shape(alpha_extinction)
    # print "1. compute_transmission: size = ", ss[0],ss[1]

    integral_alpha = np.empty_like(alpha_extinction)
    transmission = np.empty_like(alpha_extinction)
    # print np.shape(integral_alpha), np.shape(transmission)

    # print  "2. compute_transmission: ",
    # np.shape(alpha_extinction),np.shape(dz)

    integral_alpha = np.cumsum(alpha_extinction * dz[0:len(dz)], axis=1)

    # print  "3. compute_transmission: ",alpha_extinction*dz[0:len(dz)],integral_alpha
    # transmission =  integral_alpha
    transmission = np.exp(-2.0 * integral_alpha)

    return integral_alpha, transmission


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1.1 Q_ext calculation (Q_ext,dry * f(RH))

def hourly_obs_Q_ext_wet(ceil_lam, r_d, r_g, rh_frac, mod_time):
    """

    Calculate Q_ext_wet using Q_ext_dry and f(RH) for current wavelength
    Q_ext,dry and f_RH are monthly varying based on obs at NK and CH for urban and rural site default settings
    respectively. f_RH also varies with geometric radius.

    EW 15/08/18
    :param ceil_lam:
    :param r_d: dry mean volume radius [meters] - needed for Q_ext,dry LUT
    :param r_g: dry geometric mean radius [meters] - needed for f_RH LUT
    :param rh_frac: RH [fraction]
    :param mod_time (array of datetimes) datetimes for the timesteps
    :return: Q_ext, Q_ext_dry_matrix, f_RH_matrix
    """

    from ellUtils import nearest, netCDF_read

    # Reading functions
    def read_hourly_f_RH(mod_time, ceil_lam):

        """
        Read in the hourly f_RH data from netCDF file for all aerosols
        EW 21/02/17

        :param mod_time (array of datetimes) datetimes for the timesteps
        :param ceil_lam: (int) ceilometer wavelength [nm]
        :return: data = {RH:... f_RH:...}

        """

        # file name and path
        miedir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/common_data/Mie/'
        filename = 'monthly_f(RH)_NK_'+str(ceil_lam)+'nm.nc'

        # read data
        # f_RH = netCDF_read(miedir + filename, vars=['Relative Humidity', 'f(RH) MURK', 'radii_range_nm'])
        f_RH = netCDF_read(miedir + filename)
        return f_RH

    def read_hourly_Q_ext_dry(mod_time, ceil_lam):

        """
        Read in the Q_ext for dry aerosol.
        EW 21/02/17

        :param mod_time (array of datetimes) datetimes for the timesteps
        :param ceil_lam: (int) ceilometer wavelength [nm]
        :return: Q_ext_dry = {radius:... Q_ext_dry:...}

        """

        miedir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/common_data/Mie/'
        filename = 'NK_all_aerosol_Q_ext_dry_' + str(ceil_lam) + 'nm.npy'

        Q_ext_dry = np.load(miedir + filename).flat[0]

        return Q_ext_dry

    def read_hourly_rel_vol(mod_time, ceil_lam):

        """
        Read in the relative volume data from netCDF file for all aerosols
        EW 15/08/18

        :param mod_time (array of datetimes) datetimes for the timesteps
        :param ceil_lam: (int) ceilometer wavelength [nm]
        :return: rel_vol (dict): relative volume of each main aerosol component and a time array for time matching

        """

        # file name and path
        miedir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/common_data/Mie/'
        filename = 'NK_hourly_aerosol_relative_volume.npy'

        # read data
        # f_RH = netCDF_read(miedir + filename, vars=['Relative Humidity', 'f(RH) MURK', 'radii_range_nm'])
        rel_vol = np.load(miedir + filename).flat[0]
        return rel_vol

    # ---------------------------

    # convert geometric radius to nm to find f(RH)
    r_g_nm = r_g * 1.0e9

    # height idx range of r_d and RH
    height_idx_range = r_d.shape[1]

    # read in Q_ext,dry and f(RH) data
    f_RH = read_hourly_f_RH(mod_time, ceil_lam)

    # Q_ext,dry (radii)
    # still needs the relative volumes to weight each component though.
    Q_ext_dry = read_hourly_Q_ext_dry(mod_time, ceil_lam)

    # Read in the relative volume of each species (radii, RH)
    rel_vol = read_hourly_rel_vol(mod_time, ceil_lam)

    # create matric of Q_ext_dry based on r_d
    Q_ext_dry_matrix = np.empty(r_d.shape)
    Q_ext_dry_matrix[:] = np.nan
    f_RH_matrix = np.empty(r_d.shape)
    f_RH_matrix[:] = np.nan

    # find Q_ext dry, given the dry radius matrix
    # find f(RH), given the RH fraction matric
    for t, time_t in enumerate(mod_time): # time

        _, rel_vol_t_idx, _ = nearest(Q_ext_dry['time'], time_t)

        for h in range(height_idx_range): # height


            # 1. Q_ext_dry - volume mixing method as the Q_ext_dry['Q_dry_aer'] values are already calculated for the
            #   correct r_d (as otherwise the matricies would be enormous (1544, time))
            _, Q_ext_r_d_idx, _ = nearest(Q_ext_dry['r_v'], r_d[t, h]) # [m]

            Q_ext_dry_matrix[t, h] = \
                (Q_ext_dry['Q_dry_aer']['(NH4)2SO4'][Q_ext_r_d_idx] * rel_vol['(NH4)2SO4'][rel_vol_t_idx]) + \
                (Q_ext_dry['Q_dry_aer']['NH4NO3'][Q_ext_r_d_idx] * rel_vol['NH4NO3'][rel_vol_t_idx]) + \
                (Q_ext_dry['Q_dry_aer']['CORG'][Q_ext_r_d_idx] * rel_vol['CORG'][rel_vol_t_idx]) + \
                (Q_ext_dry['Q_dry_aer']['CBLK'][Q_ext_r_d_idx] * rel_vol['CBLK'][rel_vol_t_idx]) + \
                (Q_ext_dry['Q_dry_aer']['NaCl'][Q_ext_r_d_idx] * rel_vol['NaCl'][rel_vol_t_idx])

            # ------------

            # 2. f(RH) (has it's own r_idx that is in units [nm])
            # LUT uses r_g (geometric) [nm] ToDo should change this to meters...
            _, r_f_RH_idx, _ = nearest(f_RH['radii_range'], r_g_nm[t, h])
            _, rh_idx, _ = nearest(f_RH['RH'], rh_frac[t, h])
            # f_RH_matrix[t, h] = f_RH['f(RH) MURK'][month_idx, r_f_RH_idx, rh_idx]

            f_RH_matrix[t, h] = \
                (f_RH['f(RH) (NH4)2SO4'][r_f_RH_idx, rh_idx] * rel_vol['(NH4)2SO4'][rel_vol_t_idx]) + \
                (f_RH['f(RH) NH4NO3'][r_f_RH_idx, rh_idx] * rel_vol['NH4NO3'][rel_vol_t_idx]) + \
                (f_RH['f(RH) CORG'][r_f_RH_idx, rh_idx] * rel_vol['CORG'][rel_vol_t_idx]) + \
                (f_RH['f(RH) CBLK'][r_f_RH_idx, rh_idx] * rel_vol['CBLK'][rel_vol_t_idx]) + \
                (f_RH['f(RH) NaCl'][r_f_RH_idx, rh_idx] * rel_vol['NaCl'][rel_vol_t_idx])

    # calculate Q_ext_wet
    Q_ext = Q_ext_dry_matrix * f_RH_matrix

    return Q_ext, Q_ext_dry_matrix, f_RH_matrix

def calc_Q_ext_wet(ceil_lam, r_d, r_g, rh_frac, mod_time):
    """

    Calculate Q_ext_wet using Q_ext_dry and f(RH) for current wavelength
    Q_ext,dry and f_RH are monthly varying based on obs at NK and CH for urban and rural site default settings
    respectively. f_RH also varies with geometric radius.

    EW 23/02/17
    :param ceil_lam:
    :param r_d: dry mean volume radius [meters] - needed for Q_ext,dry LUT
    :param r_g: dry geometric mean radius [meters] - needed for f_RH LUT
    :param rh_frac: RH [fraction]
    :param mod_time (array of datetimes) datetimes for the timesteps
    :return: Q_ext, Q_ext_dry_matrix, f_RH_matrix
    """

    from ellUtils import nearest, netCDF_read

    # Reading functions
    def read_f_RH(mod_time, ceil_lam):

        """
        Read in the f_RH data from netCDF file
        EW 21/02/17

        :param mod_time (array of datetimes) datetimes for the timesteps
        :param ceil_lam: (int) ceilometer wavelength [nm]
        :return: data = {RH:... f_RH:...}

        """

        # file name and path
        miedir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/common_data/Mie/'
        filename = 'monthly_f(RH)_NK_'+str(ceil_lam)+'nm.nc'

        # read data
        # f_RH = netCDF_read(miedir + filename, vars=['Relative Humidity', 'f(RH) MURK', 'radii_range_nm'])
        f_RH = netCDF_read(miedir + filename, vars=['RH', 'f(RH) MURK', 'radii_range'])
        return f_RH

    def read_Q_ext_dry(mod_time, ceil_lam):

        """
        Read in the Q_ext for dry murk.
        EW 21/02/17

        :param mod_time (array of datetimes) datetimes for the timesteps
        :param ceil_lam: (int) ceilometer wavelength [nm]
        :return: Q_ext_dry = {radius:... Q_ext_dry:...}

        """

        miedir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/common_data/Mie/'
        filename = 'urban_monthly_Q_ext_dry_' + str(ceil_lam) + 'nm.csv'

        raw = np.loadtxt(miedir + filename, delimiter=',')

        # format data into a dictionary
        Q_ext_dry = {'radius_m': raw[:, 0],
                     'Q_ext_dry': raw[:, 1:]} # Q_ext_dry['Q_ext_dry'].shape(radii, month)


        return Q_ext_dry

    # ---------------------------

    # cronvert geometric radius to nm to find f(RH)
    r_g_nm = r_g * 1.0e9

    # height idx range of r_d and RH
    height_idx_range = r_d.shape[1]

    # read in Q_ext_dry and f(RH) look up tables
    f_RH = read_f_RH(mod_time, ceil_lam) # ['f_RH MURK'].shape(month, radii, RH)
    Q_ext_dry = read_Q_ext_dry(mod_time, ceil_lam) #.shape(radii, month)

    #print 'testing! reducing murk f_RH by 1/3!'
    #f_RH['f(RH) MURK'] *=0.66

    # create matric of Q_ext_dry based on r_d
    Q_ext_dry_matrix = np.empty(r_d.shape)
    Q_ext_dry_matrix[:] = np.nan
    f_RH_matrix = np.empty(r_d.shape)
    f_RH_matrix[:] = np.nan

    # find Q_ext dry, given the dry radius matrix
    # find f(RH), given the RH fraction matric
    for t, time_t in enumerate(mod_time): # time
        # get month idx (e.g. idx for 5th month = 4)
        month_idx = time_t.month - 1
        for h in range(height_idx_range): # height

            # Q_ext_dry
            # LUT uses r_d (volume) [meters]
            _, r_Q_idx, _ = nearest(Q_ext_dry['radius_m'], r_d[t, h])
            Q_ext_dry_matrix[t, h] = Q_ext_dry['Q_ext_dry'][r_Q_idx, month_idx]

            # f(RH) (has it's own r_idx that is in units [nm])
            # LUT uses r_g (geometric) [nm] ToDo should change this to meters...
            _, r_f_RH_idx, _ = nearest(f_RH['radii_range'], r_g_nm[t, h])
            _, rh_idx, _ = nearest(f_RH['RH'], rh_frac[t, h])
            f_RH_matrix[t, h] = f_RH['f(RH) MURK'][month_idx, r_f_RH_idx, rh_idx]

    # calculate Q_ext_wet
    Q_ext = Q_ext_dry_matrix * f_RH_matrix

    return Q_ext, Q_ext_dry_matrix, f_RH_matrix

def calc_Q_ext_wet_v1p0(ceil_lam, r_d, rh_frac):
    """
    Calculate Q_ext_wet using Q_ext_dry and f(RH) for current wavelength

	version 0.2 where MURK was calculated from 3 fixed aerosol species. Q_ext,dry was a function of
	radii only. f_RH was a function of RH only. Version used in paper 1.

    EW 23/02/17
    :param ceil_lam:
    :param r_d:
    :param RH:
    :return:
    """

    from ellUtils import nearest

    def read_f_RH(ceil_lam):
        """
        Read in the f_RH data from csv
        EW 21/02/17

        :param filename:
        :return: data = {RH:... f_RH:...}

        filename must be in the form of 'calculated_ext_f(RH)_[ceil_lambda]nm.csv'
        """

        # temp file name
        miedir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/clearFO/data/Mie/'
        if ceil_lam == 910:
            filename = 'sp_ew_ceil_guass_903-907_ext_f(RH)_903-907nm.csv'
        elif ceil_lam == 1064:
            filename = 'sp_ew_ceil_guass_1062-1066_ext_f(RH)_1062-1066nm.csv'
        else:
            raise ValueError('ceil lambda was not 910 or 1064 nm: not sure which f(RH) file to use!')

        # filename = 'calculated_ext_f(RH)_' + str(ceil_lam) + 'nm.csv'


        # read data
        raw = np.loadtxt(miedir + filename, delimiter=',')

        f_RH = {'RH': raw[:, 0],
                'f_RH': raw[:, 1]}

        return f_RH

    def read_Q_dry_ext(ceil_lam):
        """
        Read in the Q_ext for dry murk.
        EW 21/02/17

        :param filename:
        :param lam:
        :return: Q_ext_dry = {radius:... Q_ext_dry:...}

        Requres the wavelength to be passed, just so in the future, the 910 nm file is not incorrectly used by mistake when
        it should use the file for another wavelength.
        """

        miedir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/clearFO/data/Mie/'
        filename = 'calculated_Q_ext_' + str(ceil_lam) + 'nm.csv'

        raw = np.loadtxt(miedir + filename, delimiter=',')

        Q_ext_dry = {'radius': raw[:, 0],
                     'Q_ext': raw[:, 1]}

        return Q_ext_dry

    RH_factor = 0.01  # Relative Humidity in 0.38 not 38%

    # calculate Q_ext_wet
    f_RH = read_f_RH(ceil_lam)
    Q_ext_dry = read_Q_dry_ext(ceil_lam)

    # create matric of Q_ext_dry based on r_d
    Q_ext_dry_matrix = np.empty(r_d.shape)
    Q_ext_dry_matrix[:] = np.nan
    f_RH_matrix = np.empty(r_d.shape)
    f_RH_matrix[:] = np.nan

    # find Q_ext dry, given the dry radius matrix
    for i in range(r_d.shape[0]):
        for j in range(r_d.shape[1]):
            idx = nearest(Q_ext_dry['radius'], r_d[i, j])[1]
            Q_ext_dry_matrix[i, j] = Q_ext_dry['Q_ext'][idx]

    for i in range(rh_frac.shape[0]):
        for j in range(rh_frac.shape[1]):
            idx = nearest(f_RH['RH'], rh_frac[i, j])[1]
            f_RH_matrix[i, j] = f_RH['f_RH'][idx]

    # calculate Q_ext_wet
    Q = Q_ext_dry_matrix * f_RH_matrix

    return Q, Q_ext_dry_matrix, f_RH_matrix

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2. Reading stuff in

# 2.1 Metadata

def read_ceil_metadata(datadir, loc_filename='CeilsCSV.csv'):

    """
    Read in ceil metadata (lon, lat) into a dictionary
    :param datadir:
    :return:
    """

    # read in the ceil locations
    loc_fname = datadir + loc_filename

    ceil_rawmeta = eu.csv_read(loc_fname)

    # convert into dictionary - [lon, lat]
    # skip the headers
    ceil_metadata = {}
    for i in ceil_rawmeta[1:]:
        ceil_metadata[i[1]] = [float(i[3]), float(i[4])]

    return ceil_metadata

# -----------------
# 2.2 Model data

def get_time_idx_forecast(mod_all_data, day):

    """find the idx of all data on the same day as 'day' """

    _, startIdx_time, _ = eu.nearest(mod_all_data['time'], day)
    _, endIdx_time, _ = eu.nearest(mod_all_data['time'], (day + dt.timedelta(hours=24)))
    range_time = np.arange(startIdx_time, endIdx_time + 1)

    return range_time

def read_all_mod_data(modDatadir, day, Z):

    """ Read all the model data """

    # date string (forecast at Z starting on the previous day)
    dateStr = (day + dt.timedelta(hours=-24)).strftime('%Y%m%d')

    # temp filename for the day
    mod_filename = 'extract_prodm_op_ukv_' + dateStr + '_' + str(Z) + '_full.nc'

    # concatenate the names
    mod_fname = modDatadir + mod_filename

    # Read in the modelled data for London
    mod_all_data = eu.netCDF_read(mod_fname)

    # set timezone as UTC
    #mod_all_data['time'] = np.array([i.replace(tzinfo=tz.gettz('UTC')) for i in mod_all_data['time']])

    return mod_all_data

def get_site_loc_idx_in_mod(mod_all_data, loc, model_type, res):

        """ Get the lon and lat idx positions for the instrument"""

        # give full names to play safe
        lat = loc[0]
        lon = loc[1]

        # ToDo - projection may well change if not UKV
        if model_type == 'UKV':
            # create the rotated pole system from the UKV metadata
            rot_pole1 = iris.coord_systems.RotatedGeogCS(37.5, 177.5, ellipsoid=iris.coord_systems.GeogCS(
                6371229.0)).as_cartopy_crs()

        ll = ccrs.Geodetic()
        target_xy1 = rot_pole1.transform_point(lat, lon, ll)

        # define half grid spacing
        if res == '2p2km':
            delta = 0.01
        elif res == '0p5km':
            delta = 0.0022
        elif res == '0p2km':
            delta = 0.0009
        elif res == '0p2km':  # guess using 0p2km and that 1p5 is just OVER 1/3 of 0p5
            delta = 0.00044
        elif res == '1p5km':
            delta = 0.5 * 0.0135

        # get idx location in rotated space
        # define rotated lat and lon
        # don't know why some have 360 and others do not
        # for some reason unknown to science, 100 m does not want + 360...
        if (res == '2p2km') or (res == '0p5km') or (res == '1p5km'):
            glon = target_xy1[0] + 360
        else:
            glon = target_xy1[0]

        glat = target_xy1[1]

        # idx match with the lat and lon from the model
        mod_glon, idx_lon, diff_lon = eu.nearest(mod_all_data['longitude'], glon)
        mod_glat, idx_lat, diff_lat = eu.nearest(mod_all_data['latitude'], glat)

        return idx_lon, idx_lat, glon, glat

# main one that does the complete read in for all ceil sites
def mod_site_extract_calc(day, ceil_metadata, modDatadir, model_type, res, ceil_lam,
                          fullForecast=False, Z=21, allvars=False, m_coeff=1.0, rh_coeff=1.0,
                          version=1.1, **kwargs):

    """
    Extract MURK aerosol and calculate RH for each of the sites in the ceil metadata
    Can retain the full forecast, or just for the day

    :param day: (datetime)
    :param ceil_metadata: (dict) = {site: height}
    :param modDatadir:
    :param model_type: (str) e.g. UKV
    :param res: model resolution
    :param windVars: (bool) return u, v and w wind components
    :param Z: forecast start time, defaulted to 21
    :param version: version to run. Older versions use the original phyiscal hygroscopic growth
    :param m_coeff: (float) coefficient to rescale m if desired. Set to 1.0 normally so m is not rescaled [fraction].
                Either a single value or an array of the same shape to mod_aer
    :param rh_coeff: coefficient to rescale RH [fraction], similar to m_coeff

    kwargs
    :param fullForecast: (bool; default: False) Should the entire forecast be processed or just the main day?
    :param allvars: (bool; default: False) return all variables that were calculated in estimating the attenuated bacskcatter?
    :param obs_rh: (bool; default: False)
    :param obs_N: (bool; default: False) Use observed Number conc. from observations instead of estimating from aerosol mass
    :param obs_r: (bool; default: False) Use observed r from observations instead of estimating from aerosol mass
    :param water_vapour_absorption_from_obs: (bool; default: False) calculate the water vapour absorption from observations
                instead of the NWP model output.
    :param version: (float scaler; default: 1.1) aerFO version number:
                0.1 - original aerFO using constant Q = 2, and swelling the radii for calculating the extinction coeff.
                    No aerosol species differentiation, taken straight from Chalrton-Perez et al. 2016.
                0.2 - developed aerFO using Q = Q_ext,dry * f(RH): fixed aerosol proportions of amm. sulphate,
                    amm. nitrate and organic carbon (aged fossil fuel OC) from Haywood et al. 2008 flights.
                1.0 - same as v0.2 and the version used in paper 1.
    (Current)   1.1 - Q_ext,dry and f(RH) now monthly varying and include black carbon and salt form observations.
                    Separate default versions of Q_ext,dry, f(RH) and S for urban (based on North Kensington (NK),
                    London) and rural (based on Chilbolton (CH), UK). f(RH) now varies with radius size too. geometric
                    standard deviation for CH and NK derived from observations to help calculated Q_ext,dry and f(RH).

    :return: mod_data: (dictionary) different aerFO outputs including forward modelled attenuated backscatter.
    """

    # if 'nan_policy' in kwargs.keys():

    def calc_RH(mod_T_ceilsius, mod_q, mod_r_v):

        """
        # calculate relative humidity
        # Thermal Physics of the Atmosphere - Maarten's book.
        """

        # -----------
        # saturated vapour pressure (hPa, then Pa) - Teten's eq 5.18, pp. 98
        e_s_hpa = 6.112 * (np.exp((17.67 * mod_T_celsius) / (mod_T_celsius + 243.5)))
        e_s = e_s_hpa * 100

        # mass mixing ratio of water vapour pp. 100
        # now calculated outside function for use in water vapour absorption coefficient
        # r_v = mod_q / (1 - mod_q)

        # mass mixing ratio of water vapour at saturation eq 5.22, pp. 100
        r_vs = 0.622 * (e_s / mod_p)

        # relative humidity (variant of eq 5.24, pp 101)
        # rescale rh if requested
        mod_rh = mod_r_v / r_vs

        return mod_rh

    # Read in the modelled data for London
    mod_all_data = read_all_mod_data(modDatadir, day, Z)

    # define mod_data array
    mod_data = {}

    for site, loc in ceil_metadata.iteritems():

        # define dictionary for the site
        mod_data[site] = {}

        # get the lon and lat idx for the instrument
        idx_lon, idx_lat, _, _ = get_site_loc_idx_in_mod(mod_all_data, loc, model_type, res)

        # Time extraction - pull out just the main day's data or the full forecast?
        if fullForecast == False:

            # only extract data for the main day
            range_time = get_time_idx_forecast(mod_all_data, day)
        else:

            range_time = np.arange(len(mod_all_data['time']))

        # extract the variables for that location
        # rescale m by m_coeff. m_coeff = 1.0 by default so normally it is not rescaled
        mod_aer = mod_all_data['aerosol_for_visibility'][range_time, :, idx_lat, idx_lon] * m_coeff
        mod_q = mod_all_data['specific_humidity'][range_time, :, idx_lat, idx_lon]
        mod_p = mod_all_data['air_pressure'][range_time, :, idx_lat, idx_lon]
        mod_T = mod_all_data['air_temperature'][range_time, :, idx_lat, idx_lon]
        mod_h = mod_all_data['level_height']
        mod_time = np.array(mod_all_data['time'])[range_time] # should probably be done in the eu.netCDF_read function
        mod_u = mod_all_data['x_wind'][range_time, :, idx_lat, idx_lon]
        mod_v = mod_all_data['y_wind'][range_time, :, idx_lat, idx_lon]
        mod_w = mod_all_data['upward_air_velocity'][range_time, :, idx_lat, idx_lon]

        # extract Q_H (sensible heat flux) if it is there
        if 'boundary_layer_sensible_heat_flux' in mod_all_data:
            mod_Q_H = mod_all_data['boundary_layer_sensible_heat_flux'][range_time, :, idx_lat, idx_lon]


        # Calculate some variables from those read in

        # convert temperature to degree C
        mod_T_celsius = mod_T - 273.15
        # calculate water vapour mixing ratio [kg kg-1]: page 100 - Thermal physics of the atmosphere
        mod_r_v = mod_q / (1 - mod_q)
        # calculate virtual temperature eq. 1.31 (Ambaumm Notes), state gas constant for dry air
        mod_Tv = (1 + (0.61 * mod_q)) * mod_T
        # density of air [kg m-3] #ToDo gas constant is for dry air, should be for moist (small difference)
        mod_rho = mod_p / (286.9 * mod_Tv)

        # calculate RH [fraction 0-1] from temp [degC] specific humidity (q) [kg kg-1] ...
        # and water vapour mixing ratio (r) [kg kg-1]
        # Temp [degC] for use in the impirical equation
        # scale RH by rh_coeff. Default = 1.0 (no scaling)
        mod_rh_frac = calc_RH(mod_T_celsius, mod_q, mod_r_v) * rh_coeff

        # Replace variables with observations?
        # Use NWP model or observed RH?
        if 'obs_RH' not in kwargs:
            rh_frac = mod_rh_frac
            r_v = mod_r_v
        else: 
            if kwargs['obs_RH'] == True:
                # Read in observed RH data to take the place of model RH data
                wxt_obs = read_wxt_obs(day, mod_time, mod_rh_frac)
                obs_rh_frac = wxt_obs['RH_frac']
                rh_frac = obs_rh_frac
                r_v = wxt_obs['r_v']
            else:
                raise ValueError('obs_RH not set to True or False! Value must be a booleon')


        # prcoess forward modelled backscatter for each site
        FO_dict = forward_operator(mod_aer, rh_frac, r_v, mod_rho, mod_h, ceil_lam, version, mod_time, **kwargs)
        mod_data[site]['backscatter'] = FO_dict['bsc_attenuated']

        # store MURK aerosol, RH and heights in mod_data dictionary
        mod_data[site]['RH'] = rh_frac
        mod_data[site]['aerosol_for_visibility'] = mod_aer
        mod_data[site]['level_height'] = mod_h
        mod_data[site]['time'] = mod_time


        # check whether to return all the prognostic variables too
        # returns all variables, not just the main ones like attenuated backscatter, RH, time and height
        if allvars == True:

            # add all the vars in FO_dict to the mod_data dictionary
            mod_data[site].update(FO_dict)
            # del mod_data[site]['backscatter'] # do not need a copy of attenuated backscatter

            # add the original UKV vars into mod_data
            mod_data[site]['specific_humidity'] = mod_q
            mod_data[site]['air_temperature'] = mod_T
            mod_data[site]['air_pressure'] = mod_p

            # wind variables too
            mod_data[site]['u_wind'] = mod_u
            mod_data[site]['v_wind'] = mod_v
            mod_data[site]['w_wind'] = mod_w

            # calculate murk concentration in air(from [kg kg-1 of air] to [kg m-3 of air])
            # kg m-3_air = kg kg-1_air * kg_air m-3_air
            mod_aer_conc = mod_aer * mod_rho

            mod_data[site]['virtual_temperature'] = mod_Tv
            mod_data[site]['air_density'] = mod_rho
            mod_data[site]['aerosol_concentration_dry_air'] = mod_aer_conc

            # if Q_H is in data, extract it
            if 'boundary_layer_sensible_heat_flux' in mod_all_data:
                mod_data[site]['Q_H'] = mod_Q_H

    return mod_data

# -----------------
# 2.3 Observed data

def read_all_rh_obs(day, site_rh, rhDatadir, mod_data):

    """
    Read in day and following day's data, for all rh obs.

    :param day: day string
    :param site_rh: all rh sites
    :param rhDatadir: data directory for rh
    :return: rh obs: dictionary
    """

    # define array
    rh_obs = {}

    # get date string for obs of the main and following days
    doyStr = day.strftime('%Y%j')
    # doyStr2 = (day + dt.timedelta(hours=24)).strftime('%Y%j')

    for site, height in site_rh.iteritems():

        rh_obs[site] = {}

        # rh_fnames = [rhDatadir + site + '_' + doyStr + '_1min.nc',
        #              rhDatadir + site + '_' + doyStr2 + '_1min.nc']

        rh_fnames = rhDatadir + site + '_' + doyStr + '_1min.nc'

        # read in all data
        data_obs = eu.netCDF_read(rh_fnames, vars=['RH', 'time'])
        data_obs['height'] = height

        # find nearest time in rh time
        # pull out ALL the nearest time idxs and differences
        t_idx = np.array([eu.nearest(data_obs['time'], t)[1] for t in mod_data[mod_data.keys()[0]]['time']])
        t_diff = np.array([eu.nearest(data_obs['time'], t)[2] for t in mod_data[mod_data.keys()[0]]['time']])

        # extract hours
        rh_obs[site]['RH'] = data_obs['RH'][t_idx]
        rh_obs[site]['height'] = data_obs['height']
        rh_obs[site]['time'] = [data_obs['time'][i] for i in t_idx]

        # overwrite t_idx locations where t_diff is too high with nans
        # only keep t_idx values where the difference is below 5 minutes
        bad = np.array([abs(i.days * 86400 + i.seconds) > 10 * 60 for i in t_diff])
        rh_obs[site]['RH'][bad] = np.nan

        # change flags to nans
        rh_obs[site]['RH'][np.where(rh_obs[site]['RH'] < 0)] = np.nan

    return rh_obs

def read_all_pm10_obs(dayStart, dayEnd, site_aer, aerDatadir, mod_data):

    """
    Read in the LAQN data.
    It comes in a single file and can contain mempty strings for missing data, but the code
    will replace them with nans. Extracts pm10 and time, but inflexable if data columns would change.
    :param dayStart:
    :param dayEnd:
    :param site_aer:
    :param aerDatadir:
    :return:
    """

    # define array
    pm10_obs = {}

    # get date string for obs of the main and following days
    dateStr = dayStart.strftime('%Y%m%d') + '-' + \
              (dayEnd + dt.timedelta(hours=24)).strftime('%Y%m%d')

    for site, height in site_aer.iteritems():

        pm10_obs[site] = {}

        aer_fname = aerDatadir + site + '_' + dateStr + '.csv'

        raw_aer = np.genfromtxt(aer_fname, delimiter=',', skip_header=1, dtype="|S20")

        # convert missing sections ('' in LAQN data) to NaNs
        missing_idx = np.where(raw_aer[:, 3] == '')
        raw_aer[missing_idx, 3] = np.nan

        # extract obs and time together as a dictionary entry for the site.
        data_obs = \
            {'pm_10': np.array(raw_aer[:, 3], dtype=float),
             'time': np.array(
                 [dt.datetime.strptime(raw_aer[i, 2], '%d/%m/%Y %H:%M')
                  for i in np.arange(len(raw_aer))]),
             'height': height}


        # find nearest time in rh time
        # pull out ALL the nearest time idxs and differences
        t_idx = np.array([eu.nearest(data_obs['time'], t)[1] for t in mod_data[mod_data.keys()[0]]['time']])
        t_diff = np.array([eu.nearest(data_obs['time'], t)[2] for t in mod_data[mod_data.keys()[0]]['time']])

        # extract ALL nearest hours data, regardless of time difference
        pm10_obs[site]['pm_10'] = data_obs['pm_10'][t_idx]
        pm10_obs[site]['height'] = data_obs['height']
        pm10_obs[site]['time'] = [data_obs['time'][i] for i in t_idx]

        # overwrite t_idx locations where t_diff is too high with nans
        # only keep t_idx values where the difference is below 5 minutes
        bad = np.array([abs(i.days * 86400 + i.seconds) > 10 * 60 for i in t_diff])
        pm10_obs[site]['pm_10'][bad] = np.nan

    return pm10_obs

def read_pm10_obs(site_aer, aerDatadir, mod_data={}, matchModSample=True):

    # define array
    pm10_obs = {}


    for site, height in site_aer.iteritems():

        dir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/clearFO/data/LAQN/'
        aer_fname = dir + site + '_Hr_20150101-20170101.csv'

        pm10_obs[site] = {}

        raw_aer = np.genfromtxt(aer_fname, delimiter=',', skip_header=1, dtype="|S20")

        # convert missing sections ('' in LAQN data) to NaNs
        missing_idx = np.where(raw_aer[:, 3] == '')
        raw_aer[missing_idx, 3] = np.nan

        # extract obs and time together as a dictionary entry for the site.
        data_obs = \
            {'pm_10': np.array(raw_aer[:, 3], dtype=float),
             'time': np.array(
                 [dt.datetime.strptime(raw_aer[i, 2], '%d/%m/%Y %H:%M')
                  for i in np.arange(len(raw_aer))]),
             'height': height}

        # match the time sample resolution of mod_data
        if matchModSample == True:

            if mod_data.keys() == []:
                raise ValueError('matchModSample == True, but mod_data is empty or not given!')
            else:

                # find nearest time in rh time
                # pull out ALL the nearest time idxs and differences
                t_idx = np.array([eu.nearest(data_obs['time'], t)[1] for t in mod_data[mod_data.keys()[0]]['time']])
                t_diff = np.array([eu.nearest(data_obs['time'], t)[2] for t in mod_data[mod_data.keys()[0]]['time']])

                # extract ALL nearest hours data, regardless of time difference
                pm10_obs[site]['pm_10'] = data_obs['pm_10'][t_idx]
                pm10_obs[site]['height'] = data_obs['height']
                pm10_obs[site]['time'] = [data_obs['time'][i] for i in t_idx]

                # overwrite t_idx locations where t_diff is too high with nans
                # only keep t_idx values where the difference is below 5 minutes
                bad = np.array([abs(i.days * 86400 + i.seconds) > 10 * 60 for i in t_diff])
                pm10_obs[site]['pm_10'][bad] = np.nan

        else:
            pm10_obs[site] = data_obs



    return pm10_obs

def read_wxt_obs(day, time, z):

    """
    Read in RH observations from KSSW, time match them to the model data, and extend them in height to match the
    dimensions of model RH
    :param day:
    :param time:
    :param z:
    :return wxt_obs:
    """

    filepath = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/data/L1/' + \
               'Davis_BGH_' + day.strftime('%Y') + '_15min.nc'
    wxt_obs = eu.netCDF_read(filepath, vars=['time', 'RH', 'Tair', 'press'])

    # extract out RH obs to match mod_time
    # pull out ALL the nearest time idxs and differences
    # the mod_data time is the same for all sites so can therefore use any site
    t_idx = np.array([eu.nearest(wxt_obs['time'], t)[1] for t in time])
    t_diff = np.array([eu.nearest(wxt_obs['time'], t)[2] for t in time])

    wxt_obs['RH'] = wxt_obs['RH'][t_idx] # [%]
    wxt_obs['Tair'] = wxt_obs['Tair'][t_idx] # [degC]
    wxt_obs['press'] = wxt_obs['press'][t_idx] # [hPa]
    wxt_obs['time'] = wxt_obs['time'][t_idx]
    wxt_obs['rawtime'] = wxt_obs['rawtime'][t_idx]

    # overwrite t_idx locations where t_diff is too high with nans
    # only keep t_idx values where the difference is below 1 hour
    bad = np.array([abs(i.days * 86400 + i.seconds) > 60 * 60 for i in t_diff])

    wxt_obs['RH'][bad] = np.nan
    wxt_obs['Tair'][bad] = np.nan
    wxt_obs['press'][bad] = np.nan

    wxt_obs['time'][bad] = np.nan
    wxt_obs['rawtime'][bad] = np.nan

    # create RH_frac using RH data
    wxt_obs['RH_frac'] = wxt_obs['RH'] / 100.0

    # calculate extra variables
    e_s_hpa = 6.112 * (np.exp((17.67 * wxt_obs['Tair']) / (wxt_obs['Tair'] + 243.5)))  # [hPa] # sat. v. pressure
    e_s = e_s_hpa * 100.0  # [Pa] # sat. v. pressure
    wxt_obs['e'] = wxt_obs['RH_frac'] * e_s  # [Pa] # v. pressure
    wxt_obs['r_v'] = wxt_obs['e'] / (1.61 * ((wxt_obs['press']*100.0) - wxt_obs['e'])) # water_vapour mixing ratio [kg kg-1]
    wxt_obs['q'] =  wxt_obs['e'] / ((1.61 * ((wxt_obs['press']*100.0) - wxt_obs['e'])) + wxt_obs['e']) # specific humidity [kg kg-1]
    wxt_obs['Tv'] = (1 + (0.61 * wxt_obs['q'])) * (wxt_obs['Tair'] + 273.15) # virtual temp [K]
    wxt_obs['air_density'] = (wxt_obs['press']*100.0) / (286.9 * wxt_obs['Tv'])# [kg m-3]

    # extend the wxt obs in height to match the dimensions of model RH
    #   copy the obs so it is the same at all heights
    for var, item in wxt_obs.iteritems():
        if var not in ['time', 'rawtime']:
            # wxt_obs[var] = np.transpose(np.tile(item, (int(rh_frac.shape[1]), 1)))
            wxt_obs[var] = np.transpose(np.tile(item, (int(z.shape[-1]), 1)))

    return wxt_obs

# 2.4 Ceil obs

# new and improved versions (use this one!)
# if timeMatch is provided, it will strip ceil obs data down
# This one is specific for BSC
def read_all_ceil_BSC(day, site_bsc, ceilDatadir, timeMatch=None, calib=True):

    """
    Read in ceilometer backscatter, time, height and SNR data and strip the hours out of it.
    Calibrate data if requested

    :param day:
    :param site_bsc:
    :param ceilDatadir:
    :param mod_data:
    :param calib:
    :return: bsc_obs
    """

    import pickle

    #ToDo very inefficient application, needs reworking slightly
    def calibrate_data(bsc_obs, site):

        """
        calibrate the bsc observations

        :param bsc_obs:
        :param site:
        :return:
        """

        calib_path = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/clearFO/data/' \
                     'Calibrations_for_LUMO_Ceilometers/'

        filename = calib_path + site + '_window_trans_daily_cpro.pickle'

        # sort site name out (is in CL31-A_BSC_KSS45W format, but needs to be CL31-A_KSS45W

        # load calibration data (using pickle)
        with open(filename, 'rb') as handle:
            window_trans_daily = pickle.load(handle)

        for i, time_i in zip(np.arange(len(bsc_obs[site]['time'])), bsc_obs[site]['time']):
            # find date in window_trans_daily
            time_idx = np.where(np.array(window_trans_daily['dates']) == time_i.date())

            # apply calibration to bsc data
            bsc_obs[site]['backscatter'][i, :] *= window_trans_daily['c_pro'][time_idx]

        return bsc_obs

    # contains all the sites time-upscaled data
    bsc_obs = {}

    for site, height in site_bsc.iteritems():

        # create filename
        bsc_fname, site_id = create_filename(ceilDatadir, site, day, fType='BSC')

        # check if data is there, else skip it
        if os.path.exists(bsc_fname):

            # this sites time-upscaled data
            bsc_obs[site] = {}

            # read backscatter data
            data_obs, ceilLevel = ceil.netCDF_read_BSC(bsc_fname)

            # time match model data?
            # if timeMatch has data, then time match against it
            if timeMatch != None:

                # find nearest time in ceil time
                # pull out ALL the nearest time idxs and differences
                t_idx = np.array([eu.nearest(data_obs['time'], t)[1] for t in timeMatch[site_id]['time']])
                t_diff = np.array([eu.nearest(data_obs['time'], t)[2] for t in timeMatch[site_id]['time']])

            else:
                # extract the lot
                t_idx = np.arange(len(data_obs['backscatter']))


            # extract data
            # for var, data in data_obs.iteritems():
            bsc_obs[site]['SNR'] = data_obs['SNR'][t_idx, :]
            bsc_obs[site]['backscatter'] = data_obs['backscatter'][t_idx, :]
            bsc_obs[site]['height'] = data_obs['height'] + height
            bsc_obs[site]['time'] = [data_obs['time'][i] for i in t_idx]

            # calibrate data
            if calib == True:
                bsc_obs = calibrate_data(bsc_obs, site)

            # overwrite t_idx locations where t_diff is too high with nans
            # only keep t_idx values where the difference is below 5 minutes
            bad = np.array([abs(i.days * 86400 + i.seconds) > 10 * 60 for i in t_diff])

            bsc_obs[site]['SNR'][bad, :] = np.nan
            bsc_obs[site]['backscatter'][bad, :] = np.nan

    return bsc_obs

# Tried to make a very general version for use with any LUMO ceilometer netCDF file
def read_all_ceil_obs(day, site_bsc, ceilDatadir, fType='', timeMatch=None, calib=True):

    """
    Read in ceilometer backscatter, time, height and SNR data and strip the hours out of it.
    Calibrate data if requested

    :param day:
    :param site_bsc:
    :param ceilDatadir:
    :param mod_data:
    :param calib:
    :return: bsc_obs
    """

    # contains all the sites time-upscaled data
    ceil_obs = {}

    for site, height in site_bsc.iteritems():

        # create filename
        fname, site_id = create_filename(ceilDatadir, site, day, fType=fType)

        # check if data is there, else skip it
        if os.path.exists(fname):

            # this sites time-upscaled data
            ceil_obs[site] = {}

            # read data
            if fType == 'BSC':
                data_obs = ceil.netCDF_read_BSC(datapath, var_type='beta_tR', SNRcorrect=True)
            else:
                data_obs = ceil.netCDF_read_ceil(fname, get_height=True, get_time=True)

            # time match model data?
            # if timeMatch has data, then time match against it
            if timeMatch != None:

                # find nearest time in ceil time
                # pull out ALL the nearest time idxs and differences
                t_idx = np.array([eu.nearest(data_obs['time'], t)[1] for t in timeMatch[site_id]['time']])
                t_diff = np.array([eu.nearest(data_obs['time'], t)[2] for t in timeMatch[site_id]['time']])

            else:
                # extract the lot
                t_idx = np.arange(len(data_obs['time']))


            # get keys, trim data for all but height and time
            for var, var_data in data_obs.iteritems():
                if var_data.shape != (): # equivalent to 0 shape / scaler value
                    if var_data.shape[0] == len(data_obs['time']): # if the dimension length is equal to time's length
                        ceil_obs[site][var] = var_data[t_idx, ...]
                    else:
                        raise AttributeError('Tried to slice variable using t_idx but time was not the first dimension'
                                             'of the array! Need to reconsider this slicing approach!: ' + var + ' ' +
                                             'var shape: ' + str(var_data.shape))
                else:
                    ceil_obs[site][var] = var_data

            # add height onto the range gates in obs
            # in MLH, height is a scaler quantity, and the height of the ceilometer - not the range gates!
            if fType != 'MLH':
                ceil_obs[site]['height'] += height


            # # extract data - original approach
            # # for var, data in data_obs.iteritems():
            # bsc_obs[site]['SNR'] = data_obs['SNR'][t_idx, :]
            # bsc_obs[site]['backscatter'] = data_obs['backscatter'][t_idx, :]
            # bsc_obs[site]['height'] = data_obs['height'] + height
            # bsc_obs[site]['time'] = [data_obs['time'][i] for i in t_idx]

            # calibrate data
            if (fType == 'BSC') & (calib == True):
                ceil_obs = calibrate_data(ceil_obs, site)

            # # overwrite t_idx locations where t_diff is too high with nans
            # # only keep t_idx values where the difference is below 10 minutes
            # bad = np.array([abs(i.days * 86400 + i.seconds) > 10 * 60 for i in t_diff])
            #
            # ceil_obs[site]['SNR'][bad, :] = np.nan
            # ceil_obs[site]['backscatter'][bad, :] = np.nan

    return ceil_obs


# older versions

# read ins ceil data and strips it down to hourly values accoridng to mod_data
def read_ceil_obs(day, site_bsc, ceilDatadir, mod_data, calib=True, version=1.1):

    """
    Read in ceilometer backscatter, time, height and SNR data and strip the hours out of it.
    Calibrate data if requested

    :param day:
    :param ceilDatadir:
    :return: data_obs: dictionary
    """

    import os
    import pickle

    def calibrate_data(bsc_obs, site, day):

        """
        calibrate the bsc observations

        :param bsc_obs:
        :param site:
        :return:
        """

        # site id (short) and site str in filename
        split = site.split('_')
        site_id = split[-1]
        cal_site_name = split[0] + '_CAL_' + split[-1]

        # L2 is the interpolated calibration data (vs time, window transmission or block avg.)
        calibdir = 'C:/Users/Elliott/Documents/PhD Reading/LUMO - Sensor network/calibration/data/L2/'
        filepath = calibdir + cal_site_name + '_' + day.strftime('%Y') + '.nc'

        calib_data = eu.netCDF_read(filepath)

        # get correct calibration idx, given the bsc date
        time_idx = np.where(calib_data['time'] == day)[0][0]

        # apply calibration to bsc data
        bsc_obs[site]['backscatter'] *= calib_data['c_pro'][time_idx]

        # set timezone as UTC
        #bsc_obs[site]['time'] = np.array([i.replace(tzinfo=tz.gettz('UTC')) for i in bsc_obs[site]['time']])

        return bsc_obs

    #ToDo very inefficient application, needs reworking slightly
    def calibrate_data_v1p0(bsc_obs, site):

        """
        calibrate the bsc observations

        :param bsc_obs:
        :param site:
        :return:
        """

        calib_path = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/clearFO/data/' \
                     'Calibrations_for_LUMO_Ceilometers/'

        filename = calib_path + site + '_window_trans_daily_cpro.pickle'

        # sort site name out (is in CL31-A_BSC_KSS45W format, but needs to be CL31-A_KSS45W

        # load calibration data (using pickle)
        with open(filename, 'rb') as handle:
            window_trans_daily = pickle.load(handle)

        for i, time_i in zip(np.arange(len(bsc_obs[site]['time'])), bsc_obs[site]['time']):
            # find date in window_trans_daily
            time_idx = np.where(np.array(window_trans_daily['dates']) == time_i.date())

            # apply calibration to bsc data
            bsc_obs[site]['backscatter'][i, :] *= window_trans_daily['c_pro'][time_idx]

        return bsc_obs


    # contains all the sites time-upscaled data
    bsc_obs = {}

    for site, height in site_bsc.iteritems():

        # site id (short) and site str in filename
        split = site.split('_')
        site_id = split[-1]
        bsc_site_name = split[0] + '_BSC_' + split[-1]

        # date for the main day
        doyStr = day.strftime('%Y%j')

        # get filename
        bsc_fname = ceilDatadir + bsc_site_name + '_' + doyStr + '_15sec.nc'

        # check if data is there, else skip it
        if os.path.exists(bsc_fname):

            # this sites time-upscaled data
            bsc_obs[site] = {}

            # read backscatter data
            data_obs, ceilLevel = ceil.netCDF_read_BSC(bsc_fname)

            # find nearest time in ceil time
            # pull out ALL the nearest time idxs and differences
            # the mod_data time is the same for all sites so can therefore use any site
            t_idx = np.array([eu.nearest(data_obs['time'], t)[1] for t in mod_data[mod_data.keys()[0]]['time']])
            t_diff = np.array([eu.nearest(data_obs['time'], t)[2] for t in mod_data[mod_data.keys()[0]]['time']])

            # extract data
            # for some reason, the L1 'height' is not corrected to be above ground and is basically the range
            #   whereas L0 height is corrected to be above ground
            bsc_obs[site]['SNR'] = data_obs['SNR'][t_idx, :]
            bsc_obs[site]['backscatter'] = data_obs['backscatter'][t_idx, :]
            bsc_obs[site]['time'] = [data_obs['time'][i] for i in t_idx]
            if ceilLevel == 'L1':
                bsc_obs[site]['height'] = data_obs['height'] + height

            # calibrate data
            if calib == True:
            
                if version <= 1.0:
                    bsc_obs = calibrate_data_v1p0(bsc_obs, site)
                else:
                    bsc_obs = calibrate_data(bsc_obs, site, day)

            # overwrite t_idx locations where t_diff is too high with nans
            # only keep t_idx values where the difference is below 5 minutes
            bad = np.array([abs(i.days * 86400 + i.seconds) > 10 * 60 for i in t_diff])

            bsc_obs[site]['SNR'][bad, :] = np.nan
            bsc_obs[site]['backscatter'][bad, :] = np.nan

    return bsc_obs


def read_ceil_obs_all(day, site_bsc, ceilDatadir, calib=True, version=1.1):
    """
    Read in ceilometer backscatter, time, height and SNR data and strip the hours out of it.
    Calibrate data if requested

    :param day:
    :param ceilDatadir:
    :return: data_obs: dictionary
    """

    import os
    import pickle

    def calibrate_data(bsc_obs, site, day):

        """
        calibrate the bsc observations

        :param bsc_obs:
        :param site:
        :return:
        """

        # site id (short) and site str in filename
        split = site.split('_')
        site_id = split[-1]
        cal_site_name = split[0] + '_CAL_' + split[-1]

        # L2 is the interpolated calibration data (vs time, window transmission or block avg.)
        calibdir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/common_data/calibration/L2/'
        filepath = calibdir + cal_site_name + '_' + day.strftime('%Y') + '.nc'

        calib_data = eu.netCDF_read(filepath)

        # get correct calibration idx, given the bsc date
        time_idx = np.where(calib_data['time'] == day)[0][0]

        # apply calibration to bsc data
        bsc_obs[site]['backscatter'] *= calib_data['c_pro'][time_idx]

        # set timezone as UTC
        # bsc_obs[site]['time'] = np.array([i.replace(tzinfo=tz.gettz('UTC')) for i in bsc_obs[site]['time']])

        return bsc_obs

    # ToDo very inefficient application, needs reworking slightly
    def calibrate_data_v1p0(bsc_obs, site):

        """
        calibrate the bsc observations

        :param bsc_obs:
        :param site:
        :return:
        """

        calib_path = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/clearFO/data/' \
                     'Calibrations_for_LUMO_Ceilometers/'

        filename = calib_path + site + '_window_trans_daily_cpro.pickle'

        # sort site name out (is in CL31-A_BSC_KSS45W format, but needs to be CL31-A_KSS45W

        # load calibration data (using pickle)
        with open(filename, 'rb') as handle:
            window_trans_daily = pickle.load(handle)

        for i, time_i in zip(np.arange(len(bsc_obs[site]['time'])), bsc_obs[site]['time']):
            # find date in window_trans_daily
            time_idx = np.where(np.array(window_trans_daily['dates']) == time_i.date())

            # apply calibration to bsc data
            bsc_obs[site]['backscatter'][i, :] *= window_trans_daily['c_pro'][time_idx]

        return bsc_obs

    # contains all the sites time-upscaled data
    bsc_obs = {}

    for site, height in site_bsc.iteritems():

        # site id (short) and site str in filename
        split = site.split('_')
        site_id = split[-1]
        bsc_site_name = split[0] + '_BSC_' + split[-1]

        # date for the main day
        doyStr = day.strftime('%Y%j')

        # get filename
        bsc_fname = ceilDatadir + bsc_site_name + '_' + doyStr + '_15sec.nc'

        # check if data is there, else skip it
        if os.path.exists(bsc_fname):

            # this sites time-upscaled data
            bsc_obs[site] = {}

            # read backscatter data
            data_obs, ceilLevel = ceil.netCDF_read_BSC(bsc_fname)

            # extract data
            # for some reason, the L1 'height' is not corrected to be above ground and is basically the range
            #   whereas L0 height is corrected to be above ground
            bsc_obs[site]['SNR'] = data_obs['SNR']
            bsc_obs[site]['backscatter'] = data_obs['backscatter']
            bsc_obs[site]['time'] = data_obs['time']
            if ceilLevel == 'L1':
                bsc_obs[site]['height'] = data_obs['height'] + height

            # calibrate data
            if calib == True:

                if version <= 1.0:
                    bsc_obs = calibrate_data_v1p0(bsc_obs, site)
                else:
                    bsc_obs = calibrate_data(bsc_obs, site, day)

    return bsc_obs

# reads in all ceil data for the day and does not strip down
def read_ceil_obs_all_old(day, site_bsc, ceilDatadir, calib=True):

    """
    Read in ALL ceilometer backscatter, time, height and SNR data. No stripping down

    :param day:
    :param ceilDatadir:
    :return: data_obs: dictionary
    """

    import pickle

    def calibrate_data(bsc_obs, site):

        """
        calibrate the bsc observations

        :param bsc_obs:
        :param site:
        :return:
        """

        calib_path = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/clearFO/data/' \
                     'Calibrations_for_LUMO_Ceilometers/'

        filename = calib_path + site + '_window_trans_daily_cpro.pickle'

        # sort site name out (is in CL31-A_BSC_KSS45W format, but needs to be CL31-A_KSS45W

        # load calibration data (using pickle)
        with open(filename, 'rb') as handle:
            window_trans_daily = pickle.load(handle)

        for i, time_i in zip(np.arange(len(bsc_obs[site]['time'])), bsc_obs[site]['time']):
            # find date in window_trans_daily
            time_idx = np.where(np.array(window_trans_daily['dates']) == time_i.date())

            # apply calibration to bsc data
            bsc_obs[site]['backscatter'][i, :] *= window_trans_daily['c_pro'][time_idx]

        return bsc_obs

    # contains all the sites time-upscaled data
    bsc_obs = {}

    for site, height in site_bsc.iteritems():

        # this sites time-upscaled data
        bsc_obs[site] = {}

        # site id (short) and site str in filename
        split = site.split('_')
        site_id = split[-1]
        bsc_site_name = split[0] + '_BSC_' + split[-1]

        # date for the main day
        doyStr = day.strftime('%Y%j')

        # get filename
        bsc_fname = ceilDatadir + bsc_site_name + '_' + doyStr + '_15sec.nc'

        # read backscatter data
        bsc_obs[site] = ceil.netCDF_read_BSC(bsc_fname)

        # extract data
        # for var, data in data_obs.iteritems():
        bsc_obs[site]['height'] += height

        # calibrate data
        if calib == True:
            bsc_obs = calibrate_data(bsc_obs, site)

    return bsc_obs

# ----------------

# fast version, copy and pasted from CCW30
def read_ceil_CLD_obs(day, site_bsc, ceilDatadir, mod_data, timeMatchMod=False, dayExtract=False):

    """
    Read in raw ceilometer cloud, time and height data and strip the hours out of it.

    #ToDo currently no SNR or time QAQC

    :param day:
    :param ceilDatadir:
    :return: data_obs: [dictionary]
    """

    import os

    # contains all the sites time-upscaled data
    bsc_obs = {}

    for site, height in site_bsc.iteritems():

        # site id (short) and site str in filename
        split = site.split('_')
        site_id = split[-1]
        bsc_site_name = split[0] + '_CLD_' + split[-1]

        # date for the main day
        doyStr = day.strftime('%Y%j')

        # get filename
        bsc_fname = ceilDatadir + bsc_site_name + '_' + doyStr + '_15sec.nc'

        # check if data is there, else skip it
        if os.path.exists(bsc_fname):

            # this sites time-upscaled data
            bsc_obs[site] = {}

            # read cloud data
            # data_obs = ceil.netCDF_read_CLD(bsc_fname)
            data_obs = ceil.netCDF_read_ceil(bsc_fname)


            if timeMatchMod == True:
                # find nearest time in ceil time
                # pull out ALL the nearest time idxs and differences
                # the mod_data time is the same for all sites so can therefore use any site
                t_idx = np.array([eu.nearest(data_obs['time'], t)[1] for t in mod_data[site_id]['time']])
                t_diff = np.array([eu.nearest(data_obs['time'], t)[2] for t in mod_data[site_id]['time']])


            elif dayExtract == True:

                # get just day part of date
                day_part = np.array([i.date() for i in data_obs['time']])
                t_idx = np.where(day_part == day.date())[0]

            else:

                # extract the lot
                t_idx = np.arange(len(data_obs['CLD_Height_L1']))

            # extract data based on t_idx
            # for var, data in data_obs.iteritems():
            bsc_obs[site]['CLD_Height_L1'] = data_obs['CLD_Height_L1'][t_idx] + height

            bsc_obs[site]['height'] = data_obs['height'] + height
            bsc_obs[site]['time'] = [data_obs['time'][i] for i in t_idx]

            # overwrite t_idx locations where t_diff is too high with nans
            # only keep t_idx values where the difference is below 5 minutes
            #bad = np.array([abs(i.days * 86400 + i.seconds) > 10 * 60 for i in t_diff])
            #
            #bsc_obs[site]['backscatter'][bad, :] = np.nan

    return bsc_obs

# this is the better version atm
def read_ceil_CCW30_obs(day, site_bsc, ceilDatadir, mod_data, timeMatchMod=False, dayExtract=False):

    """
    Read in ceilometer cloud, time and height data and strip the hours out of it.

    #ToDo currently no SNR or time QAQC

    :param day:
    :param ceilDatadir:
    :return: data_obs: [dictionary]
    """

    import os

    # contains all the sites time-upscaled data
    bsc_obs = {}

    for site, height in site_bsc.iteritems():

        # site id (short) and site str in filename
        split = site.split('_')
        site_id = split[-1]
        bsc_site_name = split[0] + '_CCW30_' + split[-1]

        # date for the main day
        doyStr = day.strftime('%Y')

        # get filename
        bsc_fname = ceilDatadir + bsc_site_name + '_' + doyStr + '_15min.nc'

        # check if data is there, else skip it
        if os.path.exists(bsc_fname):

            # this sites time-upscaled data
            bsc_obs[site] = {}

            # read cloud data
            data_obs = ceil.netCDF_read_CCW30(bsc_fname)

            if timeMatchMod == True:
                # find nearest time in ceil time
                # pull out ALL the nearest time idxs and differences
                # the mod_data time is the same for all sites so can therefore use any site
                t_idx = np.array([eu.nearest(data_obs['time'], t)[1] for t in mod_data[site_id]['time']])
                t_diff = np.array([eu.nearest(data_obs['time'], t)[2] for t in mod_data[site_id]['time']])


            elif dayExtract == True:

                # get just day part of date
                day_part = np.array([i.date() for i in data_obs['time']])
                t_idx = np.where(day_part == day.date())[0]

            else:

                # extract the lot
                t_idx = np.arange(len(data_obs['CBH']))

            # extract data based on t_idx
            # for var, data in data_obs.iteritems():
            bsc_obs[site]['CBH'] = data_obs['CBH'][t_idx]
            bsc_obs[site]['CC_low'] = data_obs['CC_low'][t_idx]
            bsc_obs[site]['CC_medium'] = data_obs['CC_medium'][t_idx]
            bsc_obs[site]['CC_high'] = data_obs['CC_high'][t_idx]

            bsc_obs[site]['height'] = data_obs['height'] + height
            bsc_obs[site]['time'] = [data_obs['time'][i] for i in t_idx]

            # overwrite t_idx locations where t_diff is too high with nans
            # only keep t_idx values where the difference is below 5 minutes
            #bad = np.array([abs(i.days * 86400 + i.seconds) > 10 * 60 for i in t_diff])
            #
            #bsc_obs[site]['backscatter'][bad, :] = np.nan

    return bsc_obs


# ----------------------

# read in helper functions

def create_filename(ceilDatadir, site, day, fType):

    """ Creates filename from site stinrg and day"""

    # site id (short) and site str in filename
    split = site.split('_')
    site_id = split[-1]
    bsc_site_name = split[0] + '_' + fType + '_' + split[-1]

    # date for the main day
    doyStr = day.strftime('%Y%j')

    # time resolution of data in filename
    if fType == 'MLH':
        timestr = '15min'
    elif fType == 'BSC':
        timestr = '15sec'
    elif fType == 'CLD':
        timestr == '15sec'
    elif fType == '':
        raise ValueError('fType variable not given!')
    else:
        raise ValueError('fType argument is not recognised. Please choose MLH, BSC, CLD or add new fType')

    # get filename
    bsc_fname = ceilDatadir + bsc_site_name + '_' + doyStr + '_' + timestr + '.nc'

    return bsc_fname, site_id

