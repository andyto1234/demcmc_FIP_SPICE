from demcmc import (
    EmissionLine,
    TempBins,
    load_cont_funcs,
    plot_emission_loci,
    predict_dem_emcee,
    ContFuncDiscrete,
)
import numpy as np
import os
import re


def calc_chi2(mcmc_lines: list[EmissionLine], dem_result: np.array, temp_bins: TempBins) -> float:
    # Calculate the chi-square value for the given MCMC lines, DEM result, and temperature bins
    int_obs = np.array([line.intensity_obs for line in mcmc_lines])
    int_pred = np.array([line._I_pred(temp_bins, dem_result) for line in mcmc_lines])
    sigma_intensity_obs = np.array([line.sigma_intensity_obs for line in mcmc_lines])
    chi2 = np.sum(((int_pred - int_obs) / sigma_intensity_obs) ** 2)
    return chi2


def find_matching_file(log_density, abund_file = 'emissivities_sun_photospheric_2015_scott'):
    import platform

    if platform.system() == 'Linux':
        directory=f'/disk/solar17/st3/{abund_file}/'

    if platform.system() == 'Darwin':
        directory=f'/Users/andysh.to/Script/Data/{abund_file}/'

    # Convert log_density to float
    target_log_density = float(log_density)
    
    matching_file = None
    min_density_difference = float('inf')

    for filename in os.listdir(directory):
        if filename.startswith("emissivity_combined"):
            # Extract density from the filename using regular expression
            match = re.search(r'_([\d.e+-]+)_', filename)
            if match:
                file_density = float(match.group(1))
                
                # Calculate the absolute difference between target and file density
                density_difference = abs(target_log_density - file_density)
                
                # Check if this file is a better match than the previous ones
                if density_difference < min_density_difference:
                    min_density_difference = density_difference
                    matching_file = directory+filename

    return matching_file

def interp_emis_temp(original_array):
    # Interpolate into array with size 401
    new_size = 70
    new_indices = np.linspace(0, len(original_array) - 1, new_size)
    interpolated_array = np.interp(new_indices, np.arange(len(original_array)), original_array)
    return interpolated_array