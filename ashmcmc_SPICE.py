import xarray as xr
from astropy import units as u
import numpy as np
import sunpy.map
from mcmc_utils import find_matching_file, interp_emis_temp
from scipy.io import readsav
import astropy.units as u
from tqdm import tqdm
from pathlib import Path

from multiprocessing import Pool
import platform


from demcmc import (
    EmissionLine,
    TempBins,
    load_cont_funcs,
    plot_emission_loci,
    predict_dem_emcee,
    ContFuncDiscrete,
)
from demcmc.units import u_temp, u_dem

# Retrieve dimensions of the solar map
def process_solar_map(files: list, rebinx=3, rebiny=3):
    files = [file for file in files if 'o_1' not in file]

    y_dim = sunpy.map.Map(files[0])[0].dimensions[1].value
    x_dim = sunpy.map.Map(files[0])[0].dimensions[0].value
    output_dir = 'results/spice_'+sunpy.map.Map(files[0])[0].date_average.value.split('.')[0].replace(':','_')+ '/'

    # Check if the directory exists, and create it if it doesn't
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir+'dem_columns/').mkdir(parents=True, exist_ok=True)

    num_files = len(files)

    wcs_original = sunpy.map.Map(files[0])[0].wcs

    filenum = 0
    # Create a blank cube to store solar map data
    blank = np.zeros((int(y_dim), int(x_dim), len(files)))
    blank_error = np.zeros((int(y_dim), int(x_dim), len(files)))

    wavelengths = []
    ions = []
    # Populate the blank cube with data from each file
    for num, file in enumerate(files):
        print(file)
        solar_map = sunpy.map.Map(file)[0]
        solar_map_error = sunpy.map.Map(file)[1]
        blank[:, :, num] = solar_map.data
        blank_error[:, :, num] = solar_map_error.data
        wavelengths.append(solar_map.meta['wavelength'])
        ions.append(solar_map.meta['ion'])

    # Extract wavelengths and ions from the files
    combined_list = [f"{ion}_{wavelength:.2f}" for ion, wavelength in zip(ions, wavelengths)]

    # Create an xarray Dataset
    dataset = xr.Dataset(
        data_vars=dict(
            data=(["y", "x", "wavelength"], blank),
            error=(["y", "x", "wavelength"], blank_error)  # Add the error data variable
        ),
        coords=dict(
            y=np.arange(y_dim),
            x=np.arange(x_dim),
            wavelength=wavelengths,
            ions=ions,
            linenames = combined_list
        ),
        attrs=dict(
            description="SPICE map data"
        )
    )

    # Convert WCS object to a dictionary and add it as an attribute
    dataset.attrs["wcs"] = wcs_original.to_header_string()


    # Save the dataset to a file (optional)
    dataset.to_netcdf(output_dir+'fitted_data.nc')

    rebinned_ds = dataset.coarsen(x=rebinx, y=rebiny, boundary='trim').mean()
    rebinned_ds.to_netcdf(output_dir+'fitted_data_rebinned.nc')

    return rebinned_ds, output_dir

def find_data(dataset, desire_linenames, option = 'data'):
    # Find the indices of the desired linenames in the linenames list
    indices = [dataset.linenames.values.tolist().index(linename) for linename in desire_linenames]

    # Filter for the desired wavelengths using isel()
    data_filtered = dataset.isel(wavelength=indices)

    # Access the filtered data
    if option == 'data':
        filtered_dataArray = data_filtered['data'].values # Get the NumPy array
    elif option == 'error':
        filtered_dataArray = data_filtered['error'].values

    return filtered_dataArray

def calc_density(obs_ratio, dens_sav):
    from scipy.io import readsav
    import numpy as np

    density_ratios = readsav(dens_sav)['smooth_rat']
    density_values = readsav(dens_sav)['smooth_den']

    # Reshape obs_ratio to a 1D array
    obs_ratio_1d = obs_ratio.ravel()

    # Compute the absolute difference between density_ratios and each value in obs_ratio_1d
    diff = np.abs(density_ratios - obs_ratio_1d[:, np.newaxis])

    # Find the indices of the minimum values along the second axis
    closest_indices = np.argmin(diff, axis=1)

    # Get the corresponding density values using the closest indices
    obs_ratio_1d = density_values[closest_indices]
    obs_ratio = obs_ratio_1d.reshape(obs_ratio.shape)

    return obs_ratio

def get_density(dataset, dens_sav='density_ratios_mg_9_706_02_749_54_.sav', desired_linenames = ['mg_9_706.02', 'mg_9_749.54']):
    # Process the solar map data to get the density
    print(f'------------------------------Calculating Density------------------------------\n')
    print(f'Calculating density for {desired_linenames[0]} and {desired_linenames[1]}\n')
    obs_dens = find_data(dataset, desired_linenames, option = 'data')
    obs_dens_ratio = obs_dens[:, :, 0] / obs_dens[:, :, 1]
    ldens = calc_density(obs_dens_ratio, dens_sav)
    print(f'------------------------------Done------------------------------')
    return ldens


def combine_emissivity(data):
    # Combine emissivity of the same line together - Specific for my file configuration and SPICE

    # Initialize an empty dictionary to store the summed emissivities for each linename
    summed_emissivities = {}

    # Iterate over the linenames and emissivities
    for linename, emissivity in zip(data['linenames'], data['emissivity_combined']):
        linename = linename.decode('utf-8')

        # If the linename is already in the dictionary, add the emissivity to the existing value
        if linename in summed_emissivities:
            summed_emissivities[linename] = summed_emissivities[linename] + emissivity
        # If the linename is not in the dictionary, create a new entry with the emissivity
        else:
            summed_emissivities[linename] = emissivity.copy()

    # Convert the dictionary to separate arrays for linenames and summed emissivities
    unique_linenames = np.array(list(summed_emissivities.keys()))
    summed_emissivities_array = np.array(list(summed_emissivities.values()))

    # Create a new dictionary with the unique linenames and summed emissivities
    result = {
        'linenames': unique_linenames,
        'emissivity_combined': summed_emissivities_array,
        'logt_interpolated': data['logt_interpolated']
    }

    return result

def read_emissivity_spice(ldens, abund_file):
    # Read emissivity from .sav files
    # The abund file is the directory where the .sav files are stored - this is a bit weird

    # Find matching file based on density
    emis_file = readsav(find_matching_file(ldens, abund_file=abund_file)) # Find matching file based on density
    emis_file = combine_emissivity(emis_file) # Combine emissivity of the same line together

    logt = 10**emis_file['logt_interpolated']*u.K
    emis = emis_file['emissivity_combined']
    linenames = emis_file['linenames'].astype(str)

    return logt, emis, linenames # this has unique linenames

def emis_filter(emis, linenames, obs_Lines):
    import numpy as np
    # Filter emissivity based on specified lines
    emis_sorted = np.zeros((len(obs_Lines.linenames.data),101))
    for ind, line in enumerate(obs_Lines.linenames.data):
        emis_sorted[ind, :] = emis[np.where(linenames == line)]
    return emis_sorted

def mcmc_process(mcmc_lines: list[EmissionLine], temp_bins: TempBins, progress=False) -> np.ndarray:
    # Perform MCMC process for the given MCMC lines and temperature bins - Combination specific to SPICE
    dem_result = predict_dem_emcee(mcmc_lines, temp_bins, nwalkers=200, nsteps=300, progress=progress, dem_guess=None)
    dem_median = np.median([sample.values.value for num, sample in enumerate(dem_result.iter_binned_dems())], axis=0)
    for nstep in [300, 1000]:
        dem_result = predict_dem_emcee(mcmc_lines, temp_bins, nwalkers=200, nsteps=nstep, progress=progress,
                                        dem_guess=dem_median)
        dem_median = np.median([sample.values.value for num, sample in enumerate(dem_result.iter_binned_dems())],
                                axis=0)
    return dem_median

def calc_chi2(mcmc_lines: list[EmissionLine], dem_result: np.array, temp_bins: TempBins) -> float:
    # Calculate the chi-square value for the given MCMC lines, DEM result, and temperature bins
    int_obs = np.array([line.intensity_obs for line in mcmc_lines])
    int_pred = np.array([line._I_pred(temp_bins, dem_result) for line in mcmc_lines])
    sigma_intensity_obs = np.array([line.sigma_intensity_obs for line in mcmc_lines])
    chi2 = np.sum(((int_pred - int_obs) / sigma_intensity_obs) ** 2)
    return chi2

def calc_percentage_diff(mcmc_lines, num, temp_bins, _dem_median):
    percentage_diff = np.abs((mcmc_lines[num]._I_pred(temp_bins, _dem_median)-mcmc_lines[num].intensity_obs)/mcmc_lines[num].intensity_obs*100)
    return percentage_diff


def prep_spice_data(files):
    dataset, output_dir = process_solar_map(files)

    # Get the density
    ldens = get_density(dataset) # optional dens_sav = '/Users/andysh.to/Script/Python_Script/demcmc_FIP_SPICE/density_ratios_mg_9_706_02_749_54_.sav'; desired_linenames = ['mg_9_706.02', 'mg_9_749.54']
    dataset = dataset.assign(ldens=(["y", "x"], ldens))

    return dataset, output_dir

def emissionLine_setup(ind, emis, dataset, xpix, ypix, line, logt_interp):
    mcmc_emis = ContFuncDiscrete(logt_interp*u.K, interp_emis_temp(emis[ind, :]) *u.cm**5 / u.K,
                                name=line)
    mcmc_intensity = dataset.data[ypix, xpix, ind]*1e3
    mcmc_int_error = 0.3 * mcmc_intensity
    emissionLine = EmissionLine(
        mcmc_emis,
        intensity_obs=mcmc_intensity,
        sigma_intensity_obs=mcmc_int_error,
        name=line
    )
    return emissionLine


def process_pixel(args):
    xpix, dataset, output_dir = args

    output_file = f'{output_dir}/dem_columns/dem_{xpix}.npz'

    ycoords_out = []
    dem_results = []
    chi2_results = []
    linenames_list = []
    comp_result = []

    for ypix in tqdm(range(dataset.data[:, :, 0].shape[0])):
        logt, emis_photo, linenames = read_emissivity_spice(dataset['ldens'][ypix, xpix], abund_file='spice_emissivities_sun_photospheric_2015_scott')
        logt, emis_coro_mg, linenames = read_emissivity_spice(dataset['ldens'][ypix, xpix], abund_file='spice_emissivities_spice_sun_coronal_2015_scott_mg')

        emis_photo = emis_filter(emis_photo, linenames, dataset.linenames)
        emis_coro_mg = emis_filter(emis_coro_mg, linenames, dataset.linenames)

        logt_interp = interp_emis_temp(logt.value)

        mcmc_lines = []
        temp_bins = TempBins(logt_interp * u.K)
        chi2 = np.inf
        binary_comp = -1

        for emis in [emis_photo, emis_coro_mg]:
            for ind, line in enumerate(dataset.linenames): # setting the emissionLine variable
                if chi2 == np.inf:
                    linenames_list.append(line)  # Append the list of MCMC lines to the list
                
                # If the intensity is greater than 1 and the emissivity is not all zeros
                if dataset.data[ypix, xpix, ind]*1e3 > 1 and ~np.all(emis[ind, :] == 0): 
                    emissionLine = emissionLine_setup(ind, emis, dataset, xpix, ypix, line, logt_interp)
                    mcmc_lines.append(emissionLine)

            # Run 3 MCMC processes for SPICE and return the median DEM
            _dem_median = mcmc_process(mcmc_lines, temp_bins)  
            
            # Calculate the temporary chi2 value
            _chi2 = calc_chi2(mcmc_lines, _dem_median, temp_bins)   
            print(mcmc_lines)
            if 'mg' in [l.name.split('_')[0] for l in mcmc_lines]: # If Mg is inside the lines
                if _chi2 <= chi2*0.8:  # If the chi2 value is greater than the current chi2 value * 0.8
                    chi2 = _chi2  # Update the chi2 value
                    dem_median = _dem_median
                    binary_comp += 1  # Update the binary composition value to photospheric or coronal
                elif _chi2 > chi2*0.8 and _chi2 < chi2*1.2:
                    chi2 = _chi2  # Update the chi2 value
                    dem_median = _dem_median
                    binary_comp += 0.5  # Update the binary composition value to photospheric or coronal
            else: # If Mg is not inside the lines
                chi2 = _chi2  # Update the chi2 value
                dem_median = _dem_median
                binary_comp = np.nan  # Update the binary composition value to photospheric or coronal
                break

        dem_results.append(dem_median)
        chi2_results.append(chi2)
        ycoords_out.append(ypix)
        linenames_list.append(mcmc_lines)
        comp_result.append(binary_comp)

    dem_results = np.array(dem_results)
    chi2_results = np.array(chi2_results)
    comp_result = np.array(comp_result)
    linenames_list = np.array(linenames_list, dtype=object)

    np.savez(output_file, dem_results=dem_results, chi2=chi2_results, ycoords_out=ycoords_out,
             lines_used=linenames_list, logt=np.array(logt_interp), comp_result=comp_result)


def main(filedir):
    import glob

    # List all the files in the directory
    files = sorted(glob.glob(f'{filedir}/*int*'))

    output_dir = Path(files[0]).parent

    dataset, output_dir = prep_spice_data(files)  # get all intensity data and density data into xarray format

    # Generate a list of arguments for process_pixel function
    args_list = [(xpix, dataset, output_dir) for xpix in range(dataset.data[:, :, 0].shape[1])]

    # Determine the operating system type (Linux or macOS)
    # Set the number of processes based on the operating system
    if platform.system() == "Linux":
        process_num = 50  # above 64 seems to break the MSSL machine - probably due to no. cores = 64?
    elif platform.system() == "Darwin":
        process_num = 10
    else:
        process_num = 10
    print(f'------------------------------Calculating Composition------------------------------')

    # Create a Pool of processes for parallel execution
    with Pool(processes=process_num) as pool:
        results = list(tqdm(pool.imap(process_pixel, args_list), total=len(args_list), desc="Processing Pixels"))

    print(f'------------------------------Done------------------------------')

def process_filedir(filedir):
    # Check if the filedir is already being processed
    if filedir.endswith('[processing]'):
        print(f"Skipping {filedir} as it is already being processed.")
        return

    # Add the [processing] tag to the filedir
    with open(args.config_file, 'r') as file:
        lines = file.readlines()

    with open(args.config_file, 'w') as file:
        for line in lines:
            if line.strip() == filedir:
                file.write(f"{filedir} [processing]\n")
            else:
                file.write(line)

    # Process the filedir
    main(filedir)

    # Remove the [processing] tag from the filedir
    with open(args.config_file, 'r') as file:
        lines = file.readlines()

    with open(args.config_file, 'w') as file:
        for line in lines:
            if line.strip() == f"{filedir} [processing]":
                file.write(f"{filedir}\n")
            else:
                file.write(line)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process SPICE data.')
    parser.add_argument('config_file', type=str, help='Path to the configuration file.')
    args = parser.parse_args()

    # Read the filedirs from the configuration file
    with open(args.config_file, 'r') as file:
        filedirs = [line.strip() for line in file.readlines()]

    # Process each filedir
    for filedir in filedirs:
        process_filedir(filedir)
        