import h5py, numpy, tables, os, time, math
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.optimize import curve_fit  #fitting
from glob import glob
import shutil
import numpy as np
import datetime as dt
import pandas as pd
import re


def funct_pearson7(x, backgr, slopeLin, amplitude, center, fwhmLike, exposant):
	PI = numpy.pi
	return backgr+slopeLin*x+amplitude*(1+((x-center)/fwhmLike)**2.0)**(-exposant) #p7


def make_subplot_grid(n: int, figsize=(10, 6)):
    """ Create a near-square subplot grid for n datasets.
    args:
        n : number of datasets (and thus subplots) to create
    Returns:
        fig, axes (flattened 1D array of axes)
    """
    if not isinstance(n, int):
        print('converting n to integer')
        n = int(n)
    if n <= 0:
        raise ValueError("Number of datasets must be a positive integer.")
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    # flatten for easy iteration
    axes = axes.flatten() if n > 1 else [axes]
    # turn off unused axes
    for ax in axes[n:]:
        ax.axis("off")
    return fig, axes


def legend_columniser(len_legend: int, max_per_col=15):
    """ Determine the number of columns for a legend based on the number of items.
    args:
        len_legend : number of items in the legend
        max_per_col : maximum number of items per column
    Returns:
        ncol : number of columns for the legend
    """
    if len_legend <= max_per_col:
        return 1
    else:
        return math.ceil(len_legend / max_per_col)
    

def sort_nxs_by_sample(nxs_directory: str, sample_file: str, output_directory: str = None,
                       export_calibrations: bool | str = True, ydm_format: bool = True):
    """ Sort .nxs files in a directory by sample name.
    args:
        nxs_directory : path to the directory containing .nxs files
        sample_file : path to the file containing sample names and numbers
        output_directory : path to the directory where sorted files will be saved
        export_calibrations : whether to copy calibration files as well. True: copy calibration files, False: do not copy calibration files, 'sub': copy calibration files into subfolders of the sample folders
        ydm_format : whether to look only for subfolders in yyyy-mm-dd format (True)
    Returns:
    """
    from quixrd.nxs_export.XPAD_XRD_nxs_export import nxs_fileName_root, nxs_fileName_suffix

    try:
        if not os.path.exists(nxs_directory):
            raise ValueError(f"The specified nxs_directory does not exist: {nxs_directory}")
        if not os.path.isfile(sample_file):
            raise ValueError(f"The specified sample_file does not exist: {sample_file}")
    except Exception as e:
        print(f"Error: {e}")
        return None
    # Implementation for sorting .nxs files into subfolders by sample name
    if output_directory is None:
        output_directory = os.path.join(nxs_directory, "sorted")

    if export_calibrations not in [True, False, 'sub']:
        print(f"Invalid value for export_calibrations: {export_calibrations}. Defaulting to True.")
        export_calibrations = True

    os.makedirs(output_directory, exist_ok=True)

    # import excel file with sample names and numbers
    sample_df = pd.read_excel(sample_file)  # columns 'Sample #', 'Sample', First Scan, Last Scan, First Alignment Scan, Last Alignment Scan

    ### Check for nxs files in the directory, and if not then take all subfolders and check for nxs files in them
    sample_nxs_files = glob(os.path.join(nxs_directory, "*.nxs"))
    if not sample_nxs_files:
        subfolders = [f.path for f in os.scandir(nxs_directory) if f.is_dir()]
        for subfolder in subfolders:
            sample_nxs_files.extend(glob(os.path.join(subfolder, "*.nxs")))
        if not sample_nxs_files:
            raise ValueError(f"No .nxs files found in the specified directory or its subfolders: {nxs_directory}")
        else:
            subfolder_flag = True
    else:
        subfolder_flag = False
    
    if ydm_format:
        subfolders = [s for s in subfolders if re.match(r'\d{4}-\d{2}-\d{2}', os.path.basename(s))]  # Filter to only include subfolders called yyyy-mm-dd
    
    for _, row in sample_df.iterrows():
        sample_number = 0 if pd.isna(row['Sample #']) else int(row['Sample #'])  # 'Sample #' column, zero-padded to 4 digits, set to 0 if NaN
        sample_name = row['Sample']
        first_scan = -1 if pd.isna(row.iloc[2]) else int(row.iloc[2])  # 'First Scan' column
        last_scan = first_scan if pd.isna(row.iloc[3]) else int(row.iloc[3])  # 'Last Scan' column, if NaN assume it's the same as 'First Scan'
        alignment_first_scan = -1 if pd.isna(row.iloc[4]) else int(row.iloc[4])  # 'First Alignment Scan' column, set to -1 if NaN to indicate no alignment scans
        alignment_last_scan = -1 if (pd.isna(row.iloc[5]) or alignment_first_scan == -1) else int(row.iloc[5])  # 'Last Alignment Scan' column, set to -1 if NaN to indicate no alignment scans

        # Create a subfolder for the sample
        bad_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
        for char in bad_chars:
            sample_name = sample_name.replace(char, '_')  # Replace bad characters with underscore
        sample_folder = os.path.join(output_directory, f"{str(sample_number).zfill(2)}_{sample_name}")

        # Copy the .nxs files for the specified scan range into the sample folder
        startscan, endscan = -1, -1
        if export_calibrations == True or export_calibrations == 'sub':
            startscan = min([x for x in [first_scan, alignment_first_scan] if x >= 0])  # Start from the earliest scan, whether it's a regular scan or an alignment scan
            endscan = max([x for x in [last_scan, alignment_last_scan] if x >= 0])
        elif export_calibrations == False:
            startscan, endscan = first_scan, last_scan
            
        exported_scans = []
### CHANGE to instead use the list of sample_nxs_files as a start, and then match to the scan numbers in the sample_df
        for scan_no in range(startscan, endscan + 1):
            if scan_no < 0.1:
                continue  # Skip negative scan numbers, which indicate no scans for this sample
            # print(f"Searching for scan number {scan_no} for sample #{sample_number} - {sample_name}...")
            # Need to be able to search for .nxs files in subfolders if they are not in the main directory
            if subfolder_flag:
                nxs_file_path = None
                for subfolder in subfolders:
                    nxs_file_name = f"{nxs_fileName_root}{str(scan_no).zfill(4)}{nxs_fileName_suffix}"
                    potential_path = os.path.join(subfolder, nxs_file_name)
                    if os.path.isfile(potential_path):
                        nxs_file_path = potential_path
                        break
                if nxs_file_path:
                    os.makedirs(sample_folder.zfill(2), exist_ok=True)      # Sample numbers zero-padded to 2 digits
                    if export_calibrations == 'sub' and scan_no in range(alignment_first_scan, alignment_last_scan + 1):
                        # Create a subfolder for the calibration files within the sample folder
                        calib_subfolder = os.path.join(sample_folder, "calibrations")
                        os.makedirs(calib_subfolder, exist_ok=True)
                        shutil.copy(nxs_file_path, calib_subfolder)
                        exported_scans.append(scan_no)
                        print(f"Copied calibration file for scan number {scan_no} to {calib_subfolder}.")
                    else:
                        shutil.copy(nxs_file_path, sample_folder)
                        exported_scans.append(scan_no)
                        print(f"Copied .nxs file for scan number {scan_no} to {sample_folder}.")
                else:
                    print(f"Warning: .nxs file for scan number {scan_no} not found in any subfolder of {nxs_directory}.")
            # If the .nxs files are in the main directory, copy them directly
            else:
                nxs_file_name = f"{nxs_fileName_root}{str(scan_no).zfill(4)}{nxs_fileName_suffix}"
                nxs_file_path = os.path.join(nxs_directory, nxs_file_name)
                if os.path.isfile(nxs_file_path):
                    os.makedirs(sample_folder.zfill(2), exist_ok=True)      # Sample numbers zero-padded to 2 digits
                    if export_calibrations == 'sub' and scan_no in range(alignment_first_scan, alignment_last_scan + 1):
                        # Create a subfolder for the calibration files within the sample folder
                        calib_subfolder = os.path.join(sample_folder, "calibrations")
                        os.makedirs(calib_subfolder, exist_ok=True)
                        shutil.copy(nxs_file_path, calib_subfolder)
                        exported_scans.append(scan_no)
                        print(f"Copied calibration file for scan number {scan_no} to {calib_subfolder}.")
                    else:
                        shutil.copy(nxs_file_path, sample_folder)
                        exported_scans.append(scan_no)
                        print(f"Copied .nxs file for scan number {scan_no} to {sample_folder}.")
                else:
                    print(f"Warning: .nxs file for scan number {scan_no} not found in {nxs_directory}.")

    scan_range = range(int(min(sample_df.iloc[:, 2].min(), sample_df.iloc[:, 4].min())), int(max(sample_df.iloc[:, 3].max(), sample_df.iloc[:, 5].max())) + 1)
    # Export a summary of the exported scans and non-exported scans to a text file
    with open(os.path.join(output_directory, "export_summary.txt"), "w") as f:
        f.write("Exported Scans:\n")
        for scan_no in exported_scans:
            f.write(f"  Scan {scan_no}\n")
        f.write("\n\n\nNon-Exported Scans:\n")
        for scan_no in scan_range:
            if scan_no not in exported_scans:
                f.write(f"  Scan {scan_no}\n")

    return exported_scans