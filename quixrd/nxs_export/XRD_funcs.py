# NXS export helper functions credited to Pierre-Olivier Renault and the
# original SOLEIL XPAD-S140 NXS export-function authors; locally adapted here.

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
                       export_calibrations: bool | str = True, export_extras: bool = True, ydm_format: bool = True):
    """ Sort .nxs files in a directory by sample name.
    args:
        nxs_directory : path to the directory containing .nxs files
        sample_file : path to the file containing sample names and numbers
        output_directory : path to the directory where sorted files will be saved
        export_calibrations : whether to copy calibration files as well. True: copy calibration files, False: do not copy calibration files, 'sub': copy calibration files into subfolders of the sample folders
        export_extras : whether to copy files for scans that are not listed in the sample file
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
    
    # Create a mapping of scan numbers to file paths for faster lookup
    scan_to_file = {}
    for nxs_file in sample_nxs_files:
        match = re.search(r'(\d{4})', os.path.basename(nxs_file))  # Extract the scan number from the file name
        if match:
            scan_number = int(match.group(1))
            scan_to_file[scan_number] = nxs_file

    # create a mapping of scan numbers to sample numbers and names for faster lookup
    sample_ranges = {}
    exported_scans = []
    for _, row in sample_df.iterrows():
        sample_number = 0 if pd.isna(row['Sample #']) else int(row['Sample #'])  # 'Sample #' column, zero-padded to 4 digits, set to 0 if NaN
        sample_name = row['Sample']
        first_scan = None if pd.isna(row.iloc[2]) else int(row.iloc[2])  # 'First Scan' column
        last_scan = first_scan if pd.isna(row.iloc[3]) else int(row.iloc[3])  # 'Last Scan' column, if NaN assume it's the same as 'First Scan'
        alignment_first_scan = None if pd.isna(row.iloc[4]) else int(row.iloc[4])  # 'First Alignment Scan' column, set to None if NaN to indicate no alignment scans
        alignment_last_scan = None if (pd.isna(row.iloc[5]) or alignment_first_scan is None) else int(row.iloc[5])  # 'Last Alignment Scan' column, set to None if NaN to indicate no alignment scans
        
        print(f"Searching for scans for sample {row['Sample']} (Sample # {row['Sample #']})...")

        bad_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
        for char in bad_chars:
            sample_name = sample_name.replace(char, '_')  # Replace bad characters with underscore
        
        sample_ranges[sample_number] = {
            "name": sample_name,
            "first_scan": first_scan,
            "last_scan": last_scan,
            "alignment_first_scan": alignment_first_scan,
            "alignment_last_scan": alignment_last_scan
        }

        for scan_number in range(first_scan, last_scan + 1):
            if scan_number >= 0:
                nxs_file = scan_to_file.get(scan_number)
                if nxs_file:
                    sample_folder = os.path.join(output_directory, f"{str(sample_number).zfill(2)}_{sample_name}")
                    os.makedirs(sample_folder, exist_ok=True)
                    shutil.copy(nxs_file, sample_folder)
                    exported_scans.append(scan_number)
                    print(f"    Copied .nxs file for scan number {str(scan_number).zfill(4)} to {sample_folder}.")

        if export_calibrations == True and alignment_first_scan is not None and alignment_last_scan is not None:
            for scan_number in range(alignment_first_scan, alignment_last_scan + 1):
                if scan_number >= 0:
                    nxs_file = scan_to_file.get(scan_number)
                    if nxs_file:
                        sample_folder = os.path.join(output_directory, f"{str(sample_number).zfill(2)}_{sample_name}")
                        os.makedirs(sample_folder, exist_ok=True)
                        shutil.copy(nxs_file, sample_folder)
                        exported_scans.append(scan_number)
                        print(f"    Copied calibration file for scan number {str(scan_number).zfill(4)} to {sample_folder}.")

        elif export_calibrations == 'sub' and alignment_first_scan is not None and alignment_last_scan is not None:
            for scan_number in range(alignment_first_scan, alignment_last_scan + 1):
                if scan_number >= 0:
                    nxs_file = scan_to_file.get(scan_number)
                    if nxs_file:
                        sample_folder = os.path.join(output_directory, f"{str(sample_number).zfill(2)}_{sample_name}")
                        calib_subfolder = os.path.join(sample_folder, "calibrations")
                        os.makedirs(calib_subfolder, exist_ok=True)
                        shutil.copy(nxs_file, calib_subfolder)
                        exported_scans.append(scan_number)
                        print(f"    Copied calibration file for scan number {str(scan_number).zfill(4)} to {calib_subfolder}.")

    total_scan_range = range(int(min(sample_df.iloc[:, 2].min(), sample_df.iloc[:, 4].min())), int(max(sample_df.iloc[:, 3].max(), sample_df.iloc[:, 5].max())) + 1)
    # Export a summary of the exported scans and non-exported scans to a text file
    with open(os.path.join(output_directory, "export_summary.txt"), "w") as f:
        f.write("Exported Scans:\n")
        for scan_no in sorted(exported_scans):
            f.write(f"  Scan {str(scan_no).zfill(4)}\n")

        if export_extras:
            f.write("\n Scans exported without sample:\n")
            for scan_no in sorted(scan_to_file.keys()):
                if scan_no not in exported_scans:
                    nxs_file = scan_to_file.get(scan_no)
                    extra_folder = output_directory     # Write extra scans to the main output directory
                    os.makedirs(extra_folder, exist_ok=True)
                    shutil.copy(nxs_file, extra_folder)
                    f.write(f"  Scan {str(scan_no).zfill(4)} (exported to extra_scans folder)\n")
        else:
            f.write("\n\n\nNon-Exported Scans:\n")
            for scan_no in sorted(scan_to_file.keys()):
                if scan_no not in exported_scans:
                    f.write(f"  Scan {str(scan_no).zfill(4)}\n")
