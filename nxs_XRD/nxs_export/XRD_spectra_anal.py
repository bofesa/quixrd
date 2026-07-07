from XRD_funcs import *
from XPAD_XRD_nxs_export import *

from glob import glob

class Spectrum():
    def __init__(self, directory: str):
        """
        Args:
            directory (str): path to the directory containing exported txt or csv files (converted from nxs files)
        """
        self.file_directory = directory


    def plot_Ivs2theta(self, scanNos: int | list[int], plot_only: str | list[str] = None, offset: float = 0.0, normalise: str=None, single_chi: bool=False, label: str | list = []):
        """
        Plots the spectra for the specified scan numbers.
        args:
            scanNos (int | list): scan number(s) to plot
            plot_only (str | list[str]): list of scan types to plot (e.g. 'chi', 'delta', 'z', 'omega')
            offset (float): offset to apply to the intensity values for better visibility.
                This is a multiplier applied to the minimum intensity value of the first scan in the list. For example, an offset of 0.1 will add 10% of the minimum intensity value to all subsequent scans.
            normalise (str): method to normalise the intensity values ('zero' to normalise on the lowest-angle point, 'min' to normalise on the minimum intensity value, 'max' to normalise on the maximum intensity value)
            single_chi (bool): whether to plot only the first 'chi' scan for each scan number
            label (str): whether to add information to scan label. None for no additional information; 'type' for scan type; 'temp' for temperature, 'time' for time
        """
        if isinstance(scanNos, int):
            scanNos = [scanNos]

        if isinstance(label, str):
            label = [label]

        scan_types = ['chi', 'delta', 'z', 'omega']
        if plot_only is None:
            plot_only = ['chi', 'delta', 'z', 'omega']

        if isinstance(plot_only, str):
            plot_only = [plot_only]

        if normalise not in [None, 'zero', 'min', 'max']:
            print(f"Invalid normalisation method '{normalise}' specified. No normalisation will be applied.")
            normalise = None

        offset_value = 0.0  # Initialize the offset value for plotting

        # Set up the color map for plotting
        cmap = plt.get_cmap('jet')
        norm = plt.Normalize(vmin=0, vmax=len(scanNos)-1)
        fig, ax = plt.subplots(figsize=(10, 6))

        ask_flag = True  # Flag to determine if we should ask for scan type for unlabelled scans
        ask_again_flag = True  # Flag to determine if a selection has been made to remember the scan type for the rest of the scans
        for jdx, scanNo in enumerate(scanNos):
            idxs, twothetas, intensities = [], [], []  # Initialize arrays to hold the data for each scan
            try:
                froot = f"I_vs_2th_{scanNo}_"
                fnames = glob(os.path.join(self.file_directory, f"{froot}*.txt"))
                if not fnames:
                    print(f"File for scan number {scanNo} could not be found.")
                    continue
                # Determine the format of the scan based on the filename
                if len(os.path.basename(fnames[0]).replace(froot, '').split('_')) == 2:
                    # This is a labelled scan (e.g. chi, delta, z, etc.)
                    fformat = 'labelled'
                    scan_type = os.path.basename(fnames[0]).replace(froot, '').split('_')[0]
                    if scan_type not in scan_types:
                        print(f"Scan type '{scan_type}' for scan {scanNo} is not recognized. Skipping this scan.")
                        continue
                elif len(os.path.basename(fnames[0]).replace(froot, '').split('_')) == 1:
                    # This is an unlabelled scan
                    fformat = 'unlabelled'
                    typedict = {0: 'chi', 1: 'delta', 2: 'z', 3: 'omega'}
                    if ask_flag:
                        typeq = input(f"The {len(fnames)} files for scan {scanNo} are unlabelled. Please enter scan type (0: chi, 1: delta, 2: z, 3: omega): ")
                        if ask_again_flag:
                            ask_again = input('Enter any key to remember this selection for the rest of the scans, or "Enter" to ask again for each scan: ')
                            ask_again_flag = False
                        if ask_again:
                            ask_flag = False
                        scan_type = typedict.get(int(typeq), 'unknown')
                if scan_type == 'unknown':
                    print(f"Invalid scan type entered for scan {scanNo}. Skipping this scan.")
                    continue
                # Check if the scan type is in the list of types to plot
                if scan_type not in plot_only:
                    print(f"Scan type '{scan_type}' for scan {scanNo} is not in the list of types to plot. Skipping this scan.")
                    continue

                print(f"Processing scan {scanNo} of type '{scan_type}' with {len(fnames)} files.")
                for fname in fnames:
                    # Load the data from the files
                    idx = int(os.path.basename(fname).replace(froot, '').replace('.txt', '').split('_')[-1])
                    idxs.append(idx)
                    try:    # New files have only commented lines in the preamble and header, so try loading directly first
                        data = np.loadtxt(fname, delimiter=" ")
                    except:
                        try:    # Legacy files had a string header line, so try skipping the first line if the above fails
                            data = np.loadtxt(fname, delimiter=" ", skiprows=1)
                        except:
                            print(f"Could not load data from file {fname}. Skipping this file.")
                            continue
                    two_theta = data[:, 0]
                    intensity = data[:, 1]
                    twothetas.append(two_theta)
                    intensities.append(intensity)
                    # Look for metadata in file preamble
    ### YET TO DO
                # Apply normalisation if specified
                if scan_type == 'delta':
                    first_scan = idxs.index(min(idxs))  # Find the index of the first scan for delta scans
                    I0 = intensities[first_scan][0]
                    Imin = min(np.min(arr) for arr in intensities)
                    Imax = max(np.max(arr) for arr in intensities)
                else:
                    I0 = -1  # Placeholder for non-delta scans

                for idx, two_theta, intensity in zip(idxs, twothetas, intensities):
                    # Apply normalisation if specified
                    try:
                        if normalise is None:
                            intensity = intensity  # No normalisation
                        elif normalise == 'zero' and not I0 == 0:
                            intensity = intensity / I0  if scan_type == 'delta' else intensity/intensity[0]  # Normalise on the lowest-angle point
                        elif normalise == 'min':
                            intensity = intensity / Imin if scan_type == 'delta' else intensity/np.min(intensity)  # Normalise on the minimum intensity value
                        elif normalise == 'max':
                            intensity = intensity / Imax if scan_type == 'delta' else intensity/np.max(intensity)  # Normalise on the maximum intensity value
                    except Exception as e:
                        print(f"Error occurred while normalizing scan {scanNo}: {e}")
                        intensity = intensity  # If normalisation fails, use the original intensity values
                    
                    # intensity += offset * min(np.min(arr) for arr in intensities) * jdx  # Apply the offset to the intensity values

                    # Plot the data
                    if idx == min(idxs):  # Only label the first scan of each type for clarity
                        offset_value += offset * min(np.min(arr) for arr in intensities)  # Update the offset value for subsequent scans
                        label_str = ''
                        if 'type' in label:
                            label_str += f" {scan_type}"
                        if 'temp' in label:
                            # Data are in the preamble of the file as "# Temperature: 25"
                            try:
                                with open(fname, 'r') as f:
                                    for line in f:
                                        if line.startswith("# Temperature:"):
                                            temp = line.split(":")[1].strip()
                                            label_str += f" {temp}°C"
                                            break
                            except Exception as e:
                                print(f"Could not read temperature from file {fname}: {e}")
                        if 'time' in label:
                            # Data are in the preamble of the file as "# Start Time: 2026-06-14T09:03:45"
                            try:
                                with open(fname, 'r') as f:
                                    for line in f:
                                        if line.startswith("# Start Time:"):
                                            time = line.replace("# Start Time:", "").strip()
                                            label_str += f" {time}"
                                            break
                            except Exception as e:
                                print(f"Could not read time from file {fname}: {e}")

                        ax.plot(two_theta, intensity + offset_value, '.-', label=f"Scan {scanNo} ({label_str.strip()})", color=cmap(norm(jdx)))
                    else:
                        if single_chi and scan_type == 'chi':
                            continue  # Skip plotting additional chi scans if single_chi is True
                        ax.plot(two_theta, intensity + offset_value, '.-', color=cmap(norm(jdx)))

                ax.set_xlabel('2$\\theta$ (°)')
                ax.set_ylabel('Intensity (a.u.)')
                ax.set_yticklabels([])
                if len(scanNos) == 1:
                    scan_context = f"scan {scanNos[0]}"
                else:
                    scan_context = f"scans {min(scanNos)}-{max(scanNos)}"
                ax.set_title(f"XRD spectra - {scan_context}; types: {', '.join(plot_only)}")
                ax.grid(True, which='both')
                ax.legend(fontsize='small', ncol=legend_columniser(len(scanNos)))
                plt.show()

            except Exception as e:
                print(f"Error while attempting to plot scan {scanNo}: {e}")
                continue
