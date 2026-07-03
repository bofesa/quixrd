from XPAD_XRD_nxs_export import *
from XRD_funcs import sort_nxs_by_sample
from XRD_spectra_anal import Spectrum
# root_dir = 
root_dir = "G:\\Limit\\Barbara\\Soleil_2026_06\\"
RUNSTEPS = [5]
scanNo = 1515
scanNos = [1559, 1570, 1304]

def extract_nxs():
    # Create an instance of the S140XRD class with the specified directories and parameters
#LOCAL
    # s = S140XRD(nxs_file_directory = 'C:\\Users\\bosa\\OneDrive - empa.ch\\WFH\\Synchrotron\\SOLEIL Jun2026\\', export_directory = './export/',
    #             flat_file_directory = 'C:\\Users\\bosa\\OneDrive - empa.ch\\WFH\\Synchrotron\\SOLEIL Jun2026\\putz\\flat\\', flat_file_numbers = [39,], 
    #              daterange = [20260609, 20260615])
#BABSI LIMIT
    s = S140XRD(nxs_file_directory = "G:\\Limit\\Barbara\\Soleil_2026_06\\",
                export_directory = 'N:\\Vol1-Th\\Abt206\\Sam Bojarski N\\Synchrotron\\export\\',
                flat_file_directory = 'C:\\Users\\bosa\\OneDrive - empa.ch\\WFH\\Synchrotron\\SOLEIL Jun2026\\putz\\flat\\', flat_file_numbers = [39,], 
                daterange = [20260609, 20260615])

    
    #1 Extract one scan to single csv file
    if 1 in RUNSTEPS:
        s.extract_S140XRD_idx(scanNo=scanNo, incl_q = True)

    #2 Extract one scan with scantype labelling
    if 2 in RUNSTEPS:
        s.extract_S140XRD_chidelta(scanNo=scanNo, incl_q = True)

    #3 Batch extract a range of scans to single csv file
    if 3 in RUNSTEPS:
        s.batch_extract_S140XRD(scanNos = scanNos, incl_q = True, showGraph = False)

    #4 Batch extract a range of scans with scantype labelling
    if 4 in RUNSTEPS:
        s.batch_extract_S140XRD_chidelta(scanNos = scanNos, incl_q = True, showGraph = False, saveGraph = True)

    #5 Batch extract a range of scans with scantype labelling from a sorted directory structure (sorted by sample)
    if 5 in RUNSTEPS:
#SORTED
        s = S140XRD(nxs_file_directory = r"N:\Vol1-Th\Abt206\Sam Bojarski N\Synchrotron\Sorted_Data SOLEIL_2026_06",
                    export_directory = 'N:\\Vol1-Th\\Abt206\\Sam Bojarski N\\Synchrotron\\export_sorted\\',
                    flat_file_directory = 'C:\\Users\\bosa\\OneDrive - empa.ch\\WFH\\Synchrotron\\SOLEIL Jun2026\\putz\\flat\\', flat_file_numbers = [39,], 
                    daterange = [20260609, 20260615])
    # export to csv
        s.batch_extract_S140XRD(scanNos = scanNos, incl_q = True, showGraph = False, saveGraph = True, mirror_sorted_structure = True)
    # export each frame to txt with scantype labelling
        s.batch_extract_S140XRD_chidelta(scanNos = scanNos, incl_q = True, showGraph = False, saveGraph = False, mirror_sorted_structure = True)

    # for pI in range(0, 15):
    #     s.s140_visu_one_rawImage(scanNo = 1515, pointIndex = pI, logScale = True, mini = -1, maxi = -1, save_fig_path = './scan_1515_images/')

### TO DO:
#   PLOT A CONTINUOUS SERIES OF 2D SCANS WITH OVERLAP
#   ADD ABILITY TO IMPORT DATA FROM SORTED DIRECTORIES, MIRRORING THE STRUCTURE OF THE SORTED DIRECTORIES


def plot_nxs(scans, scan_types = ['chi', 'delta', 'z', 'omega']):
    dir = "C:\\Users\\bosa\\local_code\\33_130113 HEN\\"
    spec = Spectrum(directory = dir)
    spec.plot_Ivs2theta(scanNos = scans, plot_only = scan_types, offset = 0.0, normalise = None, single_chi = False, label = ['temp'])


def sort_nxs():
    sort_nxs_by_sample(nxs_directory = r'C:\Users\bosa\OneDrive - empa.ch\WFH\Synchrotron\SOLEIL Jun2026',
                       sample_file = r'C:\Users\bosa\OneDrive - empa.ch\WFH\Synchrotron\SOLEIL Jun26 Samples.xlsx',
                       output_directory = 'C:\\Users\\bosa\\OneDrive - empa.ch\\WFH\\Synchrotron\\SOLEIL Jun2026 sorted_nxs',
                       export_calibrations = 'sub')

    sort_nxs_by_sample(nxs_directory = r"G:\Limit\Barbara\Soleil_2026_06",
                       sample_file = r'C:\Users\bosa\OneDrive - empa.ch\WFH\Synchrotron\SOLEIL Jun26 Samples.xlsx',
                       output_directory = r"N:\Vol1-Th\Abt206\Sam Bojarski N\Synchrotron\Sorted_Data SOLEIL_2026_06",
                       export_calibrations = 'sub')


if __name__ == "__main__":
    # extract_nxs()
    delta_range = [x for x in np.arange(2150, 2200, 4)]+[x for x in np.arange(2200, 2300, 9) if x not in [2218]]+[x for x in np.arange(2300, 2330, 4)]+[x for x in np.arange(2330, 2560, 15)]
    chi_range = [x for x in np.arange(2200, 2560, 5) if x not in [2218]]
    srange = np.arange(2157, 2200, 1)
    plot_nxs(scans = 2500, scan_types = ['chi'])


    input("Press Enter to exit...")
