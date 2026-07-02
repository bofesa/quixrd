from XPAD_XRD_nxs_export import *
from XRD_funcs import sort_nxs_by_sample
from XRD_spectra_anal import Spectrum

if __name__ == "__main__":
    root_dir = 'C:\\Users\\bosa\\OneDrive - empa.ch\\WFH\\Synchrotron\\SOLEIL Jun2026\\'
    # s = S140XRD(nxs_file_directory = root_dir, export_directory = './export/',
    #             flat_file_directory = 'C:\\Users\\bosa\\OneDrive - empa.ch\\WFH\\Synchrotron\\SOLEIL Jun2026\\putz\\flat\\', flat_file_numbers = [39,], 
    #              daterange = [20260609, 20260615])
    # s.extract_S140XRD_idx(scanNo=1515, incl_q = True)
    # s.extract_S140XRD_chidelta(scanNo=1515, incl_q = True)
    # s.extract_S140XRD(scanNo=350, incl_q = True, showGraph = True)
    # s.batch_extract_S140XRD(scanNos = [1515], incl_q = True, showGraph = True)
    # s.batch_extract_S140XRD_chidelta(scanNos = range(440, 450+1), incl_q = True, showGraph = True)

    # for pI in range(0, 15):
    #     s.s140_visu_one_rawImage(scanNo = 1515, pointIndex = pI, logScale = True, mini = -1, maxi = -1, save_fig_path = './scan_1515_images/')

    ### TO DO: PLOT A CONTINUOUS SERIES OF 2D SCANS WITH OVERLAP




    spec = Spectrum(directory = './export/')
    spec.plot_Ivs2theta(scanNos = range(440, 450+1), plot_only = ['chi'], offset = 0.4, normalise = None, single_chi = False)


    # sort_nxs_by_sample(nxs_directory = r'C:\Users\bosa\OneDrive - empa.ch\WFH\Synchrotron\SOLEIL Jun2026',
    #                    sample_file = r'C:\Users\bosa\OneDrive - empa.ch\WFH\Synchrotron\SOLEIL Jun26 Samples.xlsx',
    #                    output_directory = 'C:\\Users\\bosa\\OneDrive - empa.ch\\WFH\\Synchrotron\\SOLEIL Jun2026 sorted_nxs',
    #                    export_calibrations = 'sub')

    # sort_nxs_by_sample(nxs_directory = r"G:\Limit\Barbara\Soleil_2026_06",
    #                    sample_file = r'C:\Users\bosa\OneDrive - empa.ch\WFH\Synchrotron\SOLEIL Jun26 Samples.xlsx',
    #                    output_directory = r"N:\Vol1-Th\Abt206\Sam Bojarski N\Synchrotron\Sorted_Data SOLEIL_2026_06",
    #                    export_calibrations = 'sub')



    input("Press Enter to exit...")
