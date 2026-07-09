# NXS/XPAD export routines credited to Pierre-Olivier Renault and the
# original SOLEIL XPAD-S140 NXS export-function authors; locally adapted here.

import h5py, numpy, tables, os, time, math, re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy import ndimage
from scipy.optimize import curve_fit  #fitting
from glob import glob
import shutil
import numpy as np
import datetime as dt
import pandas as pd
import math

try:
    from .XRD_funcs import *
except ImportError:
    from XRD_funcs import *

plt.ion() #interactive mode on (for plotting the figures)

DIAGNOSTIC = False # if True, print out lots of information about processing scans

# year = 2026; month1 = 6; month2 = 6; day1 = 9; day2 = 15 #date limits for which the Nxs data will be searched

### --- Fixed metadata --- ###

nxs_fileName_root = "scan_" #root name of the nxs files
nxs_fileName_suffix = "_0001.nxs" #suffix of the nxs files names

# flat_pathFolder = pathRoot + "flat/" #folder of the flat data

flat_fileName_root = "scan_" #root name of the nxs files
# flat_scanNo1 = 39; flat_scanNo2 = 39; #first and last scan of the flat timescan series (should be contiguous)

### calibration of the XPAD-S140 (pixels / deg., pos. of direct beam, ...)
### mar. 2025
calib_pixels_per_deg = 93.5  # pixels in 1 deg.
XcenDetector = 319+3*3; YcenDetector = 119+1.5 # position of direct beam on xpad at (deltaOffset, gamOffset). Use the 'corrected' positions (add 3 pixels whenever cross 80*i in X and 120 in Y)
deltaOffset = 16.; gamOffset  = 0.; # positions in diffracto angles for which the above values XcenDetector, YcenDetectors are reported

#Dec2. 022
gamShift_value = -0.1+0.063+0.077; #correction of the gam angle (after fit optimization) of the diffractometer, single point in delta scan
### correction to be implemented for all delta angles of the diffractometer
flag_gamShift_corr_deltaScan = True; #if set to True, implement the poly2 dependency of the gamShift as function of delta_diffracto
#these next parameters are meaningful only if the above flag is True. If not, the values are ignored
poly2_gamShift_corr_a = 0.001682 #poly2 = a(x-x0)^2 + b(x-x0)x +c
poly2_gamShift_corr_b = 0.418085
poly2_gamShift_corr_c = -40.188170
poly2_gamShift_corr_x0 = -51.606016

# mar. 2025
energyShift = -0.035 # cf. logbook#82 p.66

#psi-range to be extracted from the XPAD image (use very large values for full range, e.g. -1000 and +1000)
psi_mini = -1000.; psi_maxi = 1000.
psi1 = psi_mini; psi2 = psi_maxi;

Izero_thresh = -1 #value for the monitor, only data for which monitor is >= thresholds is converted into XRD curves (can prevent for example shutter close, beam loss, ...)
Izero_norm_flag = False #normalize or not with Izero (Izero monitor is data_02, set in the code, not as a parameter)

factorIdoublePixel = 1.
#   set to 1 when using the flat field, set to negative if need to mask
#   set to about 2.5 when not using the flat field but need to 'spread' the 'double' pixels
#   set to negative if need to mask

numberOfDigits = 4 #number of digits in the file Name (to be filled with zeros up to this value)
datasetIndex_max = 100 #number max of datasets in the counters to be tested to find XPAD dataset
stepTwoTh = 1.0/calib_pixels_per_deg #will be the step used for generating the XRD (I vs. 2theta) curves

#geometry information (XPAD - S140) which will be used for all the codes
numberOfModules = 2; numberOfChips = 7; # detector dimension, XPAD S-140
chip_sizeX = 80; chip_sizeY = 120; # chip dimension, in pixels (X = horiz, Y = vertical)
lines_to_remove_array = [0, -3]; # adding 3 more lines, corresponding to the double pixels on the last and 1st line of the modules

stepTwoTh = 1.0/calib_pixels_per_deg #will be the step used for generating the XRD (I vs. 2theta) curves

deg2rad = numpy.pi/180; inv_deg2rad = 1/deg2rad; #used as the functions numpy.deg2rad / numpy.rad2deg
distance = calib_pixels_per_deg/numpy.tan(1.0*deg2rad); # distance xpad to sample, in pixel units
#print "sample-detector distance = %f pixels	= %f mm" %(distance, distance*0.13)


def _scan_no_from_nxs_name(filename):
    match = re.fullmatch(
        rf"{re.escape(nxs_fileName_root)}(\d+){re.escape(nxs_fileName_suffix)}",
        os.path.basename(filename),
    )
    return int(match.group(1)) if match else None


def _scan_no_from_export_name(filename):
    basename = os.path.basename(filename)
    match = re.match(r"I_vs_2th_(\d+)_.*\.txt$", basename)
    if match:
        return int(match.group(1))
    match = re.match(r"scan_(\d+)_.*\.(csv|png)$", basename)
    if match:
        return int(match.group(1))
    return None


def _make_plot_figure(showGraph):
    if showGraph:
        return plt.subplots()
    fig = Figure()
    FigureCanvasAgg(fig)
    return fig, fig.subplots()


def _build_sorted_nxs_scan_map(sorted_nxs_directory):
    sorted_root = os.path.abspath(sorted_nxs_directory)
    scan_map = {}
    for root, _, files in os.walk(sorted_root):
        for filename in files:
            scan_no = _scan_no_from_nxs_name(filename)
            if scan_no is None:
                continue
            if scan_no in scan_map:
                continue
            full_path = os.path.join(root, filename)
            rel_folder = os.path.relpath(root, sorted_root)
            if rel_folder == ".":
                rel_folder = ""
            scan_map[scan_no] = (full_path, rel_folder)
    return scan_map


def sort_extracted_by_sample(source_export_directory, sorted_nxs_directory, output_directory=None, move=False):
    """Copy or move already-exported XRD files into folders matching a sorted NXS tree."""
    source_root = os.path.abspath(source_export_directory)
    output_root = os.path.abspath(output_directory or source_export_directory)
    scan_map = _build_sorted_nxs_scan_map(sorted_nxs_directory)
    transferred = []
    unmatched = []

    os.makedirs(output_root, exist_ok=True)
    for root, _, files in os.walk(source_root):
        if output_root == source_root and os.path.abspath(root) != source_root:
            continue
        for filename in files:
            scan_no = _scan_no_from_export_name(filename)
            if scan_no is None:
                continue
            source_path = os.path.join(root, filename)
            if scan_no not in scan_map:
                unmatched.append(source_path)
                continue
            _, rel_folder = scan_map[scan_no]
            target_folder = os.path.join(output_root, rel_folder)
            os.makedirs(target_folder, exist_ok=True)
            target_path = os.path.join(target_folder, filename)
            if os.path.abspath(source_path) == os.path.abspath(target_path):
                continue
            if move:
                shutil.move(source_path, target_path)
                action = "moved"
            else:
                shutil.copy2(source_path, target_path)
                action = "copied"
            transferred.append({"scan": scan_no, "source": source_path, "target": target_path, "action": action})

    return {
        "source_export_directory": source_root,
        "sorted_nxs_directory": os.path.abspath(sorted_nxs_directory),
        "output_directory": output_root,
        "move": bool(move),
        "transferred": transferred,
        "unmatched": unmatched,
        "scan_map": {scan: rel for scan, (_, rel) in scan_map.items()},
    }


class S140XRD():
    def __init__(self, nxs_file_directory = './', export_directory = '',
                 flat_file_directory = './flat/', flat_file_numbers: list = [],
                 daterange: list = [20260609, 20260615],
                 recursive_scan_lookup: bool = True):
        #folder of the nxs data; The NXS data are in sub-folders containing the year and the date
        if not os.path.exists(nxs_file_directory):
            if not os.path.exists(nxs_file_directory + '/'):
                raise ValueError(f"Provided nxs_file_directory '{nxs_file_directory}' does not exist.")
            else:
                self.nxsfile_directory = os.path.join(nxs_file_directory, "")
        else:
            self.nxsfile_directory = os.path.join(nxs_file_directory, "")
        #subfolder where the results are saved
        if export_directory == '' or export_directory is None: 
            self.export_directory = os.path.join(self.nxsfile_directory, "exported", "")
        else:
            self.export_directory = os.path.join(export_directory, "")
        # try:
        #     os.stat(export_directory)
        # except:
        #     os.mkdir(export_directory)
        self.flat_file_directory = os.path.join(flat_file_directory, "")
        if len(flat_file_numbers) == 0:
            print("Warning: No flat field scan number provided. Using default value of 39.")
            flat_file_numbers = [39] #default value, if not given as input parameter
        self.flat_file_numbers = flat_file_numbers
        self.daterange = self.reformat_daterange(daterange) # date limits for which the Nxs data will be searched
        self.flatImg_inv = None # will call flatImg_inv = read_flatField() when first needed, then keep it
        self.recursive_scan_lookup = recursive_scan_lookup
        self._scan_file_index = None

        # Names for different scan types, to be used in output file names
        self.scan_types = {'ascan_chi': 'chi', 'ascan_delta': 'delta', 'ascan_omega': 'omega', 'ascan_phi': 'phi', 'ascan_tzs': 'z', 'ascan_tys': 'y', 'ascan_txs': 'x',
                           'dscan_delta': 'delta', 'dscan_chi': 'chi', 'dscan_omega': 'omega', 'dscan_phi': 'phi', 'dscan_tzs': 'z', 'dscan_tys': 'y', 'dscan_txs': 'x',
                           'time_scan': 'time', }

        self.nxsdate_subfolder = '' # identified subfolder containing the data, identified from the folder names and the date limits given as input parameters. It is set when reading the metadata, and then used for reading the XPAD data

        ### OLD PATHS, FOR REFERENCE
        
        # pathRoot = "Z:/com-diffabs/2026/Run3B/putz/"
            # maps onto ...
        # pathRootData = "Z:/com-diffabs/2026/Run3/" 	#  contains the root name (on Ruche, go up to the 'Run' level) where the Nxs data is
            # maps onto ...
        # nxs_pathFolder_Root = pathRootData #folder of the nxs data; The NXS data are in sub-folders containing the year and the date
            # maps onto self.nxsfile_directory
        # pathSaveRoot = pathRoot + "exploited/" #subfolder where the results are saved

    def _scan_filename(self, scanNo):
        return nxs_fileName_root + str(scanNo).zfill(numberOfDigits) + nxs_fileName_suffix

    def _build_scan_file_index(self):
        if self._scan_file_index is not None:
            return self._scan_file_index
        self._scan_file_index = _build_sorted_nxs_scan_map(self.nxsfile_directory)
        return self._scan_file_index

    def _resolve_scan_path(self, scanNo):
        fileName = self._scan_filename(scanNo)
        direct_path = os.path.join(self.nxsfile_directory, fileName)
        if os.path.isfile(direct_path):
            return direct_path, ""

        if self.recursive_scan_lookup:
            scan_index = self._build_scan_file_index()
            if scanNo in scan_index:
                return scan_index[scanNo]

        if self.daterange:
            for datestring in self.daterange:
                dated_path = os.path.join(self.nxsfile_directory, str(datestring), fileName)
                if os.path.isfile(dated_path):
                    return dated_path, str(datestring)

        return None, ""

    def _export_path_for_scan(self, scanNo, mirror_sorted_structure=False):
        export_folder = self.export_directory
        if mirror_sorted_structure:
            _, rel_folder = self._resolve_scan_path(scanNo)
            if rel_folder:
                export_folder = os.path.join(self.export_directory, rel_folder)
        os.makedirs(export_folder, exist_ok=True)
        return os.path.join(export_folder, "")

    def _graph_path_for_scan(self, scanNo, export_folder, suffix):
        safe_suffix = str(suffix).strip().replace(" ", "_")
        if not safe_suffix.lower().endswith(".png"):
            safe_suffix += ".png"
        return os.path.join(export_folder, f"scan_{scanNo}_{safe_suffix}")

    def _date_parts_from_path(self, path):
        basename = os.path.basename(os.path.normpath(path))
        match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", basename)
        if not match:
            return None
        year, month, day = match.groups()
        return int(year), int(month), int(day)

    def reformat_daterange(self, daterange):
        if daterange is None:
            return None
        if len(daterange) == 0:
            return None
        try:
            reformatted_daterange = []
            for date in daterange:
                date = str(date)
                date = date.strip()
                for char in ['_', '-', '.']:
                    date = date.replace(char, '')
                year, month, day = date[0:4], date[4:6], date[6:8]
                reformatted_daterange.append(dt.datetime(int(year), int(month), int(day)))
            
            # reformatted_daterange = sorted(reformatted_daterange)
            startdate, enddate = min(reformatted_daterange), max(reformatted_daterange)
            contiguous_daterange = []
            for n in range((enddate-startdate).days + 1):
                contiguous_daterange.append((startdate + dt.timedelta(days=n)).strftime("%Y-%m-%d"))
            return contiguous_daterange

        except Exception as e:
            print('Error extracting daterange: ', e)
            return None


    #======================================================
    #optimizing the gam angle to be considered (shift) by minimizing the fwhm of the XRD peak
    #it will be done for a particular image of the ref powder (delta scan for ex)
    def extract_S140XRD_idx(self, scanNo, incl_q = True, sort_type=False, add_metadata=True,
                            showGraph = False, saveGraph = False, mirror_sorted_structure = False):
        """ Converts .nxs data into intensity, 2theta and q values
            One scanNo is split into all its individual scans, and saved as I_vs_2th_scanNo_pointIdx.txt files
            This is ~ the original function provided from Diffabs SOLEIL
            args:
                scanNo: the scan number to be processed
                incl_q: whether to include q values in the output file
                sort_type: whether to label the data by scan type (chi or delta or z or omega) or not
                add_metadata: whether to include metadata lines in the output file
                showGraph: whether to display the data in a graph
        """

        #pathSave, creating the corresponding folder
        # pathSave = self.export_directory + "scan_%d/"%(scanNo)
        pathSave = self._export_path_for_scan(scanNo, mirror_sorted_structure=mirror_sorted_structure) ###the extracted XRD will have 2 indexes: scanNo and pointIdx

        pointsFound, data_index_xpad, deltaArray, gamArray, phiArray, chiArray, omeArray, energyArray, TemperatureArray, mssg_metadata, mssgActuator_list, Izero, Izero_mean = self.s140_read_metadata_and_actuators(scanNo, print_Flag = True)

        image_corr1_sizeX, image_corr1_sizeY, x_matrix, y_matrix, newX_Ifactor_array, newX_array, newY_array = self.doublePixSpread()

        #reading / preparing the XPAD data
        # nxs_pathFolder = self.nxs_pathFolder
        fileName = nxs_fileName_root + str(scanNo).zfill(numberOfDigits) + nxs_fileName_suffix
        if pointsFound is None or data_index_xpad is None:
            print("Error: Failed to find XPAD dataset in the NXS file.")
            return
        file1 = tables.open_file(os.path.join(self.nxs_pathFolder, fileName))
        #fileNameRoot1 = file1.root._v_groups.keys()[0]
        #command = "file1.root.__getattr__(\""+str(fileNameRoot1)+"\")"
        command = "file1.root.scan_%s" %( str(scanNo).zfill(4)  )
        if DIAGNOSTIC:
            print('command=',command)

        scan_type = eval(command+".scan_config.name")[()].decode("utf-8")
        print(f"Scan type: {scan_type}")

        start_time = eval(command+".start_time")[()].decode("utf-8")
        timestamps = [dt.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S.%f") if not math.isnan(t) else ""
                      for t in eval(command + ".scan_data.sensors_timestamps")[:]
]
        filter = eval(command+".scan_data.data_13")[:] # Decimal value of the wanted filter combination
        filterFactor = eval(command+".scan_data.data_14")[:] # The computed attenuation of the selected filter combination

        if pointsFound.__len__() == 1: #1D scan
            error_flag = False
            if showGraph or saveGraph:
                fig, ax = _make_plot_figure(showGraph)
                cmap = plt.get_cmap('jet', deltaArray.shape[0])
            for pointIndex in range(0, deltaArray.shape[0]):
                try: #need to take into account the 'empty' XPAD images (a lot of att for ex)
                    mssg2write = "*** pointIdx = %d "%(pointIndex)
                    # mssg2write += "delta = %.3f, gam = %.3f, E = %.3f"%(deltaArray[pointIndex], gamArray[pointIndex], energyArray[pointIndex])
                    if DIAGNOSTIC:
                        print (mssg2write)

                    xpadImage = eval(command+".scan_data.data_"+str(data_index_xpad).zfill(2)+"[pointIndex]")

                    if Izero_norm_flag:
                        thisImage = 0.0+xpadImage / (0.+Izero[pointIndex])*Izero_mean
                    else:
                        thisImage = 0.0+xpadImage
                    if DIAGNOSTIC:
                        print ("xpad_size", xpadImage.shape)

                    delta = deltaArray[pointIndex];
                    gam = gamArray[pointIndex];
                    energy = energyArray[pointIndex];


                    diffracto_delta_rad = (delta+deltaOffset)*deg2rad;
                    sindelta = numpy.sin(diffracto_delta_rad); cosdelta = numpy.cos(diffracto_delta_rad);

                    #implementing the gamShift correction function of the angle of the diffractometer
                    gamShift_corr = gamShift_value
                    if flag_gamShift_corr_deltaScan:
                        #poly2 = a(x-x0)^2 + b(x-x0)x +c; x = delta (the result is in mdeg)
                        gamShift_corr = gamShift_value + 1./1000 *(poly2_gamShift_corr_a * (delta - poly2_gamShift_corr_x0) **2. + poly2_gamShift_corr_b *(delta - poly2_gamShift_corr_x0) + poly2_gamShift_corr_c )

                    diffracto_gam_rad = (gam+gamOffset+gamShift_corr)*deg2rad;
                    singamma = numpy.sin(diffracto_gam_rad); cosgamma = numpy.cos(diffracto_gam_rad);

                    #the array thisCorrectedImage contains the corrected image (double pixels corrections)
                    twoThArray = numpy.zeros((image_corr1_sizeY, image_corr1_sizeX))
                    psiArray = numpy.zeros((image_corr1_sizeY, image_corr1_sizeX))

                    x_line = numpy.linspace(0, image_corr1_sizeX-1, image_corr1_sizeX)
                    x_matrix = numpy.zeros((image_corr1_sizeX,  image_corr1_sizeY))
                    for a in range (image_corr1_sizeY):
                        x_matrix[:, a] = x_line[:]

                    y_line = numpy.linspace(0, image_corr1_sizeY-1, image_corr1_sizeY)
                    y_matrix = numpy.zeros((image_corr1_sizeX,  image_corr1_sizeY))
                    for a in range (image_corr1_sizeX):
                        y_matrix[a, :] = y_line[:]

                    corrX = distance; # for xpad3.2 like
                    corrZ = YcenDetector - y_matrix; # for xpad3.2 like
                    corrY = XcenDetector - x_matrix; # sign is reversed
                    tempX = corrX; tempY = corrZ*(-1.0); tempZ = corrY;

                    x1 = tempX*cosdelta - tempZ*sindelta; y1 = tempY; z1 = tempX*sindelta + tempZ*cosdelta;
                    corrX = x1*cosgamma + y1*singamma; corrY = -x1*singamma + y1*cosgamma; corrZ = z1;
                    corrX2 = corrX*corrX; corrY2 = corrY*corrY; corrZ2=corrZ*corrZ;
                    norm = numpy.sqrt(corrX2 + corrY2+ corrZ2);

                    thisdelta = numpy.arccos(corrX/norm)*inv_deg2rad;

                    sign = numpy.sign(corrZ);
                    cos_psi_rad = corrY/numpy.sqrt(corrY2+corrZ2);
                    psi = numpy.arccos(cos_psi_rad)*inv_deg2rad*sign;

                    psi[psi<0] +=360; 	psi -= 90;
                    psiArray = psi.T; 	twoThArray = thisdelta.T
                    #end geometry

                    #dealing now with the intensitiespointIndex
                    thisSize = image_corr1_sizeX*image_corr1_sizeY
                    thisImage = xpadImage + 0.0
                    if self.flatImg_inv is None:
                        self.flatImg_inv = self.read_flatField()
                    thisImage = self.flatImg_inv * thisImage
                    thisImage = ndimage.median_filter(thisImage, 3)

                    thisCorrectedImage = numpy.zeros((image_corr1_sizeY, image_corr1_sizeX))
                    Ifactor = newX_Ifactor_array # x
                    newY_array = newY_array.astype('int')
                    newX_array = newX_array.astype('int')

                    for x in range (0, image_corr1_sizeX):
                        thisCorrectedImage[:, x] = thisImage[newY_array[:], newX_array[x]]
                        if Ifactor[x]	< 0:
                            #print "%s %s" %(x, Ifactor[x])
                            thisCorrectedImage[:, x] = (thisImage[newY_array[:], newX_array[x]-1]+thisImage[newY_array[:], newX_array[x]+1])/2.0/factorIdoublePixel
                    thisCorrectedImage[numpy.isnan(thisCorrectedImage)] = -100000

                    # correct the double lines (last and 1st line of the modules, at their junction)
                    lineIndex1 = chip_sizeY-1; # last line of module1 = 119, is the 1st line to correct
                    lineIndex5 = lineIndex1 + 3 +1; # 1st line of module2 (after adding the 3 empty lines), becomes the 5th line tocorrect
                    lineIndex2 = lineIndex1+1; lineIndex3 = lineIndex1+2; lineIndex4 = lineIndex1+3;

                    i1 = thisCorrectedImage[lineIndex1, :]; i5 = thisCorrectedImage[lineIndex5, :];
                    i1new = i1/factorIdoublePixel; i5new = i5/factorIdoublePixel; i3 = (i1new+i5new)/2.0;
                    thisCorrectedImage[lineIndex1, :] = i1new; thisCorrectedImage[lineIndex2, :] = i1new;
                    thisCorrectedImage[lineIndex3, :] = i3;
                    thisCorrectedImage[lineIndex5, :] = i5new; thisCorrectedImage[lineIndex4, :] = i5new

                    IntensityArray = thisCorrectedImage.T.reshape(image_corr1_sizeX*image_corr1_sizeY)
                    psiArray = psiArray.T.reshape(image_corr1_sizeX*image_corr1_sizeY);
                    twoThArray = twoThArray.T.reshape(image_corr1_sizeX*image_corr1_sizeY)

                    ###psi1 = -1000.; psi2 = 1000.; #integrate the whole image, or can do a mask here
                    maskPsi = numpy.ones(psiArray.shape)
                    maskPsi[psiArray < psi1] = 0.0; maskPsi[psiArray > psi2] = 0.0;

                    print ("   ... generating XRD (I vs 2theta)")
                    miniTwoTh = twoThArray.min(); maxiTwoTh = twoThArray.max();
                    nbOfBins = int((0.0+maxiTwoTh-miniTwoTh)/stepTwoTh)+1;
                    TwoThResult = numpy.zeros(nbOfBins+1); #generate the tables for radial integration, this is delta
                    for ii in range (0, nbOfBins):
                        TwoTh_temp1 = miniTwoTh + ii*stepTwoTh; TwoTh_temp2 = TwoTh_temp1 + stepTwoTh;
                        TwoThResult[ii] = 0.5*(TwoTh_temp1+TwoTh_temp2);
                    thisBinArray = numpy.floor((twoThArray*maskPsi - miniTwoTh)/stepTwoTh);
                    thisBinArray = thisBinArray.astype('int')
                    if DIAGNOSTIC:
                        print ("		  (2th_mini=%.2f   2th_maxi=%.2f)" %(twoThArray.min(), twoThArray.max()) 		)
                        print ("		  (psi_mini=%.2f   psi_maxi=%.2f)" %(psiArray.min(), psiArray.max()) 		)

                    intensityResult = numpy.zeros(nbOfBins+1); #this will be the summed intensity
                    IntensityArray = IntensityArray * maskPsi

                    indexes_ = numpy.nonzero(IntensityArray>=0)[0]; 	my_bin = thisBinArray[indexes_]
                    my_intensity =  IntensityArray[indexes_]
                    aggregated = numpy.zeros(nbOfBins+1)
                    for i in range(my_bin.max()+1):
                        selected_intensities = my_intensity[my_bin == i]
                        intensityResult[i] = selected_intensities.mean()

                    intensityResult[numpy.isnan(intensityResult)] = -1
                    # END calculating binned data

                    line2write = ""

                    if add_metadata:
                        # Include metadata as top lines in the output file
                        line2write += f"# Scan Type: {scan_type} \n# Start Time: {start_time} \n# Frame Time: {timestamps[pointIndex]} \n# Energy: {energyArray[pointIndex]} \n# Gamma: {gamArray[pointIndex]} \n"
                        line2write += f"# Omega: {omeArray[pointIndex]} \n# Delta: {deltaArray[pointIndex]} \n# Chi: {chiArray[pointIndex]} \n# Phi: {phiArray[pointIndex]} \n"
                        line2write += f"# Attenuator Combination: {filter[pointIndex]} \n# Attenuation Factor: {filterFactor[pointIndex]} \n# Temperature: {TemperatureArray[pointIndex]} \n"
                        line2write += "#\n"

                    if energy > 0 and incl_q: #energyShift is included when reading above the energy from the energyArray
                        q = 4.0*numpy.pi/(12.3985/energy) * numpy.sin(numpy.deg2rad(0.5*TwoThResult))
                        line2write += "# 2theta intensity q\n"
                        for u in range(10, TwoThResult.shape[0]-15): #throw away some points close to the edges
                            #if energy <= 0:
                            line2write += "%f %f %f\n"%(TwoThResult[u], intensityResult[u], q[u])
                    else:
                        line2write += "# 2theta intensity\n"
                        for u in range(10, TwoThResult.shape[0]-15): #throw away some points close to the edges
                            line2write += "%f %f %f\n"%(TwoThResult[u], intensityResult[u])

                    if not sort_type:
                        with open(pathSave + "I_vs_2th_%d_%d.txt"%(scanNo, pointIndex), "w") as ff:
                            ff.write(line2write)
                    elif sort_type:
                        scan_type_label = self.scan_types.get(scan_type, 'NA')
                        with open(pathSave + "I_vs_2th_%d_%s_%d.txt"%(scanNo, scan_type_label, pointIndex), "w") as ff:
                            ff.write(line2write)
                        

                    #plotting the result
                    if showGraph or saveGraph:
                        ax.plot(TwoThResult[10:-15], intensityResult[10:-15], ".-", color=cmap(pointIndex), label=f"Point {pointIndex}",
                                linewidth=0.5, markersize=2) #throw away some points close to the edges
                        ax.set_xlabel("2$\\theta$ (°)"); ax.set_ylabel("intensity (arb. units)")
                except Exception as e:
                    print (f"    !!! XPAD image {pointIndex} could not be read properly, skipped for the XRD extraction !!!")
                    if not error_flag:
                        print(f"        Exception: {e}")
                        error_flag = True
        print("   ... closing file")
        file1.close()
        print ("FINISHED")
        if showGraph or saveGraph:
            ax.legend(fontsize=8, loc='best', ncols=legend_columniser(len(ax.get_lines())))
            ax.grid(visible = True)
            ax.set_title(f"Scan {scanNo} ({scan_type}, {deltaArray.shape[0]} points)")
            ax.set_xlabel("2$\\theta$ (°)")
            ax.set_ylabel("Intensity (arb. units)")
            ax.set_ylim(bottom=0)
            if saveGraph:
                fig.savefig(self._graph_path_for_scan(scanNo, pathSave, "I_vs_2th.png"), dpi=300)
            if showGraph:
                plt.show()
            return fig, ax
        else:
            return None


    #======================================================
    def extract_S140XRD(self, scanNo, incl_q = True, showGraph = False,
                        saveGraph = False, mirror_sorted_structure = False):
        """ Converts .nxs data into a single csv file with all data
            One scanNo is split into all its individual scans, and saved as I_vs_2th_scanNo_pointIdx.txt files
            args:
                scanNo: the scan number to be processed
                incl_q: whether to include q values in the output file
                showGraph: whether to display the data in a graph
        """

        #pathSave, creating the corresponding folder
        # pathSave = self.export_directory + "scan_%d/"%(scanNo)
        pathSave = self._export_path_for_scan(scanNo, mirror_sorted_structure=mirror_sorted_structure) ###the extracted XRD will have 2 indexes: scanNo and pointIdx

        pointsFound, data_index_xpad, deltaArray, gamArray, phiArray, chiArray, omeArray, energyArray, TemperatureArray, mssg_metadata, mssgActuator_list, Izero, Izero_mean = self.s140_read_metadata_and_actuators(scanNo, print_Flag = True)

        image_corr1_sizeX, image_corr1_sizeY, x_matrix, y_matrix, newX_Ifactor_array, newX_array, newY_array = self.doublePixSpread()

        #reading / preparing the XPAD data
        # nxs_pathFolder = self.nxs_pathFolder
        fileName = nxs_fileName_root + str(scanNo).zfill(numberOfDigits) + nxs_fileName_suffix
        if pointsFound is None or data_index_xpad is None:
            print("Error: Failed to find XPAD dataset in the NXS file.")
            return
        file1 = tables.open_file(os.path.join(self.nxs_pathFolder, fileName))
        command = "file1.root.scan_%s" %( str(scanNo).zfill(4)  )
        if DIAGNOSTIC:
            print('command=',command)

        scan_type = eval(command+".scan_config.name")[()].decode("utf-8")
        print(f"Scan type: {scan_type}")
        start_time = eval(command+".start_time")[()].decode("utf-8")
        
        if pointsFound.__len__() == 1: #1D scan
            error_flag = False
            # create the dataframe with all the per-scan metadata
            metadf = pd.DataFrame({
                'timestamp': eval(command+".scan_data.sensors_timestamps")[:],
                'omega': omeArray,
                'delta': deltaArray,
                'chi': chiArray,
                'phi': phiArray,
                'temperature': TemperatureArray,
                'filter': eval(command+".scan_data.data_13")[:],
                'attenuation_factor': eval(command+".scan_data.data_14")[:],
                'energy': energyArray,
                'gamma': gamArray,
            })
            drows = []  # list to store the rows of the dataframe

            if showGraph or saveGraph:
                fig, ax = _make_plot_figure(showGraph)
                cmap = plt.get_cmap('jet', deltaArray.shape[0])
            for pointIndex in range(0, deltaArray.shape[0]):
                try: #need to take into account the 'empty' XPAD images (a lot of att for ex)
                    mssg2write = "*** pointIdx = %d "%(pointIndex)
                    mssg2write += "delta = %.3f, gam = %.3f, E = %.3f"%(deltaArray[pointIndex], gamArray[pointIndex], energyArray[pointIndex])
                    if DIAGNOSTIC:
                        print (mssg2write)

                    metadata = metadf.iloc[pointIndex].to_dict() # extract the metadata for this point as a dictionary, to be added to each line of the output csv file

                    xpadImage = eval(command+".scan_data.data_"+str(data_index_xpad).zfill(2)+"[pointIndex]")
                    
                    if Izero_norm_flag:
                        thisImage = 0.0+xpadImage / (0.+Izero[pointIndex])*Izero_mean
                    else:
                        thisImage = 0.0+xpadImage
                    if DIAGNOSTIC:
                        print ("xpad_size", xpadImage.shape)

                    delta = deltaArray[pointIndex];
                    gam = gamArray[pointIndex];
                    energy = energyArray[pointIndex];


                    diffracto_delta_rad = (delta+deltaOffset)*deg2rad;
                    sindelta = numpy.sin(diffracto_delta_rad); cosdelta = numpy.cos(diffracto_delta_rad);

                    #implementing the gamShift correction function of the angle of the diffractometer
                    gamShift_corr = gamShift_value
                    if flag_gamShift_corr_deltaScan:
                        #poly2 = a(x-x0)^2 + b(x-x0)x +c; x = delta (the result is in mdeg)
                        gamShift_corr = gamShift_value + 1./1000 *(poly2_gamShift_corr_a * (delta - poly2_gamShift_corr_x0) **2. + poly2_gamShift_corr_b *(delta - poly2_gamShift_corr_x0) + poly2_gamShift_corr_c )

                    diffracto_gam_rad = (gam+gamOffset+gamShift_corr)*deg2rad;
                    singamma = numpy.sin(diffracto_gam_rad); cosgamma = numpy.cos(diffracto_gam_rad);

                    #the array thisCorrectedImage contains the corrected image (double pixels corrections)
                    twoThArray = numpy.zeros((image_corr1_sizeY, image_corr1_sizeX))
                    psiArray = numpy.zeros((image_corr1_sizeY, image_corr1_sizeX))

                    x_line = numpy.linspace(0, image_corr1_sizeX-1, image_corr1_sizeX)
                    x_matrix = numpy.zeros((image_corr1_sizeX,  image_corr1_sizeY))
                    for a in range (image_corr1_sizeY):
                        x_matrix[:, a] = x_line[:]

                    y_line = numpy.linspace(0, image_corr1_sizeY-1, image_corr1_sizeY)
                    y_matrix = numpy.zeros((image_corr1_sizeX,  image_corr1_sizeY))
                    for a in range (image_corr1_sizeX):
                        y_matrix[a, :] = y_line[:]

                    corrX = distance; # for xpad3.2 like
                    corrZ = YcenDetector - y_matrix; # for xpad3.2 like
                    corrY = XcenDetector - x_matrix; # sign is reversed
                    tempX = corrX; tempY = corrZ*(-1.0); tempZ = corrY;

                    x1 = tempX*cosdelta - tempZ*sindelta; y1 = tempY; z1 = tempX*sindelta + tempZ*cosdelta;
                    corrX = x1*cosgamma + y1*singamma; corrY = -x1*singamma + y1*cosgamma; corrZ = z1;
                    corrX2 = corrX*corrX; corrY2 = corrY*corrY; corrZ2=corrZ*corrZ;
                    norm = numpy.sqrt(corrX2 + corrY2+ corrZ2);

                    thisdelta = numpy.arccos(corrX/norm)*inv_deg2rad;

                    sign = numpy.sign(corrZ);
                    cos_psi_rad = corrY/numpy.sqrt(corrY2+corrZ2);
                    psi = numpy.arccos(cos_psi_rad)*inv_deg2rad*sign;

                    psi[psi<0] +=360; 	psi -= 90;
                    psiArray = psi.T; 	twoThArray = thisdelta.T
                    #end geometry

                    #dealing now with the intensitiespointIndex
                    thisSize = image_corr1_sizeX*image_corr1_sizeY
                    thisImage = xpadImage + 0.0
                    if self.flatImg_inv is None:
                        self.flatImg_inv = self.read_flatField()
                    thisImage = self.flatImg_inv * thisImage
                    thisImage = ndimage.median_filter(thisImage, 3)

                    thisCorrectedImage = numpy.zeros((image_corr1_sizeY, image_corr1_sizeX))
                    Ifactor = newX_Ifactor_array # x
                    newY_array = newY_array.astype('int')
                    newX_array = newX_array.astype('int')

                    for x in range (0, image_corr1_sizeX):
                        thisCorrectedImage[:, x] = thisImage[newY_array[:], newX_array[x]]
                        if Ifactor[x]	< 0:
                            #print "%s %s" %(x, Ifactor[x])
                            thisCorrectedImage[:, x] = (thisImage[newY_array[:], newX_array[x]-1]+thisImage[newY_array[:], newX_array[x]+1])/2.0/factorIdoublePixel
                    thisCorrectedImage[numpy.isnan(thisCorrectedImage)] = -100000

                    # correct the double lines (last and 1st line of the modules, at their junction)
                    lineIndex1 = chip_sizeY-1; # last line of module1 = 119, is the 1st line to correct
                    lineIndex5 = lineIndex1 + 3 +1; # 1st line of module2 (after adding the 3 empty lines), becomes the 5th line tocorrect
                    lineIndex2 = lineIndex1+1; lineIndex3 = lineIndex1+2; lineIndex4 = lineIndex1+3;

                    i1 = thisCorrectedImage[lineIndex1, :]; i5 = thisCorrectedImage[lineIndex5, :];
                    i1new = i1/factorIdoublePixel; i5new = i5/factorIdoublePixel; i3 = (i1new+i5new)/2.0;
                    thisCorrectedImage[lineIndex1, :] = i1new; thisCorrectedImage[lineIndex2, :] = i1new;
                    thisCorrectedImage[lineIndex3, :] = i3;
                    thisCorrectedImage[lineIndex5, :] = i5new; thisCorrectedImage[lineIndex4, :] = i5new

                    IntensityArray = thisCorrectedImage.T.reshape(image_corr1_sizeX*image_corr1_sizeY)
                    psiArray = psiArray.T.reshape(image_corr1_sizeX*image_corr1_sizeY);
                    twoThArray = twoThArray.T.reshape(image_corr1_sizeX*image_corr1_sizeY)

                    ###psi1 = -1000.; psi2 = 1000.; #integrate the whole image, or can do a mask here
                    maskPsi = numpy.ones(psiArray.shape)
                    maskPsi[psiArray < psi1] = 0.0; maskPsi[psiArray > psi2] = 0.0;

                    print ("   ... generating XRD (I vs 2theta)")
                    miniTwoTh = twoThArray.min(); maxiTwoTh = twoThArray.max();
                    nbOfBins = int((0.0+maxiTwoTh-miniTwoTh)/stepTwoTh)+1;
                    TwoThResult = numpy.zeros(nbOfBins+1); #generate the tables for radial integration, this is delta
                    for ii in range (0, nbOfBins):
                        TwoTh_temp1 = miniTwoTh + ii*stepTwoTh; TwoTh_temp2 = TwoTh_temp1 + stepTwoTh;
                        TwoThResult[ii] = 0.5*(TwoTh_temp1+TwoTh_temp2);
                    thisBinArray = numpy.floor((twoThArray*maskPsi - miniTwoTh)/stepTwoTh);
                    thisBinArray = thisBinArray.astype('int')
                    if DIAGNOSTIC:
                        print ("		  (2th_mini=%.2f   2th_maxi=%.2f)" %(twoThArray.min(), twoThArray.max()) 		)
                        print ("		  (psi_mini=%.2f   psi_maxi=%.2f)" %(psiArray.min(), psiArray.max()) 		)

                    intensityResult = numpy.zeros(nbOfBins+1); #this will be the summed intensity
                    IntensityArray = IntensityArray * maskPsi

                    indexes_ = numpy.nonzero(IntensityArray>=0)[0]; 	my_bin = thisBinArray[indexes_]
                    my_intensity =  IntensityArray[indexes_]
                    aggregated = numpy.zeros(nbOfBins+1)
                    for i in range(my_bin.max()+1):
                        selected_intensities = my_intensity[my_bin == i]
                        intensityResult[i] = selected_intensities.mean()

                    intensityResult[numpy.isnan(intensityResult)] = -1
                    # END calculating binned data

                    crop_limits = (10, -15) # to throw away some points close to the edges
                    twotheta, intensity = TwoThResult[crop_limits[0]:crop_limits[1]], intensityResult[crop_limits[0]:crop_limits[1]]
                    if energy > 0 and incl_q: #energyShift is included when reading above the energy from the energyArray
                        q = 4.0*numpy.pi/(12.3985/energy) * numpy.sin(numpy.deg2rad(0.5*TwoThResult))
                        q = q[crop_limits[0]:crop_limits[1]]
                        for u in range(len(twotheta)):
                            drows.append({
                                '2theta': twotheta[u],
                                'intensity': intensity[u],
                                'q': q[u],
                                **metadata # add the metadata for this point to each row
                            })
                    else:
                        for u in range(len(twotheta)):
                            drows.append({
                                '2theta': twotheta[u],
                                'intensity': intensity[u],
                                'q': q[u],
                                **metadata # add the metadata for this point to each row
                            })
                        
                    #plotting the result
                    if showGraph or saveGraph:
                        ax.plot(twotheta, intensity, ".-", color=cmap(pointIndex), label=f"Point {pointIndex}",
                                linewidth=0.5, markersize=2) #throw away some points close to the edges
                        ax.set_xlabel("2theta (°)"); ax.set_ylabel("intensity (arb. units)")
                except Exception as e:
                    print (f"   !!! XPAD image {pointIndex} could not be read properly, skipped for the XRD extraction !!!")
                    if not error_flag:
                        print(f"        Exception: {e}")
                        error_flag = True
        df = pd.DataFrame(drows)
        savename = pathSave + f"scan_{scanNo}_"
        savename += f"{dt.datetime.fromisoformat(start_time).strftime('%Y-%m-%dT%H%M%S')}_{scan_type.replace(' ', '_').replace('_', '-')}.csv"
        print("   ... saving data to CSV file:", savename)
        df.to_csv(savename, index=False)
        print("   ... closing file")
        file1.close()
        print ("FINISHED")
        if showGraph or saveGraph:
            ax.legend(fontsize=8, loc='best', ncols=legend_columniser(len(ax.get_lines())))
            ax.grid(visible = True)
            ax.set_title(f"Scan {scanNo} ({scan_type}, {deltaArray.shape[0]} points)")
            ax.set_xlabel(r"2$\theta$ (°)")
            ax.set_ylabel("Intensity (arb. units)")
            ax.set_ylim(bottom=0)
            if saveGraph:
                fig.savefig(os.path.splitext(savename)[0] + ".png", dpi=300)
            if showGraph:
                plt.show()
                return fig, ax
            return None
        else:
            return None


    
    #no need anymore to choose the dataset corresponding to XPAD images, this is done automatically (from the shape of the data, which should be 240 x 560)
    def s140_read_metadata_and_actuators(self, scanNo, print_Flag = False):
        """reads, from a NXS file, the metaData and (all) the actuators
        - returns the metadata message, the nb. of datapoints, the dataIndex of the XPAD and (all) the diffracto motors as tables
        - print on screen some of the above messages
        """

        mssg_metadata = ""; mssgActuator_list = "Actuators NOT detected"
        this_day = -1; this_month = -1; this_year = -1; # will be set when the subfolder containing the nxs data is identified
        fileName = nxs_fileName_root + str(scanNo).zfill(numberOfDigits) + nxs_fileName_suffix

        
        datefound_flag = False
        resolved_scan_path, resolved_rel_folder = self._resolve_scan_path(scanNo)
        # First check if the nxsfile_directory is already the correct subfolder
        nxsfile_directory_isdate = False
        try:
            if resolved_scan_path:
                nxs_pathFolder = os.path.join(os.path.dirname(resolved_scan_path), "")
            else:
                nxs_pathFolder = self.nxsfile_directory
            #try reading images in the .nxs file
            print(fileName)
            file1 = tables.open_file(os.path.join(nxs_pathFolder, fileName))
            #fileNameRoot1 = file1.root._v_groups.keys()[0]
            #command = "file1.root.__getattr__(\""+str(fileNameRoot1)+"\")"
            command = "file1.root.scan_%s"%(str(scanNo).zfill(4) )
            Izero = eval(command+".scan_data.data_02.read()")
            Izero_mean = Izero.mean()
            for datasetIndex in range (datasetIndex_max):
                try:
                    tmp_data_shape = eval(command+".scan_data.data_"+str(datasetIndex).zfill(2)+".shape")
                    if tmp_data_shape[-1] == 560 and tmp_data_shape[-2] == 240: #xpad image
                        #xpadImage = eval(command+".scan_data.data_"+str(datasetIndex).zfill(2)+"[pointIndex]")
                        pointsFound = tmp_data_shape[:-2][0]
                        data_index_xpad = datasetIndex
                except:
                    pass
            file1.close()
            nxsfile_directory_isdate = True
            date_parts = self._date_parts_from_path(nxs_pathFolder)
            if date_parts:
                this_year, this_month, this_day = date_parts
                nxsdate_subfolder = "%s-%s-%s/"%(str(this_year), str(this_month).zfill(2), str(this_day).zfill(2))
            else:
                this_year = this_month = this_day = 0
                nxsdate_subfolder = resolved_rel_folder
            self.this_timestamp = [int(this_year), int(this_month), int(this_day)]
            datefound_flag = True

        except:
            pass

        if not datefound_flag:
            #detecting th folder name (yyy-mm-dd) containing the .nxs data. The xpad images are detected from their size (240 x 560)
            if DIAGNOSTIC:
                print("Searching for the dated NXS subfolder containing the data for scan %d..."%(scanNo))
            for datestring in self.daterange:
                try:
                    nxs_pathFolder = os.path.join(self.nxsfile_directory, str(datestring), "")
                    #try reading images in the .nxs file
                    if DIAGNOSTIC:
                        print('checking for file %s'%(nxs_pathFolder+fileName))
                    file1 = tables.open_file(os.path.join(nxs_pathFolder, fileName))
                    #fileNameRoot1 = file1.root._v_groups.keys()[0]
                    #command = "file1.root.__getattr__(\""+str(fileNameRoot1)+"\")"
                    command = "file1.root.scan_%s"%(str(scanNo).zfill(4) )
                    Izero = eval(command+".scan_data.data_02.read()")
                    Izero_mean = Izero.mean()
                    for datasetIndex in range (datasetIndex_max):
                        try:
                            tmp_data_shape = eval(command+".scan_data.data_"+str(datasetIndex).zfill(2)+".shape")
                            if tmp_data_shape[-1] == 560 and tmp_data_shape[-2] == 240: #xpad image
                                # xpadImage = eval(command+".scan_data.data_"+str(datasetIndex).zfill(2)+"[pointIndex]")
                                pointsFound = tmp_data_shape[:-2][0]
                                data_index_xpad = datasetIndex
                                this_day = int(datestring[8:10]); this_month = int(datestring[5:7]); this_year=int(datestring[0:4]) # dated subfolder containin the nxs data
                                self.this_timestamp = [int(this_year), int(this_month), int(this_day)]
                                if DIAGNOSTIC:
                                    print("NXS subfolder found at %s"%(nxs_pathFolder+fileName))
                                datefound_flag = True
                                break
                        except:
                            pass
                    file1.close()
                    if datefound_flag:
                        break
                except Exception as e:
                    print("%s" % e)
                    pass
        
        if datefound_flag and this_year > 0:
            nxsdate_subfolder = "%s-%s-%s/"%(str(this_year), str(this_month).zfill(2), str(this_day).zfill(2))
        elif datefound_flag:
            nxsdate_subfolder = resolved_rel_folder or ''
        else:
            nxsdate_subfolder = "%s-%s-%s/"%(str(this_year), str(this_month).zfill(2), str(this_day).zfill(2))

        if nxsdate_subfolder == '-1--1--1/':
            print("Error: Failed to find valid NXS subfolder for the given date range.")
            return None, None, None, None, None, None, None, None, None, None, None, None, None
        
        self.this_timestamp = [int(this_year), int(this_month), int(this_day)]
    
        self.nxsdate_subfolder = nxsdate_subfolder
        # day = this_day ; month = this_month #identified the folder containing the data
        if resolved_scan_path:
            self.nxs_pathFolder = os.path.join(os.path.dirname(resolved_scan_path), "")
        elif not nxsfile_directory_isdate:
            self.nxs_pathFolder = os.path.join(self.nxsfile_directory, self.nxsdate_subfolder)
        else:
            self.nxs_pathFolder = self.nxsfile_directory
        if DIAGNOSTIC:
            print('NXS path folder found: %s'%(self.nxs_pathFolder))
        delta = numpy.nan; gam = numpy.nan; chi = numpy.nan; phi = numpy.nan; omega = numpy.nan
        energy = numpy.nan
        if DIAGNOSTIC:
            print(self.nxs_pathFolder)
            print(fileName)
        try:
            #some extra information for this image (metadata)
            with h5py.File(os.path.join(self.nxs_pathFolder, fileName),'r') as f: #will properly close it once the indented code is executed

                """
                group1 = f.get(f.keys()[0])
                try:
                    delta = numpy.array(group1['DIFFABS/d13-1-cx1__ex__dif.1-delta/raw_value'])
                except:
                    pass
                try:
                    gam = numpy.array(group1['DIFFABS/d13-1-cx1__ex__dif.1-gamma/raw_value'])
                except:
                    pass
                try:
                    chi = numpy.array(group1['DIFFABS/d13-1-cx1__ex__dif.1-chi_e/raw_value'])
                except:
                    pass
                try:
                    phi = numpy.array(group1['DIFFABS/d13-1-cx1__ex__dif.1-phi_e/raw_value'])
                except:
                    pass
                try:
                    omega = numpy.array(group1['DIFFABS/d13-1-cx1__ex__dif.1-omega_e/raw_value'])
                except:
                    pass
                try:
                    energy = numpy.array(group1['DIFFABS/d13-1-c03__op__mono/energy'])
                except:
                    pass
                """
                list_elements = [key for key in list(f["/"].keys())]
                if DIAGNOSTIC:
                    print(list_elements)
                try:
                    delta = numpy.array(f["/"+list_elements[0]+'/DIFFABS/d13-1-cx1__ex__dif.1-delta/raw_value'])
                except:
                    pass
                try:
                    gam = numpy.array(f["/"+list_elements[0]+'/DIFFABS/d13-1-cx1__ex__dif.1-gamma/raw_value'])
                except:
                    pass
                try:
                    cirpad_delta = numpy.array(f["/"+list_elements[0]+'/DIFFABS/d13-1-cx1__ex__cirpad_delta/raw_value'])
                except:
                    pass
                try:
                    cirpad_gam = numpy.array(f["/"+list_elements[0]+'/DIFFABS/d13-1-cx1__ex__dif.1-cirpad-gam/raw_value'])
                except:
                    pass
                try:
                    chi = numpy.array(f["/"+list_elements[0]+'/DIFFABS/d13-1-cx1__ex__dif-pil-sim-eulerians/chi'])
                except:
                    pass
                try:
                    phi = numpy.array(f["/"+list_elements[0]+'/DIFFABS/d13-1-cx1__ex__dif-pil-sim-eulerians/phi'])
                except:
                    pass
                try:
                    omega = numpy.array(f["/"+list_elements[0]+'/DIFFABS/d13-1-cx1__ex__dif-pil-sim-eulerians/omega'])
                except:
                    pass
                try:
                    energy = numpy.array(f["/"+list_elements[0]+'/DIFFABS/d13-1-c03__op__mono/energy']) ###read metadata Energy
                    if DIAGNOSTIC:
                        print("ENERGY read (keV) = %f"%(energy))
                except:
                    pass
                try:
                    Temperature = numpy.array(f["/"+list_elements[0]+'/scan_data/data_15']) ###read metadata Temperature
                    if DIAGNOSTIC:
                        print("Temperature read (°C) = %f"%(Temperature))
                except:
                    pass

            deltaArray = delta; gamArray = gam; omeArray = omega; chiArray = chi; phiArray = phi
            energyArray = energy; TemperatureArray= Temperature

            #reading images in the .nxs file (using tables module)
            file1 = tables.open_file(os.path.join(self.nxs_pathFolder, fileName))
            #fileNameRoot1 = file1.root._v_groups.keys()[0]
            #command = "file1.root.__getattr__(\""+str(fileNameRoot1)+"\")"
            command = "file1.root.scan_%s" %( str(scanNo).zfill(4)  )
            Izero = eval(command+".scan_data.data_02.read()")	#Izero monitor is always data_02. To put later on as parameter
            Izero_mean = Izero.mean()
            Temperature = eval(command+".scan_data.data_15.read()")


            for datasetIndex in range (datasetIndex_max):
                try:
                    tmp_data_shape = eval(command+".scan_data.data_"+str(datasetIndex).zfill(2)+".shape")
                    if tmp_data_shape[-1] == 560 and tmp_data_shape[-2] == 240: #xpad image
                        #xpadImage = eval(command+".scan_data.data_"+str(datasetIndex).zfill(2)+"[pointIndex]")

                        #modif to deal with Meshes
                        #pointsFound = tmp_data_shape[:-2][0]
                        pointsFound = tmp_data_shape[:-2]
                        if DIAGNOSTIC:
                            print("dataset",datasetIndex)

                        data_index_xpad = datasetIndex
                except:
                    pass

            #initializing the arrays with the values in the MetaData
            energyArray = numpy.zeros(pointsFound) + energy
            deltaArray = numpy.zeros(pointsFound) + delta
            gamArray = numpy.zeros(pointsFound) + gam
            omeArray = numpy.zeros(pointsFound) + omega
            chiArray = numpy.zeros(pointsFound) + chi
            phiArray = numpy.zeros(pointsFound) + phi

            mssg_metadata = "looking at file=%s; nb. of datapoints = %s\n" %(fileName, pointsFound)
            mssg_metadata += "   Metadata: delta = %.3f chi = %.3f phi = %.3f omega = %.3f\n" %(delta, chi, phi, omega)
            mssg_metadata += "			 gamma = %.3f energy = %.3f keV\n" %(gam, energy)


            mssgActuator_list = "Actuators detected:\n"; #detecting the actuators

            for actuatorIndex in range (1, 2):
                mssg2add = ""
                try:
                    if DIAGNOSTIC:
                        print (actuatorIndex, "aaaaa", command)
                        print(command+".scan_data.actuator_1_"+str(actuatorIndex)+".attrs.long_name")
                    tmp_ = eval(command+".scan_data.actuator_1_"+str(actuatorIndex)+".attrs.long_name")
                    if DIAGNOSTIC:
                        print ("aaa", tmp_)
                    try:
                        if tmp_ == b"d13-1-cx1/ex/dif.1-delta/position":
                            deltaArray = eval(command+".scan_data.actuator_1_"+str(actuatorIndex)+".read()")
                            mssg2add = "   1_%d, %s\n"%(actuatorIndex, tmp_)
                            if DIAGNOSTIC:
                                print (deltaArray)
                    except:
                        pass
                    try:
                        if tmp_ == b"d13-1-cx1/ex/dif.1-gamma/position":
                            gamArray = eval(command+".scan_data.actuator_1_"+str(actuatorIndex)+".read()")
                            mssg2add = "   1_%d, %s\n"%(actuatorIndex, tmp_)
                    except:
                        pass
                    try:
                        if tmp_ == b"d13-1-cx1/ex/dif-pil-sim-eulerians/phi":
                            phiArray = eval(command+".scan_data.actuator_1_"+str(actuatorIndex)+".read()")
                            mssg2add = "  1_%d, %s\n"%(actuatorIndex, tmp_)
                    except:
                        pass
                    try:
                        if tmp_ == b"d13-1-cx1/ex/dif-pil-sim-eulerians/omega":
                            omeArray = eval(command+".scan_data.actuator_1_"+str(actuatorIndex)+".read()")
                            mssg2add = "   1_%d, %s\n"%(actuatorIndex, tmp_)
                    except:
                        pass
                    try:
                        if tmp_ == b"d13-1-cx1/ex/dif-pil-sim-eulerians/chi":
                            chiArray = eval(command+".scan_data.actuator_1_"+str(actuatorIndex)+".read()")
                            mssg2add = "   1_%d, %s\n"%(actuatorIndex, tmp_)
                    except:
                        pass
                    try:
                        if tmp_ == b"d13-1-c03/op/mono/energy":
                            energyArray = eval(command+".scan_data.actuator_1_"+str(actuatorIndex)+".read()")
                            mssg2add = "   1_%d, %s\n"%(actuatorIndex, tmp_)
                    except:
                        pass
                    try:
                        TemperatureArray = eval(command+".scan_data.data_15.read()")
                    except:
                        pass                   
                except:
                    pass
                mssgActuator_list += mssg2add


            for actuatorIndex in range (10):
                mssg2add = ""
                try:
                    tmp_ = eval(command+".scan_data.actuator_2_"+str(actuatorIndex)+".attrs.long_name")
                    try:
                        if tmp_ == "d13-1-cx1/ex/dif.1-delta/position":
                            deltaArray = eval(command+".scan_data.actuator_2_"+str(actuatorIndex)+".read()")
                            mssg2add = "   2_%d, %s\n"%(actuatorIndex, tmp_)
                            #print deltaArray
                    except:
                        pass
                    try:
                        if tmp_ == "d13-1-cx1/ex/dif.1-gamma/position":
                            gamArray = eval(command+".scan_data.actuator_2_"+str(actuatorIndex)+".read()")
                            mssg2add = "   2_%d, %s\n"%(actuatorIndex, tmp_)
                    except:
                        pass
                    try:
                        if tmp_ == "d13-1-cx1/ex/dif.1-phi_e/position":
                            phiArray = eval(command+".scan_data.actuator_2_"+str(actuatorIndex)+".read()")
                            mssg2add = "  2_%d, %s\n"%(actuatorIndex, tmp_)
                    except:
                        pass
                    try:
                        if tmp_ == "d13-1-cx1/ex/dif.1-omega_e/position":
                            omeArray = eval(command+".scan_data.actuator_2_"+str(actuatorIndex)+".read()")
                            mssg2add = "   2_%d, %s\n"%(actuatorIndex, tmp_)
                    except:
                        pass
                    try:
                        if tmp_ == "d13-1-cx1/ex/dif.1-chi_e/position":
                            chiArray = eval(command+".scan_data.actuator_2_"+str(actuatorIndex)+".read()")
                            mssg2add = "   2_%d, %s\n"%(actuatorIndex, tmp_)
                    except:
                        pass
                    try:
                        if tmp_ == "d13-1-c03/op/mono/energy":
                            energyArray = eval(command+".scan_data.actuator_2_"+str(actuatorIndex)+".read()")
                            mssg2add = "   2_%d, %s\n"%(actuatorIndex, tmp_)
                    except:
                        pass
                except:
                    pass
                mssgActuator_list += mssg2add

            """
            mssg2write = "*** pointIdx = %d "%(pointIndex)
            mssg2write += "delta = %.3f, gam = %.3f, E = %.3f"%(deltaArray[pointIndex], gamArray[pointIndex], energyArray[pointIndex])
            print mssg2write
            """
            file1.close()

        except:
            mssg_metadata = "NOT FOUND"
            pass

        if print_Flag:
            print ("day = %d, month = %d\n***************"%(this_day, this_month) )
            print (mssg_metadata)
            print (mssgActuator_list+"\n***************" )
        energyArray += energyShift

        ### will make sure that the delta and gam arrays have the same dimmensions (for meshes)
        shape_delta = deltaArray.shape.__len__()
        shape_gam = gamArray.shape.__len__()
        if DIAGNOSTIC:
            print ("shape_delta = ", shape_delta, "shape_gam = ", shape_gam)
        if shape_delta  == 2 and shape_gam == 1: #need to generate gam Array as 2D
            new_gamArray = numpy.zeros(deltaArray.shape)
            for uu in range (gamArray.shape[0]):
                new_gamArray[uu, :] = numpy.ones(deltaArray.shape[1]) * gamArray[uu]
            gamArray = new_gamArray
            """
            print new_gamArray, new_gamArray.shape
            print "****************"
            """
        if shape_delta  == 1 and shape_gam == 2: #need to generate delta Array as 2D
            new_deltaArray = numpy.zeros(gamArray.shape)
            for uu in range (deltaArray.shape[0]):
                new_deltaArray[uu, :] = numpy.ones(gamArray.shape[1]) * deltaArray[uu]
            deltaArray = new_deltaArray
            """
            print new_deltaArray, new_deltaArray.shape
            print "****************"
            """
        #return the usefull information
        print("pointsFound = ", pointsFound)
        return pointsFound, data_index_xpad, deltaArray, gamArray, phiArray, chiArray, omeArray, energyArray, TemperatureArray, mssg_metadata, mssgActuator_list, Izero, Izero_mean


    #=============================================================
    def read_flatField(self):
        #flat field => will be later maybe replaced by a def, and read all the files in the subfolder
        flatImg = numpy.zeros((240, 560))
        for indexFlat in self.flat_file_numbers:
            try:
                fileNameFlat = flat_fileName_root+str(indexFlat).zfill(4)+nxs_fileName_suffix
                file1Flat = tables.open_file(self.flat_file_directory+fileNameFlat)

                print ("   ... looking at file flat = %s" %(fileNameFlat) )
                #fileNameRoot2Flat = file1Flat.root._v_groups.keys()[0]
                #commandFlat = "file1Flat.root.__getattr__(\""+str(fileNameRoot2Flat)+"\")"
                commandFlat = "file1Flat.root.scan_%s"%(str(indexFlat).zfill(4) )
                for flat_dataset in range (1, 100):
                    try:
                        tmp_data_flat = eval(commandFlat+".scan_data.data_"+str(flat_dataset).zfill(2)+".read()")
                        if tmp_data_flat.shape[-1] == 560 and tmp_data_flat.shape[-2] == 240: #xpad image
                            imagesFlat = numpy.sum(tmp_data_flat, axis = 0)
                            ### some median filter on the flats
                            """
                            imagesFlat = ndimage.median_filter(imagesFlat, 3)
                            imagesFlat[imagesFlat <0] = 0.
                            imagesFlat[numpy.isnan(imagesFlat)] = 0.
                            imagesFlat[numpy.isinf(imagesFlat)] = 0.
                            """

                    except:
                        pass
                flatImg += imagesFlat
                file1Flat.close()
            except:
                pass




        flatImg[numpy.isnan(flatImg)] = -10000000* 0
        flatImg[numpy.isinf(flatImg)] = -10000000* 0
        #flatImg = ndimage.median_filter(flatImg, 3)

        flatImg = 1.0*flatImg / flatImg.mean() #normalize to 1
        flatImg_inv = 1.0/flatImg #will return the inverse of the image (to be multiplied with the data)
        flatImg_inv[numpy.isnan(flatImg_inv)] = -10000000* 0
        flatImg_inv[numpy.isinf(flatImg_inv)] = -10000000* 0
        print ("   ... DONE looking at file flat image"	)
        # print (flatImg_inv.mean())
        # print(flatImg_inv)
        if flatImg_inv.mean() < 0.1:
            print ("WARNING: the mean of the flat field correction is very low, flat file has probably not been read properly")
        return flatImg_inv
    

    def showXRD_colormap(self, scan_Nos, savepath = None):
        """ Display a colormap of the XRD data for the specified scan numbers """
        fig = plt.figure("XRD colormap", figsize = (18, 10))
        pathSave = self.export_directory

        x = numpy.empty(0); y = numpy.empty(0); z = numpy.empty(0)
        for scanNo in scan_Nos:
            fileName = "I_vs_2th_%d_0.txt"%(scanNo, )
            data = numpy.genfromtxt(pathSave + fileName, skip_header = 1)

            x = numpy.append(x, data[:, 0])
            y = numpy.append(y, numpy.zeros(data.shape[0])+scanNo)
            z = numpy.append(z, data[:, 1])

        plt.tripcolor(x, y, numpy.log10(z), cmap = "jet"); plt.colorbar()
        plt.xlabel("TwoTh (deg.)"); plt.ylabel("scanNo")
        plt.grid()
        fig.show()

    
#===============================================
    def doublePixSpread(self):
        #calculate the total number of lines to remove from the image
        lines_to_remove = 0; #initialize to 0 for calculating the sum. For xpad 3.2 these lines (negative value) will be added
        for i in range (0, numberOfModules):
            lines_to_remove +=  lines_to_remove_array[i]

        #size of the resulting (corrected) image
        image_corr1_sizeY = numberOfModules * chip_sizeY - lines_to_remove;
        image_corr1_sizeX = (numberOfChips-1)*3+numberOfChips * chip_sizeX; # considers the 2.5x pixels
        #print "*********", image_corr1_sizeX, image_corr1_sizeY

        #---------- double pix corr ---------
        #=====================================
        newX_array = numpy.zeros(image_corr1_sizeX); newX_Ifactor_array = numpy.zeros(image_corr1_sizeX)
        for x in range(0, 79): # this is the 1st chip (index chip = 0)
            newX_array[x] = x;
            newX_Ifactor_array[x] = 1 # no change in intensity

        newX_array[79] = 79; newX_Ifactor_array[79] = 1/factorIdoublePixel;
        newX_array[80] = 79; newX_Ifactor_array[80] = 1/factorIdoublePixel;
        newX_array[81] = 79; newX_Ifactor_array[81] = -1

        for indexChip in range (1, 6):
            temp_index0 = indexChip * 83
            for x in range(1, 79): # this are the regular size (130 um) pixels
                temp_index = temp_index0 + x;
                newX_array[temp_index] = x + 80*indexChip;
                newX_Ifactor_array[temp_index] = 1; # no change in intensity
            newX_array[temp_index0] = 80*indexChip; newX_Ifactor_array[temp_index0] = 1/factorIdoublePixel; # 1st double column
            newX_array[temp_index0-1] = 80*indexChip; newX_Ifactor_array[temp_index0-1] = 1/factorIdoublePixel;
            newX_array[temp_index0+79] = 80*indexChip+79; newX_Ifactor_array[temp_index0+79] = 1/factorIdoublePixel; # last double column
            newX_array[temp_index0+80] = 80*indexChip+79; newX_Ifactor_array[temp_index0+80] = 1/factorIdoublePixel;
            newX_array[temp_index0+81] = 80*indexChip+79; newX_Ifactor_array[temp_index0+81] = -1;

        for x in range (6*80+1, 560): # this is the last chip (index chip = 6)
            temp_index = 18 + x;
            newX_array[temp_index] = x;
            newX_Ifactor_array[temp_index] = 1; # no change in intensity

        newX_array[497] = 480; newX_Ifactor_array[497] = 1/factorIdoublePixel;
        newX_array[498] = 480; newX_Ifactor_array[498] = 1/factorIdoublePixel;

        newY_array = numpy.zeros(image_corr1_sizeY); # correspondance oldY - newY
        newY_array_moduleID = numpy.zeros(image_corr1_sizeY); # will keep trace of module index

        newYindex = 0;
        for moduleIndex in range (0, numberOfModules):
            for chipY in range (0, chip_sizeY):
                y = chipY + chip_sizeY*moduleIndex;
                newYindex = y - lines_to_remove_array[moduleIndex]*moduleIndex;
                newY_array[newYindex] = y;
                newY_array_moduleID[newYindex] = moduleIndex;

        #print "   ... done double pixel spreading"
        x_line = numpy.linspace(0, image_corr1_sizeX-1, image_corr1_sizeX)
        x_matrix = numpy.zeros((image_corr1_sizeX,  image_corr1_sizeY))
        for a in range (image_corr1_sizeY):
            x_matrix[:, a] = x_line[:]

        y_line = numpy.linspace(0, image_corr1_sizeY-1, image_corr1_sizeY)
        y_matrix = numpy.zeros((image_corr1_sizeX,  image_corr1_sizeY))
        for a in range (image_corr1_sizeX):
            y_matrix[a, :] = y_line[:]

        return 	image_corr1_sizeX, image_corr1_sizeY, x_matrix, y_matrix, newX_Ifactor_array, newX_array, newY_array
    

    #======================================================
    # wrapper function for batch extraction of multiple scans
    def batch_extract_S140XRD(self, scanNos, incl_q = True, showGraph = False,
                              saveGraph = False, mirror_sorted_structure = False):
        """ Converts .nxs data into a single csv file with all data
            One scanNo is split into all its individual scans, and saved as I_vs_2th_scanNo_pointIdx.txt files
            args:
                scanNo: the scan number to be processed
                incl_q: whether to include q values in the output file
                showGraph: whether to display the data in a graph
        """
        outs = []
        for scanNo in scanNos:
            out = self.extract_S140XRD(
                scanNo,
                incl_q = incl_q,
                showGraph = showGraph,
                saveGraph = saveGraph,
                mirror_sorted_structure = mirror_sorted_structure,
            )
            outs.append(out)
        if showGraph:
            self.batch_plotter(outs, scanNos)
        return outs
    
    #======================================================
    # wrapper function for batch extraction of multiple scans
    def batch_extract_S140XRD_idx(self, scanNos: list, incl_q = True, sort_type = False,
                                  showGraph = False, saveGraph = False,
                                  mirror_sorted_structure = False):
        """ Converts .nxs data into intensity, 2theta and q values, and labels as chi or delta scan
            One scanNo is split into all its individual scans, and saved as I_vs_2th_scanNo_pointIdx.txt files
            args:
                scanNos: a list of scan numbers to be processed
                incl_q: whether to include q values in the output file
                showGraph: whether to display the data in a graph
        """
        outs = []
        for scanNo in scanNos:
            out = self.extract_S140XRD_idx(
                scanNo,
                incl_q = incl_q,
                sort_type = sort_type,
                showGraph = showGraph,
                saveGraph = saveGraph,
                mirror_sorted_structure = mirror_sorted_structure,
            )
            outs.append(out)
        if showGraph:
            self.batch_plotter(outs, scanNos)
        return outs
    

    #======================================================
    # wrapper function for the general extraction, with chi/delta sorting hardcoded
    def extract_S140XRD_chidelta(self, scanNo: int, incl_q = True, showGraph = False,
                                 saveGraph = False, mirror_sorted_structure = False):
        """ Converts .nxs data into intensity, 2theta and q values, and labels as chi or delta scan
            One scanNo is split into all its individual scans, and saved as I_vs_2th_scanNo_pointIdx.txt files
            args:
                scanNo: the scan number to be processed
                incl_q: whether to include q values in the output file
                sort_type: whether to sort the data by chi/delta (True) or not (False)
                showGraph: whether to display the data in a graph
        """
        out = self.extract_S140XRD_idx(
            scanNo,
            sort_type = True,
            incl_q = incl_q,
            showGraph = showGraph,
            saveGraph = saveGraph,
            mirror_sorted_structure = mirror_sorted_structure,
        )
        return out
    

    #======================================================
    # wrapper function for batch extraction of multiple scans, with chi/delta sorting hardcoded
    def batch_extract_S140XRD_chidelta(self, scanNos: list, incl_q = True, showGraph = False,
                                       saveGraph = False, mirror_sorted_structure = False):
        """ Converts .nxs data into intensity, 2theta and q values, and labels as chi or delta scan
            One scanNo is split into all its individual scans, and saved as I_vs_2th_scanNo_pointIdx.txt files
            args:
                scanNos: a list of scan numbers to be processed
                incl_q: whether to include q values in the output file
                showGraph: whether to display the data in a graph
        """
        outs = []
        for scanNo in scanNos:
            out = self.extract_S140XRD_chidelta(
                scanNo,
                incl_q = incl_q,
                showGraph = showGraph,
                saveGraph = saveGraph,
                mirror_sorted_structure = mirror_sorted_structure,
            )
            outs.append(out)
        if showGraph:
            self.batch_plotter(outs, scanNos)
        return outs
    

    #======================================================
    def batch_plotter(self, figaxs, scanNos: list):
        """ Plots the results of batch extraction in a grid of subplots
            args:
                figaxs: a list of (fig, ax) tuples from the batch extraction
                scanNos: a list of scan numbers corresponding to the figaxs
        """
        fig0, ax0 = make_subplot_grid(len(scanNos), figsize = (12, 6))
        for i, [fig, ax] in enumerate(figaxs):
            try:
                for line in ax.get_lines():
                    ax0[i].plot(line.get_xdata(), line.get_ydata(), label = line.get_label(), color = line.get_color())
                ax0[i].set_title(f"Scan {scanNos[i]}")
                ax0[i].legend(fontsize=8, ncols=legend_columniser(len(ax.get_lines())))
            except Exception as e:
                print(f"Error occurred while plotting scan {scanNos[i]}: {e}")
        fig0.tight_layout()
        fig0.suptitle(f"Batch XRD Extraction", fontsize=16)
        plt.show()
        input("Press Enter to continue...")
        plt.close('all')
        return fig0, ax0


    #======================================================
    #visualising a raw image in a scan (can be a timescan)
    #no need anymore to choose the dataset, this is done automatically (from the shape)
    def s140_visu_one_rawImage(self, scanNo, pointIndex, logScale, mini, maxi, save_fig_path = False):
        """reads and visualize one single raw image of the XPAD,
        i.e. the n-th point in the scan (supposed to be a linescan, not a XY map)
        - NB: 1st image in a scan has index 0 !!!
        - use negative values for mini and maxi to have color autoscale
        """
        pointsFound, data_index_xpad, deltaArray, gamArray, phiArray, chiArray, omeArray, energyArray, TemperatureArray, mssg_metadata, mssgActuator_list, Izero, Izero_mean = self.s140_read_metadata_and_actuators(scanNo, print_Flag = True)
        #reading the XPAD data
        this_day = self.this_timestamp[2]; this_month = self.this_timestamp[1]; this_year=self.this_timestamp[0] # dated subfolder containin the nxs data
        nxs_pathFolder = self.nxsfile_directory + "%s-%s-%s/"%(str(this_year), str(this_month).zfill(2), str(this_day).zfill(2))
        fileName = nxs_fileName_root + str(scanNo).zfill(numberOfDigits) + nxs_fileName_suffix

        #reading images in the .nxs file
        file1 = tables.open_file(nxs_pathFolder+fileName)
        #fileNameRoot1 = file1.root._v_groups.keys()[0]
        #command = "file1.root.__getattr__(\""+str(fileNameRoot1)+"\")"
        command = "file1.root.scan_%s"%(str(scanNo).zfill(4) )
        xpadImage = eval(command+".scan_data.data_"+str(data_index_xpad).zfill(2)+"[pointIndex]")

        mssg2write = "*** pointIdx = %s "%(str(pointIndex))
        mssg2write += "delta = %.3f, gam = %.3f, E = %.3f"%(deltaArray[pointIndex], gamArray[pointIndex], energyArray[pointIndex])
        print (mssg2write)
        file1.close()

        thisImg = 0.0+xpadImage

        if logScale:
            thisImg = numpy.log10(thisImg); thisImg[numpy.isinf(thisImg)] = 0.; thisImg[numpy.isnan(thisImg)] = 0.
        if mini < 0 and maxi < 0:
            mini = numpy.min(thisImg); maxi = numpy.max(thisImg);
        mini = np.percentile(thisImg, 0.5); maxi = np.percentile(thisImg, 99.5)
        print ("Img min/max (%.2f %.2f)"%(mini, maxi)	)
        print ("   ... using min/max (%.2f %.2f)"%(mini, maxi))
        plt.figure()
        plt.title("%s; pointIdx = %s"%(fileName, str(pointIndex)))
        # Select the colormap and display the image
        cmp = plt.get_cmap("jet")
        plt.imshow(thisImg, vmin = mini, vmax = maxi, cmap = cmp, interpolation = "none")
        plt.colorbar(); plt.xlabel("x-coord (pixels)"); plt.ylabel("y-coord (pixels)");
        if save_fig_path:
            os.makedirs(os.path.dirname(save_fig_path), exist_ok=True)
            plt.savefig(save_fig_path+f"scan_{scanNo}_"+str(pointIndex).zfill(2)+".png", dpi = 300)
        plt.show()

        print ("FINISHED"	)
