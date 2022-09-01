import satpy # Requires virtual environmen for reading native (.nat)  and hrit files.
import numpy as np
import datetime
import glob
from pyresample import geometry, bilinear
import os
import h5py
import sys
from tempfile import gettempdir
import numpy.ma as ma
import pickle
import pandas as pd

# The following are necessary to read compressed HRIT files
os.environ['XRIT_DECOMPRESS_PATH']='/home/users/SOFTWARE/PublicDecompWT/2.06/xRITDecompress/xRITDecompress' # Necessary to read compressed files.
my_tmpdir = gettempdir() + '/'
print ('Temporary Directory for Decompress: '+my_tmpdir)

sev_data_dir1='/gws/nopw/j04/swift/earajr/HRIT_archive/'
sev_data_dir2='/gws/nopw/j04/swift/SEVIRI/' # Second directory to check. Necessary as first directory is incomplete

unavailable_times = (
                     [datetime.datetime(2014,3,2,12,00,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)]
                    +[datetime.datetime(2014,3,3,12,00,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)]
                    +[datetime.datetime(2014,3,4,12,00,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)]
                    +[datetime.datetime(2014,3,26,2,45,0)]
                    +[datetime.datetime(2014,5,21,1,45,0)] # Problems reading
                    +[datetime.datetime(2014,5,21,7,15,0)] # Problems reading
                    +[datetime.datetime(2014,5,21,8,0,0)] # Problems reading
                    +[datetime.datetime(2014,5,21,10,30,0)] # Problems reading
                    +[datetime.datetime(2014,5,21,10,45,0)] # Problems reading
                    +[datetime.datetime(2014,5,21,11,15,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)] # Problems reading
                    +[datetime.datetime(2014,5,21,12,0,0)+datetime.timedelta(seconds=60*15*n) for n in range(6)] # Problems reading
                    +[datetime.datetime(2014,5,21,14,0,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)] # Problems reading
                    +[datetime.datetime(2014,5,21,15,0,0)] # Problems reading
                    +[datetime.datetime(2014,5,21,18,0,0)] # Problems reading
                    +[datetime.datetime(2014,5,21,19,15,0)] # Problems reading
                    +[datetime.datetime(2014,5,21,20,0,0)] # Problems reading
                    +[datetime.datetime(2014,5,21,20,30,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)] # Problems reading
                    +[datetime.datetime(2014,5,21,22,0,0)] # Problems reading
                    +[datetime.datetime(2014,5,21,23,30,0)] # Problems reading
                    +[datetime.datetime(2014,5,22,4,15,0)] # Problems reading
                    +[datetime.datetime(2014,5,22,5,45,0)+datetime.timedelta(seconds=60*15*n) for n in range(8)] # Problems reading
                    +[datetime.datetime(2014,5,22,8,15,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)] # Problems reading
                    +[datetime.datetime(2014,5,22,9,0,0)] # Problems reading
                    +[datetime.datetime(2014,5,22,9,45,0)] # Problems reading
                    +[datetime.datetime(2014,5,22,10,15,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)] # Problems reading
                    +[datetime.datetime(2014,5,22,11,0,0)] # Problems reading
                    +[datetime.datetime(2014,5,22,11,30,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)] # Problems reading
                    +[datetime.datetime(2014,5,22,12,0,0)] # Problems reading
                    +[datetime.datetime(2014,5,22,12,30,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)] # Problems reading
                    +[datetime.datetime(2014,5,22,13,15,0)+datetime.timedelta(seconds=60*15*n) for n in range(4)] # Problems reading
                    +[datetime.datetime(2014,5,22,14,30,0)] # Problems reading
                    +[datetime.datetime(2014,5,22,16,30,0)] # Problems reading
                    +[datetime.datetime(2014,5,22,17,0,0)] # Problems reading
                    +[datetime.datetime(2014,5,22,18,30,0)] # Problems reading
                    +[datetime.datetime(2014,5,22,19,30,0)] # Problems reading
                    +[datetime.datetime(2014,5,22,21,45,0)] # Problems reading
                    +[datetime.datetime(2014,5,22,23,0,0)] # Problems reading
                    +[datetime.datetime(2014,5,23,0,15,0)] # Problems reading
                    +[datetime.datetime(2014,5,23,0,45,0)] # Problems reading
                    +[datetime.datetime(2014,5,23,4,0,0)] # Problems reading
                    +[datetime.datetime(2014,5,23,4,45,0)] # Problems reading
                    +[datetime.datetime(2014,5,23,10,30,0)+datetime.timedelta(seconds=60*15*n) for n in range(9)] # Problems reading
                    +[datetime.datetime(2014,5,23,13,0,0)] # Problems reading
                    +[datetime.datetime(2014,5,23,14,0,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)] # Problems reading
                    +[datetime.datetime(2014,5,23,15,45,0)] # Problems reading
                    +[datetime.datetime(2014,5,23,16,30,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)] # Problems reading
                    +[datetime.datetime(2014,5,23,19,30,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)] # Problems reading
                    +[datetime.datetime(2014,5,23,20,15,0)] # Problems reading
                    +[datetime.datetime(2014,5,23,20,45,0)] # Problems reading
                    +[datetime.datetime(2014,5,23,22,15,0)] # Problems reading
                    +[datetime.datetime(2014,6,20,19,45,0)] # Problems reading
                    +[datetime.datetime(2014,8,15,14,45,0)] # Problems reading
                    +[datetime.datetime(2014,10,9,11,45,0)] # Problems reading
                    +[datetime.datetime(2014,10,10,11,45,0)] # Problems reading
                    +[datetime.datetime(2014,10,11,11,45,0)] # Problems reading
                    +[datetime.datetime(2014,10,12,11,45,0)] # Problems reading
                    +[datetime.datetime(2014,12,4,14,30,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)]
                    +[datetime.datetime(2014,12,8,6,45,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)]
                    +[datetime.datetime(2015,3,1,12,0,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)]
                    +[datetime.datetime(2015,3,2,12,0,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)]
                    +[datetime.datetime(2015,3,3,9,0,0)+datetime.timedelta(seconds=60*15*n) for n in range(4)]
                    +[datetime.datetime(2015,3,3,12,0,0)+datetime.timedelta(seconds=60*15*n) for n in range(3)]
                    +[datetime.datetime(2015,3,4,12,15,0)]
                    +[datetime.datetime(2015,3,16,11,30,0)]
                    +[datetime.datetime(2015,4,28,7,45,0)+datetime.timedelta(seconds=60*15*n) for n in range(4)]
                    +[datetime.datetime(2015,4,28,12,0,0)+datetime.timedelta(seconds=60*15*n) for n in range(3)]
                    +[datetime.datetime(2015,5,14,9,45,0)]
                    +[datetime.datetime(2015,5,14,10,45,0)]
                    +[datetime.datetime(2015,5,14,11,15,0)]
                    +[datetime.datetime(2015,5,29,2,15,0)]
                    +[datetime.datetime(2015,6,1,5,45,0)]
                    +[datetime.datetime(2015,6,2,2,45,0)] # Problems reading
                    +[datetime.datetime(2015,6,2,13,0,0)] # Problems reading
                    +[datetime.datetime(2015,6,3,2,15,0)] # Problems reading
                    +[datetime.datetime(2015,6,3,5,45,0)] # Problems reading
                    +[datetime.datetime(2015,6,3,7,45,0)] # Problems reading
                    +[datetime.datetime(2015,6,4,3,45,0)] # Problems reading
                    +[datetime.datetime(2015,6,4,19,0,0)] # Problems reading
                    +[datetime.datetime(2015,6,4,20,45,0)] # Problems reading
                    +[datetime.datetime(2015,6,5,12,15,0)] # Problems reading
                    +[datetime.datetime(2015,6,5,16,0,0)] # Problems reading
                    +[datetime.datetime(2015,6,8,2,15,0)] # Problems reading
                    +[datetime.datetime(2015,6,9,6,15,0)] # Problems reading
                    +[datetime.datetime(2015,6,9,10,0,0)] # Problems reading
                    +[datetime.datetime(2015,6,9,21,0,0)] # Problems reading
                    +[datetime.datetime(2015,6,10,18,0,0)] # Problems reading
                    +[datetime.datetime(2015,6,10,19,0,0)] # Problems reading
                    +[datetime.datetime(2015,6,11,4,15,0)] # Problems reading
                    +[datetime.datetime(2015,6,11,5,30,0)] # Problems reading
                    +[datetime.datetime(2015,6,12,15,0,0)] # Problems reading
                    +[datetime.datetime(2015,6,12,18,45,0)] # Problems reading
                    +[datetime.datetime(2015,6,12,22,15,0)] # Problems reading
                    +[datetime.datetime(2015,6,13,19,30,0)] # Problems reading
                    +[datetime.datetime(2015,6,14,1,0,0)] # Problems reading
                    +[datetime.datetime(2015,6,14,6,0,0)] # Problems reading
                    +[datetime.datetime(2015,6,14,7,15,0)] # Problems reading
                    +[datetime.datetime(2015,6,14,20,30,0)] # Problems reading
                    +[datetime.datetime(2015,6,14,21,15,0)] # Problems reading
                    +[datetime.datetime(2015,6,14,23,45,0)] # Problems reading
                    +[datetime.datetime(2015,6,15,2,0,0)] # Problems reading
                    +[datetime.datetime(2015,6,15,4,15,0)] # Problems reading
                    +[datetime.datetime(2015,6,15,7,0,0)] # Problems reading
                    +[datetime.datetime(2015,6,15,7,45,0)] # Problems reading
                    +[datetime.datetime(2015,6,15,14,15,0)] # Problems reading
                    +[datetime.datetime(2015,6,15,19,45,0)] # Problems reading
                    +[datetime.datetime(2015,6,15,20,0,0)] # Problems reading
                    +[datetime.datetime(2015,6,15,22,0,0)] # Problems reading
                    +[datetime.datetime(2015,6,15,22,45,0)] # Problems reading
                    +[datetime.datetime(2015,6,16,3,30,0)] # Problems reading
                    +[datetime.datetime(2015,6,16,7,0,0)] # Problems reading
                    +[datetime.datetime(2015,6,16,12,15,0)] # Problems reading
                    +[datetime.datetime(2015,6,16,15,30,0)] # Problems reading
                    +[datetime.datetime(2015,6,16,16,0,0)] # Problems reading
                    +[datetime.datetime(2015,6,17,12,30,0)] # Problems reading
                    +[datetime.datetime(2015,6,17,12,45,0)] # Problems reading
                    +[datetime.datetime(2015,6,17,14,15,0)] # Problems reading
                    +[datetime.datetime(2015,6,18,10,45,0)] # Problems reading
                    +[datetime.datetime(2015,6,18,15,30,0)] # Problems reading
                    +[datetime.datetime(2015,6,18,15,45,0)] # Problems reading
                    +[datetime.datetime(2015,6,18,16,30,0)] # Problems reading
                    +[datetime.datetime(2015,6,18,17,30,0)] # Problems reading
                    +[datetime.datetime(2015,6,18,19,15,0)] # Problems reading
                    +[datetime.datetime(2015,6,18,21,0,0)] # Problems reading
                    +[datetime.datetime(2015,6,19,7,30,0)] # Problems reading
                    +[datetime.datetime(2015,6,19,14,30,0)] # Problems reading
                    +[datetime.datetime(2015,6,19,16,30,0)] # Problems reading
                    +[datetime.datetime(2015,6,20,4,15,0)] # Problems reading
                    +[datetime.datetime(2015,6,20,13,30,0)] # Problems reading
                    +[datetime.datetime(2015,6,21,4,30,0)] # Problems reading
                    +[datetime.datetime(2015,6,21,7,15,0)] # Problems reading
                    +[datetime.datetime(2015,6,21,9,30,0)] # Problems reading
                    +[datetime.datetime(2015,6,21,15,0,0)] # Problems reading
                    +[datetime.datetime(2015,6,21,18,45,0)] # Problems reading
                    +[datetime.datetime(2015,6,22,1,45,0)] # Problems reading
                    +[datetime.datetime(2015,6,22,9,0,0)] # Problems reading
                    +[datetime.datetime(2015,6,22,16,15,0)] # Problems reading
                    +[datetime.datetime(2015,6,22,16,30,0)] # Problems reading
                    +[datetime.datetime(2015,7,1,0,0,0)+datetime.timedelta(seconds=60*15*n) for n in range(4)]
                    +[datetime.datetime(2015,10,10,11,45,0)]
                    +[datetime.datetime(2015,10,11,11,45,0)]
                    +[datetime.datetime(2015,10,12,11,45,0)]
                    +[datetime.datetime(2015,10,21,10,0,0)]
                    +[datetime.datetime(2015,11,15,3,30,0)+datetime.timedelta(seconds=60*15*n) for n in range(19)] 
                    +[datetime.datetime(2015,11,16,8,30,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)]
                    +[datetime.datetime(2015,11,25,12,15,0)]
                    +[datetime.datetime(2016,2,29,12,0,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)]
                    +[datetime.datetime(2016,1,1,8,45,0)] #Problems reading
                    +[datetime.datetime(2016,1,1,23,15,0)] #Problems reading
                    +[datetime.datetime(2016,1,2,2,15,0)] #Problems reading
                    +[datetime.datetime(2016,1,3,3,15,0)] #Problems reading
                    +[datetime.datetime(2016,1,4,3,45,0)] #Problems reading
                    +[datetime.datetime(2016,1,4,7,30,0)] #Problems reading
                    +[datetime.datetime(2016,1,4,8,45,0)] #Problems reading
                    +[datetime.datetime(2016,1,4,16,30,0)] #Problems reading
                    +[datetime.datetime(2016,1,5,11,15,0)] #Problems reading
                    +[datetime.datetime(2016,1,5,12,15,0)] #Problems reading
                    +[datetime.datetime(2016,1,5,20,15,0)] #Problems reading
                    +[datetime.datetime(2016,1,6,9,45,0)] #Problems reading
                    +[datetime.datetime(2016,1,6,15,30,0)] #Problems reading
                    +[datetime.datetime(2016,1,6,20,45,0)] #Problems reading
                    +[datetime.datetime(2016,3,8,10,30,0)]
                    +[datetime.datetime(2016,6,8,14,0,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)]
                    +[datetime.datetime(2016,7,20,15,15,0)] # Problems reading
                    +[datetime.datetime(2016,7,29,2,45,0)]
                    +[datetime.datetime(2016,8,17,9,45,0)]
                    +[datetime.datetime(2016,10,8,9,30,0)] # Unavailable from Eumetsat
                    +[datetime.datetime(2016,10,10,11,45,0)] # Problems reading
                    +[datetime.datetime(2016,10,11,11,0,0)] # Problems reading
                    +[datetime.datetime(2016,10,11,11,45,0)] # Problems reading
                    +[datetime.datetime(2016,10,12,11,45,0)] # Problems reading
                    +[datetime.datetime(2016,10,12,16,45,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)] # Supposedly available from EumetSat website, but no files in folder on more than one attempt to download
                    +[datetime.datetime(2016,10,12,22,15,0)] # Supposedly available from EumetSat website, but no files in folder on more than one attempt to download
                    +[datetime.datetime(2016,10,14,9,30,0)] # Problems reading
                    +[datetime.datetime(2016,10,15,12,30,0)+datetime.timedelta(seconds=60*15*n) for n in range(8)]
                    +[datetime.datetime(2016,10,16,13,30,0)+datetime.timedelta(seconds=60*15*n) for n in range(66)] # Weird horizontal lines on these files...
                    +[datetime.datetime(2017,2,26,12,15,0)] # Unavailable for download
                    +[datetime.datetime(2017,2,27,12,15,0)] # Unavailable for download
                    +[datetime.datetime(2017,2,28,12,15,0)]
                    +[datetime.datetime(2017,3,17,21,45,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)]
                    +[datetime.datetime(2017,4,22,22,15,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)]
                    +[datetime.datetime(2017,6,22,11,45,0)]
                    +[datetime.datetime(2017,9,22,11,30,0) + datetime.timedelta(seconds=60*15*n) for n in range(3)]
                    +[datetime.datetime(2017,10,10,11,45,0)]
                    +[datetime.datetime(2017,10,11,11,45,0)]
                    +[datetime.datetime(2017,10,12,11,45,0)]
                    +[datetime.datetime(2017,10,13,11,45,0)]
                    +[datetime.datetime(2017,11,7,7,0,0) + datetime.timedelta(seconds=60*15*n) for n in range(5)]
                    +[datetime.datetime(2018,3,6,12,15,0,0)]
                    +[datetime.datetime(2018,3,21,12,15,0,0)]
                    +[datetime.datetime(2018,5,6,20,0,0,0)]
                    +[datetime.datetime(2018,5,7,4,45,0,0)]
                    +[datetime.datetime(2018,5,7,12,0,0,0)]
                    +[datetime.datetime(2018,6,20,9,15,0) + datetime.timedelta(seconds=60*15*n) for n in range(3)]
                    +[datetime.datetime(2018,7,3,13,30,0)+datetime.timedelta(seconds=60*15*n) for n in range(2)]
                    +[datetime.datetime(2018,7,4,4,0,0)]
                    +[datetime.datetime(2018,7,4,4,45,0)]
                    +[datetime.datetime(2018,7,10,23,30,0)+datetime.timedelta(seconds=60*15*n) for n in range(16)]
                    +[datetime.datetime(2018, 9, 24, 12, 30)]
                    +[datetime.datetime(2018, 9, 27, 7, 30)+datetime.timedelta(seconds=60*15*n) for n in range(2)]
                    +[datetime.datetime(2018, 10, 10, 11, 45)]
                    +[datetime.datetime(2018, 10, 12, 11, 45)]
                    +[datetime.datetime(2018, 10, 13, 11, 45)]
                    +[datetime.datetime(2018, 10, 15, 11, 45)]
                    +[datetime.datetime(2018, 11, 19, 9, 45)]
                    +[datetime.datetime(2018, 12, 2, 15, 15)]
                    +[datetime.datetime(2018, 12, 6, 15, 0)]
                    +[datetime.datetime(2019, 1, 16, 10, 30)]
                    +[datetime.datetime(2019, 1, 22, 15, 30)+datetime.timedelta(seconds=60*15*n) for n in range(3)]
                    +[datetime.datetime(2019, 1, 29, 12, 0)+datetime.timedelta(seconds=60*15*n) for n in range(2)]
                    +[datetime.datetime(2019, 2, 14, 9, 0)]
                    +[datetime.datetime(2019, 3, 5, 12, 15)] # NaNs in file
                    +[datetime.datetime(2019, 3, 6, 12, 15)] # NaNs in file
                    +[datetime.datetime(2019, 3, 7, 12, 15)] # NaNs in file
                    +[datetime.datetime(2019, 3, 12, 13, 0)+datetime.timedelta(seconds=60*15*n) for n in range(2)] # NaNs in file
                    +[datetime.datetime(2019, 4, 12, 10, 15)] # Unavailable from EumetSat
                    +[datetime.datetime(2019, 5, 6, 15, 45)] # Unavailable from EumetSat
                    +[datetime.datetime(2019, 5, 14, 10, 15)] # Unavailable from EumetSat
                    +[datetime.datetime(2019, 5, 24, 14, 0)] # Unavailable from EumetSat
                    +[datetime.datetime(2019, 5, 28, 13, 15)] # Unavailable from EumetSat
                    +[datetime.datetime(2019, 5, 29, 2, 15)] # Unavailable from EumetSat
                    +[datetime.datetime(2019, 5, 29, 15, 15)] # Unavailable from EumetSat
                    +[datetime.datetime(2019, 8, 17, 7, 15)]
                    +[datetime.datetime(2019, 10, 8, 10, 30)]
                    +[datetime.datetime(2019, 10, 8, 10, 45)]
                    +[datetime.datetime(2019, 10, 16, 12, 15)]
                    +[datetime.datetime(2019, 11, 11, 14, 0)]
                    +[datetime.datetime(2019, 11, 11, 14, 15)]
                    +[datetime.datetime(2019, 11, 11, 14, 30)]
                    +[datetime.datetime(2019, 12, 1, 0, 0)]
                    +[datetime.datetime(2019, 12, 2, 15, 0)]
                    +[datetime.datetime(2019, 12, 17, 9, 45)]
                    +[datetime.datetime(2020, 5, 3, 8, 15)]
                    +[datetime.datetime(2020, 5, 3, 8, 45)]
                    +[datetime.datetime(2020, 5, 3, 9, 45)]
                    +[datetime.datetime(2020, 5, 3, 10, 45)]
                    +[datetime.datetime(2020, 5, 3, 11, 45)]
                    +[datetime.datetime(2020, 5, 3, 12, 45)]
                    +[datetime.datetime(2020, 5, 3, 13, 45)]
                    +[datetime.datetime(2020, 5, 3, 14, 45)]
                    +[datetime.datetime(2020, 5, 3, 15, 45)]
                    +[datetime.datetime(2020, 5, 3, 16, 45)]
                    +[datetime.datetime(2020, 5, 3, 17, 45)]
                    )# List of missing times - these are not available from the EumetSat website, or have multiple lines of missing data.

file_dict = {
             0.6 : 'VIS006',
             0.8 : 'VIS008',
             1.6 : 'IR_016',
             3.9 : 'IR_039',
             6.2 : 'WV_062',
             7.3 : 'WV_073',
             8.7 : 'IR_087',
             9.7 : 'IR_097',
             10.8 : 'IR_108',
             12.0 : 'IR_120',
             13.4 : 'IR_134'
            }

def read_seviri_channel(channel_list, time, subdomain=(), regrid=False, my_area=geometry.AreaDefinition('pan_africa', 'Pan-Africa on Equirectangular 0.1 degree grid used by GPM', 'pan_africa', {'proj' : 'eqc'}, 720, 730, (-2226389.816, -3896182.178, 5788613.521, 4230140.650)), interp_coeffs=()):

    filenames = []
    sat_names = ['MSG4', 'MSG3', 'MSG2', 'MSG1']
    sat_ind = -1
    
    
    if time in unavailable_times:
        raise UnavailableFileError("SEVIRI observations for "+time.strftime("%Y/%m/%d_%H%M")+" are not available")
    print('time=', time)
    
    
   
    while ((len(filenames) == 0) & (sat_ind < len(sat_names)-1)): 
        sat_ind += 1
        filenames=glob.glob(sev_data_dir1+time.strftime("%Y/%Y%m%d/%H/*")+sat_names[sat_ind]+time.strftime("*EPI*%Y%m%d%H%M-*"))+ glob.glob(sev_data_dir1+time.strftime("%Y/%Y%m%d/%H/*")+sat_names[sat_ind]+time.strftime("*PRO*%Y%m%d%H%M-*"))# PRO and EPI files necessary in all scenarios
        sev_dir = sev_data_dir1+time.strftime("%Y/%Y%m%d/%H/*")+sat_names[sat_ind]
    
    
    if ((len(filenames) < 2)):
        sat_ind = -1
        while ((len(filenames) == 0) & (sat_ind < len(sat_names)-1)): # Sometimes have data from multiple instruments (e.g. 20160504_1045 has MSG3 and MSG1), this ensures most recent is prioritised.
            sat_ind += 1
            filenames=glob.glob(sev_data_dir2+time.strftime("%Y/%Y%m%d/%H/*")+sat_names[sat_ind]+time.strftime("*EPI*%Y%m%d%H%M-*"))+ glob.glob(sev_data_dir2+time.strftime("%Y/%Y%m%d/%H/*")+sat_names[sat_ind]+time.strftime("*PRO*%Y%m%d%H%M-*"))# PRO and EPI files necessary in all scenarios
            sev_dir = sev_data_dir2+time.strftime("%Y/%Y%m%d/%H/*")+sat_names[sat_ind]
    
    
    if  ((time == datetime.datetime(2016,4,11,19,0,0))| (time == datetime.datetime(2018,4,23,7,15,0))): # These files are present in sev_dir1, but corrupt
        filenames=glob.glob(sev_data_dir2+time.strftime("%Y/%Y%m%d/%H/*")+sat_names[sat_ind]+time.strftime("*EPI*%Y%m%d%H%M-*"))+ glob.glob(sev_data_dir2+time.strftime("%Y/%Y%m%d/%H/*")+sat_names[sat_ind]+time.strftime("*PRO*%Y%m%d%H%M-*"))# PRO and EPI files necessary in all scenarios
        sev_dir = sev_data_dir2+time.strftime("%Y/%Y%m%d/%H/*")+sat_names[sat_ind]    
    if (len(filenames) < 2):
        
        raise MissingFileError("SEVIRI observations for "+time.strftime("%Y/%m/%d_%H%M")+" are missing. Please check if they can be downloaded and if not, add to the list of unavailable times.")
        
    
    else:
        for channel in channel_list:
            filenames=filenames + glob.glob(sev_dir+'*'+file_dict[channel]+time.strftime("*%Y%m%d%H%M-*")) # add channels required
        
        scene = satpy.Scene(reader="seviri_l1b_hrit", filenames=filenames)
        data = {}
        scene.load(channel_list)
        
        if regrid != False:
            lons, lats = my_area.get_lonlats()
            if len(interp_coeffs) == 0:
                interp_coeffs = bilinear.get_bil_info(scene[channel_list[0]].area, my_area, radius=50e3, nprocs=1)
                data.update({'interp_coeffs': interp_coeffs})
            for channel in channel_list:
                data.update({str(channel): bilinear.get_sample_from_bil_info(scene[channel].values.ravel(), interp_coeffs[0], interp_coeffs[1], interp_coeffs[2], interp_coeffs[3], output_shape=my_area.shape)})
        else:
            if len(subdomain) > 0:
                scene = scene.crop(ll_bbox=subdomain)
            lons, lats = scene[channel_list[0]].area.get_lonlats()
            lons = lons[:,::-1] # Need to invert y-axis to get longitudes increasing.
            lats = lats[:,::-1]
            for channel in channel_list:
                data.update({str(channel) : scene[channel].values[:,::-1]})
        data.update({'lons' : lons, 'lats' : lats, 'interp_coeffs' : interp_coeffs})
        # Compressed files are decompressed to TMPDIR. Now tidy up
        # This doesn't seem to be applicable to me...
        delete_list = glob.glob(my_tmpdir+'/'+time.strftime("*%Y%m%d%H%M-*"))
        for d in delete_list: os.remove(d)
        return data


class FileError(Exception):

    pass


class UnavailableFileError(FileError):

    pass


class MissingFileError(FileError):

    pass




gpm_dir = '/badc/gpm/data/GPM-IMERG-v6/' # Change as appropriate


def read_gpm(timelist, lon_min=-20., lon_max=52., lat_min=-35., lat_max=38., varname='HQprecipitation'):
 
    rain = []
    for i, time in enumerate(timelist):
        f = get_gpm_filename(time)
        dataset = h5py.File(f, 'r')
        lon = dataset['Grid']['lon'][:]
        lat = dataset['Grid']['lat'][:]
        ind_lon = np.where((lon >= lon_min) & (lon <= lon_max))[0]
        ind_lat = np.where((lat >= lat_min) & (lat <= lat_max))[0]
        if dataset['Grid'][varname].ndim == 3:
            rain += [dataset['Grid'][varname][0,ind_lon[0]:ind_lon[-1]+1, ind_lat[0]:ind_lat[-1]+1]]
        else:
            print(("dataset['Grid'][varname].ndim=", dataset['Grid'][varname].ndim))
            sys.exit()
    rain = np.ma.masked_array(np.array(rain), mask=(np.array(rain) < 0.0))
    return lon[ind_lon], lat[ind_lat], rain



def get_gpm_filename(time):

    f = glob.glob(gpm_dir + time.strftime('%Y/%j/*%Y%m%d-S%H%M*.HDF5'))
    if len(f) != 1:
        print(("gpm_dir + time.strftime('%Y/%j/*%Y%m%d-S%H%M*.HDF5')=", gpm_dir + time.strftime('%Y/%j/*%Y%m%d-S%H%M*.HDF5')))
        print(("f=", f))
        sys.exit()
    return f[0]

    with open('/home/users/random_forest_precip/training_data/interp_coeffs_SEVIRI_to_GPM.pickle', 'rb') as f:
     interp_coeffs = pickle.load(f)

outdir = "/work/scratch-pw/RF_generate_features/" 

for mm in ['00', '15', '30', '45']:
    print (mm)

    time = datetime.datetime(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(mm))   # int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6] for when used as a script

    
    data = np.ma.zeros((525600, 19))

    data[:, 0] = np.repeat(int(sys.argv[1]+sys.argv[2]+sys.argv[3]+sys.argv[4]+mm), 525600)  # time in format YYYYMMDDhhmm repeated to fill array
    # For ML, the time needs to be cyclical, so have a sin and a cos
    data[:, 1] = np.repeat(np.cos((time.timetuple().tm_yday / datetime.datetime(time.year, 12, 31).timetuple().tm_yday)*2*np.pi), 525600) # cos of DoYear / days in that year
    data[:, 2] = np.repeat(np.sin((time.timetuple().tm_yday / datetime.datetime(time.year, 12, 31).timetuple().tm_yday)*2*np.pi), 525600) # sin of DoYear / days in that year
    data[:, 3] = np.repeat(np.cos((int(sys.argv[4])*60 +int(mm))/1440*2*np.pi), 525600) # diurnal time in cyclical format (cos of minutes / 1440 in radians so *2pi)
    data[:, 4] = np.repeat(np.sin((int(sys.argv[4])*60 +int(mm))/1440*2*np.pi), 525600) # diurnal time in cyclical format (sin of minutes / 1440 in radians so *2pi)

    data[:, 5] = np.repeat(np.linspace(37.95, -34.95, 730), 720)   # latitude repeated because order is top left to bottom right
    data[:, 6] = np.tile(np.linspace(-19.95, 51.95, 720), 730)     # longitude tiled because order is top left to bottom right

    if time.minute == 15 or time.minute == 45:
        latter_15min = True
        GPMstart = time - datetime.timedelta(minutes=15)
        #print (GPMstart)
    elif time.minute == 0 or time.minute == 30:
        latter_15min = False
        GPMstart = time
        #print (GPMstart)
    else:
        raise KeyError("The time you have selected is not on Meteosat observation frequency. Please use a 15-min interval (00, 15, 30, or 45)")

    #if time is a list, then it will return 2D spacial with 3rd dimension being time
    GPM_lon, GPM_lat, GPM_rain = read_gpm([GPMstart], varname='HQprecipitation')   #HQprecipitation   precipitationCal    HQobservationTime
    # HQobservationtime
    GPM_lon, GPM_lat, GPM_time = read_gpm([GPMstart], varname='HQobservationTime')   #HQprecipitation   precipitationCal    HQobservationTime


    GPM = ((np.flip(GPM_rain[0].T, axis=0)).flatten())
    GPM_time_check = ((np.flip(GPM_time[0].T, axis=0)).flatten())

    if latter_15min:
        GPM = np.ma.masked_where(GPM_time_check < 15, GPM)
    else:
        GPM = np.ma.masked_where(GPM_time_check >= 15, GPM)
 
    data[:, 7] = GPM

    try:
        SEVIRI = read_seviri_channel(channel_list  = [0.6, 0.8, 1.6, 3.9, 6.2, 7.3, 8.7, 9.7, 10.8, 12.0, 13.4],
                                         time=time, subdomain=(), regrid=True, 
                                         my_area=geometry.AreaDefinition('pan_africa', 
                                                                         'Pan-Africa on Equirectangular 0.1 degree grid used by GPM', 
                                                                         'pan_africa', {'proj' : 'eqc'}, 720, 730, 
                                                                         (-2226389.816, -3896182.178, 5788613.521, 4230140.650)), 
                                                                         interp_coeffs=(interp_coeffs))
    except:
        print('Error with ' + int(sys.argv[1]+sys.argv[2]+sys.argv[3]+sys.argv[4]+mm) + ', continuing...')
        continue

    col = 8
    for channel in [0.6, 0.8, 1.6, 3.9, 6.2, 7.3, 8.7, 9.7, 10.8, 12.0, 13.4]:
        data[:, col] = (SEVIRI[str(channel)].flatten())
        col +=1


    features = pd.DataFrame(data=data, columns=["YYYYMMDDhhmm", "Annual_cos", "Annual_sin", "Diurnal_cos", "Diurnal_sin", "Latitude", "Longitude", "GPM_PR", "MSG_0.6", "MSG_0.8", "MSG_1.6", 
                                               "MSG_3.9", "MSG_6.2", "MSG_7.3", "MSG_8.7", "MSG_9.7", "MSG_10.8", "MSG_12.0", "MSG_13.4"])
    # Drop any row which contains a NaN
    features = features.dropna()
    # Drop any row where 10.8 µm BT > 273.15 K (because this will likely not be a cloud)
    features = features[features['MSG_10.8'] < 273.15]

    # Save the filtered Pandas DataFrame
    features.to_pickle(outdir+"/"+sys.argv[1]+"/"+sys.argv[2]+"/"+sys.argv[1]+sys.argv[2]+sys.argv[3]+sys.argv[4]+mm+".pkl")
