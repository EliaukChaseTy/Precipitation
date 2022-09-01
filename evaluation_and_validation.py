import pandas as pd
import matplotlib
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import os
import sys
import glob
import math
import pickle
import joblib
import numpy as np
from datetime import timedelta, date
import datetime
import scipy.stats as sci
np.seterr(divide='ignore', invalid='ignore')
import warnings
warnings.filterwarnings('ignore') 
import csv
from sklearn import tree
import string
from pprint import pprint
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def import_feature_data_verification(startdate, enddate, exclude_MM, exclude_hh, perc_exclude,
                                    traindir, random, solar, seviri_diff, topography, wavelets):
   
    features = None
    files = [] 

    for n in range(0, int((enddate - startdate).total_seconds()/(15*60))):
        time = (startdate + n*datetime.timedelta(minutes = 15))

        YYYY = time.strftime("%Y")
        MM = time.strftime("%m")
        DD = time.strftime("%d")
        hh = time.strftime("%H")
        mm = time.strftime("%M")

        if MM in exclude_MM:
            continue
        if hh in exclude_hh:
            continue

        try:
            files.append(glob.glob('/gws/nopw/j04/swift/bpickering/random_forest_precip/1_training_data/'+YYYY+'/'+MM+'/'+YYYY+MM+DD+hh+mm+'.pkl')[0])
        except:
            continue
            

    print ('Number of time files: ' + str(len(files)) + ' * ' + str(1-perc_exclude))
    if len(files)*(1-perc_exclude) < 1.5: print ("perc_exclude too high, zero file output is likely")

    dframe_list = []
    for file in files:
        if np.random.rand() > perc_exclude:
            frame = pd.read_pickle(file)
            dframe_list.append(frame)


    features = pd.concat(dframe_list)
    
    if random:
        print("adding random number feature...")
        features['random'] = np.random.rand(len(features['GPM_PR']))
    

    dframe_list = None
    frame = None

    print('The shape of the features array is:', features.shape)
    print('The size of the features array is: ' + str(sys.getsizeof(features)/1000000000)[:-6] + ' GB.\n')
    
    return features




def sort_feature_data(features, bin_edges):

    labels = np.array(features['GPM_PR'])

    labels = np.digitize(labels, bins=bin_edges, right=False)

    features_no_labels= features.drop('GPM_PR', axis = 1)
    features_pd= features_no_labels.drop('YYYYMMDDhhmm', axis = 1)

    feature_list = list(features_pd.columns)

    features = np.array(features_pd)
    return features, labels, feature_list


def into_range(x, range_min, range_max):
    shiftedx = x - range_min
    delta = range_max - range_min
    return (((shiftedx % delta) + delta) % delta) + range_min





def add_CRR(ver_arr, bin_edges, CRR_dir = '/gws/swift/random_forest_precip/CRR_regridded/data/'):

    lats, lat_indexes = np.linspace(-34.95, 37.95, 730), np.linspace(0,730,731)
    lons, lon_indexes = np.linspace(-19.95, 51.95, 720), np.linspace(0,720,721)
    lati = dict(zip(np.round(lats*100)/100, lat_indexes))
    loni = dict(zip(np.round(lons*100)/100, lon_indexes))

    last_strdate = 'none'

    for i in range(0, len(ver_arr)):
        strdate = (str(ver_arr[i,0])[:-2])

        if strdate != last_strdate:
            try:
                file = (glob.glob(CRR_dir+strdate[:4]+'/'+strdate[:8]
                                        +'/NWCSAF_regridded_PanAfrica_'
                                        +strdate[:8]+'T'+strdate[8:]+'.npy')[0])
                CRR = np.load(file, allow_pickle=True)

            except:
                continue

        ver_arr[i, 11] = CRR.T[int( lati[np.round(ver_arr[i,5]*100)/100] ) , int( loni[np.round(ver_arr[i,6]*100)/100] ) ]

        last_strdate = strdate

    
    digitized = np.digitize(ver_arr[:, 11], bins=bin_edges, right=False) 
    ver_arr[:, 11] = np.ma.array(data=digitized, mask=ver_arr[:, 11].mask)
    
    return ver_arr

def add_CRR_Ph(ver_arr, bin_edges, CRR_Ph_dir = '/gws/swift/random_forest_precip/CRR-Ph_regridded/data/'):

    lats, lat_indexes = np.linspace(-34.95, 37.95, 730), np.linspace(0,730,731)
    lons, lon_indexes = np.linspace(-19.95, 51.95, 720), np.linspace(0,720,721)
    lati = dict(zip(np.round(lats*100)/100, lat_indexes))
    loni = dict(zip(np.round(lons*100)/100, lon_indexes))

    last_strdate = 'none'

    for i in range(0, len(ver_arr)):
        strdate = (str(ver_arr[i,0])[:-2])

        if strdate != last_strdate:
            try:                
                file = (glob.glob(CRR_Ph_dir+strdate[:4]+'/'+strdate[:8]
                                        +'/CRR-Ph_regridded_PanAfrica_'
                                        +strdate[:8]+'T'+strdate[8:]+'.npy')[0])
                CRR_Ph = np.load(file, allow_pickle=True)

            except:
                continue

        ver_arr[i, 12] = CRR_Ph.T[int( lati[np.round(ver_arr[i,5]*100)/100] ) , int( loni[np.round(ver_arr[i,6]*100)/100] ) ]

        last_strdate = strdate

    digitized = np.digitize(ver_arr[:, 12], bins=bin_edges, right=False) 
    ver_arr[:, 12] = np.ma.array(data=digitized, mask=ver_arr[:, 12].mask) 
        
    return ver_arr



def create_bin_values_and_labels(boundaries):

    bin_edges = np.array(boundaries).astype(np.float64)
    bin_values = {}
    bin_labels = {}
    for i in range(0,len(boundaries)+1):
        if i == 0:
            bin_values.update({i: "Error_<_"+boundaries[i]})
            bin_labels.update({i: "< "+boundaries[i]})
        elif i == 1:
            bin_values.update({1: np.float64(0.0)})
            bin_labels.update({1: '0.0'})
        elif i == len(boundaries):
            bin_values.update({len(boundaries): "Error_>_"+boundaries[i-1]})
            bin_labels.update({i: "> "+boundaries[i-1]})
        else:
            bin_values.update({i: (bin_edges[i-1]+bin_edges[i])/2})
            bin_labels.update({i: boundaries[i-1]+"–"+boundaries[i]})
            
    return bin_edges, bin_values, bin_labels


def precip_bin_values(data, bin_values):

    precip_values = np.ma.copy(data)
    
    if len(precip_values[data==0]) or len(precip_values[data==len(bin_values.items())-1]) > 0:
        raise ValueError("Values less than zero mm/h or greater than the maximum rain boundary exist within the verification table data.")
    
    for k, v in bin_values.items():
        if type(v) == np.float64:
            precip_values[data==k] = v
            
    return precip_values




def multi_HSS(matrix):

    n = sum(sum(matrix))
    PC = 0
    PCR = 0
    
    for i in range (0, len(matrix)):
        PC += matrix[i,i] / n
        PCR += (sum(matrix[i,:]) / n)*(sum(matrix[:,i]) / n)

    HSS = (PC - PCR) / (1 - PCR)

    return HSS


def general_stats(truth, test, bins, PRINT=False):

    errors = (test - truth)
    abs_errors = abs(errors)
    
    if not len(abs_errors) > 1:
        mae = np.ma.masked_all(1)
        max_err = np.ma.masked_all(1)
        cov = np.ma.masked_all(1)
        mean_err = np.ma.masked_all(1)
        bias = np.ma.masked_all(1)
        p_corr = np.ma.masked_all(1)
        r2 = np.ma.masked_all(1)
        s_corr = np.ma.masked_all(1)
        mse = np.ma.masked_all(1) 
        rmse = np.ma.masked_all(1)
        hss = np.ma.masked_all(1)
        
    else:
        # Mean Absolute Error
        mae = np.round(np.ma.mean(abs_errors), 2)
        max_err = np.round(np.ma.max(abs_errors), 2) 
        
        # Covariance
        cov = np.round(np.cov(test, truth)[0,1], 2)

        # Bias
        mean_err = np.round(np.ma.mean(errors), 2)
        bias = np.round(np.ma.mean(test)/np.ma.mean(truth), 2)

        # r^2
        # calculate Pearson's correlation (must be Gaussian distributed data, which precipitation is not)
        p_corr, _ = sci.pearsonr(test, truth)
        r2 = p_corr**2
        p_corr = np.round(p_corr, 2)
        r2 = np.round(r2, 2)


        # MSE and RMSE
        mse = np.ma.mean((test-truth)**2)
        rmse = np.round(mse**0.5, 2)
        mse = np.round(mse, 2)

        # HSS 
        # Create the 2D histogram matrix using bins provided
        matrix, xedges, yedges = np.histogram2d(truth, test, bins=bins)
        hss = np.round(multi_HSS(matrix), 2)
        

    if PRINT:
        # Print out the stats
        print('Mean Absolute Error:', mae, 'mm/h.')
        print('Maximum Error:', max_err, 'mm/h.')
        print('Mean Error:', mean_err, 'mm/h.')
        print('Coeffiecient of Determination:', r2)
        print('Mean Squared Error:', mse)
        print('Root Mean Squared Error:', rmse)
        print('Multi-dimensional Heidke Skill Score:', hss)
    
    return mae, max_err, mean_err, r2, mse, rmse, hss 




def generate_stats_file(outdir, ver_labels, ver, bin_edges):
    
    stats_table = [['product', 'mae', 'max_err', 'mean_err', 'r2', 'mse', 'rmse', 'hss']]
    
    for i in range(11, len(ver[0,:])):
        mae, max_err, cov, mean_err, bias, p_corr, r2, s_corr, mse, rmse, hss = general_stats(
            truth = ver[:, 10], 
            test = ver[:, i], 
            bins=np.array([-2.5, 2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5, 57.5, 62.5]),
            PRINT=False)
        
        print(ver_labels[i])
        stats_table.append([ver_labels[i], mae, max_err, cov, mean_err, bias, p_corr, r2, s_corr, mse, rmse, hss])
        

    # Export 
    with open(outdir+"stats.csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(stats_table)
        
        
    return
    
    
def reverse_ciruclar(cos, sin):

    circ_frac = np.zeros(len(cos))
    if not len(sin) == len(cos):
        raise ValueError("The sin and cos arrays differ in length.")

    for i in range(0, len(circ_frac)):
        if sin[i] >= 0. and cos[i] > 0.:
            # first quadrant
            circ_frac[i] = np.arctan( sin[i] / cos[i] )*180/np.pi

        elif sin[i] > 0. and cos[i] <= 0.:
            # second quadrant
            circ_frac[i] = np.arctan( -cos[i] / sin[i] )*180/np.pi + 90

        elif sin[i] <= 0. and cos[i] < 0.:
            # third quadrant
            circ_frac[i] = np.arctan( sin[i] / cos[i] )*180/np.pi + 180

        elif sin[i] < 0. and cos[i] >= 0.:
            # fourth quadrant
            circ_frac[i] = np.arctan( cos[i] / -sin[i] )*180/np.pi + 270 
        
    return circ_frac / 360


def diurnal_stats(hour_of_day, truth, test, timestep=3):

    timesteps = 24. / timestep
    if not timesteps-int(timesteps) == 0.:
        raise ValueError('timestep must be a factor of 24. ' + str(timestep) + ' is not a factor of 24.')
    timesteps = int(timesteps)
    
    mae_, max_err_, mean_err_, r2_, mse_, rmse_, hss_ = np.ma.zeros(timesteps), np.ma.zeros(timesteps), np.ma.zeros(timesteps), np.ma.zeros(timesteps), np.ma.zeros(timesteps), np.ma.zeros(timesteps), np.ma.zeros(timesteps), np.ma.zeros(timesteps), np.ma.zeros(timesteps), np.ma.zeros(timesteps), np.ma.zeros(timesteps)
    
    for h in range(0, timesteps):

        test_h = test[ (hour_of_day >= h*timestep) & (hour_of_day < (h+1)*timestep) ]
        truth_h = truth[ (hour_of_day >= h*timestep) & (hour_of_day < (h+1)*timestep) ]
        
        mae_[h], max_err_[h], mean_err_[h], r2_[h], mse_[h], rmse_[h], hss_[h] = general_stats(
            truth = truth_h, test = test_h, 
            bins=np.array([-2.5, 2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5, 57.5, 62.5]),
            PRINT=False) 
    
    return mae_, max_err_, mean_err_, r2_, mse_, rmse_, hss_


def annual_stats(month_of_year, truth, test, timestep=1): 

    timesteps = 12. / timestep
    if not timesteps-int(timesteps) == 0.:
        raise ValueError('timestep must be a factor of 12. ' + str(timestep) + ' is not a factor of 12.')
    timesteps = int(timesteps)
    
    mae_, max_err_, cov_, mean_err_, bias_, p_corr_, r2_, s_corr_, mse_, rmse_, hss_ = np.ma.zeros(timesteps), np.ma.zeros(timesteps), np.ma.zeros(timesteps), np.ma.zeros(timesteps), np.ma.zeros(timesteps), np.ma.zeros(timesteps), np.ma.zeros(timesteps), np.ma.zeros(timesteps), np.ma.zeros(timesteps), np.ma.zeros(timesteps), np.ma.zeros(timesteps)
    
    # Loop through each time period in the day
    for m in range(0, timesteps):

        test_m = test[ (month_of_year >= m*timestep) & (month_of_year < (m+1)*timestep) ]
        truth_m = truth[ (month_of_year >= m*timestep) & (month_of_year < (m+1)*timestep) ]

        mae_[m], max_err_[m], cov_[m], mean_err_[m], bias_[m], p_corr_[m], r2_[m], s_corr_[m], mse_[m], rmse_[m], hss_[m] = general_stats(
            truth = truth_m, test = test_m, 
            bins=np.array([-2.5, 2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5, 57.5, 62.5]),
            PRINT=False) 
    
    return mae_, max_err_, mean_err_, r2_, mse_, rmse_, hss_





def time_stats(ver, prod_i, ver_markers, outdir, bin_edges):
    
    hour_of_day = 24 * reverse_ciruclar(cos=ver[:, 3], sin=ver[:, 4])
    
    month_of_year = 12 * reverse_ciruclar(cos=ver[:, 1], sin=ver[:, 2])
    

    for p in prod_i: # Diurnal stats
        exec (p+"_mae_h,"+p+"_max_err_h,"+p+"_cov_h,"+p+"_mean_err_h,"+p+"_bias_h,"+p+"_p_corr_h,"+p+"_r2_h,"+p+"_s_corr_h,"+p+"_mse_h,"+p+"_rmse_h,"+p+"_hss_h = diurnal_stats(hour_of_day=hour_of_day, truth=ver[:, 7], test=ver[:, "+str(prod_i[p])+"], timestep=3)")

    for p in prod_i: # Annual stats
        exec (p+"_mae_m,"+p+"_max_err_m,"+p+"_cov_m,"+p+"_mean_err_m,"+p+"_bias_m,"+p+"_p_corr_m,"+p+"_r2_m,"+p+"_s_corr_m,"+p+"_mse_m,"+p+"_rmse_m,"+p+"_hss_m = annual_stats(month_of_year=month_of_year, truth=ver[:, 7], test=ver[:, "+str(prod_i[p])+"], timestep=1)")
        
    
    # Loop through diurnal hour (h), annual month (m)
    for kind in ['h','m']:
        fig = plt.figure(figsize=(10,10))
        
        gs1 = GridSpec(4, 3, wspace=0.3, hspace=0.7)
        ax1 = fig.add_subplot(gs1[0:1, 0:1]) # mae
        ax2 = fig.add_subplot(gs1[0:1, 1:2]) # max_err
        ax3 = fig.add_subplot(gs1[1:2, 0:1]) # mean_err
        ax4 = fig.add_subplot(gs1[2:3, 0:1]) # r2
        ax5 = fig.add_subplot(gs1[2:3, 2:3]) # mse
        ax6 = fig.add_subplot(gs1[3:4, 0:1]) # rmse
        ax7 = fig.add_subplot(gs1[3:4, 1:2]) # hss

        ax_stat = {1: 'mae_',
                   2: 'max_err_',
                   3: 'mean_err_',
                   4: 'r2_',
                   5: 'mse_',
                   6: 'rmse_',
                   7: 'hss_'}

        # Plot the data
        for i in range(1,12):
            if i == 1:
                for prod in prod_i:
                    exec("ax"+str(i)+".plot("+prod+"_"+ax_stat[i]+kind+", marker='"+ver_markers[prod_i[prod]]+"', label='"+prod+"')")
            else:
                for prod in prod_i:
                    exec("ax"+str(i)+".plot("+prod+"_"+ax_stat[i]+kind+", marker='"+ver_markers[prod_i[prod]]+"')")

        # Set titles
        ax1.set_title("Mean Absolute Error")
        ax2.set_title("Maximum Error")
        ax3.set_title("Mean Error")
        ax4.set_title("Coefficient of Determination")
        ax5.set_title("Mean Squared Error")
        ax6.set_title("Root Mean Squared Error")
        ax7.set_title("Heidke Skill Score")

        # Other plot settings
        if kind == 'h':
            ticklabels = ['00-03Z','03–06Z','06–09Z','09–12Z','12-15Z','15–18Z','18–21Z','21–00Z']
            ticks = np.array([0,1,2,3,4,5,6,7])
        elif kind == 'm':
            ticklabels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
            ticks = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12])
        elif kind == 'r':
            ticklabels = ['0–5','5-10','10-15','finish the rest later...']
            ticks = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12])
        else:
            print("kind is not diurnal, annual, or rate")

        for i in range(1,12):
            exec("ax"+str(i)+".set_xticks(ticks)")
            exec("ax"+str(i)+".set_xticklabels(ticklabels, rotation=55.)")
            exec("ax"+str(i)+".grid()")

        ax1.legend(loc=2, bbox_to_anchor=(2.57, -4.05), fontsize=12, fancybox=True, shadow=True)

        filename = {'h': '3hourly', 'm': 'monthly'}
        fig.savefig(outdir+filename[kind]+'_stats.pdf', bbox_inches="tight", dpi=250)
        fig.savefig(outdir+filename[kind]+'_stats.png', bbox_inches="tight", dpi=250)
        
        plt.close()
        
    return
        
    
        

def map_stats(truth, test,
              bin_edges,
              lats, lons,
              d_lats, d_lons,
              stat_px,
              ):

    y=int((len(d_lats))/stat_px)
    x=int((len(d_lons))/stat_px)
    if not y-int(y)==0. and x-int(x)==0.:
        raise ValueError('stat_px must be a factor of domain dimensions. Either ' + str(stat_px) + ' is not a factor of ' + str(len(d_lats)) + ' or ' + str(stat_px) + ' is not a factor of ' + str(len(d_lons)) + '.')
    y=int(y)
    x=int(x)
    mae_, max_err_, mean_err_, r2_, mse_, rmse_, hss_ = np.ma.zeros((y,x)), np.ma.zeros((y,x)), np.ma.zeros((y,x)), np.ma.zeros((y,x)), np.ma.zeros((y,x)), np.ma.zeros((y,x)), np.ma.zeros((y,x)), np.ma.zeros((y,x)), np.ma.zeros((y,x)), np.ma.zeros((y,x)), np.ma.zeros((y,x))

    num_samples = np.ma.zeros((y,x))
    
    for lat in range(0, y):
        for lon in range(0, x):
            test_r = test[ (lats > d_lats[stat_px*(lat+1)]) & (lats < d_lats[stat_px*lat]) & (lons > d_lons[stat_px*lon]) & (lons < d_lons[stat_px*(lon+1)])]
            truth_r = truth[ (lats > d_lats[stat_px*(lat+1)]) & (lats < d_lats[stat_px*lat]) & (lons > d_lons[stat_px*lon]) & (lons < d_lons[stat_px*(lon+1)])]

            num_samples[lat,lon] = test_r.count()
            
            if lon == 0:
                print (lat+1, 'out of 73. Num px in region:', len(test_r), 'out of', len(truth), 'or', round(len(test_r)/len(truth)*720*730, 1), '% of one whole map.')

            mae_[lat,lon], max_err_[lat,lon], cov_[lat,lon], mean_err_[lat,lon], bias_[lat,lon], p_corr_[lat,lon], r2_[lat,lon], s_corr_[lat,lon], mse_[lat,lon], rmse_[lat,lon], hss_[lat,lon] = general_stats(
                truth = truth_r, test = test_r, 
                bins=bin_edges,
                PRINT=False) 
            
        
    return num_samples, mae_, max_err_, mean_err_, r2_, mse_, rmse_, hss_


def plot_map(ax, data):

    ax.add_feature(cfeature.COASTLINE, edgecolor=(0,0,0,1), linewidth=1., zorder=3)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=1., zorder=3)
    ax.add_feature(cfeature.LAKES, edgecolor=(0,0,0,0), linewidth=0.25, facecolor=(1,1,1,0.4), zorder=1)
    ax.add_feature(cfeature.LAKES, edgecolor=(0,0,0,1), linewidth=0.25, facecolor=(0,0,0,0), zorder=3)

    cmap = matplotlib.cm.viridis
    cmap.set_under((0,0,0,0))
    img = ax.imshow(data, extent=[-20, 52, -35, 38], 
                       vmin=0.01, interpolation='nearest',
                       cmap=cmap, alpha=1., zorder=2)

    plt.colorbar(img, ax=ax, shrink=0.6, pad = 0.05, cmap=cmap, extend='max')
    
    ax.set_xticks(np.array([-20, -10, 0, 10, 20, 30, 40, 50]))
    ax.set_yticks(np.array([-30, -20, -10, 0, 10, 20, 30]))
    ax.grid()
    
    return
    
    
    


def plot_map_stats(prod_i, statlabels, outdir):

    for p in prod_i:
        fig = plt.figure(figsize=(16,20))
        
        gs1 = GridSpec(4, 3, wspace=0.15, hspace=0.)
        ax1 = fig.add_subplot(gs1[0:1, 0:1], projection=ccrs.PlateCarree()) # num_samples
        ax2 = fig.add_subplot(gs1[0:1, 1:2], projection=ccrs.PlateCarree()) # mae
        ax3 = fig.add_subplot(gs1[0:1, 2:3], projection=ccrs.PlateCarree()) # max_err
        ax5 = fig.add_subplot(gs1[1:2, 1:2], projection=ccrs.PlateCarree()) # r2
        ax7 = fig.add_subplot(gs1[2:3, 0:1], projection=ccrs.PlateCarree()) # hss
        ax8 = fig.add_subplot(gs1[2:3, 1:2], projection=ccrs.PlateCarree()) # precip_data
        ax9 = fig.add_subplot(gs1[2:3, 2:3], projection=ccrs.PlateCarree()) # num_samples (other method)

        
        
        # Plot the data
        for i in range(0, len(statlabels)):
            exec(p + statlabels[i] + "=np.load('"+outdir+'/map_stats/'+p+statlabels[i]+"', allow_pickle=True)")
            exec("plot_map(ax=ax"+str(i+1)+", data="+ p + statlabels[i] +")")
        
        
        # Set titles
        ax1.set_title("Number of Samples")
        ax2.set_title("Mean Absolute Error")
        ax3.set_title("Maximum Error")
        ax5.set_title("Coefficient of Determination")
        ax7.set_title("Heidke Skill Score")
        ax8.set_title("Total Rainfall")
        ax9.set_title("No. of Samples (should be same as ax1)")
        
        fig.savefig(outdir+'map_stats_'+p+'.pdf', bbox_inches="tight", dpi=250)
        fig.savefig(outdir+'map_stats_'+p+'.png', bbox_inches="tight", dpi=250)
        plt.close()
        
    return


def generate_map(rain_col, lat_data, lon_data):

    precip_data = np.zeros((730 ,720))
    precip_num = np.zeros((730 ,720))

    lats = np.linspace(37.95, -34.95, 730)
    rows = np.linspace(0, 729, 730)
    lat_2_row = dict(zip(lats.tolist(), rows.tolist()))

    lons = np.linspace(-19.95, 51.95, 720)
    cols = np.linspace(0, 719, 720)
    lon_2_col = dict(zip(lons.tolist(), cols.tolist()))
    
    counter = 0
    for i in range(0, len(rain_col)):
        if not np.ma.is_masked(rain_col[i]):
            precip_data[ int(lat_2_row[ lat_data[i] ] ) , int(lon_2_col[ lon_data[i] ]) ] += rain_col[i]
            precip_num[ int(lat_2_row[ lat_data[i] ] ) , int(lon_2_col[ lon_data[i] ]) ] += 1
            counter += 1
    
    return precip_data, precip_num



def grid(outdir, ver, ver_labels, bin_edges, bin_labels, truthname): ## REMOVED truth and test, need to define inside here in loop starting at [:, 11] to len([0,:])

    for p in range(11, len(ver[0,:])):
        
        fig = plt.figure(figsize=(10,10))

        gs1 = GridSpec(21, 21, wspace=0.5, hspace=0.1)
        ax1 = fig.add_subplot(gs1[0:3, 6:18]) # top hit rate bar chart
        ax3 = fig.add_subplot(gs1[3:15, 6:18]) # grid
        ax4 = fig.add_subplot(gs1[5:13, 18]) # colorbar
        ax5 = fig.add_subplot(gs1[17:21, 6:18]) # x-axis histogram

        heatmap, xedges, yedges = np.histogram2d(ver[:, 10], ver[:, p], bins=bin_edges)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        hit_rate = np.diagonal(heatmap) / (np.sum(heatmap, axis=1)) * 100
        steps = np.linspace(0, 200, len(bin_edges))
        width = 0.9*(steps[1]-steps[0])
        midbins = (steps[1:] + steps[:-1]) / 2
        ax1.bar(midbins, hit_rate, width=width, color='grey', zorder=1)
        ax1.set_yscale('log')
        ax1.set_ylim([0.05, 500])
        ax1.set_ylabel("Hit Rate (%)")  
        ax1.set_yticks([0.1, 1, 10, 100])
        ax1.set_yticklabels([0.1, 1, 10, 100])
        ax1.set_xticks(midbins)
        ax1.set_xlim([xedges[0], xedges[-1]])
        ax1.set_title(ver_labels[p]+' vs. '+truthname)
        for i in range(0, len(midbins)):
            ax1.text(x=midbins[i], y=200, s=str(hit_rate[i])[:3], fontsize='small', fontweight='bold', ha='center', va='center', color='grey')
        ax1.grid(which='both', axis='y', zorder=10)


        cs = ax3.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect="auto")

        for i in range(0, len(steps)-2):
            ax3.plot( steps[i:i+3], np.repeat(steps[i+1], 3), 'k-', linewidth=1., zorder=2.)
            ax3.plot( np.repeat(steps[i+1], 3), steps[i:i+3], 'k-', linewidth=1., zorder=2.)

        ax3.set_xticks(midbins)
        ax3.set_xticklabels(list(bin_labels.values())[1:-1], rotation=35.)
        ax3.set_xlabel(truthname+" Rainfall (mm h$^{-1}$)", labelpad=-5)
        ax3.set_yticks(midbins)
        ax3.set_yticklabels(list(bin_labels.values())[1:-1])
        ax3.set_ylabel(ver_labels[p]+" Rainfall (mm h$^{-1}$)", labelpad=-5)

        ax4.plot(0.5,0.5, 'r-', marker='o')
        cbar = fig.colorbar(cs, shrink=0.8, panchor=(-50., 0.5), anchor=(-50., 0.5), label="Log Frequency", cax=ax4)

        truthdist = (np.sum(heatmap, axis=1))
        testdist = (np.sum(heatmap, axis=0))
        ax5.bar(midbins - width/4, truthdist, width=width/2, zorder=1, label=truthname)
        ax5.bar(midbins + width/4, testdist, width=width/2, zorder=1, label=ver_labels[p])
        ax5.set_yscale('log')
        ax5.set_ylabel("Frequency")
        ax5.set_xlim([xedges[0], xedges[-1]])
        ax5.set_xticks(midbins)
        ax5.set_xticklabels(list(bin_labels.values())[1:-1], rotation=35.)
        ax5.set_xlabel("Rainfall (mm h$^{-1}$)", labelpad=-5)
        ax5.grid(which='major', axis='y', zorder=10)
        ax5.legend()


        # Save and display
        fig.savefig(outdir+'grid_'+ver_labels[p]+'_vs_'+truthname+'.pdf', bbox_inches="tight", dpi=250)
        fig.savefig(outdir+'grid_'+ver_labels[p]+'_vs_'+truthname+'.png', bbox_inches="tight", dpi=250)
        #plt.show()
        plt.close()
    
    return

    

    
def labelAtEdge(levels, cs, ax, fmt, side='all', pad=0.005, **kwargs):

    from matplotlib.transforms import Bbox
    collections = cs.collections
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    bbox = Bbox.from_bounds(xlim[0], ylim[0], xlim[1]-xlim[0], ylim[1]-ylim[0])
    eps = 1e-5  # error for checking boundary intersection
    # -----------Loop through contour levels-----------
    for ii, lii in enumerate(levels):
        cii = collections[ii]  # contours for level lii
        pathsii = cii.get_paths()  # the Paths for these contours
        if len(pathsii) == 0:
            continue
        for pjj in pathsii:
            # check first whether the contour intersects the axis boundary
            if not pjj.intersects_bbox(bbox, False):  # False significant here
                continue
            xjj = pjj.vertices[:, 0]
            yjj = pjj.vertices[:, 1]
            # intersection with the left edge
            if side in ['left', 'all']:
                inter_idx = np.where(abs(xjj-xlim[0]) <= eps)[0]
                for kk in inter_idx:
                    inter_x = xjj[kk]
                    inter_y = yjj[kk]
                    ax.text(inter_x-pad, inter_y, fmt % lii,
                            ha='right',
                            va='center',
                            **kwargs)
            # intersection with the right edge
            if side in ['right', 'all']:
                inter_idx = np.where(abs(xjj-xlim[1]) <= eps)[0]
                for kk in inter_idx:
                    inter_x = xjj[kk]
                    inter_y = yjj[kk]
                    ax.text(inter_x+pad, inter_y, fmt % lii,
                            ha='left',
                            va='center',
                            **kwargs)
            # intersection with the bottom edge
            if side in ['bottom', 'all']:
                inter_idx = np.where(abs(yjj-ylim[0]) <= eps)[0]
                for kk in inter_idx:
                    inter_x = xjj[kk]
                    inter_y = yjj[kk]
                    ax.text(inter_x-pad, inter_y, fmt % lii,
                            ha='center',
                            va='top',
                            **kwargs)
            # intersection with the top edge
            if side in ['top', 'all']:
                inter_idx = np.where(abs(yjj-ylim[-1]) <= eps)[0]
                for kk in inter_idx:
                    inter_x = xjj[kk]
                    inter_y = yjj[kk]
                    ax.text(inter_x+pad, inter_y, fmt % lii,
                            ha='center',
                            va='bottom',
                            **kwargs)
    return




def performance_stats(ver, truth_i, ver_labels, ver_markers, ver_colors, bin_edges):
    '''
    Produce the performance dictionary, looping through all columns 11 onwards
    '''
    performance={}
    for i in range(truth_i+1, len(ver[0])):
        # Create the 2D histogram using bins provided
        heatmap, xedges, yedges = np.histogram2d(ver[:, truth_i], ver[:, i], bins=bin_edges)
        
        # Calculate the hit rate for each column
        POD = np.diagonal(heatmap) / (np.sum(heatmap, axis=1))
        
        # Calculate the FAR and then SR for each column
        FAR = ((np.sum(heatmap, axis=0)) - np.diagonal(heatmap)) / (np.sum(heatmap, axis=0))
        SR = 1 - FAR
        
        # print(SR, FAR)
        
        # Add the scores of the product to the overall dictionary
        performance.update({ver_labels[i]: 
                       {'pod': POD,
                        'sr': SR,
                        'marker': ver_markers[i],
                        'color': ver_colors[i]
                       }})
    
    return performance



                                         
def performance_diagram(outdir, p, bin_labels, description=''):  

    figure = plt.figure(figsize=(10, 10), dpi=100)
    ax1 = figure.add_subplot(1, 1, 1)

    ticks=np.arange(0, 1.01, 0.1)
    csi_levels=np.arange(0.1, 1.1, 0.1)  
    b_levels=np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.66, 2.5, 5])
    grid_ticks = np.arange(0.0, 1.01, 0.01)
    sr_g, pod_g = np.meshgrid(grid_ticks, grid_ticks)
    bias = pod_g / sr_g
    csi = 1.0 / (1.0 / sr_g + 1.0 / pod_g - 1.0)


    csi_contour = ax1.contour(sr_g, pod_g, csi, levels=csi_levels, extend="max", colors=[(0,0,0,0.2)])
    csi_contour.collections[0].set_label('Critical Success Index') # to make CSI show on plot legend
    labelAtEdge(levels=csi_levels, cs=csi_contour, ax=ax1, fmt='%.1f', side='top', pad=0.005, rotation=0., color=(0,0,0,0.5), fontstyle='oblique')
    labelAtEdge(levels=csi_levels, cs=csi_contour, ax=ax1, fmt='%.1f', side='right', pad=0.005, rotation=0., color=(0,0,0,0.5), fontstyle='oblique')

    # Bias lines
    b_contour = ax1.contour(sr_g, pod_g, bias, levels=b_levels, colors=[(0,0,0,0.2)], linestyles="dashed")
    b_contour.collections[0].set_label('Frequency Bias') # to make CSI show on plot legend
    
    ax1.clabel(b_contour, fmt="%.2f", manual=[(0.08,0.69), (0.20,0.67), (0.32,0.63), (0.45,0.54), (0.5,0.5),
                                              (0.55,0.44), (0.59,0.38), (0.64,0.28), (0.68,0.13)]) # mid-curve bias labels

    ABC = list(string.ascii_uppercase)
    for i in p:
        ax1.plot(p[i]['sr'], p[i]['pod'], marker=p[i]['marker'], color=p[i]['color'], label=i, linewidth=0., markersize=15., alpha=0.8)
        for j in range(0,len(p[i]['pod'])):
            ax1.text(p[i]['sr'][j], p[i]['pod'][j], ABC[j], ha='center', va='center', fontsize=8.)
    
    legend_left = 'Key:\n'
    legend_right = '\n'
    for j in range(0,len(p[i]['pod'])):
        legend_left = legend_left + ABC[j] + ':\n' 
        legend_right= legend_right + bin_labels[j+1] + ' mm h$^{-1}$\n'
    ax1.text(1.07, 0.00, legend_left, ha='left', va='bottom', fontsize=12)
    ax1.text(1.12, 0.00, legend_right, ha='left', va='bottom', fontsize=12)

    # Other plot parameters
    ax1.set_xlabel(xlabel="Success Ratio (1-FAR)", fontsize=14)
    ax1.set_ylabel(ylabel="Probability of Detection", fontsize=14)
    ax1.set_title("Performance Diagram\n"+description+"\n", fontsize=14, fontweight="bold")
    ax1.set_xticks(ticks)
    ax1.set_yticks(ticks)
    ax1.grid(alpha=0.1, which='both')
    ax1.legend(loc=2, bbox_to_anchor=(1.04, 1.0), fontsize=12, fancybox=True, shadow=True)

    # Save and display the figure
    figure.savefig(outdir+'performance_diagram'+description+'.pdf', dpi=250, bbox_inches="tight")
    figure.savefig(outdir+'performance_diagram'+description+'.png', dpi=250, bbox_inches="tight")
    #plt.show()
    plt.close()
    
    return
    


def performance_diagram_zoomed(outdir, p, bin_labels, description=''):

    figure = plt.figure(figsize=(10, 10), dpi=100)
    ax1 = figure.add_subplot(1, 1, 1)

    ticks=np.arange(0, 0.11, 0.01)
    csi_levels=np.arange(0.01, 0.1, 0.01)
    b_levels=np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.66, 2.5, 5])#([0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 3.3, 10])

    grid_ticks = np.arange(0, 0.101, 0.001)
    sr_g, pod_g = np.meshgrid(grid_ticks, grid_ticks)
    bias = pod_g / sr_g
    csi = 1.0 / (1.0 / sr_g + 1.0 / pod_g - 1.0)

    
    csi_contour = ax1.contour(sr_g, pod_g, csi, levels=csi_levels, extend="max", colors=[(0,0,0,0.2)])
    csi_contour.collections[0].set_label('Critical Success Index') # to make CSI show on plot legend
    labelAtEdge(levels=csi_levels, cs=csi_contour, ax=ax1, fmt='%.2f', side='top', pad=0.0005, rotation=0., color=(0,0,0,0.5), fontstyle='oblique')
    labelAtEdge(levels=csi_levels, cs=csi_contour, ax=ax1, fmt='%.2f', side='right', pad=0.0005, rotation=0., color=(0,0,0,0.5), fontstyle='oblique')

    # Bias lines
    b_contour = ax1.contour(sr_g, pod_g, bias, levels=b_levels, colors=[(0,0,0,0.2)], linestyles="dashed")
    b_contour.collections[0].set_label('Frequency Bias') # to make CSI show on plot legend
   
    ax1.clabel(b_contour, fmt="%.2f", manual=[(0.008,0.069), (0.020,0.067), (0.032,0.063), (0.045,0.054), (0.05,0.05),
                                              (0.055,0.044), (0.059,0.038), (0.064,0.028), (0.068,0.013)]) # mid-curve bias labels

    ABC = list(string.ascii_uppercase)
    for i in p:
        counter = 0
        for j in range(0,len(p[i]['pod'])):
            if p[i]['sr'][j] <= 0.1 and p[i]['pod'][j] <= 0.1:
                if counter == 0: 
                    ax1.plot(p[i]['sr'][j], p[i]['pod'][j], marker=p[i]['marker'], color=p[i]['color'], label=i, linewidth=0., markersize=15., alpha=0.8)
                    counter = 1
                else:
                    ax1.plot(p[i]['sr'][j], p[i]['pod'][j], marker=p[i]['marker'], color=p[i]['color'], linewidth=0., markersize=15., alpha=0.8)
                  
                ax1.text(p[i]['sr'][j], p[i]['pod'][j], ABC[j], ha='center', va='center', fontsize=8.)
    
    legend_left = 'Key:\n'
    legend_right = '\n'
    for j in range(0,len(p[i]['pod'])):
        legend_left = legend_left + ABC[j] + ':\n' 
        legend_right= legend_right + bin_labels[j+1] + ' mm h$^{-1}$\n'
    ax1.text(0.107, 0.00, legend_left, ha='left', va='bottom', fontsize=12)
    ax1.text(0.112, 0.00, legend_right, ha='left', va='bottom', fontsize=12)

    ax1.set_xlabel(xlabel="Success Ratio (1-FAR)", fontsize=14)
    ax1.set_ylabel(ylabel="Probability of Detection", fontsize=14)
    ax1.set_title("Performance Diagram Zoomed\n"+description+"\n", fontsize=14, fontweight="bold")
    ax1.set_xticks(ticks)
    ax1.set_yticks(ticks)
    ax1.grid(alpha=0.1, which='both')
    ax1.legend(loc=2, bbox_to_anchor=(1.04, 1.0), fontsize=12, fancybox=True, shadow=True)

    ax1.set_ylim(0., 0.1)
    ax1.set_xlim(0., 0.1)

    figure.savefig(outdir+'performance_diagram_zoomed'+description+'.pdf', dpi=250, bbox_inches="tight")
    figure.savefig(outdir+'performance_diagram_zoomed'+description+'.png', dpi=250, bbox_inches="tight")
    plt.close()
    
    return


def plot_importances(RF_dict, outdir):
    for RF in RF_dict:

        # Extract model and flabels from RF_dict
        clf, flabels = RF_dict[RF]['model'], RF_dict[RF]['labels']
        
        # Get numerical feature importances
        importances = list(clf.feature_importances_)

        # List of tuples with variable and importance
        feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(flabels, importances)]

        # Sort the feature importances by most important first
        feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

        # Export the features and importances as csv
        with open(outdir+"feature_importances_"+str(RF)+".csv","w+") as my_csv:
            csvWriter = csv.writer(my_csv,delimiter=',')
            csvWriter.writerows(feature_importances)


        # Set the style
        plt.style.use('fivethirtyeight')

        # list of x locations for plotting
        x_values = list(range(len(importances)))

        # Make a bar chart (differs if random is included) 
        if 'random' in flabels:
            # Work out where it is
            rand_index = flabels.index('random')
            print(rand_index)
            # Plot red line instead
            plt.plot([-0.5, -1.5+len(importances)],[importances[rand_index], importances[rand_index]], color='red', linestyle='--', linewidth=3., label='Random')
            # Remove from importances and plot
            del importances[rand_index]
            flabels.remove('random')

            plt.bar(x_values[:-1], importances, orientation = 'vertical')
            plt.legend()

        else:
            plt.bar(x_values, importances, orientation = 'vertical')

        # Tick labels for x axis
        plt.xticks(x_values, flabels, rotation='vertical')
        plt.axis([None, None, -0.01, None])

        # Axis labels and title
        plt.ylabel('Importance'); plt.xlabel('Feature'); plt.title('Feature Importances');
        plt.savefig(outdir+'feature_importances_'+str(RF)+'.png', bbox_inches="tight", dpi=250)
        
        plt.close()
    


    
    
def make_thresholds(RF_dict):

    thresholds_dict = {}
    for RF in RF_dict:
        feature_list, clf = RF_dict[RF]['labels'], RF_dict[RF]['model']
        
        thresholds = {}
        for i in range(0, len(feature_list)):
            thresholds.update({i: {'value': [], 'weight': []} })


        t_nodes = 0
        for i in range(0, len(clf.estimators_)):
            for n in range(0, clf.estimators_[i].tree_.node_count):
                t_nodes += 1

                if not clf.estimators_[i].tree_.feature[n] == -2:

                    thresholds[ clf.estimators_[i].tree_.feature[n] ]['value'].append( clf.estimators_[i].tree_.threshold[n] )

                    thresholds[ clf.estimators_[i].tree_.feature[n] ]['weight'].append( clf.estimators_[i].tree_.n_node_samples[n] / clf.estimators_[i].tree_.n_node_samples[0])
                    
        thresholds_dict.update({RF: {'thresholds': thresholds,
                                   't_nodes': t_nodes,
                                   'feature_list': feature_list}})
            
    return thresholds_dict
    

def threshold_hist(ax, data, minn, maxx, rangee, xlabel):
    ax.hist(data, bins=20, zorder=2)
    ax.yaxis.tick_right()
    ax.set_xlim([minn - rangee*0.0, maxx + rangee*0.0])
    ax.set_xlabel(xlabel)
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(ax.get_yticks()[1]/5))
    ax.grid(which='both', zorder=-1)
    

def threshold_heatmap(ax, thresholds, i, minn, maxx, rangee, cm):
    h, xedges, yedges, image = ax.hist2d(np.array(thresholds[i]['value']), np.array(thresholds[i]['weight']), bins=[40,20],  range=[[minn, maxx],[0,1]],  cmin=1.0, cmap=cm, zorder=2)
    ax.set_ylim([0.0,1.0])
    ax.set_xlim([minn - rangee*0.0, maxx + rangee*0.0])
    ax.set_xticklabels([''])
    ax.set_ylabel('')
    ax.set_facecolor((0,0,0,0.2))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
    ax.grid(which='both', zorder=0)
    
    return np.nanmax(h)


def threshold_colorbar(ax, max_h, cm):
    a = np.array([[1.0, max_h]])
    img = ax.imshow(a, cmap=cm)
    ax.set_visible(False)
    cb = plt.colorbar(mappable=img, orientation="vertical", ax=ax, shrink=0.7, aspect=10)
    cb.ax.minorticks_on()
    


def plot_feature_thresholds(RF_dict, thresholds_dict, feature_units, outdir): # was thresholds, feature_list, outdir !!!!!

    for RF in thresholds_dict:
        thresholds = thresholds_dict[RF]['thresholds']
        feature_list = thresholds_dict[RF]['feature_list']
        
        # Create custom colormap (logarithmic magma)
        magma, bob, counter, cmap_name = matplotlib.cm.get_cmap('magma'), list(), 0.0, 'log_magma'       
        for step in ['aa','bb','cc','dd','ee','ff','gg','hh','ii']:
            exec(step+"=("+ str((counter)**3) +", magma(counter))")
            exec("bob.append("+step+")")
            counter += 0.125
        cm = matplotlib.colors.LinearSegmentedColormap.from_list(cmap_name, bob)

        c = {'1s':0, '1e':40, '2s':0, '2e':40, '3s':37, '3e':42}
        r = {'1s': 0, '1e': 20, '2s': 20, '2e':30, '3s': 0, '3e': 18}

        cols = 3

        h = 45
        w = 50

        # Define matplotlib style
        plt.style.use('default')

        # Start figure and GridSpec
        fig = plt.figure(figsize=(20,20))
        gs1 = GridSpec(320, 165, wspace=0.0, hspace=0.0)

        # Loop through each feature (feature list and thresholds could be ranked by importance)
        for i in range(1, len(feature_list)+1):
            #print ("\n\n\n", feature_list[i])

            # Calculate the column and row that the figure should be in
            col = (i-1)%cols+1
            row = int(np.floor((i-1)/3)+1)

            # Calculate the GridSpec index to add to each row and column
            c_ = (col-1)*w
            r_ = (row-1)*h

            # Create the three subplots for the figure
            exec("ax"+str(i)+" = fig.add_subplot(gs1["+str(r_ + r['1s'])+":"+str(r_ + r['1e'])+", "+str(c_ + c['1s'])+":"+str(c_ + c['1e'])+"])")
            exec("ax"+str(i)+"_ = fig.add_subplot(gs1["+str(r_ + r['2s'])+":"+str(r_ + r['2e'])+", "+str(c_ + c['2s'])+":"+str(c_ + c['2e'])+"])")
            exec("ax"+str(i)+"__ = fig.add_subplot(gs1["+str(r_ + r['3s'])+":"+str(r_ + r['3e'])+", "+str(c_ + c['3s'])+":"+str(c_ + c['3e'])+"])")

            # Calculate the min, max and range of thresholds
            minn = np.min(thresholds[i-1]['value'])
            maxx = np.max(thresholds[i-1]['value'])
            rangee = maxx - minn

            # Plot heatmap                 
            exec("heat_ax = ax"+str(i))
            exec("max_h = threshold_heatmap(ax=heat_ax, thresholds=thresholds, i=i-1, minn=minn, maxx=maxx, rangee=rangee, cm=cm)")

            # Plot histogram
            exec("hist_ax = ax"+str(i)+"_")
            exec("threshold_hist(ax=hist_ax, data=thresholds[i-1]['value'], minn=minn, maxx=maxx, rangee=rangee, xlabel=(feature_list[i-1] + feature_units[feature_list[i-1]]))")

            # Plot heatmap colorbar
            exec("hist_ax = ax"+str(i)+"__")
            exec("threshold_colorbar(ax=hist_ax, max_h=max_h, cm=cm)")

        # Save and display
        fig.savefig(outdir+'feature_thresholds_'+str(RF)+'.pdf', bbox_inches="tight", dpi=250)
        fig.savefig(outdir+'feature_thresholds_'+str(RF)+'.png', bbox_inches="tight", dpi=250)

        #plt.show()
        plt.close()



def plot_rf_tree(RF_dict, outdir, number=0):
    for RF in RF_dict:
        clf, flabels = RF_dict[RF]['model'], RF_dict[RF]['labels']
        tree.plot_tree(clf.estimators_[number], feature_names=flabels, node_ids=True, filled=True, proportion=True, rounded=True, precision=2) 
        plt.savefig(outdir+'tree.png')
        plt.savefig(outdir+'tree.pdf')
        plt.close()

np.random.seed(42)

start_time = datetime.datetime.now() # For tracking the performance of this verification script.


param_file = sys.argv[1] 
RF_parameters = pickle.load( open(param_file, "rb") )

pprint(RF_parameters)

np.random.seed(RF_parameters["random_state"])

  
bin_edges, bin_values, bin_labels = create_bin_values_and_labels(boundaries=RF_parameters["bin_edges"])
print("\nRain rate boundaries:", list(bin_labels.values()))


modelpaths = {
        1: RF_parameters['modeldir']
    + 'RF_model___base_seviri_plus_diff_plus_topog_plus_wavelets___201501-201912_all_0.5perc___bins_0-0.5-2-5-10-20-35-60-100-200__'
    + 'est100_balanced_subsample_gini_maxfeat4_minsplit5_maxdepth10_bootstrap_0.0001_time-20220306-0647.pkl'}



prod_i = {'CRR': 11,
          'CRR_Ph': 12,
          'Random_Forest': 13}#,


ver_labels={0: 'YYYYMMDDhhmm',
            1: 'Annual_cos',
            2: 'Annual_sin',
            3: 'Diurnal_cos',
            4: 'Diurnal_sin',
            5: 'Latitude',
            6: 'Longitude',
            7: 'Solar_elevation',
            8: 'Solar_azimuth_cos',
            9: 'Solar_azimuth_sin',
            10: 'IMERG',
            11: 'CRR',
            12: 'CRR_Ph',
            13:'Random_Forest'}

feature_units = {
    'Annual_cos': ' ',
    'Annual_sin': ' ',
    'Diurnal_cos': ' ',
    'Diurnal_sin': ' ',
    'Latitude': ' (º)',
    'Longitude': ' (º)',
    'Solar_elevation': ' (º)',
    'Solar_azimuth_cos': ' ',
    'Solar_azimuth_sin': ' ',
    'MSG_0.6': ' Reflectance (%)',
    'MSG_0.8': ' Reflectance (%)',
    'MSG_1.6': ' Reflectance (%)',
    'MSG_3.9': ' Brightness Temperature (K)',
    'MSG_6.2': ' Brightness Temperature (K)',
    'MSG_7.3': ' Brightness Temperature (K)',
    'MSG_8.7': ' Brightness Temperature (K)',
    'MSG_9.7': ' Brightness Temperature (K)',
    'MSG_10.8': ' Brightness Temperature (K)',
    'MSG_12.0': ' Brightness Temperature (K)',
    'MSG_13.4': ' Brightness Temperature (K)'
}

# Markers for time stats plots and performance diagram
ver_markers={11: 'h', # hexagon (vertical point)
             12: '*', # 5-point star
             13:'P', # plus (filled)
             14:'s', # square
             15:'p', # pentagon
             16:'v', # tringle down
             17:'o', # circle
             18:'X', # X (filled)
             19:'d', # thin diamond
             20:'^'} # triangle_up

# Colours for performance diagram
ver_colors={11:'tab:blue',
            12:'tab:orange',
            13:'tab:green',
            14:'tab:red',
            15:'tab:purple',
            16:'tab:brown',
            17:'tab:pink',
            18:'tab:gray',
            19:'tab:olive',
            20:'tab:cyan'}


features = import_feature_data_verification(
    # general settings
    startdate=RF_parameters["verify_startdate"], 
    enddate=RF_parameters["verify_enddate"], 
    exclude_MM=RF_parameters["exclude_MM"], 
    exclude_hh=RF_parameters["exclude_hh"], 
    perc_exclude=RF_parameters["verify_perc_exclude"], 
    traindir=RF_parameters["traindir"], 
    # base_features
    random=RF_parameters["base_features"]["random"]
)


ver = np.ma.masked_all(( (len(features)), 13+len(modelpaths) )) # 13 is the length of the array with just metadata, IMERG, CRR, CRR-Ph

ver[:,:7] = np.array(features)[:,:7]
ver[:, 7], ver[:, 8], ver[:, 9] = np.array(features['Solar_elevation']), np.array(features['Solar_azimuth_cos']), np.array(features['Solar_azimuth_sin'])


print (features.columns)

features, labels, all_features_list = sort_feature_data(
    features=features, 
    bin_edges=np.array(RF_parameters["bin_edges"]).astype(np.float64)
)

ver[:, 10] = labels[:]

print("loading RF models...")

RF_dict = {}

for i in range(1, len(modelpaths)+1):
    start = 12 # starting index of models !!!! this "12+" will have to change if verification products (NIPE), or metadata like topography are added
    
    # Load model
    RF_model = joblib.load(modelpaths[i])
    
    feature_list = RF_model.feature_names_in_
    print(feature_list)
    
    if len(feature_list) > len(all_features_list):
        raise ValueError("Model has more features in it's list than are avaialble from the main features array.")
    f_index = []
    for f in feature_list: # loop through all the feature labels for this particular model
        f_index.append(all_features_list.index(f)) # find the position of the feature label in the wider/main features array
    features_ = features[:,f_index] # slice out only the features needed for this array
    
   
    RF_predictions = RF_model.predict(features_) #provide only the features this particular model expects to see.
    ver[:, start+i]= RF_predictions[:] 
    
    exec("RF_dict.update({'"+ver_labels[start+i]+"': {'model': RF_model, 'labels': feature_list} })")
    print(ver_labels[start+i], "loaded")
    exec("print('Last tree count:', RF_dict['"+ver_labels[start+i]+"']['model'].estimators_[-1].tree_.node_count)")
    
pprint(RF_dict)

ver = add_CRR(ver_arr = ver, bin_edges=np.array(RF_parameters["bin_edges"]).astype(np.float64), CRR_dir = '/gws/nopw/j04/swift/bpickering/random_forest_precip/4_verify_model/CRR_regridded/data/')

ver = add_CRR_Ph(ver_arr = ver, bin_edges=np.array(RF_parameters["bin_edges"]).astype(np.float64), CRR_Ph_dir = '/gws/nopw/j04/swift/bpickering/random_forest_precip/4_verify_model/CRR-Ph_regridded/data/')

ver[:, 10:] = precip_bin_values(data=ver[:, 10:], bin_values=bin_values)

for i in range(0, len(ver[0, :])):
    if len(ver_labels[i]) < 11:
           print (ver_labels[i] + ':\t\t' + str(ver[0, i]))
    else:
           print (ver_labels[i] + ':\t' + str(ver[0, i]))
            

generate_stats_file(outdir=RF_parameters['verdir'], ver_labels=ver_labels, ver=ver, bin_edges=np.array(RF_parameters["bin_edges"]).astype(np.float64))
time_stats(ver=ver, prod_i=prod_i, ver_markers=ver_markers, outdir=RF_parameters['verdir'], bin_edges=np.array(RF_parameters["bin_edges"]).astype(np.float64))

if not os.path.exists(RF_parameters['verdir']+"/map_stats"):
    os.mkdir(RF_parameters['verdir']+"/map_stats")
     
for p in prod_i:
    print ('\n\n\n', p)
    
    exec (p+"_num_samples_r,"+p+"_mae_r,"+p+"_max_err_r,"+p+"_cov_r,"+p+"_mean_err_r,"+p+"_bias_r,"+p+"_p_corr_r,"+p+"_r2_r,"+p+"_s_corr_r,"+p+"_mse_r,"+p+"_rmse_r,"+p+"_hss_r = map_stats(truth=ver[:, 10], test=ver[:, "+str(prod_i[p])+"], bin_edges=np.array(RF_parameters['bin_edges']).astype(np.float64), lats = ver[:, 5], lons = ver[:, 6], d_lats = np.linspace(38., -35., 731), d_lons = np.linspace(-20., 52., 721), stat_px = 10)")
    
    for stat in ["_num_samples_r","_mae_r","_max_err_r","_cov_r","_mean_err_r","_bias_r","_p_corr_r","_r2_r","_s_corr_r","_mse_r","_rmse_r","_hss_r"]:
        exec(p+stat+".dump('"+RF_parameters['verdir']+"/map_stats/"+p+stat+"')")

for i in range(10, len(ver[0,:])): # loop through all the products, including the truth label, IMERG
    print(ver_labels[i])
    
    exec(ver_labels[i]+"_precip_data, "+ver_labels[i]+"_precip_num = generate_map(rain_col=ver[:, "+str(i)+"], lat_data=ver[:, 5], lon_data=ver[:, 6])")
    
    exec(ver_labels[i]+"_precip_data.dump('"+RF_parameters['verdir']+"/map_stats/"+ver_labels[i]+"_precip_data')")
    exec(ver_labels[i]+"_precip_num.dump('"+RF_parameters['verdir']+"/map_stats/"+ver_labels[i]+"_precip_num')")


statlabels = ["_num_samples_r", "_mae_r","_max_err_r","_r2_r","_rmse_r","_hss_r","_precip_data","_precip_num"] # new simplified list for ease of interpretation

plot_map_stats(prod_i, statlabels, outdir=RF_parameters['verdir'])


grid(
    outdir=RF_parameters['verdir'], 
    ver=ver, 
    ver_labels=ver_labels,
    bin_edges=np.array(RF_parameters["bin_edges"]).astype(np.float64), 
    bin_labels=bin_labels,
    truthname='IMERG'
)
performance = performance_stats(ver=ver, truth_i=10, ver_labels=ver_labels, ver_markers=ver_markers, ver_colors=ver_colors, bin_edges=np.array(RF_parameters["bin_edges"]).astype(np.float64))

performance_diagram(outdir=RF_parameters['verdir'], p=performance, bin_labels=bin_labels, description='')
performance_diagram_zoomed(outdir=RF_parameters['verdir'], p=performance, bin_labels=bin_labels, description='')
plot_importances(RF_dict=RF_dict, outdir=RF_parameters['verdir'])
thresholds_dict = make_thresholds(RF_dict=RF_dict)
plot_feature_thresholds(RF_dict=RF_dict, thresholds_dict=thresholds_dict, feature_units=feature_units, outdir=RF_parameters['verdir'])
plot_rf_tree(RF_dict=RF_dict, outdir=RF_parameters['verdir'], number=0)
