import pandas as pd
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import sys
import glob
import math
import pickle
import joblib
import numpy as np
from datetime import timedelta, date
import datetime
import joblib
import scipy.stats as sci
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint

np.random.seed(42)


def import_feature_data(startdate, enddate, exclude_MM, exclude_hh, perc_keep, traindir, random, solar, seviri_diff, topography, wavelets):
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
            files.append(glob.glob(traindir+YYYY+'/'+MM+'/'+YYYY+MM+DD+hh+mm+'.pkl')[0])
        except:
            continue

    print ('Number of time files: ' + str(len(files)))
    if len(files)*perc_keep < 1.5: print ("perc_keep too low, zero file output is likely")

    
    dframe_list = []
    for file in files:
        if np.random.rand() <= perc_keep:
            # print (file)
            frame = pd.read_pickle(file)
            dframe_list.append(frame)
    
    features = pd.concat(dframe_list)
    
    if random:
        print("adding random number feature...")
        features['random'] = np.random.rand(len(features['GPM_PR']))
    
    

    dframe_list = None
    frame = None
    
    return features


def sort_feature_data(features, bin_edges):
    labels = np.array(features['GPM_PR'])

    labels = np.digitize(labels, bins=bin_edges, right=False)

    features_no_labels= features.drop('GPM_PR', axis = 1)
    features_pd= features_no_labels.drop('YYYYMMDDhhmm', axis = 1)
    print (features_pd.head(1))

    feature_list = list(features_pd.columns)
    

    features = np.array(features_pd)
    return features, labels, feature_list


def into_range(x, range_min, range_max):
    shiftedx = x - range_min
    delta = range_max - range_min
    return (((shiftedx % delta) + delta) % delta) + range_min


start = datetime.datetime.now() 


param_file = sys.argv[1]

RF_parameters = pickle.load( open(param_file, "rb") )
RF_parameters['train_perc_keep'] = float(sys.argv[2])
pprint(RF_parameters)

np.random.seed(RF_parameters["random_state"])


file = open(RF_parameters['settingsdir']+"settings___"+RF_parameters['name']+".pkl", "rb")
RF_settings = pickle.load(file)
pprint(RF_settings)


features = import_feature_data(
    startdate=RF_parameters["train_startdate"], 
    enddate=RF_parameters["train_enddate"], 
    exclude_MM=RF_parameters["exclude_MM"], 
    exclude_hh=RF_parameters["exclude_hh"], 
    perc_keep=RF_parameters["train_perc_keep"], 
    traindir=RF_parameters["traindir"], 
    random=RF_parameters["base_features"]["random"], 

    )


all_features, labels, all_features_list = sort_feature_data(
    features=features, 
    bin_edges=np.array(RF_parameters["bin_edges"]).astype(np.float64)
)
print(all_features)
print(all_features.shape)
print(all_features_list)


feature_list = RF_settings['chosen_features']
print(feature_list)
if len(feature_list) > len(all_features_list):
    raise ValueError("Model has more features in it's list than are avaialble from the main features array.")
f_index = []
for f in feature_list:
    f_index.append(all_features_list.index(f)) 
features = all_features[:,f_index] 
print(features)


# Instantiate model
rf = RandomForestClassifier(
    n_jobs       = RF_parameters['train_n_jobs'],
    criterion    = RF_parameters['train_criterion'],
    random_state = RF_parameters["random_state"], 
    
    n_estimators      = RF_settings['hyperparameters']['n_estimators'], 
    min_samples_split = RF_settings['hyperparameters']['min_samples_split'], 
    max_features      = RF_settings['hyperparameters']['max_features'], 
    max_depth         = RF_settings['hyperparameters']['max_depth'], 
    max_samples       = RF_settings['hyperparameters']['max_samples'],
    bootstrap         = RF_settings['hyperparameters']['bootstrap'],
    class_weight      = RF_settings['hyperparameters']['class_weight'] 
)

start_rf = datetime.datetime.now()
rf.fit(features, labels);
end_rf = datetime.datetime.now()

time_taken = end_rf-start_rf
print("RF Training Time Taken = ", time_taken)


modelname = 'RF_model___' + RF_parameters['name'] + '___' + RF_parameters["train_startdate"].strftime('%Y%m') + '-' + RF_parameters["train_enddate"].strftime('%Y%m') + '_' + RF_parameters["diurnal"] + '_' + str(RF_parameters["train_perc_keep"]*100) + 'perc___bins_'
# Bin spacings
for binn in RF_parameters["bin_edges"]:
    if binn == RF_parameters["bin_edges"][-1]:
        modelname = modelname + str(binn) 
    else:
        modelname = modelname + str(binn) + "-"

modelname = modelname + '__est' + str(RF_settings['hyperparameters']["n_estimators"]) + '_' + RF_settings['hyperparameters']["class_weight"] + '_' + RF_parameters['train_criterion'] + '_maxfeat' + str(RF_settings['hyperparameters']["max_features"]) + '_minsplit' + str(RF_settings['hyperparameters']["min_samples_split"]) + '_maxdepth' + str(RF_settings['hyperparameters']["max_depth"])

if RF_settings['hyperparameters']['bootstrap']:
    modelname = modelname + '_bootstrap_' + str(RF_settings['hyperparameters']["max_samples"])


name = RF_parameters['modeldir'] + modelname + '_time-' + datetime.datetime.now().strftime("%Y%m%d-%H%M")
print (name)
rf.feature_names_in_ = feature_list
joblib.dump(rf, name + '.pkl')

# Report execution time
end = datetime.datetime.now()
time_taken = end-start
print("Full Script Time Taken = ", time_taken)
print("SCRIPT SUCCESSFUL")