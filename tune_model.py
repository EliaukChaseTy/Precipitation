import pandas as pd
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import sys
import csv
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


from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from itertools import compress
from pprint import pprint

np.random.seed(42)


# Advanced data import
def import_feature_data(startdate, enddate, exclude_MM, exclude_hh, perc_keep, traindir, random, solar, seviri_diff, topography, wavelets):
    print("adding base SEVIRI channel features...")
    features = None
    files = [] # Initialise empty list for the file paths to be appended to

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
    
    
    key_features = []
    
    if seviri_diff:
        print("adding SEVIRI channel difference features...")
        
        key_features.append(['MSG_6.2-7.3',
                            'MSG_3.9-10.8',
                            'MSG_1.6-0.6',
                            'MSG_0.6-10.8',
                            'MSG_0.8-10.8',
                            'MSG_1.6-10.8',
                            'MSG_6.2-10.8',
                            'MSG_7.3-10.8',
                            'MSG_8.7-10.8',
                            'MSG_9.7-10.8',
                            'MSG_12.0-10.8',
                            'MSG_13.4-10.8'])
        
 
    # Save some memory
    dframe_list = None
    frame = None

    print('The shape of the features array is:', features.shape)
    print('The size of the features array is: ' + str(sys.getsizeof(features)/1000000000)[:-6] + ' GB.')
    
    return features, key_features


def sort_feature_data(features, bin_edges, force_features, desired_feature_list):
    labels = np.array(features['GPM_PR'])

    labels = np.digitize(labels, bins=bin_edges, right=False)


    features_no_labels= features.drop('GPM_PR', axis = 1)
    features_pd= features_no_labels.drop('YYYYMMDDhhmm', axis = 1)
    print (features_pd.head(1))
    
    
    if force_features:
        features_pd = features_pd[desired_feature_list]


    feature_list = list(features_pd.columns)
    

    features = np.array(features_pd)
    return features, labels, feature_list


def into_range(x, range_min, range_max):
    shiftedx = x - range_min
    delta = range_max - range_min
    return (((shiftedx % delta) + delta) % delta) + range_min



def plot_importances(RF_dict, outdir):

    for model in RF_dict:
        clf = RF_dict[model]['model']
        flabels = RF_dict[model]['labels'].copy()
        
        importances = list(clf.feature_importances_)
        
        feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(flabels, importances)]
        
        feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
        
        with open(outdir+'feature_importances_'+model+'.csv',"w+") as my_csv:
            csvWriter = csv.writer(my_csv,delimiter=',')
            csvWriter.writerows(feature_importances)


        plt.style.use('fivethirtyeight')
        plt.figure(figsize=(15,10))

        x_values = list(range(len(importances)))

        if 'random' in flabels:
            rand_index = flabels.index('random')

            plt.plot([-0.5, -1.5+len(importances)],[importances[rand_index], importances[rand_index]], color='red', linestyle='--', linewidth=3., label='Random')

            del importances[rand_index]
            flabels.remove('random')

            plt.bar(x_values[:-1], importances, orientation = 'vertical')
            plt.legend()

        else:
            plt.bar(x_values, importances, orientation = 'vertical')

        plt.xticks(x_values, flabels, rotation='vertical')

        plt.ylabel('Importance'); plt.xlabel('Feature'); plt.title(model+' Feature Importances');
        plt.axis([None, None, -0.01, None])

        plt.savefig(outdir+'feature_importances_'+model+'.png', bbox_inches="tight", dpi=250)
        
    return feature_importances


    start = datetime.datetime.now()


param_file = sys.argv[1] 

RF_parameters = pickle.load( open(param_file, "rb") )


pprint(RF_parameters)

np.random.seed(RF_parameters["random_state"])



features, key_features = import_feature_data(
    startdate=RF_parameters["tune_startdate"], 
    enddate=RF_parameters["tune_enddate"], 
    exclude_MM=RF_parameters["exclude_MM"], 
    exclude_hh=RF_parameters["exclude_hh"], 
    perc_keep=RF_parameters["tune_perc_keep"], 
    traindir=RF_parameters["traindir"],
    random=RF_parameters["base_features"]["random"], 
    
    # key_features:
    seviri_diff=RF_parameters["key_features"]["seviri_diff"]
    )

force_features, desired_feature_list = False, None
# This step is optional, only here for final publication results
if RF_parameters["name"] == 'base_seviri_plus_diff_plus_topog_plus_wavelets':
    force_features=True
    desired_feature_list = [
            'MSG_13.4-10.8',
            'MSG_7.3-10.8',
            'MSG_6.2-7.3',
            'Longitude', 
            'MSG_3.9-10.8',
            'Solar_azimuth_sin',
            'Solar_elevation',
            'Latitude',
            'w_30.0',
            'w_60.0'
        ]



features, labels, feature_list = sort_feature_data(
    features = features, 
    bin_edges = np.array(RF_parameters["bin_edges"]).astype(np.float64),
    force_features = force_features,
    desired_feature_list = desired_feature_list
)
print(features)
print(features.shape)
print(feature_list)


random_grid = RF_parameters["random_grid"]
pprint(random_grid)

total=1
for arg in random_grid:
    total = total * len(random_grid[arg]) 
total = total
print("Total number of RandomizedSearchCV ensembles:", total) 

rf_random = RandomizedSearchCV(
    estimator           = RandomForestClassifier(
        random_state    = RF_parameters["random_state"], # Keeps the results repeatable by locking the randomness
        criterion       = RF_parameters["train_criterion"],
        verbose         = 0
        ), 
    param_distributions = random_grid, 
    n_iter              = int(total * RF_parameters["tune_n_iter"] ), #!!!! change to total * 0.1 (10% or something)
    cv                  = RF_parameters["tune_cv"], # cross validation. Default=5. No. of repeat experiments to do through splitting dataset 5 times
    verbose             = 2, 
    random_state        = RF_parameters["random_state"], 
    n_jobs              = RF_parameters["tune_n_jobs"]
    )

start_train = datetime.datetime.now()
rf_random.fit(features, labels)
end_train = datetime.datetime.now()

print("RF Training Time Taken = ", end_train-start_train)


file = open(RF_parameters["settingsdir"]+"DEBUG_MODEL___"+RF_parameters["name"]+".pkl", "wb")
pickle.dump(rf_random, file)
file.close()

pprint(rf_random.best_params_)


feature_importances = plot_importances(
    RF_dict={
        RF_parameters["name"]: {
            'model': rf_random.best_estimator_,
            'labels': feature_list
        }
      }, 
    outdir=RF_parameters["settingsdir"])


n_features = rf_random.best_params_['max_features']

chosen_features = []
for i in range(0,len(feature_importances)):
    if not feature_importances[i][0] == 'random':
        chosen_features.append(feature_importances[i][0])

print(chosen_features[:n_features])


RF_settings = {'name': RF_parameters["name"],
               'chosen_features': chosen_features,
               'hyperparameters': rf_random.best_params_}
pprint(RF_settings)

# Dump the dictionary to disk
file = open(RF_parameters["settingsdir"]+"settings___"+RF_parameters["name"]+".pkl", "wb")
pickle.dump(RF_settings, file)
file.close()

