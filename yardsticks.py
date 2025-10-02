import pickle
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import os
import torch
import requests
from urllib.parse import quote_plus
import json
import pickle
import json
from flask import Flask, request
import yaml
import sys
from scipy.stats import skew, kurtosis
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


app = Flask(__name__)
service_name = "yardsticks"

@app.route('/about')
def about():
    return 'This is a service for annotating French text with yardsticks scores (i.e. Descriptive, Lexicon, Syntactic, Semantic,).'

import warnings

# Suppress only FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


features = {'structure': ['word_count', 'sentence_count', 'sentence_length', 'word_length', 'word_syllables'],
           'lexicon': ['complexity', 'lexical_frequency', 'age_of_acquisition', 'lexical_diversity'],
           'syntax': ['parse_depth', 
            'max_size_subordination', 
            'ratio_subordination_per_token',
             'ratio_subordination_per_verb',
             'total_token_ratio_subordination',
            'max_size_np_pp_modifiers', 
            'max_size_passive', 
            'max_size_passive',
             'ratio_passive_per_token',
             'ratio_passive_per_verb',
             'total_token_ratio_passive',
            'max_size_coordination',
             'ratio_coordination_per_token',
             'total_token_ratio_coordination',
            'max_size_aux_verbs',
             'ratio_aux_verbs_per_token',
             'ratio_aux_verbs_per_verb',
             'total_token_ratio_aux_verbs'],
           'semantics': ['concrete_ratio']}


def aggregate(feature_vector, type="mean"):

    if type == "mean":
        return [np.mean(feature_vector)]
    elif type == "mean+std":
        return [np.mean(feature_vector), np.std(feature_vector)]
    elif type == "mean+std+per+skew":
        return [
            np.mean(feature_vector),
            np.std(feature_vector),
            np.percentile(feature_vector, 25),
            np.percentile(feature_vector, 75),
            np.percentile(feature_vector, 90),
            skew(feature_vector) if not np.isnan(skew(feature_vector)) else 0.0,
        ]
    elif type == "mean+std+max+per+skew":
        return [
            np.mean(feature_vector),
            np.std(feature_vector),
            np.max(feature_vector),
            np.percentile(feature_vector, 25),
            np.percentile(feature_vector, 75),
            np.percentile(feature_vector, 90),
            skew(feature_vector) if not np.isnan(skew(feature_vector)) else 0.0,
        ]
    elif type == "full":
        return [
            np.mean(feature_vector),
            np.std(feature_vector),
            np.max(feature_vector),
            np.percentile(feature_vector, 25),
            np.percentile(feature_vector, 75),
            np.percentile(feature_vector, 90),
            skew(feature_vector) if not np.isnan(skew(feature_vector)) else 0.0,
            kurtosis(feature_vector) if not np.isnan(kurtosis(feature_vector)) else 0.0,
        ]

def find_full_key_path(d, target_key, path=None):
    if path is None:
        path = []
    if isinstance(d, dict):
        for k, v in d.items():
            new_path = path + [k]
            if k == target_key:
                return new_path
            result = find_full_key_path(v, target_key, new_path)
            if result is not None:
                return result
    return None


def load_models(path = './resources/yardsticks_models'):
    best_gmm_models = {}
    scalers = {}
    
    for yardstick in features.keys():
        with open('%s/best_gmm_models_%s.pkl' %(path, yardstick), 'rb') as f:
            best_gmm_models[yardstick] = pickle.load(f)
        with open('%s/scaler_%s.pkl' %(path, yardstick), 'rb') as f:
            scalers[yardstick] = pickle.load(f)
            
    return best_gmm_models, scalers

keys_paths = {}
best_gmm_models, scalers = (None, None)


def get_features(phenomena_output, yardtick = 'lexicon', aggregation_type="full"):
    X_list = []
    x = []
    for feat in features[yardtick]:
        path_in_dico = find_full_key_path(phenomena_output, feat)
        tmp = phenomena_output
        if '0' not in path_in_dico:
            for key in path_in_dico:
                tmp = tmp[key]
            x.append(tmp)
        elif path_in_dico.count('0') == 1:
            xi = [tmp['sentences'][str(s)]['features'][feat] for s in range(len(tmp['sentences']))]  
            #print(feat, xi)
            xi = [x for x in xi if x != 'na']
            xi = aggregate(xi, type=aggregation_type)
            x.extend(xi)
        elif path_in_dico.count('0') == 2:
            xi = [ tmp['sentences'][str(s)]['words'][str(w)][feat]
                    for s in range(len(tmp['sentences']))
                    for w in range(len(tmp['sentences'][str(s)]['words']))] 
            xi = [x for x in xi if x != 'na']
            xi = aggregate(xi, type=aggregation_type)
            x.extend(xi)

    X_list.append(x)  
    X = np.array(X_list)
    return X


@app.route('/processing', methods=['POST'])
def processing():  # parameters is a dict. See /test
    phenomena_output = request.json['phenomena']
    
    output = {}
    
    for yardstick in features.keys():
        scaler = scalers[yardstick]
        best_gmm_model = best_gmm_models[yardstick]
        
        X = get_features(phenomena_output, yardstick, aggregation_type='full')
        X_scaled = scaler.transform(X)
    
        for x in X_scaled:
            log_likelihoods = {cls: gmm.score_samples(x.reshape(1, -1))[0] for cls, gmm in best_gmm_model.items()}
            predicted_class = max(log_likelihoods, key=log_likelihoods.get)

        output[yardstick] = predicted_class

    output_json = json.dumps(output, indent=4)
    return output_json



if __name__ == '__main__':
    yaml_file = sys.argv[1] # "services_fr.yaml"
    services = yaml.safe_load(open(yaml_file, "r"))
    port = services[service_name]["port"]

    yardsticks_models = services[service_name]["internal_parameters"]["yardsticks_models"]
    best_gmm_models, scalers = load_models(yardsticks_models)
    
    app.run(port=port,debug=True, host=services[service_name]["ip"])
    #serve(app, host=services[service_name]["ip"], port=port)
