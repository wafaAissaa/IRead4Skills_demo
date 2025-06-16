import numpy as np
import pickle
import pandas as pd
import json
import os
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix


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


df = pd.read_csv('./Qualtrics_Annotations_B.csv', delimiter="\t", index_col="text_indice")
classes_to_level = {'Très Facile':'N1', 'Facile': 'N2', 'Accessible':'N3','+Complexe':'N4'}
df['classe'] = df['gold_score_20_label'].map(classes_to_level)


def find_full_key_path(d, target_key, path=None):
    # gets the path of the feature in the outputs.json
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


def get_keys_paths(yardstick):
    keys_paths = {}
    with open('./outputs/0.json', 'r') as file:
        dico = json.load(file)
    for feat in features[yardstick]:
        keys_paths[feat] = find_full_key_path(dico, feat)
    return keys_paths


def get_data(outputs_json_path, yardstick):
    X_list = []
    y_list = []

    keys_paths = get_keys_paths(yardstick)
    for index, row in df.iterrows():
        print(index)
        if index == 2074: continue
        file_path = os.path.join(outputs_json_path, f"{index}.json")
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Extract features
        x = []
        for feat in features[yardstick]:
            print(feat)
            path_in_dico = keys_paths[feat]
            #print(path_in_dico)
            # print(find_full_key_path(thresholds, feat))
            tmp = data
            if '0' not in path_in_dico:
                for key in path_in_dico:
                    tmp = tmp[key]
                x.append(tmp)
            elif path_in_dico.count('0') == 1:
                xi = [tmp['sentences'][str(s)]['features'][feat] for s in range(len(tmp['sentences']))]
                #print(feat, xi)
                xi = [x for x in xi if x not in ['-1', 'na']] # to replace it to -1 for age of aquisition
                xi = np.mean(xi)
                x.append(xi)
            elif path_in_dico.count('0') == 2:
                xi = [tmp['sentences'][str(s)]['words'][str(w)][feat]
                      for s in range(len(tmp['sentences']))
                      for w in range(len(tmp['sentences'][str(s)]['words']))]
                #print(feat, xi)
                xi = [x for x in xi if x not in ['-1', 'na']] # to replace it to -1 for age of aquisition
                # if feat in ['lexical_frequency']: # maybe do this later instead of mean
                #    xi = np.percentile(xi, 20)
                # elif featu in ['complexity', 'age_of_acquisition']:
                #    xi = np.percentile(xi, 80)
                # else:
                xi = np.mean(xi)
                x.append(xi)

        X_list.append(x)
        yi = row['classe']
        y_list.append(yi)

    X = np.array(X_list)
    y = np.array(y_list)
    # X: feature matrix (shape [n_samples, n_features])
    # y: class labels (shape [n_samples])
    return X, y


def train_model(X, y, yardstick):
    best_gmm_models = {}
    best_params = {}
    
    classes = np.unique(y)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dump the fitted scaler
    with open('./yardsticks_models/scaler_%s.pkl' % yardstick, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Train GMMs per class with BIC selection
    for cls in classes:
        X_cls = X_scaled[y == cls]
        lowest_bic = np.inf
        best_gmm = None
        best_setting = None
    
        for n in range(1, 6):  # Try 1 to 5 components
            for cov_type in ['full', 'tied', 'diag', 'spherical']:
                gmm = GaussianMixture(n_components=n, covariance_type=cov_type, n_init=10)  # , init_params='k-means++')
                gmm.fit(X_cls)
                bic = gmm.bic(X_cls)
                if bic < lowest_bic:
                    lowest_bic = bic
                    best_gmm = gmm
                    best_setting = {'n_components': n, 'covariance_type': cov_type}
    
        best_gmm_models[cls] = best_gmm
        best_params[cls] = best_setting
        print(f"Best GMM for class {cls}: {best_setting} with BIC={lowest_bic:.2f}")
    
    with open('./yardsticks_models/best_gmm_models_%s.pkl' % yardstick, 'wb') as f:
        pickle.dump(best_gmm_models, f)
    
    with open('./yardsticks_models/best_gmm_models_%s.pkl' % yardstick, 'rb') as f:
        best_gmm_models = pickle.load(f)
    
    # Predict class of each sample based on maximum log-likelihood
    y_pred = []
    
    # Compute class priors from training labels
    class_counts = Counter(y)
    total_samples = len(y)
    class_priors = {cls: np.log(count / total_samples) for cls, count in class_counts.items()}
    
    for x in X_scaled:
        log_likelihoods = {cls: gmm.score_samples(x.reshape(1, -1))[0] + class_priors[cls] for cls, gmm in
                           best_gmm_models.items()}
        predicted_class = max(log_likelihoods, key=log_likelihoods.get)
        y_pred.append(predicted_class)
    
    # Compute accuracy
    accuracy = accuracy_score(y, y_pred)
    print(f"\nPrediction accuracy on training set: {accuracy:.3f}")
    
    f1_macro = f1_score(y, y_pred, average='macro')
    print(f"\nPrediction f1_macro on training set: {f1_macro:.3f}")

    report = classification_report(y, y_pred)
    print("\nClassification Report:\n", report)

    cm = confusion_matrix(y, y_pred, labels=classes)
    cm_df = pd.DataFrame(cm, index=[f"True: {c}" for c in classes],
                         columns=[f"Pred: {c}" for c in classes])
    print("Confusion Matrix:\n", cm_df)


if __name__ == "__main__":
    # Work in progress: split to train, val, test sets
    outputs_json_path = './outputs'  # path to the folder containing the jsons outputs of the annotator
    # 'structure', 'lexicon','syntax', 'semantics'
    yardstick = 'syntax'
    X, y = get_data(outputs_json_path, yardstick)
    train_model(X, y, yardstick)

