import numpy as np
import pickle
import pandas as pd
import json
import os
import itertools
import copy
from collections import Counter
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, cohen_kappa_score



FEATURES = {'structure': ['word_count', 'sentence_count', 'sentence_length', 'word_length', 'word_syllables'],
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
level_to_int = {'N1': 1, 'N2': 2, 'N3': 3, 'N4': 4}
yardsticks = ['structure', 'lexicon','syntax', 'semantics']

RESULTS_DICO = {'config': (),'structure': {'mad': 0, 'qwk': 0, 'acc': 0, 'macro-F1': 0}, 'lexicon' : {'mad': 0, 'qwk': 0, 'acc': 0, 'macro-F1': 0},'syntax' : {'mad': 0, 'qwk': 0, 'acc': 0, 'macro-F1': 0}, 'semantics' : {'mad': 0, 'qwk': 0, 'acc': 0, 'macro-F1': 0}}


def load_annotations(yardstick, yardticks_filename="aggregated_yardstick_annotations.csv"):

    yardstick_annotations = pd.read_csv(yardticks_filename, index_col=0)
    yardstick_annotations[yardstick] = yardstick_annotations[yardstick].map(classes_to_level)
    df = pd.read_csv('./Qualtrics_Annotations_B.csv', delimiter="\t", index_col="text_indice")
    # Get the index values to locate
    target_indices = yardstick_annotations.index
    # Find ilocs of these indices in df
    iloc_positions = [df.index.get_loc(idx) for idx in target_indices]
    return iloc_positions, yardstick_annotations[yardstick].tolist()



def save_results_to_csv(results, filename):
    """
    Save one row of results to a CSV file, where each row represents a config.

    Parameters:
        results (dict): Evaluation results by yardstick.
        filename (str): Path to the CSV file.
        prior (str): Prior type used in the model (e.g., 'uniform', 'empirical').
        aggregation_type (str): Type of aggregation used.
        random_state (int): Random seed used for cross-validation.
    """
    # Flatten the result data
    data = [
        round(results['structure']['mad'], 3), round(results['structure']['qwk'], 3), round(results['structure']['acc'], 3), round(results['structure']['macro-F1'], 3),
        round(results['lexicon']['mad'], 3), round(results['lexicon']['qwk'], 3), round(results['lexicon']['acc'], 3), round(results['lexicon']['macro-F1'], 3),
        round(results['syntax']['mad'], 3), round(results['syntax']['qwk'], 3), round(results['syntax']['acc'], 3), round(results['syntax']['macro-F1'], 3),
        round(results['semantics']['mad'], 3), round(results['semantics']['qwk'], 3), round(results['semantics']['acc'], 3), round(results['semantics']['macro-F1'], 3),
    ]

    row = [results['config']] + data

    # Define column names
    columns = ['config'] + [
        'Structure mad', 'Structure qwk', 'Structure acc', 'Structure macro-F1',
        'Lexicon mad', 'Lexicon qwk', 'Lexicon acc', 'Lexicon macro-F1',
        'Syntax mad', 'Syntax qwk', 'Syntax acc', 'Syntax macro-F1',
        'Semantics mad', 'Semantics qwk', 'Semantics acc', 'Semantics macro-F1',
    ]

    # Create DataFrame for a single row
    df = pd.DataFrame([row], columns=columns)

    # Append to CSV or create it if it doesn't exist
    try:
        existing_df = pd.read_csv(filename, sep='\t')
        df = pd.concat([existing_df, df], ignore_index=True)
    except FileNotFoundError:
        pass  # file will be created below

    df.to_csv(filename, index=False, sep='\t')


def evaluate(all_y_true, all_y_pred, results, yardstick):
    # Final evaluation
    mad = mean_absolute_difference(all_y_true, all_y_pred)
    print(f"mad: {mad:.3f}")
    results[yardstick]['mad'] = mad
    qwk = cohen_kappa_score([level_to_int[y] for y in all_y_true], [level_to_int[y] for y in all_y_pred], weights="quadratic")
    print(f"qwk: {qwk:.3f}")
    results[yardstick]['qwk'] = qwk
    accuracy = accuracy_score(all_y_true, all_y_pred)
    print(f"accuracy: {accuracy:.3f}")
    results[yardstick]['acc'] = accuracy
    f1_macro = f1_score(all_y_true, all_y_pred, average='macro', zero_division=0)
    print(f"F1 score: {f1_macro:.3f}")
    results[yardstick]['macro-F1'] = f1_macro

    return results


def results_baseline(results):
    for yardstick in yardsticks:
        positions, y = load_annotations(yardstick)
        classes = df['classe'].iloc[positions].to_list()
        evaluate(y, classes, results, yardstick)


def mean_absolute_difference(prediction, ref):
    pred_levels = [level_to_int[label] for label in prediction]
    ref_levels = [level_to_int[label] for label in ref]
    pred_levels = np.array(pred_levels)
    ref_levels = np.array(ref_levels)
    return np.mean(np.abs(pred_levels - ref_levels))


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
    for feat in FEATURES[yardstick]:
        keys_paths[feat] = find_full_key_path(dico, feat)
    return keys_paths


def get_data(outputs_json_path, yardstick, aggregation_type="mean", train=True):
    X_list = []
    y_list = []

    keys_paths = get_keys_paths(yardstick)
    for index, row in df.iterrows():
        #print(index)
        file_path = os.path.join(outputs_json_path, f"{index}.json")
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Extract features
        x = []
        for feat in FEATURES[yardstick]:
            #print(feat)
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
                xi = [x for x in xi if x not in [-1, 'na']]
                # print(xi)
                xi = aggregate(xi, type=aggregation_type)
                if np.isnan(xi).sum(): print(feat, xi)
                x.extend(xi)
            elif path_in_dico.count('0') == 2:
                xi = [tmp['sentences'][str(s)]['words'][str(w)][feat]
                      for s in range(len(tmp['sentences']))
                      for w in range(len(tmp['sentences'][str(s)]['words']))]
                xi = [x for x in xi if x not in [-1, 'na']]
                xi = aggregate(xi, type=aggregation_type)
                if np.isnan(xi).sum(): print('here', xi)
                x.extend(xi)

        X_list.append(x)
        if train:
            yi = row['classe']
        else:
            yi = row[yardstick]
        y_list.append(yi)

    X = np.array(X_list) # X: feature matrix (shape [n_samples, n_features])
    y = np.array(y_list) # y: class labels (shape [n_samples])
    return X, y




def train_model(X, y, yardstick, prior, results, results_of = 'train'):
    best_gmm_models = {}
    best_params = {}
    
    classes = np.unique(y)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dump the fitted scaler
    with open('./yardsticks_models/gmm/scaler_%s.pkl' % yardstick, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Train GMMs per class with BIC selection
    for cls in classes:
        X_cls = X_scaled[y == cls]
        lowest_bic = np.inf
        best_gmm = None
        best_setting = None
    
        for n in range(1, 6):  # Try 1 to 5 components
            for cov_type in ['full', 'tied', 'diag', 'spherical']:
                gmm = GaussianMixture(n_components=n, covariance_type=cov_type, n_init=1, random_state=random_state)  # init_params='k-means++',
                gmm.fit(X_cls)
                bic = gmm.bic(X_cls)
                if bic < lowest_bic:
                    lowest_bic = bic
                    best_gmm = gmm
                    best_setting = {'n_components': n, 'covariance_type': cov_type}
    
        best_gmm_models[cls] = best_gmm
        best_params[cls] = best_setting
        print(f"Best GMM for class {cls}: {best_setting} with BIC={lowest_bic:.2f}")
    
    with open('./yardsticks_models/gmm/best_gmm_models_%s.pkl' % yardstick, 'wb') as f:
        pickle.dump(best_gmm_models, f)
    
    with open('./yardsticks_models/gmm/best_gmm_models_%s.pkl' % yardstick, 'rb') as f:
        best_gmm_models = pickle.load(f)
    
    # Predict class of each sample based on maximum log-likelihood
    y_pred = []

    # Compute class priors from training labels
    if prior == 'empirical':
        class_counts = Counter(y)
        total_samples = len(y)
        class_priors = {cls: np.log(count / total_samples) for cls, count in class_counts.items()}
    else:
        class_priors = {cls: 0.0 for cls in classes}

    if results_of == "train":
        for x in X_scaled:
            log_likelihoods = {cls: gmm.score_samples(x.reshape(1, -1))[0] + class_priors[cls] for cls, gmm in
                               best_gmm_models.items()}
            predicted_class = max(log_likelihoods, key=log_likelihoods.get)
            y_pred.append(predicted_class)

        evaluate(y, y_pred, results, yardstick)

    elif results_of == "test":
        positions, y = load_annotations(yardstick)
        for x in X_scaled[positions]:
            log_likelihoods = {cls: gmm.score_samples(x.reshape(1, -1))[0] + class_priors[cls] for cls, gmm in
                               best_gmm_models.items()}
            predicted_class = max(log_likelihoods, key=log_likelihoods.get)
            y_pred.append(predicted_class)
        evaluate(y, y_pred, results, yardstick)
    elif results_of == "baseline":
        print('please provide split name!')


    return results


def train_model_crossval(X, y, yardstick, prior, results, random_state=2, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_y_true, all_y_pred = [], []
    all_y_train_true, all_y_train_pred = [], []

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"--- Fold {fold + 1} ---")
        best_gmm_models = {}
        best_params = {}

        X_trainval_scaled, y_trainval = X_scaled[train_idx], y[train_idx]
        X_test_scaled, y_test = X_scaled[test_idx], y[test_idx]

        # Split training data into train and validation
        X_train_scaled, X_val_scaled, y_train, y_val = train_test_split(
            X_trainval_scaled, y_trainval, test_size=0.2, stratify=y_trainval, random_state=random_state)

        # Train GMMs per class using validation BIC
        classes = np.unique(y_train)

        # Compute class priors from training data
        if prior == 'empirical':
            class_counts = Counter(y)
            total_samples = len(y)
            class_priors = {cls: np.log(class_counts[cls] / total_samples) for cls in classes}
            #print("class_priors:\n", class_priors)
        else:
            class_priors = {cls: 0.0 for cls in classes}

        for cls in classes:
            X_cls_train = X_train_scaled[y_train == cls]
            X_cls_val = X_val_scaled[y_val == cls]
            lowest_bic = np.inf
            best_gmm = None
            best_setting = None

            for n in range(1, 6):
                for cov_type in ['full', 'tied', 'diag', 'spherical']:
                    gmm = GaussianMixture(n_components=n, covariance_type=cov_type, n_init=5, random_state=random_state)
                    gmm.fit(X_cls_train)
                    bic = gmm.bic(X_cls_val)
                    if bic < lowest_bic:
                        lowest_bic = bic
                        best_gmm = gmm
                        best_setting = {'n_components': n, 'covariance_type': cov_type}

            best_gmm_models[cls] = best_gmm
            best_params[cls] = best_setting
            #print(f"Best GMM for class {cls}: {best_setting} with BIC={lowest_bic:.2f}")


        # Predict on training set
        y_train_pred_fold = []
        for x in X_train_scaled:
            log_likelihoods = {cls: gmm.score_samples(x.reshape(1, -1))[0] + class_priors[cls]
                               for cls, gmm in best_gmm_models.items()}
            predicted_class = max(log_likelihoods, key=log_likelihoods.get)
            y_train_pred_fold.append(predicted_class)

        all_y_train_true.extend(y_train)
        all_y_train_pred.extend(y_train_pred_fold)

        report = classification_report(y_train, y_train_pred_fold, zero_division=0)
        #print("\nCross-validated TRAIN Classification Report:\n", report)

        # Predict on test set
        y_pred_fold = []
        for x in X_test_scaled:
            log_likelihoods = {cls: gmm.score_samples(x.reshape(1, -1))[0] + class_priors[cls]
                               for cls, gmm in best_gmm_models.items()}
            predicted_class = max(log_likelihoods, key=log_likelihoods.get)
            y_pred_fold.append(predicted_class)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred_fold)

        report = classification_report(y_test, y_pred_fold, zero_division=0)
        #print("\nCross-validated TEST Classification Report:\n", report)


    # Final evaluation
    mad = mean_absolute_difference(all_y_true, all_y_pred)
    print(f"Cross-validated TEST prediction mad: {mad:.3f}")
    results[yardstick]['mad'] = mad
    accuracy = accuracy_score(all_y_true, all_y_pred)
    print(f"Cross-validated TEST prediction accuracy: {accuracy:.3f}")
    results[yardstick]['acc'] = accuracy
    f1_macro = f1_score(all_y_true, all_y_pred, average='macro', zero_division=0)
    print(f"Cross-validated TEST macro F1 score: {f1_macro:.3f}")
    results[yardstick]['macro-F1'] = f1_macro

    report = classification_report(all_y_true, all_y_pred, zero_division=0)
    #print("\nCross-validated TEST Classification Report:\n", report)

    cm = confusion_matrix(all_y_true, all_y_pred, labels=np.unique(y))
    cm_df = pd.DataFrame(cm, index=[f"True: {c}" for c in np.unique(y)],
                         columns=[f"Pred: {c}" for c in np.unique(y)])
    # print("Cross-validated TEST Confusion Matrix:\n", cm_df)

    return results

def train_joint_gmm_crossval(X, y, n_splits=5, use_priors=True):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_y_true, all_y_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1} ---")
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train one GMM per class on all features
        classes = np.unique(y_train)
        gmm_models = {}
        for cls in classes:
            X_cls = X_train_scaled[y_train == cls]
            gmm = GaussianMixture(n_components=1, covariance_type='full', n_init=5, random_state=42)
            gmm.fit(X_cls)
            gmm_models[cls] = gmm
            print(f"Trained joint GMM for class {cls} on {len(X_cls)} samples.")

        # Compute class priors
        if use_priors:
            class_counts = Counter(y_train)
            total_samples = len(y_train)
            class_priors = {cls: np.log(class_counts[cls] / total_samples) for cls in classes}
        else:
            class_priors = {cls: 0.0 for cls in classes}
        print("class_priors:\n", class_priors)

        # Predict on test set
        y_pred_fold = []
        for x in X_test_scaled:
            log_likelihoods = {cls: gmm.score_samples(x.reshape(1, -1))[0] + class_priors[cls]
                               for cls, gmm in gmm_models.items()}
            predicted_class = max(log_likelihoods, key=log_likelihoods.get)
            y_pred_fold.append(predicted_class)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred_fold)

    # Evaluation
    accuracy = accuracy_score(all_y_true, all_y_pred)
    f1_macro = f1_score(all_y_true, all_y_pred, average='macro', zero_division=0)
    report = classification_report(all_y_true, all_y_pred, zero_division=0)
    cm = confusion_matrix(all_y_true, all_y_pred, labels=np.unique(y))
    cm_df = pd.DataFrame(cm, index=[f"True: {c}" for c in np.unique(y)],
                         columns=[f"Pred: {c}" for c in np.unique(y)])

    print(f"\nCross-validated TEST prediction accuracy: {accuracy:.3f}")
    print(f"\nCross-validated TEST macro F1 score: {f1_macro:.3f}")
    print("\nClassification Report:\n", report)
    print("Confusion Matrix:\n", cm_df)


if __name__ == "__main__":
    random_state = 42
    np.random.seed(random_state)

    results_of = "test"
    get_best_model = True

    if results_of == "baseline":
        results = copy.deepcopy(RESULTS_DICO)
        results_baseline(results)
        filename = "./results/results_gmm_%s_qwk.csv" % results_of
        results['config'] = f"(baseline)"
        save_results_to_csv(results, filename=filename)
    else:

        outputs_json_path = './outputs'  # path to the folder containing the jsons outputs of the annotator

        aggregation_types = [ "mean", "mean+std", "mean+std+per+skew", "mean+std+max+per+skew", "full"]
        priors = ["uniform", "empirical"]
        for prior, agg_type in itertools.product(priors, aggregation_types):
            if get_best_model and (prior != "empirical" or agg_type != "full"):
                continue
            results = copy.deepcopy(RESULTS_DICO)
            for yardstick in yardsticks:
                X, y = get_data(outputs_json_path, yardstick, agg_type)
                print(f"--- CrossVal | Yardstick: {yardstick} | Prior: {prior} | Aggregation: {agg_type} | Seed: {random_state} ---")
                #results = train_model_crossval(X, y, yardstick, prior, results, random_state=random_state ,n_splits=5)
                results = train_model(X, y, yardstick, prior, results, results_of=results_of)
                #print(results)
            #filename = f"./results/cv_rs{random_state}_prior-{prior}_agg-{agg_type}.csv"
            filename = "./results/results_gmm_%s_best.csv" %results_of

            extra= " " # "kmeans++"
            results['config'] = f"(random_state: {random_state}, prior: {prior}, aggregation_type: {agg_type}, extra: {extra})"
            save_results_to_csv(results, filename=filename)
