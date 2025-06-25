import copy

import pandas as pd
import json
import os
from IPython.display import display, Markdown
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from collections import Counter,defaultdict
import statistics
from yardsticks_GMM_train import FEATURES, RESULTS_DICO, save_results_to_csv, mean_absolute_difference
from compute_thresholds import thresholds_init , distrib_levels


df = pd.read_csv('./Qualtrics_Annotations_B.csv', delimiter="\t", index_col="text_indice")
df = df[~df.index.duplicated(keep='first')]
classes_to_level = {'Très Facile':'N1', 'Facile': 'N2', 'Accessible':'N3','+Complexe':'N4'}
df['classe'] = df['gold_score_20_label'].map(classes_to_level)
level_to_int = {'N1': 1, 'N2': 2, 'N3': 3, 'N4': 4}
classes = ['N1', 'N2', 'N3', 'N4']


AGGREGATION_RULES = {
    # --- Structure ---
    'word_count': 'sum',  # total number of words in document
    'sentence_count': 'sum',  # total number of sentences
    'sentence_length': 'mean',  # average sentence length
    'word_length': 'mean',  # average word length
    'word_syllables': 'mean',  # average syllables per word

    # --- Lexicon ---
    'complexity': 'mean',  # average lexical complexity
    'lexical_frequency': 'mean',  # average word frequency
    'age_of_acquisition': 'mean',  # average AoA
    'lexical_diversity': 'mean',  # mean diversity per sentence or document

    # --- Syntax ---
    'parse_depth': 'mean',  # average syntactic parse depth
    'max_size_subordination': 'max',  # max depth of subordination
    'ratio_subordination_per_token': 'mean',
    'ratio_subordination_per_verb': 'mean',
    'total_token_ratio_subordination': 'mean',
    'max_size_np_pp_modifiers': 'max',
    'max_size_passive': 'max',
    'ratio_passive_per_token': 'mean',
    'ratio_passive_per_verb': 'mean',
    'total_token_ratio_passive': 'mean',
    'max_size_coordination': 'max',
    'ratio_coordination_per_token': 'mean',
    'total_token_ratio_coordination': 'mean',
    'max_size_aux_verbs': 'max',
    'ratio_aux_verbs_per_token': 'mean',
    'ratio_aux_verbs_per_verb': 'mean',
    'total_token_ratio_aux_verbs': 'mean',

    # --- Semantics ---
    'concrete_ratio': 'mean'  # average concreteness per token or sentence
}


def precision_class_1(y_true, y_pred):
    tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def evaluate(results, all_y_true, all_y_pred):
    mad = mean_absolute_difference(all_y_true, all_y_pred)
    print(f"Cross-validated TEST prediction mad: {mad:.3f}")
    results[yardstick]['mad'] = mad
    accuracy = accuracy_score(all_y_true, all_y_pred)
    print(f"Cross-validated TEST prediction accuracy: {accuracy:.3f}")
    results[yardstick]['acc'] = accuracy
    f1_macro = f1_score(all_y_true, all_y_pred, average='macro', zero_division=0)
    print(f"Cross-validated TEST macro F1 score: {f1_macro:.3f}")
    results[yardstick]['macro-F1'] = f1_macro
    return results


def get_data(distributions_json_path='./outputs/distributions.json', features = FEATURES, yardstick='structure'):
    with open(distributions_json_path) as json_data:
        distributions = json.load(json_data)

    distributions_yardstick = defaultdict(dict)
    for feat in features[yardstick]:
        found = False
        for n in ['N1', 'N2', 'N3', 'N4']:
            for level in distributions[n]:
                if feat in distributions[n][level]:
                    distributions_yardstick[n][feat] = distributions[n][level][feat]
                    found = True
                    break
            if not found: print(f"{feat} not found")

    return distributions_yardstick


def get_prob_threshold_f1(y_labels, y_pred_prob):
    precision, recall, thresholds = precision_recall_curve(y_labels, y_pred_prob)
    #print(precision, recall)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores[np.isnan(f1_scores)] = 0
    #print(f1_scores[f1_scores<0])
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    return best_threshold


def train_regression(x_values, y_labels):
    x_values = np.array(x_values)
    y_labels = np.array(y_labels)

    # Scale input
    scaler = RobustScaler()
    x_values = scaler.fit_transform(x_values)

    # Logistic Regression with GridSearch
    base_model = LogisticRegression(class_weight='balanced', max_iter=1000)
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs'],
        'penalty': ['l2']
    }

    grid = GridSearchCV(base_model, param_grid, scoring='precision', cv=5)
    grid.fit(x_values, y_labels)

    model = grid.best_estimator_

    # Evaluate precision on training set
    y_pred_class = model.predict(x_values)
    precision = precision_score(y_labels, y_pred_class, pos_label=1)

    # Compute decision threshold (for P=0.7)
    threshold = 0.7
    w = model.coef_[0][0]
    b = model.intercept_[0]
    x_thresh = -(np.log(1 / threshold - 1) + b) / w

    return model, precision, x_thresh


def get_features(output, features):
    output_features = defaultdict(dict)
    for feat in features:
        # need to find the level of the features
        for level in thresholds_init:
            if feat in thresholds_init[level]:
                #print(f"'{feat}' is inside '{level}'")
                break
        values = []
        if distrib_levels[level] == 'document':
            values.append(output['features'][feat])

        elif distrib_levels[level] == 'sentence':
            for k, v in output['sentences'].items():
                values.append(v['features'][feat])

        elif distrib_levels[level] == 'token':
            for k, v in output['sentences'].items():
                for k1, v1 in v['words'].items():
                    if v1[feat] != 'na': values.append(v1[feat])

        # output_features[feat] = np.mean(values)
        output_features[feat] = values

    return output_features


def get_predictions(thresholds):
    #get the avg of the classes
    features = list(thresholds['N1'].keys())
    #print('features ------------------->', features)
    y_trues, y_preds = [], []
    for index, row in df.iterrows():
        predictions = defaultdict(dict)

        file_path = os.path.join(outputs_json_path, f"{index}.json")
        with open(file_path, 'r') as file:
            output = json.load(file)

        output_features = get_features(output, features)
        #if 119 in output_features['word_length']: print("index ", index)

        # print('output_features----------->', output_features)
        for feat in features:
            predictions[feat] = []
            for v in output_features[feat]:
                if v in ['-1', 'na', -1]: continue
                if v > thresholds['N1'][feat]:
                    predictions[feat].append('N2')
                elif v > thresholds['N2'][feat]:
                    predictions[feat].append('N3')
                elif v > thresholds['N3'][feat]:
                    predictions[feat].append('N4')
                else:
                    predictions[feat].append('N1')

                #print(feat, thresholds['N1'][feat], thresholds['N2'][feat], thresholds['N3'][feat], v, predictions[feat][-1])

            #print(predictions[feat])
            #predictions[feat] = statistics.mode(predictions[feat])
            predictions[feat] = classes[int(round(np.mean([level_to_int[p] for p in predictions[feat]])))-1]
            #print(predictions[feat])

        # prediction = statistics.mode(list(predictions.values()))

        prediction = classes[int(round(np.mean([level_to_int[p] for p in list(predictions.values())]))) - 1]
        y_trues.append(df.loc[index]['classe'])
        #print(predictions, prediction, df.loc[index]['classe'])
        y_preds.append(prediction)
    return y_trues, y_preds


def get_thresholds(distributions, yardstick):
    thresholds = defaultdict(dict)

    for classe, dico in distributions.items():
        if classe == 'N4': continue
        for heuristic, values in dico.items():
            x_values = []
            y_labels = []
            # if heuristic != 'words_after_verb': continue
            for c in classes:
                values = distributions[c][heuristic]
                x_values += [[v] for v in values]
                if classes.index(c) > classes.index(classe):
                    y_labels += [1 for _ in range(len(values))] # 1 for complex
                else:
                    y_labels += [0 for _ in range(len(values))] # 0 for simple

            #print(classe, heuristic)
            _, precision, x_thresh = train_regression(x_values, y_labels)
            #print(round(thresh, 3))

            thresholds[classe][heuristic] = round(x_thresh, 3)

    return thresholds


def select_features(distributions, yardstick, precision_threshold=0.7):
    selected_features = []

    for heuristic, values in distributions['N1'].items():
        x_values = []
        y_labels = []
        for c in classes:
            values = distributions[c][heuristic]
            x_values += [[v] for v in values]
            if classes.index(c) > classes.index('N1'):
                y_labels += [1 for _ in range(len(values))]  # 1 for complex
            else:
                y_labels += [0 for _ in range(len(values))]  # 0 for simple

        #print('N1', heuristic)

        precision, x_thresh = train_regression(x_values, y_labels)
        # print(round(thresh, 3))

        if precision >= precision_threshold:
            selected_features.append(heuristic)
        else:
            print('not selected feature:', heuristic)
    return selected_features


if __name__ == '__main__':
    random_state = 2
    np.random.seed(random_state)
    outputs_json_path = './outputs'  # path to the folder containing the jsons outputs of the annotator
    yardsticks = ['structure', 'lexicon', 'syntax', 'semantics']
    results = copy.deepcopy(RESULTS_DICO)
    for yardstick in yardsticks:
        #if yardstick != 'structure': continue
        print(f"--- Yardstick: {yardstick} | Seed: {random_state} ---")
        #distributions = get_data(yardstick=yardstick, features={'structure': ['word_length']})
        distributions = get_data(yardstick=yardstick)#, features={'structure': ['word_length']})
        print('DONE loading data')
        #selected_features = select_features(distributions, yardstick, precision_threshold=0.7)
        #print('DONE selecting features', selected_features)
        thresholds = get_thresholds(distributions, yardstick)

        y_trues, y_preds = get_predictions(thresholds)
        print('ACCURACY ', np.mean(np.array(y_trues) == np.array(y_preds)))
        #print(thresholds)
        results = evaluate(results, y_trues, y_preds)
        print(results)


