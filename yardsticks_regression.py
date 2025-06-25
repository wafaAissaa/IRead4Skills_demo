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
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from collections import Counter
from yardsticks_GMM_train import FEATURES, RESULTS_DICO
from collections import defaultdict


df = pd.read_csv('./Qualtrics_Annotations_B.csv', delimiter="\t", index_col="text_indice")
classes_to_level = {'Très Facile':'N1', 'Facile': 'N2', 'Accessible':'N3','+Complexe':'N4'}
df['classe'] = df['gold_score_20_label'].map(classes_to_level)
level_to_int = {'N1': 1, 'N2': 2, 'N3': 3, 'N4': 4}
classes = ['N1', 'N2', 'N3', 'N4']


def precision_class_1(y_true, y_pred):
    tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def get_data(distributions_json_path='./outputs/distributions.json', features = FEATURES, yardstick='structure'):
    with open(distributions_json_path) as json_data:
        distributions = json.load(json_data)

    distributions_yardstick = defaultdict(dict)
    for feat in features[yardstick]:
        for n in ['N1', 'N2', 'N3', 'N4']:
            for level in distributions[n]:
                if feat in distributions[n][level]:
                    distributions_yardstick[n][feat] = distributions[n][level][feat]
                    break

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
    # Example data (inputs and labels)
    x_values = np.array(x_values)  # input values (single feature, reshaped)
    y_labels = np.array(y_labels)  # labels (0 or 1)

    # Create a Logistic Regression model
    model = LogisticRegression(class_weight='balanced')

    # Train the model on the data
    model.fit(x_values, y_labels)

    # Make predictions (probability and class prediction)
    y_pred_prob = model.predict_proba(x_values)[:, 1]  # Probability for class 1
    y_pred_class = model.predict(x_values)  # Predicted class labels
    #print(y_labels)
    #print(y_pred_class)
    precision = precision_score(y_labels, y_pred_class, pos_label=1)
    print(f"Precision (class 1): {precision:.3f}")
    #precision_tmp = precision_class_1(y_labels, y_pred_class)
    #print(f"Precision tmp (class 1): {precision_tmp:.3f}")
    # Decision boundary (where probability is 0.5)
    # plt.axvline(x=model.intercept_ / -model.coef_, color='green', linestyle='--', label='Decision Boundary')

    threshold = 0.7
    w = model.coef_[0][0]
    b = model.intercept_[0]
    # Compute x for desired threshold
    #print(f"w: {w:.3f}")
    x_thresh = -(np.log(1 / threshold - 1) + b) / w
    return model, precision, x_thresh


def get_level(thresholds, x_values, y_labels):
    #get the avg of the classes
    pass




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

        print('N1', heuristic)

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
    for yardstick in yardsticks:
        print(f"--- Yardstick: {yardstick} | Seed: {random_state} ---")
        distributions = get_data(yardstick=yardstick)
        print('DONE loading data')
        #selected_features = select_features(distributions, yardstick, precision_threshold=0.7)
        #print('DONE selecting features', selected_features)
        thresholds = get_thresholds(distributions, yardstick)
        #print(thresholds)

