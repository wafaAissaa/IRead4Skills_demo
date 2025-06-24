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


df = pd.read_csv('./Qualtrics_Annotations_B.csv', delimiter="\t", index_col="text_indice")
classes_to_level = {'Très Facile':'N1', 'Facile': 'N2', 'Accessible':'N3','+Complexe':'N4'}
df['classe'] = df['gold_score_20_label'].map(classes_to_level)
level_to_int = {'N1': 1, 'N2': 2, 'N3': 3, 'N4': 4}


RESULTS_DICO = {'structure': {'mad': 0, 'acc': 0, 'macro-F1': 0}, 'lexicon' : {'mad': 0, 'acc': 0, 'macro-F1': 0},'syntax' : {'mad': 0, 'acc': 0, 'macro-F1': 0}, 'semantics' : {'mad': 0, 'acc': 0, 'macro-F1': 0}}


def get_prob_threshold_f1(y_labels, y_pred_prob):
    precision, recall, thresholds = precision_recall_curve(y_labels, y_pred_prob)
    #print(precision, recall)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores[np.isnan(f1_scores)] = 0
    #print(f1_scores[f1_scores<0])
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    return best_threshold


def get_input_threshold(x_values, y_labels):
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

    # Decision boundary (where probability is 0.5)
    # plt.axvline(x=model.intercept_ / -model.coef_, color='green', linestyle='--', label='Decision Boundary')

    # Desired threshold
    threshold = 0.5
    print(threshold)
    # Get weight and bias from model
    w = model.coef_[0][0]
    b = model.intercept_[0]
    # Compute x for desired threshold
    x_thresh = -(np.log(1 / threshold - 1) + b) / w
    # print(model.intercept_ , -model.coef_)
    # x_thresh = (model.intercept_ / -model.coef_).item()
    return x_thresh










classes = ['N1', 'N2', 'N3', 'N4']
values = []
for classe, dico in data_json.items():
    for level, dico2 in dico.items():
        for heuristic, values in dico2.items():
            x_values = []
            y_labels = []
            # if heuristic != 'words_after_verb': continue
            for c in classes:
                values = data_json[c][level][heuristic]
                x_values += [[v] for v in values]
                if classes.index(c) > classes.index(classe):
                    y_labels += [1 for _ in range(len(values))]
                else:
                    y_labels += [0 for _ in range(len(values))]

            print(classe, level, heuristic)
            thresh = get_input_threshold(x_values, y_labels)
            print(round(thresh, 3))

            thresh_json[classe][level][heuristic] = round(thresh, 3)