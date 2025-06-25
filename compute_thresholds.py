import json
import pandas as pd
import numpy as np
import os
import copy

from tqdm import tqdm

sentence_token_level = {
    "max_size_aux_verbs": None,
    "max_size_passive": None,
    "max_size_named_entities": None,
    "max_size_np_pp_modifiers": None,
    "max_size_subordination": None,
    "max_size_coordination": None,
    }
document_token_level = {
    "total_token_ratio_aux_verbs": None,
    "total_token_ratio_passive": None,
    "total_token_ratio_named_entities": None,
    "total_token_ratio_subordination": None,
    "ratio_clitic_per_token": None,
    "ratio_post_clitic_pronouns_per_token": None,
    "ratio_pre_clitic_pronouns_per_token": None,
    "ratio_named_entities_per_token": None,
    "ratio_aux_verbs_per_token": None,
    "ratio_subordination_per_token": None,
    "ratio_passive_per_token": None,
    "sophisticated_ratio": None,
    "concrete_ratio": None,
    "hapax_legomena_lemma_ratio": None,
    # "p0-p75_freq_ratio": None, missing from output
    # "p0-p75_freq_lemma_ratio": None, missing from output
    "ratio_aux_verbs_per_verb": None,
    "ratio_subordination_per_verb": None,
    "ratio_passive_per_verb": None,
    "ratio_coordination_per_token": None,
    "total_token_ratio_coordination": None,
    "ratio_coordination_per_token": None,
}

sentence_level= {
      "parse_depth": None,
      "words_after_verb": None,
      "words_before_verb": None,
      "sentence_length": None
    }

document_document_level = {
    "word_count": None,
    "sentence_count": None,
    "lexical_diversity": None}

token_level_high = {
      "word_length": None,
      "word_syllables": None,
      "ortho_neighbors": None,
      "ortho_neighbors_+freq_cum": None,
      "age_of_acquisition": None,
      "complexity": None,
    }

token_level_low = {
      "familiarity": None,
      "lexical_frequency": None
    }

thresholds_init = {"sentence-token-level": copy.deepcopy(sentence_token_level), "document-token-level": copy.deepcopy(document_token_level), "sentence-level": copy.deepcopy(sentence_level), "document-document-level":copy.deepcopy(document_document_level), "token-level-high": copy.deepcopy(token_level_high), "token-level-low": copy.deepcopy(token_level_low)}

classes = {'Très Facile':'N1', 'Facile': 'N2', 'Accessible':'N3','+Complexe':'N4'}

thresholds = {'N1':copy.deepcopy(thresholds_init), 'N2':copy.deepcopy(thresholds_init), 'N3': copy.deepcopy(thresholds_init), 'N4':copy.deepcopy(thresholds_init)}
densities = {'N1':copy.deepcopy(thresholds_init), 'N2':copy.deepcopy(thresholds_init), 'N3': copy.deepcopy(thresholds_init), 'N4':copy.deepcopy(thresholds_init)}

distrib_levels = {"sentence-token-level": "document", "document-token-level": "document", "sentence-level": "sentence", "document-document-level": "document", "token-level-high": "token", "token-level-low": "token"}


def check_for_errors(folder_path):
    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):  # Check if the file has a .json extension
            file_path = os.path.join(folder_path, filename)

            # Open and load the JSON file
            try:
                with open(file_path, "r") as file:
                    data = json.load(file)
                    print(data['features'].keys())
                    # Check if "error" key exists in the JSON data
                    if "error" in data:
                        print(f"'error' key found in: {filename}")

            except json.JSONDecodeError:
                print(f"Could not decode JSON in file: {filename}")


def get_bounds(values):
    Q1 = np.percentile(values, 25)  # First quartile (25th percentile)
    Q3 = np.percentile(values, 75)  # Third quartile (75th percentile)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound


def compute_thresholds(thresholds, df, outputs_path, densities = None):

    for classe, niveau in classes.items():
        print(classe)
        indexes = df[df["gold_score_20_label"] == classe].index
        dictionnary = thresholds[niveau]
        for key, dictionnary2 in tqdm(dictionnary.items(), desc="Processing keys"):
            for phenomenon in dictionnary2.keys():
                #print(phenomenon)
                values = []
                for i in indexes:
                    with open('%s/%s.json' %(outputs_path,i), 'r') as file:
                        data = json.load(file)
                    if 'error' in data.keys():
                        print('error in index %s' %i)
                        continue
                    if distrib_levels[key] == 'document':
                        if data['features'][phenomenon] not in ['-1', 'na', -1]: values.append(data['features'][phenomenon])
                    elif distrib_levels[key] == 'sentence':
                        for k, v in data['sentences'].items():
                            if v['features'][phenomenon] not in ['-1', 'na', -1]: values.append(v['features'][phenomenon])
                    elif distrib_levels[key] == 'token':
                        for k, v in data['sentences'].items():
                            for k1, v1 in v['words'].items():
                                if v1[phenomenon] not in ['-1', 'na', -1]: values.append(v1[phenomenon])
                if densities:
                    densities[niveau][key][phenomenon] = values

                lower_bound, upper_bound = get_bounds(values)
                #print(phenomenon, lower_bound, upper_bound)
                if key == 'token-level-low':
                    dictionnary2[phenomenon] = round(lower_bound,4)
                else:
                    dictionnary2[phenomenon] = round(upper_bound,4)

    if densities:
        with open('./outputs/distributions.json', 'w') as f:
            json.dump(densities, f)
        return thresholds, densities
    return thresholds


if __name__ == '__main__':
    df = pd.read_csv('./Qualtrics_Annotations_B.csv', delimiter="\t", index_col="text_indice")
    folder_path = "./outputs"
    thresholds, densities = compute_thresholds(thresholds, df, folder_path, densities)
    with open('./outputs/thresholds.json', 'w') as f:
        json.dump(thresholds, f)