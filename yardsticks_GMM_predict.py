from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.stats import skew, kurtosis
import numpy as np
import pickle
import requests
import json


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
        if len(feature_vector) == 0:
            # return 8 NaNs, since that's the expected size
            return [np.nan] * 8
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


def load_models(path='./yardsticks_models'):
    best_gmm_models = {}
    scalers = {}

    for yardstick in features.keys():
        with open('%s/best_gmm_models_%s.pkl' % (path, yardstick), 'rb') as f:
            best_gmm_models[yardstick] = pickle.load(f)
        with open('%s/scaler_%s.pkl' % (path, yardstick), 'rb') as f:
            scalers[yardstick] = pickle.load(f)

    return best_gmm_models, scalers


keys_paths = {}
best_gmm_models, scalers = (None, None)

def get_features(phenomena_output, yardtick='lexicon', aggregation_type="full"):
    X_list = []
    x = []
    for feat in features[yardtick]:
        path_in_dico = find_full_key_path(phenomena_output, feat)
        tmp = phenomena_output
        print(path_in_dico)
        if '0' not in path_in_dico:
            for key in path_in_dico:
                tmp = tmp[key]
            x.append(tmp if tmp not in [-1, 'na', 'NA'] else np.nan)
        elif path_in_dico.count('0') == 1:
            xi = [tmp['sentences'][str(s)]['features'][feat] for s in range(len(tmp['sentences']))]
            # print(feat, xi)
            xi = [x for x in xi if x not in [-1, 'na', 'NA']]
            xi = aggregate(xi, type=aggregation_type)
            x.extend(xi)
        elif path_in_dico.count('0') == 2:
            xi = [tmp['sentences'][str(s)]['words'][str(w)][feat]
                  for s in range(len(tmp['sentences']))
                  for w in range(len(tmp['sentences'][str(s)]['words']))]
            xi = [x for x in xi if x not in [-1, 'na', 'NA']]
            print(xi)
            xi = aggregate(xi, type=aggregation_type)
            print(xi)
            x.extend(xi)

    X_list.append(x)
    X = np.array(X_list)
    return X


# TODO this needs phenomena bio output of the text to predict
def predict(phenomena_output):
    output = {}

    best_gmm_models, scalers = load_models(path='yardsticks_models/gmm_unaligned_in_paper')

    for yardstick in features.keys():
        print(yardstick)

        scaler = scalers[yardstick]
        #print(scaler.mean_)
        best_gmm_model = best_gmm_models[yardstick]

        #for cls, gmm in best_gmm_model.items():
        #    print(f"Class {cls}: expects {gmm.n_features_in_} features")

        X = get_features(phenomena_output, yardstick, aggregation_type='full')

        X_scaled = scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        for x in X_scaled:
            # print(x)
            log_likelihoods = {cls: gmm.score_samples(x.reshape(1, -1))[0] for cls, gmm in best_gmm_model.items()}
            predicted_class = max(log_likelihoods, key=log_likelihoods.get)

        output[yardstick] = predicted_class

    return output

if __name__ == '__main__':


    message = "DISCOURS DU PRÉSIDENT DE LA RÉPUBLIQUE AUX ARMÉES. […] Cette année, comme l'an prochain, comme depuis 1790, le défilé du 14 juillet nous rappellera que certaines choses méritent qu'on s'engage et qu'on se batte pour elles, que la paix n'est pas un confort qu'on achète par des concessions. C'est un idéal de justice qu'il faut être capable de défendre. La France du 14 juillet est une France souveraine, rayonnant en Europe et dans le monde, capable de maîtriser son destin pour que chaque Français ait la possibilité de décider du sien à son tour. C'est la France fidèle à l'esprit des compagnons et à leurs cendres. Et c'est une France que vous faites vivre. Et en concluant mon propos, je veux ici vous redire ma confiance et ma fierté. Il y a la stratégie, il y a le cap, il y a les moyens qu'on donne, il y a les choix qu'on fait, il y a les partenariats qu'on noue, il y a les innovations que l'on conduit. Mais chaque petit matin et chaque soir, il n'y a que les femmes et les hommes qui se lancent et qui, jusqu'au sacrifice ultime, retrouvent le sens de ce lien sacré entre l'armée et la Nation. Et c'est vous qui le portez. Ce lien, cet engagement est un trésor. Et ce trésor, j'invite chacun de nos compatriotes à en penser l'intensité, la transcendance, la singularité. C'est celle qui doit nous inspirer chaque jour. C'est celle qui force le respect, l'admiration mais c'est celle aussi qui doit nous conduire dans chacune de nos décisions. Car la République comme la Nation sont un bloc. Vous avez ma confiance et ma fierté. Vive la République, vive la France !"  # ou:ville,village

    difficulty_level = "A1"
    #message_json = json.dumps(message, ensure_ascii=False)
    #print(message)
    r = requests.post(url="http://192.168.249.77:8080/process_phenomena", data={"raw_text": message,  # server
                                                                                "difficulty_level": difficulty_level})
    # r = requests.post(url="http://0.0.0.0:8080/process_phenomena",data={"raw_text": message_json,      # local
    #                         "difficulty_level": difficulty_level})
    # "services": services})

    output_dict = json.loads(r.text)

    # yardsticks call
    # phenomena_output = json.dumps(output_dict, ensure_ascii=False)
    phenomena_output = output_dict

    if list(phenomena_output.keys()) == ["error"]:
        result = {k: "N1" for k in ["structure", "lexicon", "syntax", "semantics"]}
        print(result)
    else:
        print(predict(phenomena_output))