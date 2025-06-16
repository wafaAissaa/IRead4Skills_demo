
import numpy as np
import pickle


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


def load_models(path = './yardsticks_models'):
    best_gmm_models = {}
    scalers = {}
    
    for yardstick in features.keys():
        with open('%s/best_gmm_models_%s.pkl' %(path, yardstick), 'rb') as f:
            best_gmm_models[yardstick] = pickle.load(f)
        with open('%s/scaler_%s.pkl' %(path, yardstick), 'rb') as f:
            scalers[yardstick] = pickle.load(f)
            
    return best_gmm_models, scalers

keys_paths ={}


def get_features(phenomena_output, yardtick = 'lexicon'):
    
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
            xi = np.mean(xi)
            x.append(xi)
        elif path_in_dico.count('0') == 2:
            xi = [ tmp['sentences'][str(s)]['words'][str(w)][feat]
                    for s in range(len(tmp['sentences']))
                    for w in range(len(tmp['sentences'][str(s)]['words']))] 
            xi = [x for x in xi if x != 'na']
            xi = np.mean(xi)
            x.append(xi)

    X_list.append(x)  
    X = np.array(X_list)
    return X


def predict(phenomena_output):
    
    output = {}
    
    best_gmm_models, scalers = load_models()
    
    for yardstick in features.keys():
        #print(yardstick)
        
        scaler = scalers[yardstick]
        best_gmm_model = best_gmm_models[yardstick]
        
        X = get_features(phenomena_output, yardstick)
        #print(X)
        
        X_scaled = scaler.fit_transform(X)

        # TODO add class priors

        for x in X_scaled:
            log_likelihoods = {cls: gmm.score_samples(x.reshape(1, -1))[0] for cls, gmm in best_gmm_model.items()}
            predicted_class = max(log_likelihoods, key=log_likelihoods.get)

        output[yardstick] = predicted_class
    
    return output
