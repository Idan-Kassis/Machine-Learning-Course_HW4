from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score

def PCA_IG_SVM(X, y):
    X = pd.DataFrame(X)
    all_features = list(X.columns)
    all_features_dict = {x:0 for x in all_features}
    best_150_features = PCA_IG(X,y)
    X = X[best_150_features]
    svm_dict = SVM (X, y)
    final_list = []
    for f in all_features_dict:
        if f in svm_dict.keys():
            final_list.append(svm_dict[f])
        else:
            final_list.append(all_features_dict[f])
    return np.array(final_list)

def PCA_IG(X, y):
    X = pd.DataFrame(X)
    all_features = list(X.columns)
    n_samples, n_features = X.shape
    k = n_samples/2
    pca = PCA(n_components=n_samples)
    X_new = pca.fit_transform(X,y)
    new_features = list(pca.get_feature_names_out())
    X_new = pd.DataFrame(X_new, columns=new_features)
    new_features_most_important = IG(X_new,y,k)
    components = pca.components_
    all_scores_list = []
    sum_of_IG_scores = 0
    for x,y in new_features_most_important:
        index = new_features.index(x)
        scores_list = components[index]
        all_scores_list.append(scores_list*y)
        sum_of_IG_scores += y
    scores = []
    for f in all_features:
        index = all_features.index(f)
        sum = 0
        for i in all_scores_list:
            sum += i[index]
        scores.append(sum/sum_of_IG_scores)
    features_score_list = list(zip(scores, all_features))
    sorted_list = sorted(features_score_list, key=lambda x: x[0], reverse=True)
    final = [x for y, x in sorted_list[0:150]]
    return final

def IG (X, y, k):
    importance = mutual_info_classif(X, y)
    all_features = list(X.columns)
    features_score_list = list(zip(importance, all_features))
    sorted_list = sorted(features_score_list, key=lambda x: x[0], reverse=True)
    final = [(x,y) for y, x in sorted_list]
    return final

def SVM (X, y):
    feature_names = list(X.columns)
    kfold = KFold(n_splits=10, shuffle=True)
    sol_dict = {}
    for feature in feature_names:
        svc = SVC(kernel='rbf', C=2)
        X_temp = X.drop([feature], axis=1)
        results = cross_val_score(svc, X_temp, y, cv=kfold)
        acc = results.mean()
        sol_dict[feature] = 1 - acc
    return sol_dict
