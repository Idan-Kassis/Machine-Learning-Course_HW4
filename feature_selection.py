# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 16:10:48 2022

@author: Tom and Idan
"""
# Setup
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_selection import mutual_info_classif
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_selection import SelectFdr, f_classif
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
#%% Article A - IG_SVM

def IG_SVM(X, y):
    X = pd.DataFrame(X)
    all_features = list(X.columns)
    all_features_dict = {x:0 for x in all_features}
    best_150_features = IG(X,y)
    X = X[best_150_features]
    svm_dict = SVM (X, y)
    final_list = []
    for f in all_features_dict:
        if f in svm_dict.keys():
            final_list.append(svm_dict[f])
        else:
            final_list.append(all_features_dict[f])
    return np.array(final_list)

def IG (X, y, k=150):
    importance = mutual_info_classif(X, y)
    all_features = list(X.columns)
    features_score_list = list(zip(importance, all_features))
    sorted_list = sorted(features_score_list, key=lambda x: x[0], reverse=True)
    final = [x for y,x in sorted_list][0:k]
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

#%% Article B - PCA_IG

def PCA_IG(X, y):
    X = pd.DataFrame(X)
    all_features = list(X.columns)
    n_samples, n_features = X.shape
    k = n_samples/2
    pca = PCA(n_components=n_samples)
    X_new = pca.fit_transform(X,y)
    new_features = list(pca.get_feature_names_out())
    X_new = pd.DataFrame(X_new, columns=new_features)
    new_features_most_important = IG_B(X_new,y,k)
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
    return np.array(scores)


def IG_B (X, y, k):
    importance = mutual_info_classif(X, y)
    all_features = list(X.columns)
    features_score_list = list(zip(importance, all_features))
    sorted_list = sorted(features_score_list, key=lambda x: x[0], reverse=True)
    final = [(x,y) for y, x in sorted_list]
    return final

#%% mRMR
def mRMR(X,y):
    # inputs:
    #    X: pandas.DataFrame, features
    #    y: pandas.Series, target variable
    #    K: number of features to select
    X = pd.DataFrame(X)
    all_features = X.columns
    # compute F-statistics and correlations
    F = pd.Series(f_regression(X, y)[0], index=X.columns)
    corr = X.corr().abs().clip(.00001)  # minimum value of correlation set to .00001 (to avoid division by zero)

    # initialize list of selected features and list of excluded features
    selected = []
    not_selected = list(X.columns)
    sol_dict = {}
    # repeat K times:
    # compute FCQ score for all the features that are currently excluded,
    # then find the best one, add it to selected, and remove it from not_selected
    for i in range(100):
        # compute FCQ score for all the (currently) excluded features (this is Formula 2)
        score = F.loc[not_selected] / corr.loc[not_selected, selected].mean(axis=1).fillna(.00001)

        # find best feature, add it to selected and remove it from not_selected
        best = score.index[score.argmax()]
        selected.append(best)
        not_selected.remove(best)
        sol_dict[best] = score.max()
    sol_list = []
    for feature in all_features:
        if feature in sol_dict.keys():
            sol_list.append(sol_dict[feature])
        else:
            sol_list.append(0)
    return np.array(sol_list)

#%% SelectFdr with f_classif
def seletFdr_fclassif(X,y):
    slc = SelectFdr(f_classif,alpha=0.1)
    slc.fit_transform(X,y)
    return np.array(slc.scores_)

#%% RFE with SVM
def rfe_svm(X,y):
    n_samples, n_features = X.shape
    svc = SVC(kernel='linear')
    selector = RFE(svc, n_features_to_select=100, step=1)
    selector = selector.fit(X,y)
    return np.array(selector.ranking_)

#%% ReliefF
def reliefF(X, y, mode="rank", **kwargs):
    if "k" not in list(kwargs.keys()):
        k = 5
    else:
        k = kwargs["k"]
    n_samples, n_features = X.shape

    # calculate pairwise distances between instances
    distance = pairwise_distances(X, metric="manhattan")

    score = np.zeros(n_features)

    # the number of sampled instances is equal to the number of total instances
    for idx in range(n_samples):
        near_hit = []
        near_miss = dict()

        self_fea = X[idx, :]
        c = np.unique(y).tolist()

        stop_dict = dict()
        for label in c:
            stop_dict[label] = 0
        del c[c.index(y[idx])]

        p_dict = dict()
        p_label_idx = float(len(y[y == y[idx]])) / float(n_samples)

        for label in c:
            p_label_c = float(len(y[y == label])) / float(n_samples)
            p_dict[label] = p_label_c / (1 - p_label_idx)
            near_miss[label] = []

        distance_sort = []
        distance[idx, idx] = np.max(distance[idx, :])

        for i in range(n_samples):
            distance_sort.append([distance[idx, i], int(i), y[i]])
        distance_sort.sort(key=lambda x: x[0])

        for i in range(n_samples):
            # find k nearest hit points
            if distance_sort[i][2] == y[idx]:
                if len(near_hit) < k:
                    near_hit.append(distance_sort[i][1])
                elif len(near_hit) == k:
                    stop_dict[y[idx]] = 1
            else:
                # find k nearest miss points for each label
                if len(near_miss[distance_sort[i][2]]) < k:
                    near_miss[distance_sort[i][2]].append(distance_sort[i][1])
                else:
                    if len(near_miss[distance_sort[i][2]]) == k:
                        stop_dict[distance_sort[i][2]] = 1
            stop = True
            for (key, value) in list(stop_dict.items()):
                if value != 1:
                    stop = False
            if stop:
                break

        # update reliefF score
        near_hit_term = np.zeros(n_features)
        for ele in near_hit:
            near_hit_term = np.array(abs(self_fea - X[ele, :])) + np.array(near_hit_term)

        near_miss_term = dict()
        for (label, miss_list) in list(near_miss.items()):
            near_miss_term[label] = np.zeros(n_features)
            for ele in miss_list:
                near_miss_term[label] = np.array(abs(self_fea - X[ele, :])) + np.array(near_miss_term[label])
            score += near_miss_term[label] / (k * p_dict[label])
        score -= near_hit_term / k
    return score

