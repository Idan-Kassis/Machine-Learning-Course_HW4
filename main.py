# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 12:16:38 2022

@author: Tom and Idan
"""
# Setup
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import average_precision_score, roc_auc_score, make_scorer, accuracy_score, matthews_corrcoef, precision_score
from sklearn.utils import shuffle
import time
import os
from feature_selection import mRMR, reliefF, IG_SVM, seletFdr_fclassif, rfe_svm, PCA_IG
from New_PCA_IG_SVM import PCA_IG_SVM
from imblearn.over_sampling import SMOTE 
from imblearn.over_sampling import RandomOverSampler 
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
from tensorflow.keras.utils import to_categorical

#%% 

# Learning Methos
classifiers = [KNeighborsClassifier(), RandomForestClassifier(), LogisticRegression(), SVC(probability=True), GaussianNB()]
clf_names = ['KNN', 'RF', 'LogisticReg','SVM','NB']

# Feature selection Methos
FS_algo = [IG_SVM, PCA_IG, mRMR, seletFdr_fclassif, rfe_svm, reliefF, PCA_IG_SVM]
FS_names = ['IG_SVM - Article A', 'PCA_IG - Article B', 'mRMR', 'SelectFdr - fclassif', 'FRE-SVM','ReliefF', 'New_PCA-IG-SVM']

# Data Names List
all_data_names = os.listdir(os.path.join(os.getcwd(),'Data'))

# Parameters
k_array = np.array([1,2,3,4,5,10,15,20,25,30,50,100])
scoring = {'AUC' : 'roc_auc_ovr_weighted',
           'PR-AUC': make_scorer(precision_score, average = 'weighted', zero_division=1),
           'acc': 'accuracy',
           'MCC': make_scorer(matthews_corrcoef)}

# Results DataFrame
res_B_df = pd.DataFrame(columns = ['Data', 'Number of samples', 'Number of features', 'FS method', 'learning method', 'k',
                                 'CV method', 'Number of folds', 'AUC', 'PR-AUC', 'ACC', 'MCC', 'Selected features','Features scores',
                                 'FS time', 'Train time', 'Test time'])
res_C_df = pd.DataFrame(columns = ['Data', 'Number of samples', 'Number of features', 'FS method', 'learning method', 'k',
                                 'CV method', 'Number of folds', 'AUC', 'PR-AUC', 'ACC', 'MCC', 'Selected features','Features scores',
                                 'FS time', 'Train time', 'Test time'])
#%% Loop of part B + C

for data_name in all_data_names:
    #%% Data
    data = pd.read_csv(os.path.join(os.getcwd(),'Data',data_name))
    y = data['y']
    X = data.iloc[:,:-1]
    if min(y)==1:
        y = y-1
    X, y = shuffle(X, y)
    # Data Parameters 
    feature_names = np.array(X.keys())
    _, counts = np.unique(y, return_counts=True)
    # Folds
    if X.shape[0]>1000:
        F=5
    else:
        F=10
    num_Folds = min(min(counts),F)
    
    #%% FS and Classification
    for FS_method, FS_name in zip(FS_algo, FS_names):
        for k in k_array:
            # Feature selection
            start = time.time()
            select = SelectKBest(FS_method, k=k)
            X_selected = select.fit_transform(X, y)
            end = time.time()
            FS_time = end-start
            selected_features_names = feature_names[select.get_support()]
            selected_features_scores = select.scores_[select.get_support()]
            
            # CV Classification
            for model, name in zip(classifiers, clf_names):
                scores = cross_validate(model, X_selected, y, scoring=scoring,
                                         cv = num_Folds, return_train_score=False)
                # Create DF
                if FS_name != 'New_PCA-IG-SVM':
                    res_B_df = res_B_df.append({'Data': data_name, 'Number of samples': X.shape[0], 'Number of features': X.shape[1], 
                                            'FS method': FS_name, 'learning method': name, 'k': k,
                                             'CV method': 'KFold', 'Number of folds': num_Folds,
                                             'AUC': np.mean(scores['test_AUC']), 'PR-AUC': np.mean(scores['test_PR-AUC']), 'ACC': np.mean(scores['test_acc']),
                                             'MCC': np.mean(scores['test_MCC']), 'Selected features': selected_features_names,
                                             'Features scores': selected_features_scores, 'FS time': FS_time, 
                                             'Train time': np.mean(scores['fit_time']), 'Test time': np.mean(scores['score_time'])}, ignore_index=True)
                    # Write to excel
                    res_B_df.to_excel("Results_Part-B.xlsx")
                else:
                    res_C_df = res_C_df.append({'Data': data_name, 'Number of samples': X.shape[0], 'Number of features': X.shape[1], 
                        'FS method': FS_name, 'learning method': name, 'k': k,
                         'CV method': 'KFold', 'Number of folds': num_Folds,
                         'AUC': np.mean(scores['test_AUC']), 'PR-AUC': np.mean(scores['test_PR-AUC']), 'ACC': np.mean(scores['test_acc']),
                         'MCC': np.mean(scores['test_MCC']), 'Selected features': selected_features_names,
                         'Features scores': selected_features_scores, 'FS time': FS_time, 
                         'Train time': np.mean(scores['fit_time']), 'Test time': np.mean(scores['score_time'])}, ignore_index=True)
                    # Write to excel
                    res_C_df.to_excel("Results_Part-C.xlsx")
        print('Done - ',data_name)


#%% Create One Results file - b+c
all_files_names_B = os.listdir(os.path.join(os.getcwd(),'Part B'))
all_files_names_C = os.listdir(os.path.join(os.getcwd(),'Part C'))

all_res_df = pd.DataFrame(columns = ['Data', 'Number of samples', 'Number of features', 'FS method', 'learning method', 'k',
                                 'CV method', 'Number of folds', 'AUC', 'PR-AUC', 'ACC', 'MCC', 'Selected features','Features scores',
                                 'FS time', 'Train time', 'Test time'])

for name in all_files_names_B:
    Res = pd.read_excel(os.path.join(os.getcwd(),'Part B',name)) 
    all_res_df = all_res_df.append(Res, ignore_index=True)

for name in all_files_names_C:
    Res = pd.read_excel(os.path.join(os.getcwd(),'Part C',name)) 
    all_res_df = all_res_df.append(Res, ignore_index=True)

for idx in range(len(all_res_df)):
    head, tail = os.path.split(all_res_df['Data'].iloc[idx])
    all_res_df['Data'][idx] = tail
    
# write to excel
all_res_df.to_excel("Results.xlsx")


#%% Part D
# Best Configuration and augmented DF
best_df = pd.DataFrame(columns = ['Data', 'Number of samples', 'Number of features', 'FS method', 'learning method', 'k',
                                 'CV method', 'Number of folds', 'AUC', 'PR-AUC', 'ACC', 'MCC', 'Selected features','Features scores',
                                 'FS time', 'Train time', 'Test time'])
aug_df = pd.DataFrame(columns = ['Data', 'Number of samples', 'Number of features', 'FS method', 'learning method', 'k',
                                 'CV method', 'Number of folds', 'AUC', 'PR-AUC', 'ACC', 'MCC', 'Selected features','Features scores',
                                 'FS time', 'Train time', 'Test time'])

# Read results excel
Res = pd.read_excel('Results.xlsx') 
# Loop on data names
for data_name in all_data_names:
    # Data - Reading
    data = pd.read_csv(os.path.join(os.getcwd(),'Data',data_name))
    y = data['y']
    X = data.iloc[:,:-1]
    if min(y)==1:
        y = y-1
    X, y = shuffle(X, y)
    
    # data reading
    # Get and save the best configuration
    AUC_list = pd.Series.to_list(Res['AUC'][Res['Data']==data_name])
    max_auc_idx = AUC_list.index(max(AUC_list))
    best_configuration = Res[Res['Data']==data_name].iloc[[max_auc_idx]]
    best_df = best_df.append(best_configuration, ignore_index=True)
    best_df.to_excel("Results_Part-D-Best.xlsx")
    
    # FS
    FS_method = FS_algo[FS_names.index(best_configuration['FS method'].iloc[0])]
    k = best_configuration['k'].iloc[0]
    start = time.time()
    select = SelectKBest(FS_method, k=k)
    X_selected = select.fit_transform(X, y)
    end = time.time()
    FS_time = end-start
    
    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=1, stratify=y)
    
    # PCA
    # Linear
    linear_KPCA = KernelPCA(n_components=k, kernel="linear")
    X_train_linear = linear_KPCA.fit_transform(X_train)
    X_test_linear = linear_KPCA.fit(X_train).transform(X_test)
    # rbf
    rbf_KPCA = KernelPCA(n_components=k, kernel="rbf")
    X_train_rbf = rbf_KPCA.fit_transform(X_train)
    X_test_rbf = rbf_KPCA.fit(X_train).transform(X_test)
    # Concatenation
    X_train_PCA = np.concatenate((X_train, X_train_linear, X_train_rbf), axis =1)
    X_test_PCA = np.concatenate((X_test, X_test_linear, X_test_rbf), axis =1)
    
    
    # SMOTE - Only for train
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X_train_PCA, y_train)
    
    sm = SMOTE(random_state=42)
    X_train_aug, y_train_aug = sm.fit_resample(X_res, y_res)
    
    # Learning algo
    clf = classifiers[clf_names.index(best_configuration['learning method'].iloc[0])]
    start = time.time()
    clf.fit(X_train_aug, y_train_aug)
    end = time.time()
    fit_time = end-start
    
    # Prediction
    start = time.time()
    y_pred = clf.predict(X_test_PCA)
    end = time.time()
    y_prob = clf.predict_proba(X_test_PCA)
    test_time = end-start
    
    # Evaluation
    AUC = roc_auc_score(to_categorical(y_test), y_prob, average='weighted', multi_class='ovr')
    PR_AUC = precision_score(y_test, y_pred, average='weighted',zero_division=1)
    ACC = accuracy_score(y_test, y_pred)
    MCC = matthews_corrcoef(y_test, y_pred)
    
    # Save to Excel
    aug_df = aug_df.append({'Data': str('Aug_'+data_name), 'Number of samples': X.shape[0], 'Number of features': X.shape[1], 
                            'FS method': best_configuration['FS method'].iloc[0], 'learning method': best_configuration['learning method'].iloc[0], 
                            'k': k, 'CV method': 'KFold', 'Number of folds': best_configuration['Number of folds'].iloc[0],
                            'AUC': AUC, 'PR-AUC': PR_AUC, 'ACC': ACC, 'MCC': MCC, 
                            'Selected features': best_configuration['Selected features'].iloc[0],
                            'Features scores': best_configuration['Features scores'].iloc[0], 'FS time': FS_time, 
                            'Train time': fit_time, 'Test time': test_time}, ignore_index=True)
    aug_df.to_excel("Results_Part-D-Augmented.xlsx")

#%% Part E - Statistics

from scipy import stats
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns

# Read results excel
Res = pd.read_excel('Results.xlsx') 
# Loop on data names

for data_name in all_data_names:

    # Get and save the best configuration
    AUC_IG_SVM = pd.Series.to_list(Res['AUC'][np.logical_and(np.array(Res['Data']==data_name) ,np.array(Res['FS method']=='IG_SVM - Article A'))])
    AUC_PCA_IG = pd.Series.to_list(Res['AUC'][np.logical_and(np.array(Res['Data']==data_name) ,np.array(Res['FS method']=='PCA_IG - Article B'))])
    AUC_mRMR = pd.Series.to_list(Res['AUC'][np.logical_and(np.array(Res['Data']==data_name) ,np.array(Res['FS method']=='mRMR'))])
    AUC_fclassif = pd.Series.to_list(Res['AUC'][np.logical_and(np.array(Res['Data']==data_name) ,np.array(Res['FS method']=='SelectFdr - fclassif'))])
    AUC_SVM = pd.Series.to_list(Res['AUC'][np.logical_and(np.array(Res['Data']==data_name) ,np.array(Res['FS method']=='FRE-SVM'))])
    AUC_relieF = pd.Series.to_list(Res['AUC'][np.logical_and(np.array(Res['Data']==data_name) ,np.array(Res['FS method']=='ReliefF'))])
    AUC_new = pd.Series.to_list(Res['AUC'][np.logical_and(np.array(Res['Data']==data_name) ,np.array(Res['FS method']=='New_PCA-IG-SVM'))])
    
    # Boxplot
    df_AUC = pd.DataFrame({'IG_SVM': AUC_IG_SVM, 'PCA_IG': AUC_PCA_IG, 'mRMR': AUC_mRMR, 'f classif': AUC_fclassif,
                           'SVM': AUC_SVM, 'reliefF':AUC_relieF, 'New': AUC_new})
    ax = sns.boxplot(data=df_AUC, color='#99c2a2')
    plt.title(data_name)
    plt.show()

    _, p = stats.friedmanchisquare(AUC_IG_SVM, AUC_PCA_IG, AUC_mRMR, AUC_fclassif, AUC_SVM, AUC_relieF, AUC_new)
    
    
    if p<0.05:
        print('Data - ',data_name ,' statistically significant with P value = ', p)
        # Post Hoc test
        df = pd.DataFrame({'AUC': np.array(Res['AUC'][Res['Data']==data_name]), 'FS method': np.array(Res['FS method'][Res['Data']==data_name])})
        posthoc=sp.posthoc_conover(df, val_col='AUC', group_col='FS method', p_adjust = 'holm')
        
#%% Statistics
# Check the improvement
imp=0
nimp=0
Res = pd.read_excel('Results.xlsx') 
for data_name in all_data_names:
    old = pd.Series.to_list(Res['AUC'][np.logical_and(np.array(Res['Data']==data_name) ,np.array(Res['FS method']=='PCA_IG - Article B'))])
    new = pd.Series.to_list(Res['AUC'][np.logical_and(np.array(Res['Data']==data_name) ,np.array(Res['FS method']=='New_PCA-IG-SVM'))])
    print(data_name, ' , old - ',np.mean(np.array(old)),' , new - ', np.mean(np.array(new)), ' , difference = ',np.mean(np.array(new))-np.mean(np.array(old)))
    if np.mean(np.array(new))-np.mean(np.array(old))>0:
        print('Improve')
        imp+=np.mean(np.array(new))-np.mean(np.array(old))
    else:
        nimp+=np.mean(np.array(new))-np.mean(np.array(old))
        
# D
Res_best = pd.read_excel('Results_Part-D-Best.xlsx')['AUC']
Res_aug = pd.read_excel('Results_Part-D-Augmented.xlsx')['AUC']
Res_aug-Res_best

