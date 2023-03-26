from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import argparse
from typing import Dict, List
import numpy as np
import sys 
dataset_path = '/Users/shubhamojha/Desktop/Medical Diagnosis (Breast Cancer)/Breast_Cancer/Dataset'
sys.path.append(dataset_path)
import Breast_Cancer_Wisconsin
data = Breast_Cancer_Wisconsin.Dataset




def tuning_knn_hyperparameter(Hyperparamters : Dict ) -> int:
    """
    """
    acc_rate=[]
    for i in range(Hyperparamters['min_nearest_neighbor'], Hyperparamters['max_nearest_neighbor']):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(data.x_train, data.y_train) 
        pred = knn.predict(data.x_test)
        acc_rate.append(np.mean(pred == data.y_test)) 

    for i in range(len(acc_rate)):
        if max(acc_rate) == acc_rate[i]:
            knn_best_params = i + 1
            break
    return knn_best_params



def tuning_decision_tree_hyperparameter(Hyperparamters : Dict) -> List:
    """
    """
    acc_rate_1=[]
    acc_rate_2=[]
    for i in range(Hyperparamters['min_depth'], Hyperparamters['max_depth']):
        dt1 = DecisionTreeClassifier(criterion="entropy", max_depth = i) 
        dt2 = DecisionTreeClassifier(criterion="gini", max_depth = i)
        dt1.fit(data.x_train, data.y_train) 
        dt2.fit(data.x_train, data.y_train) 
        predl = dt1.predict(data.y_test) 
        pred2 = dt2.predict(data.y_test)
        acc_rate_1.append(np. mean (predl == data.y_test)) 
        acc_rate_2.append(np. mean (pred2 == data.y_test))

    for i in range(len(acc_rate_1)):
        if max(acc_rate_1) == acc_rate_1[i]:
            dt_entropy_best_params = i + 1
            break

    for i in range(len(acc_rate_2)):
        if max(acc_rate_2) == acc_rate_2[i]:
            dt_gini_best_params = i + 1
            break

    return [dt_entropy_best_params, dt_gini_best_params]


def tuning_svm_hyperparameter(Hyperparamters : Dict) -> Dict:
    """
    """
    svc = SVC() 
    grid_search = GridSearchCV(svc, Hyperparamters.SVM_parameters) 
    grid_search.fit(data.x_train, data.y_train)
    SVM_best_params = grid_search.best_params_
    return SVM_best_params  #### {'C': 10, 'gamma': 0.01}

def tuning_gradient_boosting_hyperparameter(Hyperparamters : Dict) -> Dict:
    """
    """
    gbc = GradientBoostingClassifier() 
    grid_search_gbc = GridSearchCV(gbc, Hyperparamters.GradientBoosting_parameters,  cv = 5, n_jobs = -1, verbose = 1)
    grid_search_gbc.fit(data.x_train, data.y_train)
    GradientBoosting_best_params = grid_search_gbc.best_params_
    return GradientBoosting_best_params   #####   {'learning_rate': 1, 'loss': 'exponential', 'n_estimators': 200}


def _parse_args():
    """
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--knn_hyperparameter",    ### knn_hyperparameter = {'min_nearest_neighbor' : 1, 'max_nearest_neighbor' : 11} 
        type = str
    )

    parser.add_argument(
        "--decision_tree_hyperparameter",        ### decision_tree = {'min_depth' : 1, 'max_depth' : 11}
        type = str
    )

    parser.add_argument(
        "--SVM_parameters",                     ### SVM_parameters = {'gamma' : [0.0001, 0.001, 0.01, 0.1], 'C' : [0.01, 0.05, 0.5, 0.1, 1, 10, 15, 20]}
        type = str
    )

    parser.add_argument(
        "--GradientBoosting_parameters",      ### GradientBoosting_parameters = {'loss': ['deviance', 'exponential'], 'learning_rate': [0.001, 0.1, 1, 10], 'n_estimators': [100, 150, 180, 200]}
        type = str
    )


    args = parser.parse_args()
    return args
