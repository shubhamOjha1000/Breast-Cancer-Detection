from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from typing import List
import sys 
path = '/Users/shubhamojha/Desktop/Medical Diagnosis (Breast Cancer)/Breast_Cancer/Classical_Models'
sys.path.append(path)
import best_classical_ml_models

selected_models = best_classical_ml_models.Classical_Models_main()

def estimate_creator(l):     ##### put in utl
    estm = []
    for i in l:
        estm.append(selected_models[i]) 
    return estm

def model_stacking_order(order : List , model_list : List = selected_models):
    """
    """
    estimator_list = []
    for i in range(len(order)):
        estimator_list.append(estimate_creator(order[i]))

    return estimator_list

def stacking_order_main():
    """
    """
    estimator_list = model_stacking_order()  ###### input CLI -> order
    stack_model_list = [] 
    for i in range(len(estimator_list)):
        stack_model_list.append(StackingClassifier(estimators  = estimator_list[i], final_estimator = LogisticRegression())) #### final_estimator -> CLI, stacked models can have diff final_estimator
    
    return stack_model_list