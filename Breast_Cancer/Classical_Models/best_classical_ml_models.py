from typing import List
import hyperparameter_tuning
import classical_ml_models
import base_class
import sys 
dataset_path = '/Users/shubhamojha/Desktop/Medical Diagnosis (Breast Cancer)/Breast_Cancer/Dataset'
sys.path.append(dataset_path)
import Breast_Cancer_Wisconsin
data = Breast_Cancer_Wisconsin.Dataset



def choose_best_model(metrics : List, threshold_value : float, model_list : List):
    selected_models = []
    selected_acc = [] 
    #model_list = []
    for i in range(10):
        if metrics[i] > threshold_value :
            selected_models.append(model_list[i])
            #selected_acc.append(metrics[i])


def Classical_Models_main():
    """
    """
    knn_best_params = hyperparameter_tuning.tuning_knn_hyperparameter()   #### input :- CLI
    dt_best_params = hyperparameter_tuning.tuning_decision_tree_hyperparameter()   #### input :- CLI
    SVM_best_params = hyperparameter_tuning.tuning_svm_hyperparameter()     #### input :- CLI
    GradientBoosting_best_params = hyperparameter_tuning.tuning_gradient_boosting_hyperparameterr()   #### input :- CLI

    model_list = []
    model_list.append(classical_ml_models.model1())
    model_list.append(classical_ml_models.model2())
    model_list.append(classical_ml_models.model3())
    model_list.append(classical_ml_models.model4())
    model_list.append(classical_ml_models.model5())
    model_list.append(classical_ml_models.model6())
    model_list.append(classical_ml_models.model7())
    model_list.append(classical_ml_models.model8())
    model_list.append(classical_ml_models.model9())
    model_list.append(classical_ml_models.model10()) 

    acc_train = []
    acc_test = []
    pres_train = []
    pres_test = []
    rec_train = []
    rec_test = []
    f1_train = []
    f1_test = []    

    for i in range(len(model_list)):
        model = base_class.classical_ml_models(model = model_list[i], dataset = data)
        model.fit_predict()
        acc_train.append(model.train_accuracy)
        acc_test.append(model.test_accuracy)
        pres_train.append(model.train_precision)
        pres_test.append(model.test_precision)
        rec_train.append(model.train_recall)
        rec_test.append(model.test_recall)
        f1_train.append(model.train_f1)
        f1_test.append(model.test_f2)

    selected_models =  choose_best_model(metrics = acc_test, threshold_value = 0.95, model_list  = model_list)

    return selected_models

    


    

    





