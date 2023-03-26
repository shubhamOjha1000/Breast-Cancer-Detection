"""Script to run an experiment."""
import argparse 
from typing import Dict
import sys
data_path = '/Users/shubhamojha/Desktop/Medical Diagnosis (Breast Cancer)/Breast_Cancer/Dataset'
sys.path.append(data_path)
import Breast_Cancer_Wisconsin
classical_models_path = '/Users/shubhamojha/Desktop/Medical Diagnosis (Breast Cancer)/Breast_Cancer/Classical_Models'
sys.path.append(classical_models_path)
import best_classical_ml_models
stacking_path = '/Users/shubhamojha/Desktop/Medical Diagnosis (Breast Cancer)/Breast_Cancer/Stacking'
sys.path.append(stacking_path)
import input_to_ANN_model
DL_Model_path = '/Users/shubhamojha/Desktop/Medical Diagnosis (Breast Cancer)/Breast_Cancer/'
sys.path.append(DL_Model_path)
import DL_Model_Class



def run_experimentexperiment_config(experiment_config: Dict):
    """
    Run a training experiment.

    Parameters
    ----------
    experiment_config (dict)
        Of the form
        {

            "Dataset" : {   
        
                "dataset_args": {
            
                    'threshold_value_for_dropping_highly_correlated_features' : 0.4,
                    'threshold_value_for_dropping_low_importance_features' : 0.02,
                    'spliting_ratio' : 0.30

                }

            },

            "Classical_Models" : {
            
                "hyperparameter_tuning" : {
                
                    'knn_hyperparameter' : {
                    
                        'min_nearest_neighbor' : 1,
                        'max_nearest_neighbor' : 10

                    },

                    'decision_tree_hyperparamete' : {
                        
                        'min_depth' : 1,
                        'max_depth' : 15

                    },

                    'svm_hyperparamete' : {
                        
                        'C': [0.01, 0.05, 0.5, 0.1, 1, 10, 15, 20], 
                        'gamma': [0.0001, 0.001, 0.01, 0.1]

                    }

                    'gradient_boosting_hyperparameter' : {
                    
                        'learning_rate': [0.001, 0.1, 1, 10], 
                        'loss': ['deviance', 'exponential'], 
                        'n_estimators': [100, 150, 180, 200]
                    
                    }
                    
                }

                'classical_ml_parameters' : {
                
                    'RandomForestClassifier' : {
                    
                        'n_estimators' = 60,
                        'random_state' = 0
                    },

                    'xgboost' : {
                    
                        'random_state' = 0,
                        'booster' = "gbtree"    
                    
                    }
                
                }

                'choose_best_model' : {
                
                    'model_selection_metrics' : acc_test,     ### (acc_train, acc_test, pres_train, pres_test, rec_train, rec_test, f1_train, f1_test)
                    'threshold_value' : 0.95
                } 
            
            },
            
            'Stacking' : {
            
                'classical_ml_model_stacking_order' : [[0,1,2,3,4], [3,1,5,6,2], [0,3,5], [0,1,3,5], [0,1,2,3,4,5,6], [3,6,1,2,4], [3,2,1,6,5], [3,1]],
                'stacked_model_training_predictions_csv_file_path' : './....',
                'stacked_model_test_predictions_csv_file_path' : './....'

            }

            'DL_Model' : {
            
                'DL_networks_parameters' : {
                
                    'input_shape' : 8,
                    'activation_function' : 'relu',
                    'no_of_output_classes' : 1,   ### select from 1 or 2
                    'last_layer_activation' : 'sigmoid'  ##### select from softmax, sigmoid

                },

                "train_args_for_DL_model" : {
                
                    'dataset' : {
                        'input_features' : ['stacked_model_training_predictions', 'stacked_model_test_predictions'],
                        'target_label' : ['training_target_label', 'testing_target_label']
                    },
                    'network' : dl_model1,
                    'loss_function' : 'binary_crossentropy',
                    'metrics' : "accuracy",
                    'optimizer' : 'adam',
                    'batch_size' : 100,
                    'epochs' : 150, 
                }
                'DL_model_eval' : {
                
                    'threshold_value' : 0.5 

                }

            }


        }

    """
    dataset_class_  = Breast_Cancer_Wisconsin.Dataset()
    dataset_args = experiment_config["Dataset"]["dataset_args"]
    dataset = dataset_class_(dataset_args['threshold_value_for_dropping_highly_correlated_features'],
                             dataset_args['threshold_value_for_dropping_low_importance_features'],
                             dataset_args['spliting_ratio'])
    dataset.load_or_generate_data()


    selected_models = best_classical_ml_models.Classical_Models_main(
        

    )


    input_to_ANN_model.input_to_ANN_main()



    dl_model = DL_Model_Class.DL_base_model(

    )
    dl_model.fit()












def _parse_args(): 
    """
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment_config",
        type=str,
    )

    args = parser.parse_args()
    return args



def main():
    """Run experiment."""

    args = _parse_args()



if __name__ == "__main__":
    main()