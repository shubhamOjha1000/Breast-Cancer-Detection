import stacking_order
import stacking_model_class
import sys 
dataset_path = '/Users/shubhamojha/Desktop/Medical Diagnosis (Breast Cancer)/Breast_Cancer/Dataset'
sys.path.append(dataset_path)
import Breast_Cancer_Wisconsin
data = Breast_Cancer_Wisconsin.Dataset
import sys 
util_path = '/Users/shubhamojha/Desktop/Medical Diagnosis (Breast Cancer)/Breast_Cancer'
sys.path.append(util_path)
import util



def input_to_ANN_main():
    """
    """
    stack_model_list = stacking_order.stacking_order_main()

    stack_acc_train = []
    stack_acc_test = []
    stack_pres_train = []
    stack_pres_test = []
    stack_rec_train = []
    stack_rec_test = []
    stack_f1_train = []
    stack_f1_test = []
    test_prediction = []
    train_prediction = []

    for i in range(len(stack_model_list)):
        """
        """
        stacked_model = stacking_model_class.stacked_model_metrics(model = stack_model_list[i], dataset = data)
        stacked_model.fit(data.x_train, data.y_train)
        train_prediction.append(stacked_model.train_predict)
        test_prediction.append(stacked_model.test_predict)
        stack_acc_train.append(stacked_model.train_accuracy)
        stack_acc_test.append(stacked_model.test_accuracy)
        stack_pres_train.append(stacked_model.train_precision)
        stack_pres_test.append(stacked_model.test_precision)
        stack_rec_train.append(stacked_model.train_recall)
        stack_rec_test.append(stacked_model.test_recall)
        stack_f1_train.append(stacked_model.train_f1)
        stack_f1_test.append(stacked_model.test_f2)

    stack_model_name = []
    for i in range(len(stack_model_list)):
        stack_model_name.append(str("Meta Model " + str(i+1)))
    
    ### train_prediction :-
    util.creating_dataframe(stack_model_name, train_prediction, path) #### path :- CLI

    ### test_prediction
    util.creating_dataframe(stack_model_name, test_prediction, path)  #### path :- CLI

        
        






