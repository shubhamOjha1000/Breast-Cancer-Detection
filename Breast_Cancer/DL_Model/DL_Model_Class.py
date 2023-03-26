import DL_networks
import sys 
path = '/Users/shubhamojha/Desktop/Medical Diagnosis (Breast Cancer)/Breast_Cancer/Stacking'
sys.path.append(path)
import input_to_ANN_model 
from typing import List


class DL_base_model:
    """
    """
    def __init__(self, input_features : List, 
                 target_label : List, 
                 network, loss_function : str, 
                 metrics : str, 
                 optimizer : str):
        """
        """
        self.x_train = input_features[0]
        self.x_test = input_features[1]
        self.y_train = target_label[0]
        self.y_test = input_features[1]
        self.network = network
        self.pred_train = None
        self.pred_test = None


    
    def fit(self, batch_size : int, epochs : int):
        """
        """
        self.network.compile(loss=self.loss(), optimizer=self.optimizer(), metrics=self.metrics())

        self.network.fit(self.x_train, self.y_train, batch_size = batch_size , epochs = 150)
        

    def save_weights(self):      ######
        """
        """

    def loss(self, loss_function):   ######
        """
        """
        return loss_function


    def optimizer(self, optimizer):   ######
        """
        """
        return optimizer

    def metrics(self, metrics):     ######
        """
        """
        return metrics

    def evaluate(self, threshold_value : int):    ####
        """
        """
        self.pred_train = self.network.predict(self.x_train)
        self.pred_train = (self.pred_train > threshold_value)

        self.pred_test = self.network.predict(self.x_train)
        self.pred_test = (self.pred_test > threshold_value)

    

    
