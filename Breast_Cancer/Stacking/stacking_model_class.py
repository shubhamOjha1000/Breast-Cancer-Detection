import sys 
dataset_path = '/Users/shubhamojha/Desktop/Medical Diagnosis (Breast Cancer)/Breast_Cancer/Dataset'
sys.path.append(dataset_path)
import Breast_Cancer_Wisconsin
data = Breast_Cancer_Wisconsin.Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class stacked_model_metrics:
    def __init__(self, model, dataset = data):
        self.model = model
        self.dataset = dataset

        self.pred_train = None
        self.pred_test = None


    def fit(self, x_train, y_train):
        self.model = self.model.fit(x_train, y_train)

    def train_predict(self):
        self.pred_train = self.model.predict(self.dataset.x_train)

    def test_predict(self):
        self.pred_test = self.model.predict(self.dataset.x_test)

    def train_accuracy(self):
        return accuracy_score(self.dataset.y_train, self.pred_train)

    def test_accuracy(self):
        return accuracy_score(self.dataset.y_test, self.pred_test)
        
    def train_precision(self):
        return precision_score(self.dataset.y_train, self.pred_train)
        

    def test_precision(self):
        return accuracy_score(self.dataset.y_test, self.pred_test)

    def train_recall(self):
        return recall_score(self.dataset.y_train, self.pred_train)

    def test_recall(self):
        return recall_score(self.dataset.y_test, self.pred_test)

    def train_f1(self):
        return f1_score(self.dataset.y_train, self.pred_train)

    def test_f2(self):
        return f1_score(self.dataset.y_test, self.pred_test)

    

    


    