"""
Breast Cancer Wisconsin Dataset Class.

"""
import argparse
import pandas as pd
import sys 
util_path = '/Users/shubhamojha/Desktop/Medical Diagnosis (Breast Cancer)/Breast_Cancer'
sys.path.append(util_path)
import util
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



class Dataset:
    """

    """
    def __init__(
        self, 
        threshold_val_correlated_features : float = None,
        threshold_val_low_importance_features : float = None,
        dataset_splitting_ratio : float = None):

        self.splitting_ratio = dataset_splitting_ratio 

        self.df = None
        self.y = None
        self.x = None

        self.x_1 = None

        self.x_12 = None


        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def load_or_generate_data(self):
        """
        """
        self.extract()
        self.load()
        self.transformation()
        self.feature_scaling()
        self.encode()
        self.data_split()

    def extract(self):
        """
        call util.download_url function to extract data from the url 
        
        """
        self.df = pd.read_csv('/Users/shubhamojha/Desktop/Breast Cancer/data.csv')


    def load(self):
        """
        Choose a directory to load the data in h5py.file format.

        unzip the file .

        separating target variable from training features into two separate files.

        """
        self.y = self.df['diagnosis']

        self.x = self.df.drop(['Unnamed: 32','id','diagnosis'],axis = 1)


    def transformation(self):
        """
        calling functions utils to calculate(highly correlated features, feature importance) and dropping them from input features
        """
        # Dropping highly correlated features :-
        #corr_list = util.calculating_correlation_bw_features(self.x.values)
        #self.x_1 = util.dropping_highly_correlated_features(self.x.columns, corr_list, self.x)
        self.x_1 = self.x.drop(['perimeter_mean','radius_mean','compactness_mean','concave points_mean','radius_se','perimeter_se','radius_worst','perimeter_worst','compactness_worst','concave points_worst','compactness_se','concave points_se','texture_worst','area_worst'],axis=1)

        # Removing low importance feature:-
        #util.calculating_feature_importance()
        #self.x_12 = util.dropping_low_importance_features()
        self.x_12 = self.x_1.drop(['symmetry_mean','texture_se','symmetry_se','smoothness_se'], axis=1)




    def data_split(self):
        """
        splitting the data into training and testing data
        """
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_12,self.y,test_size=0.36,random_state=18)



    def encode(self):
        """
        encoding the target variable 
        """
        self.y = self.y.replace({'M':1,'B':0})



    def feature_scaling(self):
        """
        scaling the input features using Standard Scaler
        """
        scaler = StandardScaler()
        self.x_12 = scaler.fit_transform(self.x_12) 


    
def _parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--threshold_value_for_dropping_highly_correlated_features",
        type = int
    )

    parser.add_argument(
        "--threshold_value_for_dropping_low_importance_features",
        type = int
    )

    parser.add_argument(
        "--spliting_ratio",
        type = int
    )

    return parser.parse_args()

def main():

    args = _parse_args()
    dataset = Dataset(
        threshold_val_correlated_features = args.threshold_value_for_dropping_highly_correlated_features,
        threshold_val_low_importance_features = args.threshold_value_for_dropping_low_importance_features,
        dataset_splitting_ratio = args.spliting_ratio)

    dataset.load_or_generate_data() 
    
    



if __name__ == "__main__":
    main()



        
