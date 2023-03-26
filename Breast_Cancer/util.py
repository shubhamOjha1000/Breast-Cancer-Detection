from typing import List
import numpy as np
import pandas as pd


def download_url(url):
    """
    download dataset from a given url 
    """
    

def calculating_correlation_bw_features(input_features : np.ndarray ) -> List:
    """
    calculating Pearson correlation coefficient bw input features
    """
    return np.corrcoef(input_features.T)

    
    

def dropping_highly_correlated_features(input_features_column_name : pd.core.indexes.base.Index, corr_list : np.ndarray, input_features : pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """
    dropping highly correalted features from input features
    """

def calculating_feature_importance():
    """

    """

def dropping_low_importance_features():
    """
    
    """

def creating_dataframe(model_name : List, model_predictions : List, export_path : str):
    """
    """
    creator = {}
    for i in range(len(model_name)):
        creator[model_name[i]] = model_predictions[i] 
    
    df_ann = pd.DataFrame(creator)
    df_ann.to_csv(export_path, index=False) 