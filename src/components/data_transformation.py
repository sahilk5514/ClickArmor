import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import (
    save_object,
    calculate_entropy,
    is_ip_address_in_domain,
    extract_sensitive_words_with_freq,
    add_sensitive_word_feature,
    extract_url_features,
    count_subdomains,
    extract_url_features1
)

from tqdm import tqdm
tqdm.pandas()

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()


    def feature_extraction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Custom feature engineering from raw 'url' column.
        """
        try:
            # URL length
            df['UrlLen'] = df['url'].apply(lambda x: len(repr(x)) - 2)

            # Example: count of dots in URL
            df['count_dot'] = df['url'].apply(lambda x: x.count('.'))

            # Shannon entropy
            df['entropy'] = df['url'].apply(calculate_entropy)

            # no of Slashes 
            df['NoOfSlashes']=df['url'].apply(lambda s:s.count("/"))

            # ip address is in domain or not
            df['IsIpAddressinDomain'] = df['url'].apply(is_ip_address_in_domain)

            #
            sensitive_words = extract_sensitive_words_with_freq(df)
            df = add_sensitive_word_feature(df,sensitive_words)

            # 
            feature_columns = [
            'SymbolCount_URL', 'executable', 'NumberRate_URL',
            'Querylength', 'argPathRatio', 'charcompace', 'CharacterContinuityRate',
            'Entropy_Domain', 'Entropy_Filename', 'pathurlRatio']

            df[feature_columns] = df['url'].progress_apply(extract_url_features).apply(pd.Series)

            df = df.drop('tokens',axis=1)
            
            df = extract_url_features1(df)

            df['NumSubDomains'] = df['url'].apply(count_subdomains)

            df = df.drop(['url'],axis=1)
            return df
    
        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Apply feature extraction
            train_df = self.feature_extraction(train_df)
            test_df = self.feature_extraction(test_df)

            target_column_name = "label"

            X_train = train_df.drop(columns=[target_column_name])
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=[target_column_name])
            y_test = test_df[target_column_name]

            y_train = y_train.map({'benign': 0, 'phishing': 1})
            y_test  = y_test.map({'benign': 0, 'phishing': 1})
            logging.info("Feature extraction and split complete.")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys)
        

