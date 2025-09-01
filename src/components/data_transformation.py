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
    load_object,
    calculate_entropy,
    is_ip_address_in_domain,
    extract_sensitive_words_with_freq,
    add_sensitive_word_feature,
    extract_url_features,
    count_subdomains,
    extract_url_features1,
    save_object
)

from tqdm import tqdm
tqdm.pandas()

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()


    def feature_extraction(self, df: pd.DataFrame, sensitive_words=None) -> pd.DataFrame:
        """
        Custom feature engineering from raw 'url' column.
        Keeps 'label' if it exists, so splitting can be done later.
        """
        try:
            df = df.copy()
            # URL length
            df['UrlLen'] = df['url'].apply(lambda x: len(repr(x)) - 2)

            # count of dots
            df['count_dot'] = df['url'].apply(lambda x: x.count('.'))

            # Shannon entropy
            df['entropy'] = df['url'].apply(calculate_entropy)

            # no of Slashes 
            df['NoOfSlashes'] = df['url'].apply(lambda s: s.count("/"))

            # ip address in domain
            df['IsIpAddressinDomain'] = df['url'].apply(is_ip_address_in_domain)

            # --- Sensitive words ---
            if sensitive_words is None:
                # prediction case → load pre-saved
                sensitive_words = load_object(os.path.join("artifacts", "sensitive_words.pkl"))

            df = add_sensitive_word_feature(df, sensitive_words)

            # extract url features
            feature_columns = [
                'SymbolCount_URL', 'executable', 'NumberRate_URL',
                'Querylength', 'argPathRatio', 'charcompace', 'CharacterContinuityRate',
                'Entropy_Domain', 'Entropy_Filename', 'pathurlRatio'
            ]
            df[feature_columns] = df['url'].progress_apply(extract_url_features).apply(pd.Series)

            df = extract_url_features1(df)

            df['NumSubDomains'] = df['url'].apply(count_subdomains)

            # cleanup
            if "tokens" in df.columns:
                df = df.drop("tokens", axis=1)
            if "url" in df.columns:
                df = df.drop("url", axis=1)

            # NOTE: Do NOT drop label here
            return df

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Sensitive words from training set
            sensitive_words = extract_sensitive_words_with_freq(train_df)
            save_object(os.path.join("artifacts", "sensitive_words.pkl"), sensitive_words)
            logging.info("Sensitive words extracted and saved.")

            # Apply feature extraction
            train_df = self.feature_extraction(train_df, sensitive_words)
            test_df  = self.feature_extraction(test_df, sensitive_words)

            target_column_name = "label"

            # Ensure label exists
            if target_column_name not in train_df.columns:
                raise CustomException("Label column missing in training data", sys)

            X_train = train_df.drop(columns=[target_column_name])
            y_train = train_df[target_column_name].copy()

            X_test = test_df.drop(columns=[target_column_name])
            y_test = test_df[target_column_name].copy()

            # Map to numeric
            y_train = y_train.map({'benign': 0, 'phishing': 1})
            y_test  = y_test.map({'benign': 0, 'phishing': 1})

            logging.info("Feature extraction and split complete.")
            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys)


    def transform_for_prediction(self, url: str) -> pd.DataFrame:
        """
        Transform a single URL for prediction.
        """
        try:
            df = pd.DataFrame({"url": [url]})
            features = self.feature_extraction(df, sensitive_words=None)  # will auto-load
            return features
        except Exception as e:
            raise CustomException(e, sys)

