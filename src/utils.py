import re
import pandas as pd
import numpy as np
import math
from urllib.parse import urlparse
from collections import Counter
from itertools import groupby
from tqdm import tqdm
import pickle
from src.exception import CustomException
import sys
import os
import dill
import joblib

def calculate_entropy(url: str) -> float:
    """
    Calculate Shannon entropy of characters in a URL.
    """
    if not url:
        return 0.0
    
    freq = {}
    for char in url:
        freq[char] = freq.get(char, 0) + 1
        
    total_len = len(url)
    entropy = 0.0
    for count in freq.values():
        p = count / total_len
        entropy -= p * math.log2(p)
    
    return entropy




def is_ip_address_in_domain(url: str) -> int:
    # Checks if ip_address in domain
    if not isinstance(url, str):
        return 0
    
    ip_regex = re.compile(r'^(https?://)?(\d{1,3}\.){3}\d{1,3}([:/]?|$)')
    return 1 if ip_regex.match(url) else 0



def extract_sensitive_words_with_freq(df, ratio_threshold=2):
    # Tokenizer function
    def tokenize_url(url):
        return [t for t in re.split(r'[\/\?\=\-\_\.\&\:\;\%]', url.lower()) if t]

    # Apply tokenization to all URLs at once
    df['tokens'] = df['url'].apply(tokenize_url)

    # Separate phishing and benign tokens using explode
    phishing_tokens = df[df['label'] == 'phishing']['tokens'].explode()
    benign_tokens = df[df['label'] == 'benign']['tokens'].explode()

    # Count frequencies
    phishing_freq = Counter(phishing_tokens.dropna())
    benign_freq = Counter(benign_tokens.dropna())

    # Extract sensitive words based on ratio
    sensitive_words = {}
    for word, p_count in phishing_freq.items():
        b_count = benign_freq.get(word, 0)
        if p_count / (b_count + 1) >= ratio_threshold:
            sensitive_words[word] = {
                'phishing_count': p_count,
                'benign_count': b_count,
                'ratio': p_count / (b_count + 1)
            }

    # Sort by ratio descending
    sensitive_words = dict(sorted(sensitive_words.items(), key=lambda x: x[1]['ratio'], reverse=True))

    return sensitive_words

def add_sensitive_word_feature(df, sensitive_words):
    # Precompile regex for splitting
    pattern = re.compile(r'[\/\?\=\-\_\.\&\:\;\%]')
    
    # Convert sensitive_words to a set for O(1) lookup
    sensitive_set = set(sensitive_words)

    def count_sensitive_words(url):
        tokens = [t for t in pattern.split(url.lower()) if t]  # tokenization
        return sum(1 for t in tokens if t in sensitive_set)     # count matches

    df['SensitiveWordCount'] = df['url'].apply(count_sensitive_words)
    return df

def extract_url_features(url):
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path
    query = parsed.query
    filename = path.split('/')[-1] if '/' in path else ''

    # 1. SymbolCount_URL
    symbol_count = len(re.findall(r'[^\w\s]', url))

    # 2. executable
    is_executable = int(bool(re.search(r'\.(exe|bat|zip|scr|msi|apk)$', filename.lower())))

    # 3. NumberRate_URL
    num_digits = sum(c.isdigit() for c in url)
    number_rate = num_digits / len(url) if len(url) > 0 else 0

    # 5. Querylength
    query_length = len(query)

    # 6. argPathRatio
    arg_path_ratio = len(query) / len(path) if len(path) > 0 else 0

    # 7. charcompace (Character Complexity = unique chars / total chars)
    charcompace = len(set(url)) / len(url) if len(url) > 0 else 0

    # 8. CharacterContinuityRate (max sequence of same char / url length)
    max_seq = max((len(list(g)) for _, g in groupby(url)), default=1)
    continuity_rate = max_seq / len(url) if len(url) > 0 else 0

    # 9. Entropy_Domain
    domain_entropy = calculate_entropy(domain)

    # 10. Entropy_Filename
    filename_entropy = calculate_entropy(filename)

    # 11. pathurlRatio
    pathurl_ratio = len(path) / len(url) if len(url) > 0 else 0

    return [
        symbol_count, is_executable, number_rate,
        query_length, arg_path_ratio, charcompace, continuity_rate,
        domain_entropy, filename_entropy, pathurl_ratio]


def extract_url_features1(df):
    # 1. HasRedirection
    df['HasRedirection'] = df['url'].apply(lambda x: 1 if '//' in x.split('//', 1)[-1] else 0)

    # 2. HasAtSymbol
    df['HasAtSymbol'] = df['url'].apply(lambda x: 1 if '@' in x else 0)

    # 3. HasHyphenInDomain
    df['HasHyphenInDomain'] = df['url'].apply(lambda x: 1 if '-' in urlparse(x).netloc else 0)
    return df

    # 4. NumSubDomains

def count_subdomains(url):
    try:
        hostname = urlparse(url).hostname
        if hostname:
            return max(0, hostname.count('.') - 1)  # clamp at 0
        return 0
    except:
        return 0


# if __name__ == "__main__":
#     print("Testing utils...")

#     # Test calculate_entropy
#     print("Entropy:", calculate_entropy("https://google.com"))

#     # Test IP address check
#     print("IP in domain:", is_ip_address_in_domain("http://192.168.1.1/login"))
#     print("IP in domain:", is_ip_address_in_domain("http://example.com"))

#     # Test sensitive words
#     import pandas as pd
#     data = {
#         "url": ["http://phishing-bank.com/login", "http://goodsite.com/home"],
#         "label": ["phishing", "benign"]
#     }
#     df = pd.DataFrame(data)
#     sensitive_words = extract_sensitive_words_with_freq(df)
#     print("Sensitive words:", sensitive_words)

#     # Test subdomain counter
#     print("Subdomains:", count_subdomains("http://mail.google.com"))
#     print("Subdomains:", count_subdomains("http://google.com"))

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return joblib.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def save_object(file_path, obj):
    """
    Save a Python object to a file using dill.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)