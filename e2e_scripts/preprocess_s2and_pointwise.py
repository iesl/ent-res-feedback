"""
Run from command line:
    python e2e_scripts/preprocess_s2and_pointwise.py --data_home_dir="./data" --dataset_name="pubmed"
"""
import sys

from typing import Union, Dict
from typing import Tuple

from s2and.consts import PREPROCESSED_DATA_DIR
from s2and.featurizer import FeaturizationInfo, store_featurized_pickles, many_pairs_featurize, pointwise_featurize
from os.path import join
from s2and.data import ANDData
import pickle
import os
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from utils.parser import Parser

from s2and.data import ANDData
import logging
from s2and.featurizer import FeaturizationInfo, featurize
from IPython import embed

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def save_pointwise_in_different_splits(AND_dataset, sparse_matrix, label_encoder_signatures, random_seed):
    logger.info('extracting signature depending on different split')

    train_block, val_block, test_block = AND_dataset.split_cluster_signatures()

    train_pointwise_features = {}
    validation_pointwise_features = {}
    test_pointwise_features = {}

    # The above three should have a key-list(val) (where val is a list of signature IDs) under them. 
    
    # Doing for training block : 
    for block_id, list_of_signatures in train_block.items():
        # Let us transform each of those using label encoder and index them from the sparse matrix.
        encoded_signature_id_list = label_encoder_signatures.transform(list_of_signatures)
        train_pointwise_features[block_id] = sparse_matrix[encoded_signature_id_list, :]

    # Doing for validation block : 
    for block_id, list_of_signatures in val_block.items():
        # Let us transform each of those using label encoder and index them from the sparse matrix.
        encoded_signature_id_list = label_encoder_signatures.transform(list_of_signatures)
        validation_pointwise_features[block_id] = sparse_matrix[encoded_signature_id_list, :]
    
    for block_id, list_of_signatures in test_block.items():
        # Let us transform each of those using label encoder and index them from the sparse matrix.
        encoded_signature_id_list = label_encoder_signatures.transform(list_of_signatures)
        test_pointwise_features[block_id] = sparse_matrix[encoded_signature_id_list, :]

    if(not os.path.exists(f"{PREPROCESSED_DATA_DIR}/{AND_dataset.name}/pointwise/seed{random_seed}")):
        os.makedirs(f"{PREPROCESSED_DATA_DIR}/{AND_dataset.name}/pointwise/seed{random_seed}")

    train_pkl = f"{PREPROCESSED_DATA_DIR}/{AND_dataset.name}/pointwise/seed{random_seed}/train_features.pkl"
    val_pkl = f"{PREPROCESSED_DATA_DIR}/{AND_dataset.name}/pointwise/seed{random_seed}/val_features.pkl"
    test_pkl = f"{PREPROCESSED_DATA_DIR}/{AND_dataset.name}/pointwise/seed{random_seed}/test_features.pkl"

    with open(train_pkl,"wb") as _pkl_file:
        pickle.dump(train_pointwise_features, _pkl_file)
    with open(val_pkl,"wb") as _pkl_file:
        pickle.dump(validation_pointwise_features, _pkl_file)
    with open(test_pkl,"wb") as _pkl_file:
        pickle.dump(test_pointwise_features, _pkl_file)
    

def save_pickled_pointwise_features(data_home_dir, dataset_name, random_seed):
    """
    Fetch pointwise feature for dataset and store in a pickle.
    """
    processed_data = {}
    parent_dir = f"{data_home_dir}/{dataset_name}"
    """
    AND_dataset = ANDData(
        signatures=join(parent_dir, f"{dataset_name}_signatures.json"),
        papers=join(parent_dir, f"{dataset_name}_papers.json"),
        mode="inference",
        clusters=join(parent_dir, f"{dataset_name}_clusters.json"),
        block_type="s2",
        train_pairs_size=100000,
        val_pairs_size=10000,
        test_pairs_size=10000,
        name=dataset_name,
        n_jobs=16,
        random_seed=random_seed
    )
    
    print("Storing pickled dataset....")
    with open(f'preprocess_dataset_{dataset_name}.pkl', 'wb') as f:
        pickle.dump(AND_dataset, f)
    """
    # Use below line carefully. 
    print("Loading pickled dataset...")
    with open(f'preprocess_dataset_{dataset_name}.pkl', 'rb') as f:
        AND_dataset = pickle.load(f)
    print("Loaded pickle dataset...")
    
    

    point_features_row, point_features_col,  point_features_data, num_feats, num_points, le_feature_dict = pointwise_featurize(AND_dataset,
                                                                                                              n_jobs=16,
                                                                                                            use_cache=False)
    logger.info('converting feature indices to csr_matrix')
    point_features = coo_matrix(
            (point_features_data, (point_features_row, point_features_col)),
            shape=(num_points, num_feats)
    ).tocsr()
    print("Matrix creation done.")
    save_pointwise_in_different_splits(AND_dataset, point_features, le_feature_dict, random_seed)

if __name__=='__main__':
    # Creates the pickles that store the preprocessed data
    # Read cmd line args
    
    parser = Parser(add_preprocessing_args=True)
    parser.add_preprocessing_args()

    args = parser.parse_args()
    print(args)

    params = args.__dict__
    data_home_dir = params["data_home_dir"]
    dataset = params["dataset_name"]
    random_seed = 1000

    print("Preprocessing started")
    save_pickled_pointwise_features(data_home_dir, dataset, random_seed)
    print("Matrix")
