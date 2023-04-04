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

def save_pickled_pointwise_features(AND_dataset, sparse_matrix, 
                                    label_encoder_signatures,
                                    random_seed: int = None):
    """
    Fetch pointwise feature for dataset and store in a pickle.
    """
    
    if random_seed:
        # This splits the signatures per three different blocks
        train_block, val_block, test_block = AND_dataset.split_cluster_signatures()

        train_pointwise_features = {}
        validation_pointwise_features = {}
        test_pointwise_features = {}

        # The above three should have a key-list(val) (where val is a list of signature IDs) under them. 
        # Below three for loops go through the blocks, gets the corresponding row index of the signature 
        # from the label encoder, splices the matrix with only those rows and stores per block.
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

        train_pkl = f"{PREPROCESSED_DATA_DIR}/{AND_dataset.name}/pointwise/seed{random_seed}/train_signature_features.pkl"
        val_pkl = f"{PREPROCESSED_DATA_DIR}/{AND_dataset.name}/pointwise/seed{random_seed}/val_signature_features.pkl"
        test_pkl = f"{PREPROCESSED_DATA_DIR}/{AND_dataset.name}/pointwise/seed{random_seed}/test_signature_features.pkl"

        with open(train_pkl,"wb") as _pkl_file:
            pickle.dump(train_pointwise_features, _pkl_file)
        with open(val_pkl,"wb") as _pkl_file:
            pickle.dump(validation_pointwise_features, _pkl_file)
        with open(test_pkl,"wb") as _pkl_file:
            pickle.dump(test_pointwise_features, _pkl_file)
    else:
        processed_data = {}
        point_features_mat, _ = create_signature_features_matrix(data_home_dir, AND_dataset.name)
        processed_data['mention_level_features'] = point_features_mat

        logger.info('Dumping processed data')
        file_name = f"{PREPROCESSED_DATA_DIR}/{AND_dataset.name}/{AND_dataset.name}_all_signature_features.pkl"

        with open(file_name, 'wb') as f:
            pickle.dump(processed_data, f)
    

def create_signature_features_matrix(data_home_dir, dataset_name):
    """
    Generate pointwise feature set for the entire dataset and return sparse matrix 
    representation for each signature and their respective features.
    """
    logger.info("Signature features pre-procesing started")
    parent_dir = f"{data_home_dir}/{dataset_name}"
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
        n_jobs=16
    )
    
#     print("Storing pickled dataset....")
#     with open(f'preprocess_dataset_{dataset_name}.pkl', 'wb') as f:
#         pickle.dump(AND_dataset, f)
    
#     # Use below line carefully. 
#     print("Loading pickled dataset...")
#     with open(f'preprocess_dataset_{dataset_name}.pkl', 'rb') as f:
#         AND_dataset = pickle.load(f)
#     print("Loaded pickle dataset...")
    
    point_features_mat, le_signatures = pointwise_featurize(AND_dataset,
                                                          n_jobs=16,
                                                        use_cache=False)
    
    matrix_pickle_file_location = f'preprocess_matrix_{dataset_name}.pkl'
    print("Storing pickled matrix ....")
    with open(matrix_pickle_file_location, 'wb') as f:
        pickle.dump((point_features_mat, le_signatures), f)

    print("### loading from pickle")
    with open(matrix_pickle_file_location, 'rb') as f:
        point_features_mat, le_signatures = pickle.load(f)
    
    
    logger.info("Signature features pre-procesing completed")
    return point_features_mat, le_signatures
    
    
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
    save_pickled_pointwise_features(data_home_dir, dataset)
