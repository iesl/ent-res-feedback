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

def save_pickled_pointwise_features(data_home_dir, dataset_name):
    """
    Fetch pointwise feature for dataset and store in a pickle.
    """
    processed_data = {}
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
        n_jobs=16,
        random_seed=random_seed,
    )
    
    # print("Storing pickled dataset....")
    # with open(f'preprocess_dataset_{dataset_name}.pkl', 'wb') as f:
    #     pickle.dump(AND_dataset, f)
    
    # print("Loading pickled dataset...")
    # with open(f'preprocess_dataset_{dataset_name}.pkl', 'rb') as f:
    #     AND_dataset = pickle.load(f)
    # print("Loaded pickle dataset...")
    
    

    point_features_row, point_features_col,  point_features_data, num_feats, num_points = pointwise_featurize(AND_dataset,
                                                                                                              n_jobs=16,
                                                                                                            use_cache=False)
    logger.info('converting feature indices to csr_matrix')
    point_features = coo_matrix(
            (point_features_data, (point_features_row, point_features_col)),
            shape=(num_points, num_feats)
    ).tocsr()
    print("Matrix creation done.")
    processed_data['mention_level_features'] = point_features
    
    logger.info('Dumping processed data')
    
    with open(f'{dataset_name}_feature_processed.pkl', 'wb') as f:
        pickle.dump(processed_data, f)

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

    print("Preprocessing started")
    save_pickled_pointwise_features(data_home_dir, dataset)
    print("Matrix")
