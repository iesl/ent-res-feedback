"""
Run from command line:
    python e2e_scripts/preprocess_s2and_data.py --data_home_dir="./data" --dataset_name="pubmed"
"""
from typing import Union, Dict
from typing import Tuple

from s2and.consts import PREPROCESSED_DATA_DIR
from s2and.featurizer import FeaturizationInfo, store_featurized_pickles, many_pairs_featurize
from os.path import join
from s2and.data import ANDData
import pickle
import numpy as np
from utils.parser import Parser

from s2and.data import ANDData
import logging
import json
from s2and.featurizer import FeaturizationInfo, featurize
from preprocess_s2and_pointwise import save_pickled_pointwise_features, create_signature_features_matrix

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_order_all_block(pointwise_block, features_block):
    # First, check the blockwise similarity
    keys_pointwise = list(pointwise_block.keys())
    keys_features = list(features_block.keys())

    if len(keys_pointwise) == len(keys_features):
        print("The number of blocks are same across the features")
        ordered = True
        index = 0
        len_of_blocks = len(keys_pointwise)
        while ordered and index < len_of_blocks:
            if keys_pointwise[index] == keys_features[index]:
                index+=1
            else:
                ordered = False
        
        if not ordered:
            print("The blocks are not in order.")
        else:
            print("The blocks are in order.")
        return ordered
    else:
        print("The number of blocks in seed : ", seed, "are not the same across features")
        return False

def validate_order_inside_block(pointwise_block, features_block):
    pointwise_signature_list = []
    signature_id_list = []

    for block, val in pointwise_block.items():
        list_of_sig = val[0]
        pointwise_signature_list.extend(list_of_sig)
    
    #print("pointwise_signature_list : ", pointwise_signature_list)
    
    for block, val in features_block.items():
        list_of_sig = [sigs.signature_id for sigs in val]
        signature_id_list.extend(list_of_sig)
    
    #print("signature_id_list : ", signature_id_list)

    # Now for validation part. 
    ordered = True
    index = 0
    len_of_sigs = len(pointwise_signature_list)
    while ordered and index < len_of_sigs:
        if pointwise_signature_list[index] == signature_id_list[index]:
            index += 1
        else:
            ordered = False
    if not ordered:
        print("The Signatures are not in order.")
    else:
        print("The Signatures are in order.")
    return ordered

def validate_pointwise_featurizer(dataset):
    print("### --- Validating the pointwise matrix creation")
    # Need to go through each pickle file in all the seeds.
    seeds = [1]

    for seed in seeds:
        train_point_loc = f"{PREPROCESSED_DATA_DIR}/{dataset.name}/pointwise/seed{seed}/train_signature_features.pkl"
        val_point_loc = f"{PREPROCESSED_DATA_DIR}/{dataset.name}/pointwise/seed{seed}/val_signature_features.pkl"
        test_point_loc = f"{PREPROCESSED_DATA_DIR}/{dataset.name}/pointwise/seed{seed}/test_signature_features.pkl"


        train_loc = f"{PREPROCESSED_DATA_DIR}/{dataset.name}/seed{seed}/train_signatures.pkl"
        val_loc = f"{PREPROCESSED_DATA_DIR}/{dataset.name}/seed{seed}/val_signatures.pkl"
        test_loc = f"{PREPROCESSED_DATA_DIR}/{dataset.name}/seed{seed}/test_signatures.pkl"

        # This is what the pointwise has created. 
        with open(train_point_loc, 'rb') as f:
            train_block_pointwise_data = pickle.load(f)
        
        with open(train_loc, 'rb') as f:
            train_block_data = pickle.load(f)


        # For training block.
        train_blocks_in_order = validate_order_all_block(train_block_pointwise_data, train_block_data)

        if train_blocks_in_order:
            print("Training block of seed ", seed, "in order")
        else:
            print("Training block of seed ", seed, " are not in order")

        is_signature_train_in_order = validate_order_inside_block(train_block_pointwise_data, train_block_data)
        
        if is_signature_train_in_order:
            print("Training signature of seed ", seed , "in order")
        else:
            print("Training signature of seed ", seed , "are not in order")

        

        # For validation parts..
        with open(val_point_loc, 'rb') as f:
            val_block_pointwise_data = pickle.load(f)
        with open(val_loc, 'rb') as f:
            val_block_data = pickle.load(f)

        val_blocks_in_order = validate_order_all_block(val_block_pointwise_data, val_block_data)

        if val_blocks_in_order:
            print("Validation block of seed ", seed, "in order")
        else:
            print("Validation block of seed ", seed, " are not in order")

        is_signature_val_in_order = validate_order_inside_block(val_block_pointwise_data, val_block_data)
        
        if is_signature_val_in_order:
            print("Validation signature of seed ", seed , "in order")
        else:
            print("Validation signature of seed ", seed , "are not in order")
        

        # For testing parts.. 
        with open(test_point_loc, 'rb') as f:
            test_block_pointwise_data = pickle.load(f)
        with open(test_loc, 'rb') as f:
            test_block_data = pickle.load(f)

        test_blocks_in_order = validate_order_all_block(test_block_pointwise_data, test_block_data)

        if test_blocks_in_order:
            print("Test block of seed ", seed, "in order")
        else:
            print("Test block of seed ", seed, " are not in order")

        is_signature_test_in_order = validate_order_inside_block(test_block_pointwise_data, test_block_data)
        
        if is_signature_test_in_order:
            print("Test signature of seed ", seed , "in order")
        else:
            print("Test signature of seed ", seed , "are not in order")
        

def save_featurized_data(data_home_dir, dataset_name, random_seed, point_features_mat, le_signatures):
    parent_dir = f"{data_home_dir}/{dataset_name}"
    AND_dataset = ANDData(
        signatures=join(parent_dir, f"{dataset_name}_signatures.json"),
        papers=join(parent_dir, f"{dataset_name}_papers.json"),
        mode="train",
        specter_embeddings=join(parent_dir, f"{dataset_name}_specter.pickle"),
        clusters=join(parent_dir, f"{dataset_name}_clusters.json"),
        block_type="s2",
        train_pairs_size=100000,
        val_pairs_size=10000,
        test_pairs_size=10000,
        name=dataset_name,
        n_jobs=16,
        random_seed=random_seed,
    )
    logger.info("Loaded ANDData object")
    # Load the featurizer, which calculates pairwise similarity scores
    featurization_info = FeaturizationInfo()
    logger.info("Loaded featurization info")

    #save_pickled_pointwise_features(AND_dataset, point_features_mat, le_signatures, random_seed)

    

    train_pkl, val_pkl, test_pkl = store_featurized_pickles(AND_dataset,
                                                            featurization_info,
                                                            n_jobs=16,
                                                            use_cache=False,
                                                            random_seed=random_seed,
                                                            pointwise_matrix=point_features_mat,
                                                            le_signatures=le_signatures)
    
    validate_pointwise_featurizer(AND_dataset)
    print(" ## Validation and save process completed.")

    return train_pkl, val_pkl, test_pkl


def read_blockwise_features(pkl):
    blockwise_data: Dict[str, Tuple[np.ndarray, np.ndarray]]
    with open(pkl,"rb") as _pkl_file:
        blockwise_data = pickle.load(_pkl_file)

    print("Total num of blocks:", len(blockwise_data.keys()))
    return blockwise_data

def find_total_num_train_pairs(blockwise_data):
    count = 0
    for block_id in blockwise_data.keys():
        count += len(blockwise_data[block_id][0])

    print("Total num of signature pairs", count)    
    
# def verify_diff_with_s2and(dataset_name, random_seed):
#     parent_dir = f"{DATA_HOME_DIR}/{dataset_name}"
#     AND_dataset = ANDData(
#         signatures=join(parent_dir, f"{dataset_name}_signatures.json"),
#         papers=join(parent_dir, f"{dataset_name}_papers.json"),
#         mode="train",
#         specter_embeddings=join(parent_dir, f"{dataset_name}_specter.pickle"),
#         clusters=join(parent_dir, f"{dataset_name}_clusters.json"),
#         block_type="s2",
#         train_pairs_size=100,
#         val_pairs_size=100,
#         test_pairs_size=100,
#         # train_pairs_size=100000,
#         # val_pairs_size=10000,
#         # test_pairs_size=10000,
#         name=dataset_name,
#         n_jobs=2,
#         random_seed=random_seed,
#     )
#
#     # Load the featurizer, which calculates pairwise similarity scores
#     featurization_info = FeaturizationInfo()
#     # the cache will make it faster to train multiple times - it stores the features on disk for you
#     train, val, test = featurize(AND_dataset, featurization_info, n_jobs=2, use_cache=False)
#     X_train, y_train, _ = train
#     X_val, y_val, _ = val
#
#     logger.info("Done loading and featurizing")
#
#     #Verify the 2 sets are equal
#     with open("s2and_data_subsample.pkl", "rb") as _pkl_file:
#         s2and_set = pickle.load(_pkl_file)
#
#     with open("our_data_subsample.pkl", "rb") as _pkl_file:
#         our_set = pickle.load(_pkl_file)
#
#     print("VERIFICATION STATUS: ", s2and_set==our_set)

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
    
    point_features_mat, le_signatures = create_signature_features_matrix(data_home_dir, dataset)

    # Added this for speeding up while testing.
    matrix_pickle_file_location = "./matrix_pickle.pkl"

    with open(matrix_pickle_file_location,"wb") as _pkl_file:
        pickle.dump((point_features_mat, le_signatures), _pkl_file)

    """
    with open(matrix_pickle_file_location, 'rb') as f:
        point_features_mat, le_signatures = pickle.load(f)
    """
    
    random_seeds = [1] if params["dataset_seed"] is None else [params["dataset_seed"]]
    for seed in random_seeds:
        print("Preprocessing started for seed value", seed)
        save_featurized_data(data_home_dir, dataset, seed, point_features_mat, le_signatures)
        

        # Check the pickles are created OK
        train_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset}/seed{seed}/train_features.pkl"
        val_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset}/seed{seed}/val_features.pkl"
        test_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset}/seed{seed}/test_features.pkl"
        #blockwise_features = read_blockwise_features(train_pkl)
        #find_total_num_train_pairs(blockwise_features)
        #verify_diff_with_s2and(dataset, seed)
