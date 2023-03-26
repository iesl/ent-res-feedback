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

def validate_pointwise_featurizer(dataset, pointwise_matrix, le_signature_ids):
    # This function is here to validate two things : 
    # 1. Whether the length of the signtaures in the matrix is same as the length of the json file. 
    #   Also the block id. 
    # 2. Check whether the order is the same between the json file and the matrix.
    #   The order needs to be checked on block level as well as the signature level. 

    print("### --- Validating the pointwise matrix creation")
    print("The shape of the matrix : ",pointwise_matrix.shape)
    file_signature_location= f"../data/{dataset.name}/{dataset.name}_signatures.json"
    with open(file_signature_location, 'r') as myfile:
        data=myfile.read() 
    dict_obj = json.loads(data)
    file_keys = list(dict_obj.keys())
    print("Length of the signatures file : ", len(dict_obj))
    print("### -- Validating the signature order. ")
    indices = list(range(pointwise_matrix.shape[0]))
    inverse_transformed_signature_ids = list(le_signature_ids.inverse_transform(indices))
    ordered = True
    length = len(dict_obj)
    if len(dict_obj) == pointwise_matrix.shape[0]:
        print("The lengths are same")
        index = 0
        while (ordered and index < length):
            if inverse_transformed_signature_ids[index] == file_keys[index]:
                index += 1
            else:
                print("inverse_transformed_signature_ids[index] :", inverse_transformed_signature_ids[index])
                print("file_keys[index] : ", file_keys[index])
                ordered = False
                print("The order is not same")
        if ordered:
            print("The order is same.")
        
    else:
        print("The lengths are not same..")

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

    save_pickled_pointwise_features(AND_dataset, point_features_mat, le_signatures, random_seed)

    validate_pointwise_featurizer(AND_dataset, point_features_mat, le_signatures)

    train_pkl, val_pkl, test_pkl = store_featurized_pickles(AND_dataset,
                                                            featurization_info,
                                                            n_jobs=16,
                                                            use_cache=False,
                                                            random_seed=random_seed)

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
    
    random_seeds = [1, 2, 3, 4, 5] if params["dataset_seed"] is None else [params["dataset_seed"]]
    for seed in random_seeds:
        print("Preprocessing started for seed value", seed)
        save_featurized_data(data_home_dir, dataset, seed, point_features_mat, le_signatures)
        

        # Check the pickles are created OK
        train_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset}/seed{seed}/train_features.pkl"
        val_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset}/seed{seed}/val_features.pkl"
        test_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset}/seed{seed}/test_features.pkl"
        blockwise_features = read_blockwise_features(train_pkl)
        find_total_num_train_pairs(blockwise_features)
        #verify_diff_with_s2and(dataset, seed)
