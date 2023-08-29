import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import argparse
from data_preprocessing import utils

def main(args: dict) -> None:
    dataset = args["dataset"]    
    if dataset == "churn":
        churn_raw = pd.read_csv(f'{args["path_to_raw_data"]}/{dataset}/train.csv')
        
        features_df = utils.preprocess_features(
            churn_raw, "MCC", "cl_id", "TRDATETIME", "amount", date_format='%d%b%y:%H:%M:%S', churn_horizon_months = 1.
        )
        targets_df = utils.preprocess_targets(churn_raw, "cl_id", "target_flag", "target_sum")
        
    elif dataset == "default":
        default_trans = pd.read_csv(f'{args["path_to_raw_data"]}/{dataset}/transactions_finetune.csv')
        default_target = pd.read_csv(f'{args["path_to_raw_data"]}/{dataset}/target_finetune.csv')
        
        features_df = utils.preprocess_features(
            default_trans, "mcc_code", "user_id", "transaction_dttm", "transaction_amt", date_format='%Y-%m-%d %H:%M:%S'
        )
        targets_df = utils.preprocess_targets(default_target, "user_id", "target")

        
    elif dataset == "raif":
        raif_trans = pd.read_csv(
            f'{args["path_to_raw_data"]}/{dataset}/transactions_last_2.csv', sep=";", engine="python", on_bad_lines='skip'
        )
        raif_clients = pd.read_csv(
            f'{args["path_to_raw_data"]}/{dataset}/clients_last_2_fixed.csv', sep=";", engine="python", on_bad_lines='skip'
        )
        
        features_df = utils.preprocess_features(
            raif_trans, "mcc", "cnum", "purchdate", "amount", date_format='%Y-%m-%d %H:%M:%S'
        )
        targets_df = utils.preprocess_targets(raif_clients, "cnum_", "gender", "age", "married_", "residenttype")
        
    else:
        raise ValueError(f"Unknown dataset name {dataset}.")
       
    df_prepr = utils.merge(features_df, targets_df)
    utils.save_parquet(df_prepr, path_to_folder=args["path_to_save_folder"], dataset_name=args["dataset"])
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare and local validation dataset.")
    
    parser.add_argument("--path_to_raw_data", type=str, required=True, help="Path to folder with raw data")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name", choices=["churn", "default", "raif"])
    
    parser.add_argument("--path_to_save_folder", type=str, required=True, help="Path to folder to save preprocessed data")
    
    args = parser.parse_args()
    args = dict(vars(args))
    
    main(args)
