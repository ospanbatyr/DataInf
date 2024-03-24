from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import warnings
import argparse
import logging
import torch
import os
import sys
import json

from lora_model import LORAEngineGeneration
from influence import IFEngineGeneration


import pandas as pd

def dataset_to_dataframe(dataset):
    """
    Converts a custom dataset object to a pandas DataFrame.
    
    This function assumes that the dataset can be iterated over to
    retrieve individual records as dictionaries, which is a common
    structure for many dataset implementations. If your dataset structure
    is different, you will need to adjust the logic accordingly.
    
    Parameters:
    - dataset: Your custom dataset object.
    
    Returns:
    - A pandas DataFrame containing the data from the dataset.
    """
    # Check if the dataset is already a DataFrame
    if isinstance(dataset, pd.DataFrame):
        return dataset
    
    # Try to convert the dataset to a DataFrame directly (works if the dataset is a list of dicts)
    try:
        return pd.DataFrame(dataset)
    except Exception as e:
        print(f"Error converting dataset to DataFrame: {e}")
        # Implement custom conversion logic here if direct conversion fails
        # This might involve iterating over the dataset and collecting
        # the data into a list of dicts or a similar structure that can be
        # directly converted to a DataFrame.
        # Example placeholder logic (you need to replace this with actual logic):
        data = []
        for item in dataset:
            # Assuming 'item' can be directly converted to a dict or is a dict
            data.append(dict(item))
        return pd.DataFrame(data)


def initialize_lora_engine(args):
    project_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    adapters_path = os.path.abspath(os.path.join(project_path, args.adapter_path))
    lora_engine = LORAEngineGeneration(base_path=args.base_path,
                                       adapter_path=adapters_path,
                                       project_path=project_path,
                                       seq_len= args.seq_len,
                                       train_dataset_name=args.train_dataset,
                                       validation_dataset=args.validation_dataset,
                                       n_train_samples = args.n_train_samples,
                                       n_val_samples = args.n_val_samples,
                                       device="cuda",
                                       load_in_8bit=False,
                                       load_in_4bit=False)
    return lora_engine

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Influence Function Analysis")
    parser.add_argument("--base_path", type=str, default="mistralai/Mistral-7B-v0.1", help="Base path for the model")
    parser.add_argument("--adapter_path", type=str, default="adapters/mistral-lora-sft-only", help="Adapters path")
    parser.add_argument("--train_dataset", type=str, default="open-platypus_medqa-instruction-train_merged_no_prompt.json", help="Train dataset filename")
    parser.add_argument("--validation_dataset", type=str, default="eval_datasets/medqa_usmle.json", help="Validation dataset filename")
    parser.add_argument("--n_train_samples", type=int, default=400, help="Number of samples from the training dataset")
    parser.add_argument("--n_val_samples", type=int, default=100, help="Number of samples from the validation dataset")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    parser.add_argument("--seq_len", type=int, default=4096, help="Maximum Sequence Length")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    
    lora_engine = initialize_lora_engine(args)

    # Model and GPU Settings
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    lora_engine.model = lora_engine.model.to("cuda")

    ### Example: model prediction
    prompt = """
Question: Tensor veli palatini is supplied by:
(A) Facial nerve (B) Trigeminal nerve (C) Glossopharyngeal nerve (D) Pharyngeal plexus
Answer:"""
    inputs = lora_engine.tokenizer(prompt, return_tensors="pt").to("cuda")

    # Generate
    generate_ids = lora_engine.model.generate(input_ids=inputs.input_ids, 
                                            max_length=128,
                                            pad_token_id=lora_engine.tokenizer.eos_token_id)
    output = lora_engine.tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]

    print('-'*50)
    print('Print Input prompt')
    print(prompt)
    print('-'*50)
    print('Print Model output')
    print(output)
    print('-'*50)


    # Gradient Computation

    tokenized_datasets, collate_fn = lora_engine.create_tokenized_datasets()
    tr_grad_dict, val_grad_dict = lora_engine.compute_gradient(tokenized_datasets, collate_fn)

    print("Computation of the gradients is done.")
    print("Computing now the HVPS")
    ### Compute the influence function
    influence_engine = IFEngineGeneration()
    influence_engine.preprocess_gradients(tr_grad_dict, val_grad_dict)
    influence_engine.compute_hvps()
    print("Computing now the influence scores")
    influence_engine.compute_IF()
    influence_engine.IF_dict['proposed'].to_csv("proposed _scores_no_prompt.csv", index= False)

    print("Computation of the influence scores is done.")
    print("Conmputing now the most and least influencing examples")

    # Computing top 5 most and least influential training samples for each validation sample
    top_n = 5  # Specify top N samples to select

    most_influential_data_points_proposed = influence_engine.IF_dict['proposed'].apply(lambda x: x.abs().nlargest(top_n).index.tolist(), axis=1)
    least_influential_data_points_proposed = influence_engine.IF_dict['proposed'].apply(lambda x: x.abs().nsmallest(top_n).index.tolist(), axis=1)
    # val_id = 0
    # print(f'Validation Sample ID: {val_id}')
    # print(lora_engine.validation_dataset[val_id])
    # print('The most influential training sample:')
    # print(lora_engine.train_dataset[int(most_influential_data_points_proposed.iloc[val_id])])
    # print('The least influential training sample:')
    # print(lora_engine.train_dataset[int(least_influential_data_points_proposed.iloc[val_id])])

    # Adjusting the DataFrame construction for saving
    df_data = {
        "Validation Sample": [lora_engine.validation_dataset[val_id] for val_id in range(len(lora_engine.validation_dataset))],
        "Most Influential Training Samples": [lora_engine.train_dataset[most_influential_data_points_proposed.iloc[val_id]] for val_id in range(len(lora_engine.validation_dataset))],
        "Least Influential Training Samples": [lora_engine.train_dataset[least_influential_data_points_proposed.iloc[val_id]] for val_id in range(len(lora_engine.validation_dataset))]
    }

    df = pd.DataFrame(df_data)

    # Save to JSON and CSV formats
    json_file_path = '/kuacc/users/hpc-rbech/hpc_run/DataInf/src/influential_samples_top5_no_prompt.json'
    csv_file_path = '/kuacc/users/hpc-rbech/hpc_run/DataInf/src/influential_samples_top5_no_prompt.csv'

    df.to_json(json_file_path, orient='records', lines=True)
    df.to_csv(csv_file_path, index=False)
    print("Saved dataframe with top 5 most and least influential training samples")

    # Also save the indices of these samples if needed
    most_influential_data_points_proposed.to_csv('/kuacc/users/hpc-rbech/hpc_run/DataInf/src/most_influential_data_points_proposed_top5_no_prompt.csv', index=False)
    least_influential_data_points_proposed.to_csv('/kuacc/users/hpc-rbech/hpc_run/DataInf/src/least_influential_data_points_proposed_top5_no_prompt.csv', index=False)
