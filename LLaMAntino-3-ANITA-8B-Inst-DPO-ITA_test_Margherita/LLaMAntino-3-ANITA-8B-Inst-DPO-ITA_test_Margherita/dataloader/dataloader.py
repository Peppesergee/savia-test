import os
import datasets

# from utils.helper_functions import generate_prompt_for_training_Mixtral_8x7B
from utils.prompts import generate_training_sample, generate_training_sample_no_prompt


def create_dataloader(configs, tokenizer):
    dataset_folder = configs["dataset"]["dataset_folder"]

    print("loading dataset from json")

    prompt_fn = (
        generate_training_sample_no_prompt
        if configs["training"]["loss"]["ignore_prompt"]
        else generate_training_sample
    )

#    prompt_fn = generate_training_sample

    train_dataset = datasets.load_dataset(
        "json", data_files=os.path.join(dataset_folder, "IB_dataset_train.json")
    )
    train_dataset = train_dataset["train"].map(
        lambda samples: prompt_fn(samples, tokenizer), batched=False
    )
    
    eval_dataset = datasets.load_dataset(
        "json", data_files=os.path.join(dataset_folder, "IB_dataset_val.json")
    )
    eval_dataset = eval_dataset["train"].map(
        lambda samples: prompt_fn(samples, tokenizer), batched=False
    )
    

    #    eval_dataset = eval_dataset[0:2]
    #    eval_dataset = eval_dataset.select(range(5))
    

    return train_dataset, eval_dataset
