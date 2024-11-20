#import json
#import os

import torch
#from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
#from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
#from datasets import load_dataset, load_from_disk
#import transformers
#import random
#import numpy as np
import sys
#from notebooks.helper_functions import print_trainable_parameters, generate_prompt_for_training, generate_prompt_for_training_Mixtral_8x7B

from utils_LLM.logger import Logger
from utils_LLM.helper_functions import load_configs
from tokenizer.tokenizer import load_tokenizer
from dataloader.dataloader import create_dataloader
from model.model import load_model, save_adapters
from trainer.trainer import load_trainer#, load_trainer_orig


def main(configs):

    print("CUDA available:", torch.cuda.is_available())
    print("num GPUs:", torch.cuda.device_count())
    
    tokenizer = load_tokenizer(configs)
    train_dataset, eval_dataset = create_dataloader(configs, tokenizer)

    model = load_model(configs)

    trainer = load_trainer(configs, tokenizer, model, train_dataset, eval_dataset)

    trainer.train()

    save_adapters(configs, model)

if __name__ == "__main__":

    sys.stdout = Logger()
    configs = load_configs()

#    print(configs)
    main(configs)

