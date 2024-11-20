import os
#import torch
#from transformers import AutoModelForCausalLM, BitsAndBytesConfig
#from peft import (
#    prepare_model_for_kbit_training,
#    LoraConfig,
#    get_peft_model,
#    PeftModel,
#    PeftConfig,
#)
#from Meta_Llama_3_1_8B_Instruct.utils_LLM.helper_functions import print_trainable_parameters
#from huggingface_hub import hf_hub_url, hf_hub_download
from huggingface_hub import snapshot_download

import os

def download_from_hf_and_save(configs, model_path):

    print("downloading full-precision model from HF")

#    model_folder_full_prec = os.path.join(configs["model"]["checkpoints_folder"], "model")

#    print(model_folder_full_prec)

    snapshot_download(
        repo_id = configs["model"]["model_name"],
        local_dir = model_path,
        use_auth_token = configs['HF_auth_token']
        )
 
    print("saving full-precision model in:", model_path)
