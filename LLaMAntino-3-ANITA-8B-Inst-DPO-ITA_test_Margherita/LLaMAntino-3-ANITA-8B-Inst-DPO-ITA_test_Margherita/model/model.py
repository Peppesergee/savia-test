import os
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel,
    PeftConfig,
)
from utils.helper_functions import print_trainable_parameters
#from huggingface_hub import hf_hub_url, hf_hub_download
from huggingface_hub import snapshot_download

import os


def load_model(configs):
#    load_and_save(configs)

    #download full-precision model
    model_folder_full_prec = os.path.join(configs["training"]["checkpoints_folder"], "model")
    if not os.path.exists(model_folder_full_prec):
        download_from_hf_and_save(configs)

    model_folder = os.path.join(configs["training"]["checkpoints_folder"], "model_4bit")

    #convert full-precision into 4-bits
    if not os.path.exists(model_folder):
        load_and_save_4bit(configs)

    model = AutoModelForCausalLM.from_pretrained(model_folder, device_map="auto")

    if configs["training"]["load_model"]["restore_qlora_weights"]:
        print("restoring pre-trained qlora weights")

        qlora_weights_folder = os.path.join(
            configs["training"]["checkpoints_folder"],
            configs["training"]["qlora_folder"],
        )
        model = PeftModel.from_pretrained(
            model, qlora_weights_folder, is_trainable=True
        )

    else:
        print("applying new qlora adapters")

        model = apply_qlora_matrices(model)
        model.config.use_cache = False

    print_trainable_parameters(model)

    return model


def apply_qlora_matrices(model):
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    #    param = model.model.embed_tokens.weight
    #    param.data = param.data.to(torch.float32)

    config = LoraConfig(
        r = 32,
        lora_alpha = 32,
        target_modules = ['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj'],
        lora_dropout = 0.1,
        bias = "none",
        task_type = "CAUSAL_LM",
    )

    model = get_peft_model(model, config)

    print("added qlora adapters")

    return model


def save_adapters(configs, model):
    # save adapters
    out_folder = os.path.join(
        configs["training"]["checkpoints_folder"], configs["training"]["qlora_folder"]
    )
    print("saving adapters in:", out_folder)

    os.makedirs(out_folder, exist_ok=True)

    model.save_pretrained(out_folder)

def download_from_hf_and_save(configs):

    print("downloading full-precision model from HF")

    model_folder_full_prec = os.path.join(configs["training"]["checkpoints_folder"], "model")

#    print(model_folder_full_prec)

    snapshot_download(
        repo_id=configs["training"]["load_model"]["pretrained_model"],
        local_dir = model_folder_full_prec,
        use_auth_token = "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"
        )
 
    print("saving full-precision model in:", model_folder_full_prec)


def load_and_save_4bit(configs):
    print("converting in 4-bits")

    model_folder_full_prec = os.path.join(
        configs["training"]["checkpoints_folder"], "model"
    )

#    if torch.cuda.get_device_capability()[0] >= 8:
#        !pip install -qqq flash-attn
#        attn_implementation = "flash_attention_2"
#        torch_dtype = torch.bfloat16
#    else:
    attn_implementation = "eager"
    torch_dtype = torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
                model_folder_full_prec, 
                quantization_config = bnb_config, 
                device_map = "auto", 
                attn_implementation = attn_implementation 
            )

    model_folder_4bit = os.path.join(
        configs["training"]["checkpoints_folder"], "model_4bit"
    )
    os.makedirs(model_folder_4bit, exist_ok=True)

    model.save_pretrained(model_folder_4bit)
    print("saving model in:", model_folder_4bit)


def load_and_save_old(configs):
    model = AutoModelForCausalLM.from_pretrained(
        configs["training"]["load_model"]["pretrained_model"], device_map="auto"
    )

    model_folder_full_prec = os.path.join(
        configs["training"]["checkpoints_folder"], "model"
    )
    os.makedirs(model_folder_full_prec, exist_ok=True)

    model.save_pretrained(model_folder_full_prec)
    print("saving full-precision model in:", model_folder_full_prec)
