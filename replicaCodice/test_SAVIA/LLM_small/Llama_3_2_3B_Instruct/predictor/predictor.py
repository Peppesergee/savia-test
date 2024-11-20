from Llama_3_2_3B_Instruct.tokenizer.tokenizer import load_tokenizer

import torch
import os

# import random
from transformers import TextStreamer, BitsAndBytesConfig, AutoModelForCausalLM

import sys

sys.path.append("../")
# from Mixtral_Instruct.utils.helper_functions import generate_prompt_for_inference, generate_prompt_for_inference_from_question
from Llama_3_2_3B_Instruct.utils_LLM.prompts import generate_inference_sample
from Llama_3_2_3B_Instruct.model.model import download_from_hf_and_save


class Predictor:
    def __init__(self, configs, load_4_bits = False):
        self.configs = configs
        self.tokenizer = load_tokenizer(self.configs)
        #load configs for inference
        self.inference_configs = self.load_inference_configs()
        self.model = self.load_model()

    def load_model(self):

        model_name = self.configs['model']['model_name']
        model_path = os.path.join(self.configs['model']['checkpoints_folder'], model_name.split("/")[-1], "model")

        print("model path", model_path)

        #download full-precision model
        if not os.path.exists(model_path):
            download_from_hf_and_save(self.configs, model_path)

        if self.configs['quantization']['load_in_4bit']:                          
            print("quantization in 4 bits")
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype = torch.bfloat16)
        elif self.configs['quantization']['load_in_8bit']:
            print("quantization in 8 bits")
            quantization_config = BitsAndBytesConfig(load_in_8bit = True)
        else:
            print("loading full precision model")
            quantization_config = None
    
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map = "auto", trust_remote_code = True, quantization_config = quantization_config)
        model.eval()

        return model

    def load_inference_configs(self):
        
        inference_configs = self.configs['inference']

        inference_configs['pad_token_id'] = self.tokenizer.pad_token_id
        inference_configs['eos_token_id'] = self.tokenizer.eos_token_id

        return inference_configs


    def inference(self, sample, use_streamer=False, clean_pred = True):

        inference_sample = generate_inference_sample(sample, self.tokenizer)

        if use_streamer:
            streamer = TextStreamer(
                self.tokenizer,
                skip_prompt=True,
                decode_kwargs={"skip_special_tokens": True},
            )
        else:
            streamer = None

        inference_configs = self.inference_configs
        inference_configs['streamer'] = streamer

        input_ids = inference_sample["input_ids"].to(self.model.device)        
        generated_ids = None
        
        try:
            generated_ids = self.model.generate(input_ids, **inference_configs)

        except torch.cuda.OutOfMemoryError as e:
            print(e)
#            print("retrying with cache_implementation='offloaded'")
            torch.cuda.empty_cache()
            generated_ids = None
            pred = "out of memory"

#        if oom:
#            print("retrying")
#            torch.cuda.empty_cache()
#            inference_configs["cache_implementation"] = "offloaded"
#            generated_ids = self.model.generate(input_ids, **inference_configs)
#                        )
        if generated_ids is not None:
            pred = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False) if generated_ids != None else None
            if clean_pred:
                pred = self.clean_pred(pred)

        return pred
 

    def clean_pred(self, pred):
        pred_cleaned = pred.split("<|end_header_id|>")[-1].replace("<|eot_id|>", "").strip()

        return pred_cleaned
    