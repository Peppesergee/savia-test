from tokenizer.tokenizer import load_tokenizer

# from dataloader.dataloader import load_dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
import os

# import random
from transformers import TextStreamer

import sys

sys.path.append("../")
# from Mixtral_Instruct.utils.helper_functions import generate_prompt_for_inference, generate_prompt_for_inference_from_question
from utils.prompts import generate_inference_sample


class Predictor:
    def __init__(self, configs):
        self.configs = configs
        self.tokenizer = load_tokenizer(configs)
        self.model = self.load_model()

    def load_model(self):

        model_folder = os.path.join(
            self.configs["training"]["checkpoints_folder"], "model_4bit"
        )

#        model_folder = os.path.join(
#            self.configs["training"]["checkpoints_folder"], "model"
#        )

#        print("loading model from:", model_folder)

        model = AutoModelForCausalLM.from_pretrained(model_folder, device_map="auto")

        qlora_weights = os.path.join(self.configs["training"]["checkpoints_folder"], self.configs["training"]["qlora_folder"])#, checkpoint)
        print("loading qlora adapters from:", qlora_weights)

        model = PeftModel.from_pretrained(model, qlora_weights)
        model.eval()
        model = model.merge_and_unload()

        print("merged qlora adapters")

        return model

    def inference(self, sample, use_streamer=False):
        inference_sample = generate_inference_sample(sample, self.tokenizer)

#        print(inference_sample['prompt'])

        if use_streamer:
            streamer = TextStreamer(
                self.tokenizer,
                skip_prompt=True,
                decode_kwargs={"skip_special_tokens": True},
            )
        else:
            streamer = None

        input_ids = inference_sample["input_ids"].to("cuda")

        generated_ids = self.model.generate(
            input_ids,
            streamer = streamer,
            pad_token_id = self.tokenizer.pad_token_id,
            eos_token_id = self.tokenizer.eos_token_id,
            do_sample = False,
            temperature = None, 
            num_beams = 1,
            top_p = None,
#            max_new_tokens = self.configs["inference"]["max_new_tokens"]
            max_new_tokens = 3000

        )

        pred = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        
        return pred
        