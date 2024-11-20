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
from utils_LLM.prompts import generate_inference_sample


class Predictor:
    def __init__(self, configs):
        self.configs = configs
        self.tokenizer = load_tokenizer(configs)
        self.model = self.load_model()

    def load_model(self):

#        model_folder = os.path.join(
#            self.configs["training"]["checkpoints_folder"], "model_4bit"
#        )

        model_folder = os.path.join(
            self.configs["training"]["checkpoints_folder"], "model"
        )

#        print("loading model from:", model_folder)

        model = AutoModelForCausalLM.from_pretrained(model_folder, device_map="auto")

#        qlora_weights = os.path.join(self.configs["training"]["checkpoints_folder"], self.configs["training"]["qlora_folder"])#, checkpoint)
#        print("loading qlora adapters from:", qlora_weights)#

#        model = PeftModel.from_pretrained(model, qlora_weights)
#        model.eval()
#        model = model.merge_and_unload()

#        print("merged qlora adapters")

        return model

    def inference(self, sample, use_streamer=False):
#        inference_sample = generate_inference_sample(sample, self.tokenizer)

        sys = """
            Sei l'assistente AI in lingua italiana dell'Assemblea legislativa dell'Emilia-Romagna.
            Il tuo compito è quello di estrarre i metadati dalla query dell'utente. 
            Rispondi in maniera concisa utilizzando il seguente formato:
            tema = "tema della legge"
            """

    #    print("here", sys)

#        question_with_context = add_context_to_question(sample)

    #    print(question_with_context)
        
        messages = [
            {"role": "system", "content": sys},
            {"role": "user", "content": sample['question']},
    #        {"role": "assistant", "content": answer}
        ]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(prompt)

        inference_sample = self.tokenizer(prompt, return_tensors="pt")    

    #    output = {}
        
#        output["prompt"] = prompt
    #    output["answer"] = answer

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
            top_p = None,
            num_beams = 1,
            max_new_tokens = 3000
        )

#        pred = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        
        pred = None

        return pred
        