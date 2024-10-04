# from fastapi import FastAPI

# app = FastAPI()

# @app.get("/")

# async def root():
#     return {"message": "Hello MarghERita"}

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

base_model = "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
#    bnb_4bit_quant_type="nf4",
#    bnb_4bit_compute_dtype=torch.bfloat16,
#    bnb_4bit_use_double_quant=False,
)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto", # metti "auto" se c'è la gpu
)
tokenizer = AutoTokenizer.from_pretrained(base_model)

sys = "Sei un an assistente AI per la lingua Italiana di nome LLaMAntino-3 ANITA " \
    "(Advanced Natural-based interaction for the ITAlian language)." \
    " Rispondi nella lingua usata per la domanda in modo chiaro, semplice ed esaustivo."

messages = [
    {"role": "system", "content": sys},
    {"role": "user", "content": "Chi è Carlo Magno?"}
]

pipe = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=False, # langchain expects the full text
    task='text-generation',
    max_new_tokens=512, # max number of tokens to generate in the output
    temperature=0.6,  #temperature for more or less creative answers
    do_sample=True,
    top_p=0.9,
)

sequences = pipe(messages)
for seq in sequences:
    print(f"{seq['generated_text']}")
