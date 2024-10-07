import time
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
    # bnb_4bit_quant_type="nf4",
    # bnb_4bit_compute_dtype=torch.bfloat16,
    # bnb_4bit_use_double_quant=False,
)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",  # "auto" usa la GPU se disponibile
)
tokenizer = AutoTokenizer.from_pretrained(base_model)

sys = "Sei un assistente AI per la lingua Italiana di nome LLaMAntino-3 ANITA " \
      "(Advanced Natural-based interaction for the ITAlian language)." \
      " Rispondi nella lingua usata per la domanda in modo chiaro, semplice ed esaustivo."

pipe = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    task='text-generation',
    max_new_tokens=512,
    temperature=0.6,
    do_sample=True,
    top_p=0.9,
)

while True:
    user_question = input("Inserisci la tua domanda (o digita 'esci' per uscire): ")

    # uscita
    if user_question.lower() == 'esci':
        print("Uscita dal programma.")
        break

    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": user_question}
    ]

    start_time = time.time()
    sequences = pipe(messages)
    end_time = time.time()

    response_time = end_time - start_time
    for seq in sequences:
        print(f"Risposta: {seq['generated_text']}")
    
    print(f"Tempo di risposta: {response_time:.2f} secondi\n")
