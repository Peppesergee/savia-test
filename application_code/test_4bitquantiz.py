import time
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

model = None
tokenizer = None

def initialize_model():
    global model, tokenizer

    if model is None or tokenizer is None:
        print("Caricamento del modello...")
        base_model = "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_compute_dtype=torch.bfloat16,  
        )
   
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",  # "auto" usa la GPU se disponibile
            torch_dtype=torch.bfloat16,  
        )

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        print("Modello caricato correttamente.")
    else:
        print("Modello già caricato, utilizzo quello esistente.")

sys = "Sei un assistente AI per la lingua Italiana di nome LLaMAntino-3 ANITA " \
      "(Advanced Natural-based interaction for the ITAlian language)." \
      " Rispondi nella lingua usata per la domanda in modo chiaro, semplice ed esaustivo."

initialize_model()

pipe = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    task='text-generation',
    max_new_tokens=3000,  
    temperature=0.4,  
    do_sample=True,
    top_p=0.8,  
)

while True:
    user_question = input("Inserisci la tua domanda (o digita 'esci' per uscire): ")

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