import time
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# Variabili globali per tenere traccia dello stato del modello e tokenizer
model = None
tokenizer = None

def initialize_model():
    global model, tokenizer

    if model is None or tokenizer is None:
        print("Caricamento del modello...")
        base_model = "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",  # "auto" usa la GPU se disponibile
            attn_implementation="eager",
            # torch_dtype=torch.bfloat16, # test
        )
        tokenizer = AutoTokenizer.from_pretrained(
            base_model, 
            token = "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"
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_bos_token = False
        print("Modello caricato correttamente.")
    else:
        print("Modello già caricato, utilizzo quello esistente.")

sys = "Sei un assistente AI per la lingua Italiana di nome LLaMAntino-3 ANITA " \
      "(Advanced Natural-based interaction for the ITAlian language)." \
      " Rispondi nella lingua usata per la domanda in modo chiaro, semplice ed esaustivo."

# Inizializza il modello solo se necessario
initialize_model()

while True:
    # Richiede la domanda all'utente e aspetta che premi invio
    user_question = input("Inserisci la tua domanda (o digita 'esci' per uscire): ")

    # uscita dal ciclo
    if user_question.lower() == 'esci':
        print("Uscita dal programma.")
        break

    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": user_question}
    ]
  
    start_time = time.time()

    tokenizer_start_time = time.time()
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokenizer_end_time = time.time()
    print(f" tokenizer: {tokenizer_end_time - tokenizer_start_time}")

    inputtokenizer_start_time = time.time()
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputtokenizer_end_time = time.time()
    print(f"input tokenizer: {inputtokenizer_end_time - inputtokenizer_start_time}")

    for k,v in inputs.items():
        inputs[k] = v.cuda()

    generate_start_time = time.time()
    outputs = model.generate(
        **inputs, 
        streamer = None,
        pad_token_id = tokenizer.pad_token_id,
        eos_token_id = tokenizer.eos_token_id,
        do_sample=False,
        top_p=None,
        num_beams = 1,
        max_new_tokens = 3000,
        temperature=0.6)

    generatetokenizer_end_time = time.time()
    print(f"generate time: {generatetokenizer_end_time - generate_start_time}")

    sequences = tokenizer.batch_decode(outputs)
    
    end_time = time.time()

    response_time = end_time - start_time
    for seq in sequences:
        print(f"Risposta: {seq}")
    
    print(f"Tempo di risposta: {response_time:.2f} secondi\n")
