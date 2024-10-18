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
            device_map="auto",  
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

def generate_response_streaming(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    max_new_tokens = 300  
    temperature = 0.4  
    top_p = 0.8  
    
    # Genera token per token
    output_ids = model.generate(
        input_ids, 
        max_new_tokens=max_new_tokens, 
        temperature=temperature, 
        top_p=top_p,
        do_sample=True,  
        pad_token_id=tokenizer.eos_token_id,  
        use_cache=True,
        output_scores=True, 
        return_dict_in_generate=True,
    )
    
    # Decodifica i token man mano e mostralo in tempo reale
    generated_text = ""
    for i in range(len(output_ids.sequences[0])):
        # Decodifica solo il nuovo token
        new_token = tokenizer.decode(output_ids.sequences[0][i], skip_special_tokens=True)
        if new_token in prompt:
            filtered_token = ''.join([char for char in new_token if char not in prompt])
            new_token = filtered_token
        generated_text += new_token
        print(new_token, end="", flush=True)  # Stampa progressivamente i nuovi token
        time.sleep(0.05)  # Aggiungi un leggero ritardo per simulare lo streaming
    return generated_text

while True:
    user_question = input("Inserisci la tua domanda (o digita 'esci' per uscire): ")

    if user_question.lower() == 'esci':
        print("Uscita dal programma.")
        break

    messages = sys + "\n" + user_question

    start_time = time.time()
    
    print("\nRisposta: ", end="")  # Inizia a stampare la risposta
    generate_response_streaming(messages)
    
    end_time = time.time()
    response_time = end_time - start_time
    print(f"\nTempo di risposta: {response_time:.2f} secondi\n")
