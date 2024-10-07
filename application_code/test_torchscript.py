import time
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from torch import cuda

base_model = "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"

def load_scripted_model(base_model):
    # Carica il modello preaddestrato da Hugging Face
    model = AutoModelForCausalLM.from_pretrained(base_model).to('cuda')

    # Esegui lo scripting del modello
    scripted_model = torch.jit.script(model)
    
    # Salva il modello ottimizzato (opzionale)
    torch.jit.save(scripted_model, f"{base_model}_scripted.pt")

    return scripted_model

def infer_with_scripted_model(model, inputs):
    with torch.no_grad():
        return model(**inputs)

scripted_model = load_scripted_model(base_model)

tokenizer = AutoTokenizer.from_pretrained(base_model)

def generate_response(input_text):
    inputs = tokenizer(input_text, return_tensors="pt").to('cuda')

    output = infer_with_scripted_model(scripted_model, inputs)

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

while True:
    user_question = input("Inserisci la tua domanda (o digita 'esci' per uscire): ")

    if user_question.lower() == 'esci':
        print("Uscita dal programma.")
        break

    start_time = time.time()
    response = generate_response(user_question)
    end_time = time.time()

    print(f"Risposta: {response}")
    print(f"Tempo di risposta: {end_time - start_time:.2f} secondi\n")
