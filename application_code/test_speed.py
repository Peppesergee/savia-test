import time
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from torch import cuda

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
            device_map="auto",  # Usa la GPU se disponibile
        )
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            token="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"
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_bos_token = False
        print("Modello caricato correttamente.")
    else:
        print("Modello già caricato, utilizzo quello esistente.")

sys = (
    "Sei un assistente AI per la lingua Italiana di nome LLaMAntino-3 ANITA "
    "(Advanced Natural-based interaction for the ITAlian language). "
    "Rispondi nella lingua usata per la domanda in modo chiaro, semplice ed esaustivo."
)

# Inizializza il modello solo se necessario
initialize_model()

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

    # Ottimizzazione: Prealloca il prompt senza template complesso
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)

    # Ottimizzazione della tokenizzazione
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to('cuda')

    # Generazione ottimizzata: riduzione max_new_tokens e temperature più alta
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,  # Limita la lunghezza generata
        temperature=0.7,  # Un po' più alta per generare più velocemente
        top_p=0.9,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,  # Campiona in modo da ridurre la complessità
    )

    sequences = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    end_time = time.time()

    response_time = end_time - start_time
    for seq in sequences:
        print(f"Risposta: {seq}")
    
    print(f"Tempo di risposta: {response_time:.2f} secondi\n")

# Limitazione dei token generati (max_new_tokens=512): riduce la lunghezza massima di output.
# Incremento della temperatura (temperature=0.7): leggermente più alta per velocizzare la generazione.
# Utilizzo di do_sample=True: campionamento casuale anziché cercare di ottimizzare con il fascio di ricerca.
# Movimento diretto dei tensori su GPU: inputs.to('cuda'), per evitare la duplicazione dei dati.
