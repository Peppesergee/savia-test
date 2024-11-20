
def add_context_to_question(sample):
    
    context = sample['context']

    prompt = f""" \nINIZIO_CONTESTO: \n##### \n"""

    for elem in context:
        for k, v in elem.items():
           if v is not None:
               prompt += k + ": " + str(v).replace("\n", " ") + " \n"
    
#        prompt += " \n "
        prompt += "##### \n"

    prompt += "FINE_CONTESTO "

    question_with_context = sample['question'] + prompt

    return question_with_context 

def generate_inference_sample(sample, tokenizer):

    question_with_context = add_context_to_question(sample)

#    print(question_with_context)
    
    sys = """
        Sei l'assistente AI in lingua italiana dell'Assemblea legislativa dell'Emilia-Romagna.
        Rispondi nella lingua usata per la domanda in modo chiaro, semplice ed esaustivo, utilizzando il contesto fornito.
        Se non sono presenti informazioni di contesto, utilizza la conoscenza già acquisita, senza informare l'utente.
        Non fornire informazioni che non sono presenti nel contesto.
        Usa tutte le informazioni fornite nel contesto se sono rilevanti, ed elenca gli atti 
        citati specificando sempre il numero dell'atto e l'oggetto.
        """

    question_with_context = add_context_to_question(sample)

#    print(question_with_context)
    
    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": question_with_context},
#        {"role": "assistant", "content": answer}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    output = tokenizer(prompt, return_tensors="pt")    

    output["prompt"] = prompt

    return output