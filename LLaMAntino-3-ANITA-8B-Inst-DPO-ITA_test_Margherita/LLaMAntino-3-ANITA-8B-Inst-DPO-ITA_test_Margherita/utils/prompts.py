
def generate_training_sample_no_prompt(sample, tokenizer):
#    question = sample["question"]

    context = sample["context"]
    answer = " " + sample["answer"].strip() # + " "

    sys = "Sei l'assistente AI in lingua italiana dell'Assemblea legislativa." \
        " Rispondi nella lingua usata per la domanda in modo chiaro, semplice ed esaustivo, utilizzando il contesto fornito."

    question_with_context = add_context_to_question(sample)

#    print(question_with_context)
    
    messages_prompt = [
        {"role": "system", "content": sys},
        {"role": "user", "content": question_with_context},
#        {"role": "assistant", "content": answer}
    ]

    chat_template_prompt = tokenizer.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=False)
    chat_template_prompt_ids = tokenizer(chat_template_prompt)["input_ids"]
    prompt_mask = [0 for _ in range(0, len(chat_template_prompt_ids))]

    messages_answer = [
        {"role": "assistant", "content": answer}
    ]

    #workaround, replace manually BOS token, with the current transformer version there is no other way
    chat_template_answer = tokenizer.apply_chat_template(messages_answer, tokenize=False, add_generation_prompt=False).replace("<|begin_of_text|>", "")
    chat_template_answer_ids = tokenizer(chat_template_answer)["input_ids"]
    prompt_mask += [1 for _ in range(0, len(chat_template_answer_ids))]

    prompt = chat_template_prompt + chat_template_answer
    input_ids = chat_template_prompt_ids + chat_template_answer_ids

    attention_mask = [1 for _ in range(0, len(input_ids))]

#    output = tokenizer(prompt)    
#    output['input_ids'] = output['input_ids'][:-1]

    output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_ids": prompt_mask,
            "prompt": prompt,
            "context": context
        }

    return output



def generate_training_sample(sample, tokenizer):

    context = sample["context"]
    answer = " " + sample["answer"].strip() # + " "

    sys = "Sei l'assistente AI in lingua italiana dell'Assemblea legislativa." \
        " Rispondi nella lingua usata per la domanda in modo chiaro, semplice ed esaustivo, utilizzando il contesto fornito."

    question_with_context = add_context_to_question(sample)

#    print(question_with_context)
    
    messages_prompt = [
        {"role": "system", "content": sys},
        {"role": "user", "content": question_with_context},
        {"role": "assistant", "content": answer}
    ]

    prompt = tokenizer.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=False)

    output = tokenizer(prompt)    

    output["prompt"] = prompt
#    output["question_with_context"] = question_with_context
    output["context"] = context

    return output



def add_context_to_question(sample):
    
    context = sample['context']

    prompt = f""" \nINIZIO_CONTESTO: \n##### \n"""

    for elem in context:
        for k, v in elem.items():
           if v is not None:
               prompt += k + ": " + str(v).replace("\n", " ") + " \n"
    
#        prompt += " \n "
        prompt += "##### \n"

#    prompt = prompt.strip()

    prompt += "FINE_CONTESTO "

#    print(prompt)

    question_with_context = sample['question'] + prompt

    return question_with_context 


"""
def generate_training_sample_with_prompt_mask(sample, tokenizer):
#    print("Here")

    context = sample["context"]
    answer = " " + sample["answer"].strip() # + " "

#    sys = "Sei l'assistente AI in lingua italiana dell'Assemblea legislativa." \
#        " Rispondi nella lingua usata per la domanda in modo chiaro, semplice ed esaustivo, utilizzando il contesto fornito."

    question_with_context = add_context_to_question(sample)

#    print(question_with_context)
    
    messages = [
#        {"role": "system", "content": sys},
        {"role": "user", "content": question_with_context},
#        {"role": "assistant", "content": answer}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    output = tokenizer(prompt)    

    prompt_ids = tokenizer(prompt)["input_ids"]
    # set prompt mask to 0 in prompt
    prompt_mask = [0 for _ in range(0, len(prompt_ids))]

    answer = answer + " </s>"
    answer_ids = tokenizer(answer)["input_ids"]
    # set prompt mask to 1 in answer
    prompt_mask += [1 for _ in range(0, len(answer_ids))]

    input_ids = prompt_ids + answer_ids
    attention_mask = [1 for _ in range(0, len(input_ids))]
    full_prompt = prompt + answer

    output = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label_ids": prompt_mask,
        "prompt": full_prompt,
#        "question": question,
#        "context": context,
#        "answer": answer,
    }


#    output['input_ids'] = output['input_ids'][:-1]

#    output = {}
    
#    output = prompt
#    output["prompt"] = prompt
#    output["question_with_context"] = question_with_context
#    output["context"] = context


    question = sample["question"]
    context = sample["context"]
    answer = sample["answer"]

    prompt_ids = tokenizer(prompt)["input_ids"]
    # set prompt mask to 0 in prompt
    prompt_mask = [0 for _ in range(0, len(prompt_ids))]

    answer = answer + " </s>"
    answer_ids = tokenizer(answer)["input_ids"]
    # set prompt mask to 1 in answer
    prompt_mask += [1 for _ in range(0, len(answer_ids))]

    input_ids = prompt_ids + answer_ids
    attention_mask = [1 for _ in range(0, len(input_ids))]

    #    output = tokenizer(prompt)
    #    output = {}
    full_prompt = prompt + answer

    output = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label_ids": prompt_mask,
        "prompt": full_prompt,
        "question": question,
        "context": context,
        "answer": answer,
    }

    return output
"""

def generate_inference_sample(question, tokenizer):
        
    sys = "Sei l'assistente AI in lingua italiana dell'Assemblea legislativa." \
        " Rispondi nella lingua usata per la domanda in modo chiaro, semplice ed esaustivo, utilizzando il contesto fornito."

    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": question}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    output = tokenizer(prompt, return_tensors="pt")    

    return output
