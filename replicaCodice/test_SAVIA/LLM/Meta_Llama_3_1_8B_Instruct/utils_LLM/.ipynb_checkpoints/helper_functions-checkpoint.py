import os
import yaml
import re


def load_configs(configs_path="./configs/configs.yml"):
    with open(configs_path) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    return configs


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def get_context(RAG_dataset, context_ids):
    context = ""

    for context_id in context_ids:
        for RAG_dataset_v1 in RAG_dataset.values():
            for RAG_dataset_item in RAG_dataset_v1:
                if RAG_dataset_item["id"] == context_id:
                    for k, v in RAG_dataset_item.items():
                        if type(v) != int:
                            context += "\n- " + k + ": " + v + " "

        context += "\n"

    return context


def clean_answer(text):
    answer_ind = re.search(r"\[/INST\]", text, flags=0).end()
    answer = text[answer_ind:].strip()

    return answer
