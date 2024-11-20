import os
from transformers import AutoTokenizer


def load_tokenizer(configs):

    model_name = configs["model"]["model_name"]
    tokenizer_folder = os.path.join(configs["model"]["checkpoints_folder"], model_name.split("/")[-1], "tokenizer")

#    embedding_model_path = os.path.join(self.configs['models']['models_folder'], embedding_model_name.split("/")[-1])

    os.makedirs(tokenizer_folder, exist_ok=True)

    if "tokenizer.json" in os.listdir(tokenizer_folder):
        print("loading tokenizer from:", tokenizer_folder)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_folder)
        tokenizer.add_bos_token = False

    else:
        print("loading tokenizer from HF")
        tokenizer = AutoTokenizer.from_pretrained(
            configs["model"]["model_name"],  
            token = configs['HF_auth_token']
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_bos_token = False

        print("saving tokenizer in:", tokenizer_folder)
        tokenizer.save_pretrained(tokenizer_folder)

    return tokenizer
