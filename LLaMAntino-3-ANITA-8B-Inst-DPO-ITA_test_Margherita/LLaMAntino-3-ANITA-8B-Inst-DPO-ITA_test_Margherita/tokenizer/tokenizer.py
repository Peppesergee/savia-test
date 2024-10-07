import os
from transformers import AutoTokenizer


def load_tokenizer(configs):
    tokenizer_folder = os.path.join(
        configs["training"]["checkpoints_folder"], "tokenizer"
    )

    os.makedirs(tokenizer_folder, exist_ok=True)

    if "tokenizer.json" in os.listdir(tokenizer_folder):
        print("loading tokenizer from:", tokenizer_folder)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_folder)
#        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_bos_token = False

#        tokenizer.eos_token_id = 2


    else:
        print("loading tokenizer from HF")
        tokenizer = AutoTokenizer.from_pretrained(
            configs["training"]["load_model"]["pretrained_model"],  
            token = "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"
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_bos_token = False

        print("saving tokenizer in:", tokenizer_folder)
        tokenizer.save_pretrained(tokenizer_folder)

    return tokenizer
