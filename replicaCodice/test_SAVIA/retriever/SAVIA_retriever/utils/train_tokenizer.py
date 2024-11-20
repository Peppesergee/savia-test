import json
import os
from transformers import AutoTokenizer
from datasets import load_dataset
import time


def train_tokenizer():
    model_name = "mistralai/Mistral-7B-v0.1"

    orig_tokenizer = AutoTokenizer.from_pretrained(model_name)


    datasets_folder = "./"
    data_path = os.path.join(datasets_folder, "merged_data.json")
    with open(data_path, 'r') as f:
        data = json.load(f)

    all_texts = []

    for k in data.keys():
        for item in data[k][0:]:
            all_texts.append(item['text'])


    batch_size = 500

    training_corpus = (
        all_texts[i : i + batch_size]
        for i in range(0, len(all_texts), batch_size)
    )

    tokenizer = orig_tokenizer.train_new_from_iterator(training_corpus, 52000)


if __name__ == '__main__':

    print("training tokenizer")

    start_time = time.time()

    train_tokenizer()

    elapsed = time.time() - start_time

    print("--- %s seconds ---" % elapsed)

    print("end")
