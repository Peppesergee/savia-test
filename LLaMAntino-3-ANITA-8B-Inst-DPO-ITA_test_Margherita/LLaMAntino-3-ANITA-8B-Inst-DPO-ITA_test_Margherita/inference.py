import os

# from datasets import load_from_disk
# from transformers import TextStreamer
import torch
import random
import json
import time

# from peft import PeftModel
import sys

sys.path.append("../")

from utils.helper_functions import load_configs, clean_answer

# from utils.prompts import generate_sample_for_inference
from predictor.predictor import Predictor


def inference_on_test_set(configs, num_questions):

    dataset_folder = configs["dataset"]["dataset_folder"]

    with open(os.path.join(dataset_folder, "IB_dataset_train.json"), "r") as f:
        IB_dataset_train = json.load(f)

    with open(os.path.join(dataset_folder, "IB_dataset_test.json"), "r") as f:
        IB_dataset_test = json.load(f)


    predictor = Predictor(configs)

    res = []

    for num_item in range(0, num_questions):

        print("pred", str(num_item), "/", num_questions)

        ind = random.randint(0, len(IB_dataset_test))
        #print(ind)
        item = IB_dataset_test[ind]
#        print(item['question'])
#        print("-------------")
#        print(item['answer'])

        start_time = time.time()

        pred = predictor.inference(item, use_streamer = False)

        inference_time = time.time() - start_time

        res.append({"question": item['question'],
                    "gt": item['answer'],
                    "pred": pred,
                    "context": item['context'],
                    "inference_time": inference_time
            })

    return res


if __name__ == '__main__':

    configs = load_configs(configs_path="./configs/configs.yml")

    num_questions = 10

    res = inference_on_test_set(configs, num_questions)

    print("saving predictions")

    with open('results.json', 'w', encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
