from transformers import (
    Trainer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    EarlyStoppingCallback,
)
#from trainer.trainer_no_prompt import TrainerNoPrompt
#from dataloader.datacollator_no_prompt import DataCollatorNoPrompt
import os
#from trainer.new_custom_trainer import NewCustomTrainer, NewDataCollator
from trainer.datacollator import DataCollatorIgnorePrompt

# import evaluate
# import numpy as np
# from torch.utils.tensorboard import SummaryWriter

# metric = evaluate.load("perplexity")
# metric = evaluate.load("glue", "mrpc")
# tb_writer = SummaryWriter()


def load_trainer(configs, tokenizer, model, train_dataset, eval_dataset):
    # print(configs['training']['loss']['ignore_prompt'])
    training_configs = configs["training"]

    # create folder for intermediate checkpoints
    # temp_checkpoints_folder = os.path.join(
    #     training_configs['checkpoints_folder'], training_configs['temp_checkpoints_folder']
    # )
    temp_checkpoints_folder = os.path.join(
        training_configs["checkpoints_folder"], training_configs["qlora_folder"], "temp"
    )
    os.makedirs(temp_checkpoints_folder, exist_ok=True)

    if training_configs["early_stopping"]["use_early_stopping"]:
        metric_for_best_model = "eval_loss"
        load_best_model_at_end = training_configs["load_best_model_at_end"]
    else:
        metric_for_best_model = None
        load_best_model_at_end = None

    args = TrainingArguments(
        per_device_train_batch_size = training_configs["per_device_train_batch_size"],
        per_device_eval_batch_size = training_configs["per_device_train_batch_size"],
        gradient_accumulation_steps = training_configs["gradient_accumulation_steps"],
        warmup_steps = training_configs["warmup_steps"],
        max_steps = training_configs["max_steps"],
        learning_rate = float(training_configs["optimizer"]["learning_rate"]),
        lr_scheduler_type = "cosine",
        fp16 = training_configs["optimizer"]["fp16"],
        logging_steps = training_configs["logging"]["logging_steps"],
        save_strategy = "steps",
        save_steps = training_configs["save_steps"],
        output_dir = temp_checkpoints_folder,
        optim = training_configs["optimizer"]["optim"],
        evaluation_strategy = "steps",
        eval_steps = training_configs["eval_steps"],
        load_best_model_at_end = load_best_model_at_end,
#        weight_decay = training_configs["optimizer"]["weight_decay"],
        metric_for_best_model = metric_for_best_model
        # report_to='tensorboard'
    )

    if configs["training"]["loss"]["ignore_prompt"]:
        print("ignoring prompt in loss calculation")
        data_collator = DataCollatorIgnorePrompt(tokenizer, mlm=False)
#        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    else:
        print("keeping prompt in loss calculation")
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    if configs["training"]["early_stopping"]["use_early_stopping"]:
        callbacks = [
            EarlyStoppingCallback(early_stopping_patience=configs["training"]["early_stopping"]["patience"])
        ]
    
    else:
        callbacks = None

    trainer = Trainer(
        model = model,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        args = args,
        data_collator = data_collator,
        callbacks = callbacks,
    )

    return trainer

"""
def load_trainer_new(configs, args, tokenizer, model, train_dataset, eval_dataset):
    
    trainer = NewCustomTrainer(
        model = model,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        args = args,
        data_collator = NewDataCollator(tokenizer, mlm=False),
        callbacks = None,
    )

    return trainer


def load_trainer_no_prompt(
    configs, args, tokenizer, model, train_dataset, eval_dataset
):
    if configs["training"]["early_stopping"]["use_early_stopping"]:
        callbacks = [
            EarlyStoppingCallback(
                early_stopping_patience=configs["training"]["early_stopping"][
                    "patience"
                ]
            )
        ]
    else:
        callbacks = None

    trainer = TrainerNoPrompt(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=args,
        data_collator=DataCollatorNoPrompt(tokenizer, mlm=False),
        # compute_metrics = compute_metrics,
        callbacks=callbacks,
    )

    return trainer


def load_trainer_with_prompt(
    configs, args, tokenizer, model, train_dataset, eval_dataset
):
    if configs["training"]["early_stopping"]["use_early_stopping"]:
        callbacks = [
            EarlyStoppingCallback(
                early_stopping_patience=configs["training"]["early_stopping"][
                    "patience"
                ]
            )
        ]
    else:
        callbacks = None

    trainer = Trainer(
        model = model,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        args = args,
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks = callbacks,
    )

    return trainer
"""

"""
def compute_metrics(eval_preds):

    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    return metric.compute(predictions=predictions, references=labels)
"""
