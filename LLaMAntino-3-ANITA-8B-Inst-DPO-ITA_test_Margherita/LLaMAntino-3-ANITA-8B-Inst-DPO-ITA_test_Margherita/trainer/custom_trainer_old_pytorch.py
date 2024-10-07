from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler, DataCollatorForLanguageModeling
import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset
import numpy as np
from torch.nn import CrossEntropyLoss
import random
from accelerate import Accelerator


class CustomTrainer:
    def __init__(self, train_set, val_set, tokenizer, model):
        print("init")

        self.tokenizer = tokenizer
        self.model = model
        self.train_set, self.val_set = self.create_dataset(train_set, val_set)
        self.train_dataloader, self.val_dataloader = self.create_dataloader(
            self.train_set, self.val_set
        )
        self.accelerator = Accelerator()

    def create_dataset(self, train_set, val_set):
        train_set = CustomDataset(train_set, self.tokenizer)
        val_set = CustomDataset(val_set, self.tokenizer)

        return train_set, val_set

    def create_dataloader(self, train_dataset, valid_dataset):
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=False,
            collate_fn=self.data_collator,
            pin_memory=True,
        )
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=1, shuffle=False, collate_fn=None, pin_memory=True
        )

        return train_dataloader, valid_dataloader

    def data_collator(self, batch):
        """
        Function to pad inputs (right-padding)
        """

        batch_dim = len(batch)
        max_len = max([len(x["input_ids"]) for x in batch])

        input_ids_padded = []
        attention_mask_padded = []
        prompt_mask_padded = []

        for ind_batch in range(0, batch_dim):
            batch_elem = batch[ind_batch]
            batch_elem_len = len(batch_elem["input_ids"])

            input_ids_padded.append(
                batch_elem["input_ids"]
                + [self.tokenizer.pad_token_id] * (max_len - batch_elem_len)
            )
            attention_mask_padded.append(
                batch_elem["attention_mask"] + [0] * (max_len - batch_elem_len)
            )
            prompt_mask_padded.append(
                batch_elem["prompt_mask"] + [0] * (max_len - batch_elem_len)
            )

        input_ids_padded = np.asarray(input_ids_padded)
        attention_mask_padded = np.asarray(attention_mask_padded)

        out_batch = {
            "input_ids": torch.as_tensor(input_ids_padded),
            "attention_mask": torch.as_tensor(attention_mask_padded),
            "prompt_mask": torch.as_tensor(prompt_mask_padded),
        }

        return out_batch

    def train(self):
        """
        training loop
        """
        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        num_epochs = 3
        num_training_steps = num_epochs * len(self.train_dataloader)

        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        #        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        #        print("using device:", device)

        #        self.model.to(device)

        progress_bar = tqdm(range(num_training_steps))

        self.model.train()

        (
            self.train_dataloader,
            self.val_dataloader,
            self.model,
            optimizer,
        ) = self.accelerator.prepare(
            self.train_dataloader, self.val_dataloader, self.model, optimizer
        )

        #        for epoch in range(num_epochs):
        for ind, batch in enumerate(self.train_dataloader):
            if ind > 20:
                break

            labels = batch["input_ids"]  # .to(device)
            prompt_mask = batch["prompt_mask"]  # .to(device)

            #            print("ind_first_token", ind_first_token)
            #            input_keys = ['input_ids', 'attention_mask']

            #            batch = {k: v.to(device) for k, v in batch.items() if k in input_keys}
            batch = {
                k: v for k, v in batch.items() if k in ["input_ids", "attention_mask"]
            }

            outputs = self.model(**batch)
            logits = outputs.get("logits")
            loss = self.custom_loss(labels, logits, prompt_mask)
            #            loss.backward()
            self.accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            print(loss)

        return

    def custom_loss(self, labels, logits, prompt_mask):
        """
        Ignore prompt token
        in loss calculation
        """

        # set labels to -100 in prompt
        labels = torch.where(prompt_mask == 0, -100, labels)

        # shift by 1 the labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # reshape
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits, shift_labels)

        #        return (loss, outputs) if return_outputs else loss

        return loss


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, shuffle=True):
        self.data = data
        self.tokenizer = tokenizer
        self.shuffle = shuffle

    #        if self.shuffle:
    #            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        output = self.generate_input(self.data[idx])

        return output

    def generate_input(self, sample):
        """
        Generate prompt, input_ids, attention mask
        and prompt mask (to remove prompt in loss calculation)
        """
        prompt = (
            f"""<s> [INST] Fornisci una risposta utilizzando il contesto seguente: """
        )
        prompt += " [CONTEXT] " + sample["context"] + " [/CONTEXT] "
        prompt += sample["question"] + " [/INST] "

        answer = sample["answer"] + " </s>"

        prompt_ids = self.tokenizer(prompt)["input_ids"]
        prompt_mask = [0 for _ in range(0, len(prompt_ids))]

        answer_ids = self.tokenizer(answer)["input_ids"]
        prompt_mask += [1 for _ in range(0, len(answer_ids))]

        input_ids = prompt_ids + answer_ids
        attention_mask = [1 for _ in range(0, len(input_ids))]

        full_prompt = prompt + answer

        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt_mask": prompt_mask,
            "prompt": full_prompt,
        }

        return output
