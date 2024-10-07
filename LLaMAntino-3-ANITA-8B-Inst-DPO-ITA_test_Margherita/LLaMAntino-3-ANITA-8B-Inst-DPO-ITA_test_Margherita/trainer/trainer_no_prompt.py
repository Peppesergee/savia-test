from transformers import Trainer
from torch.nn import CrossEntropyLoss

# import os
import torch


class TrainerIgnorePrompt(Trainer):
    def __init__(self, *args, **kwargs):
        super(TrainerIgnorePrompt, self).__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        batch = {
            k: v
            for k, v in inputs.items()
            if k in ["input_ids", "attention_mask", "labels"]
        }
        # print(batch.keys())

        labels = inputs.get("labels")
        outputs = model(**batch)
        logits = outputs.get("logits")

        prompt_mask = inputs.get("prompt_mask")

        # clone labels tensor and set to -100, otherwise you need to create a tensor in the correct device manually
        ignore_mask = torch.clone(labels)
        ignore_mask[...] = -100

        # set labels to -100 in prompt
        labels = torch.where(prompt_mask > 0, ignore_mask, labels)

        # shift by 1 the labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # reshape
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits, shift_labels)

        return (loss, outputs) if return_outputs else loss
