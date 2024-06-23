import argparse
import json
import os
import pickle
import random

import json_repair
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from datasets import Dataset, concatenate_datasets

from peft import LoraConfig, TaskType, get_peft_model
from scipy.stats import entropy
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoConfig, EarlyStoppingCallback, PreTrainedModel, \
    T5EncoderModel, TrainerCallback
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments
from transformers.modeling_outputs import TokenClassifierOutput

from llm import label_with_llm_concurrently, llm_parser
from utils import align_tokens_and_labels


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100, reduction='mean', num_classes=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                assert len(alpha) == num_classes, "alpha vector must match the number of classes"
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            elif isinstance(alpha, (float, int)):
                self.alpha = torch.tensor([alpha] * num_classes, dtype=torch.float32)
            else:
                raise TypeError("alpha must be float, int, list, or np.ndarray")
        else:
            self.alpha = torch.tensor([1.0] * num_classes, dtype=torch.float32)
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        # Ensure alpha is properly broadcasted
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
        else:
            alpha = torch.ones(self.num_classes, device=inputs.device, dtype=torch.float32)

        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float().to(inputs.device)
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets_one_hot, reduction='none')

        # Apply alpha weighting
        alpha_factor = alpha[None, None, :].expand_as(BCE_loss)  # Ensure it's broadcastable

        pt = torch.exp(-BCE_loss)
        focal_loss = alpha_factor * ((1 - pt) ** self.gamma) * BCE_loss

        # Masking ignore_index
        active = targets.unsqueeze(-1).expand_as(BCE_loss) != self.ignore_index
        focal_loss = focal_loss * active

        if self.reduction == 'mean':
            return focal_loss[active].mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


from transformers import Trainer


class CustomTrainer(Trainer):
    def __init__(self, *args, focal_loss_alpha, **kwargs):
        self.focal_loss_alpha = focal_loss_alpha
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        is_training = model.training

        if is_training:
            outputs = model(**inputs)
        else:
            with torch.no_grad():
                outputs = model(**inputs)

        logits = outputs.logits
        labels = inputs['labels']
        # Instantiate your custom focal loss
        # print(f"num_classes={len(focal_loss_alpha)}")
        # Assume you have a method to instantiate your focal loss only once, not every time
        if not hasattr(self, 'focal_loss'):
            self.focal_loss = FocalLoss(alpha=self.focal_loss_alpha, num_classes=len(self.focal_loss_alpha))

        loss = self.focal_loss(logits, labels)

        return (loss, outputs) if return_outputs else loss


def compute_metrics(p, labels):
    predictions = np.argmax(p.predictions, axis=2)
    true_labels = p.label_ids.flatten()
    pred_labels = predictions.flatten()

    ignore_index = -100
    true_labels = true_labels[true_labels != ignore_index]
    pred_labels = pred_labels[true_labels != ignore_index]

    # Exclude the 'other' label for class-wise metrics calculation
    # 'other' label is assumed to have the id 0
    class_labels = [label for label in labels.values() if label != labels['other']]

    # Calculate class-wise metrics excluding 'other'
    class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, labels=class_labels, average=None)

    # Calculate weighted metrics excluding 'other'
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, labels=class_labels, average='weighted')

    metrics = {
        "weighted_precision_excluding_other": weighted_precision,
        "weighted_recall_excluding_other": weighted_recall,
        "weighted_f1_excluding_other": weighted_f1,
    }

    # Update the metrics dict with individual class metrics
    # Exclude 'other' from aspect names for individual class metrics
    label_to_aspect = {value: key for key, value in labels.items() if key != "other"}
    for label in class_labels:
        aspect = label_to_aspect[label]
        try:
            metrics[f"{aspect}_precision"] = class_precision[label - 1]
            metrics[f"{aspect}_recall"] = class_recall[label - 1]
            metrics[f"{aspect}_f1"] = class_f1[label - 1]
        except Exception as ex:
            print(ex)
            pass

    return metrics


def annotations_to_dataset(annotations_df, tokenizer, labels):
    tokenized_texts = []
    aligned_labels = []
    skipped = 0
    for i, row in tqdm(annotations_df.iterrows(), total=len(annotations_df), desc="Converting annotations to dataset"):
        tokenizer_outputs = tokenizer(row['text'], truncation=True, padding='max_length', max_length=256,
                                      return_tensors="pt", return_offsets_mapping=True)
        input_ids, input_labels, worst_score = align_tokens_and_labels(
            {'input_ids': tokenizer_outputs['input_ids'][0], 'tokens': tokenizer_outputs.tokens(),
             "offset_mapping": tokenizer_outputs["offset_mapping"][0]}, row['text'], row['annotations'], labels)
        if worst_score < 75:
            skipped += 1
            continue
        tokenized_texts.append(input_ids)
        aligned_labels.append(input_labels)
    data = {"input_ids": tokenized_texts, "labels": aligned_labels}
    dataset = Dataset.from_dict(data)
    if skipped > 0:
        print(f"Skipped {skipped} texts of {len(annotations_df)} with low matching scores")
    return dataset


def initialize_model(model_name, use_t5_encoder, num_labels):
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    if use_t5_encoder:
        model = T5EncoderForTokenClassification(config=config, num_labels=num_labels)
    else:
        model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Initialized the model {model_name}")
    return tokenizer, model


def sample_uncertain_instances(model, tokenizer, unlabeled_texts, num_samples, batch_size=8):
    all_entropy_values = []

    for i in tqdm(list(range(0, len(unlabeled_texts), batch_size)), desc="Processing samples"):
        batch_texts = unlabeled_texts[i:i + batch_size]
        tokenized_batch = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**{k: v.to(model.device) for k, v in tokenized_batch.items()})
            predictions = torch.softmax(outputs.logits, dim=-1)

        # Calculate entropy for each instance in the batch
        batch_entropy = entropy(predictions.cpu().numpy(), base=2, axis=2).mean(axis=1)
        all_entropy_values.extend(batch_entropy)

    # Select instances with the highest entropy
    uncertain_indices = np.argsort(all_entropy_values)[-num_samples:]

    return [unlabeled_texts[i] for i in uncertain_indices]


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"


class UnfreezeCallback(TrainerCallback):
    def __init__(self, unfreeze_after_epoch):
        self.unfreeze_after_epoch = unfreeze_after_epoch

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.unfreeze_after_epoch is not None and state.epoch == self.unfreeze_after_epoch:
            print("Unfreezing all layers")
            for param in model.base_model.parameters():
                param.requires_grad = True
            for param in model.classifier.parameters():
                param.requires_grad = True
            control.should_training_stop = False
            print(print_number_of_trainable_model_parameters(model))


def train_model(train_data, model, tokenizer, labels, args):
    if args.lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            task_type=TaskType.TOKEN_CLS
        )
        model = get_peft_model(model, lora_config)

    if args.freeze_layers is not None:
        for param in model.base_model.parameters():
            param.requires_grad = False
        # unfreeze classification head
        for param in model.classifier.parameters():
            param.requires_grad = True
        # unfreeze also last N layers
        if args.freeze_layers > 0:
            for param in model.base_model.encoder.layer[-args.freeze_layers:].parameters():
                param.requires_grad = True

    print(print_number_of_trainable_model_parameters(model))

    all_labels = []
    for lab in train_data_part_labeled['labels']:
        for l in lab:
            all_labels.append(l)
    label_counts = pd.Series(all_labels).value_counts()
    total_count = label_counts.sum()
    class_frequency = (label_counts / total_count).to_dict()
    alpha = [1 - class_frequency[label] for label in sorted(class_frequency)]
    focal_loss_alpha = [a / sum(alpha) for a in alpha]

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        evaluation_strategy='epoch',
        logging_dir="./logs",
        logging_steps=min(len(train_data) // args.train_batch_size // 10, 50),
        dataloader_num_workers=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_strategy="epoch",  # Save a model checkpoint at the end of each epoch.
        save_total_limit=3,  # Only save the last 3 checkpoints to save disk space.
        load_best_model_at_end=True,  # Load the best model (lowest validation loss) at the end of training.
        metric_for_best_model="weighted_f1_excluding_other",  # Use validation loss to determine the best model.
        greater_is_better=True,  # Lower validation loss is better.
        lr_scheduler_type=args.lr_scheduler,
    )
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    print("Prepared for training")

    callbacks = []
    if args.early_stopping is not None:
        callbacks.append(EarlyStoppingCallback(args.early_stopping))
    if args.unfreeze_after_epoch is not None:
        callbacks.append(UnfreezeCallback(args.unfreeze_after_epoch))

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, labels),
        tokenizer=tokenizer,
        focal_loss_alpha=focal_loss_alpha,
        callbacks=callbacks
    )

    trainer.train()
    model.config.id2label = {v: k for k, v in labels.items()}
    model.config.label2id = {k: v for k, v in labels.items()}

    return model, trainer.state.best_metric


def active_learning_loop(model, tokenizer, train_dataset, unlabeled_texts, labels, aspect_map, llm_args,
                         iterations=5, samples_per_iteration=1000, labeling_ratio=25,
                         active_learning_strategy="continue-train"):
    for iteration in range(iterations):
        print(f"Iteration {iteration + 1}/{iterations}")

        random.seed(300)
        unlabeled_texts_part = random.choices(unlabeled_texts, k=samples_per_iteration * labeling_ratio)
        print(f"Will selected uncertain samples from {len(unlabeled_texts_part)} texts")
        # Use the model to label a batch of unlabeled data based on uncertainty
        uncertain_samples = sample_uncertain_instances(model, tokenizer, unlabeled_texts_part,
                                                       num_samples=samples_per_iteration,
                                                       batch_size=args.eval_batch_size)

        # Remove the selected samples from the unlabeled pool
        unlabeled_texts = [text for text in unlabeled_texts if text not in uncertain_samples]

        print(f"Prepared {len(uncertain_samples)} uncertain texts")

        uncertain_samples_df = pd.DataFrame({'text': uncertain_samples})

        # Label the uncertain samples with the LLM or other means (e.g., manual labeling, semi-supervised methods)
        # Here, replace `label_with_llm` with your method of obtaining labels for the selected samples
        uncertain_samples_df_labeled = label_with_llm_concurrently(uncertain_samples_df, llm_args, aspect_map)

        # Combine the newly labeled data with the existing training data
        new_data = annotations_to_dataset(uncertain_samples_df_labeled, tokenizer, labels)

        if active_learning_strategy == "fine-tune":
            train_dataset = new_data
        else:
            train_dataset = concatenate_datasets([train_dataset, new_data])

        print(f"New training dataset size: {len(train_dataset)}")

        if active_learning_strategy == "retrain":
            tokenizer, model = initialize_model(args.student_model_name, args.use_t5_encoder, num_labels=len(labels))

        model, best_metrics = train_model(train_dataset, model, tokenizer, labels, args)

        print(
            f"Trained and saving to ./models/active-learning-{iteration}, model weighted f1: {best_metrics}")

        # Now save your model and tokenizer
        model.save_pretrained(f'./models/active-learning-{iteration}')
        tokenizer.save_pretrained(f'./models/active-learning-{iteration}')

    print("Active learning loop completed.")


# Example usage
# initial_training_data, validation_data = prepare_datasets()


class T5EncoderForTokenClassification(PreTrainedModel):
    def __init__(self, config, num_labels=None):
        super().__init__(config)
        self.num_labels = num_labels if num_labels is not None else config.num_labels

        # Load the T5 Encoder
        self.t5_encoder = T5EncoderModel(config)

        # Classification head
        self.classification_head = nn.Linear(config.d_model, self.num_labels)

        # Initialize weights and biases for better performance
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, labels=None, return_dict=False):
        # Process inputs through the T5 encoder
        outputs = self.t5_encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Get the last hidden states
        sequence_output = outputs.last_hidden_state

        # Apply the classification head
        logits = self.classification_head(sequence_output)

        # If labels are provided, calculate the loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Label reviews using LLM')
    parser.add_argument('--input_file', type=str, required=True,
                            help='Input CSV file containing reviews in column "text"')
    parser.add_argument('--max_input_rows', type=int, default=None,
                        help='Maximum number of input rows to process')
    parser.add_argument('--validation_input_file', type=str, required=True,
                        help='Input CSV file containing validation reviews in column "text"')
    parser.add_argument('--aspect_map', type=str, required=True,
                        help='Input JSON file with aspect map: json array with {name, description} fields')
    parser.add_argument('--output_file', type=str, required=True, help='Output CSV file for the labeled reviews')
    parser.add_argument('--student_model_name', type=str, default="cointegrated/rubert-tiny2",
                        help='Model name for token classification')
    parser.add_argument('--active_learning_strategy', type=str, default="retrain",
                        help='Active learning strategy: retrain, continue-train, fine-tune')
    parser.add_argument('--active_learning_iterations', type=int, default=5,
                        help='Number of active learning iterations')
    parser.add_argument('--samples_per_iteration', type=int, default=1000,
                        help='Number of samples to label in each iteration')
    parser.add_argument('--load_student_model', type=str, default=None,
                        help='Load a pre-trained student model')
    parser.add_argument('--continue_training', type=bool, default=False)
    parser.add_argument('--lora', action='store_true', help='Use LoRA for training')
    parser.add_argument('--lora_r', type=int, default=32, help='LoRA hyperparameter r')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA hyperparameter alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout rate')
    parser.add_argument('--lora_bias', action='store_true', help='Use bias in LoRA', default="lora_only")
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--train_batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='Evaluation batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--lr_scheduler', type=str, default="linear", help='Learning rate scheduler')
    parser.add_argument('--warmup_steps', type=int, default=0, help='Warmup steps')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--early_stopping', type=int, help='Early stopping patience')
    parser.add_argument('--use_t5_encoder', type=bool, default=False, help='Use T5 encoder for token classification')
    parser.add_argument('--max_text_length', type=int, default=512)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--freeze_layers', type=int, help='Number of layers to freeze, 0 to freeze head only')
    parser.add_argument('--unfreeze_after_epoch', type=int, help='Unfreeze layers after this epoch')

    args, unknown = parser.parse_known_args()
    llm_args = llm_parser().parse_args(unknown)

    with open(args.aspect_map, "r", encoding="utf-8") as f:
        aspect_map = json.load(f)
    # aspect_map is a dict of dicts {name, description}, and we need to use names + sentiment(positive/negative),
    # to create labels, 2 x aspect count, and special label other with key 0
    aspect_names = [aspect['name'] for aspect in aspect_map.values()]
    labels = {aspect + "_positive": i * 2 + 1 for i, aspect in enumerate(aspect_names)}
    labels.update({aspect + "_negative": i * 2 + 2 for i, aspect in enumerate(aspect_names)})
    labels['other'] = 0

    train_data = pd.read_csv(args.input_file, encoding="utf-8")
    if args.max_text_length is not None:
        print(f"Filtering out reviews longer than {args.max_text_length} characters. Original size: {len(train_data)}")
        train_data = train_data[train_data["text"].str.len() <= args.max_text_length]
        print(f"Filtered size: {len(train_data)}")
    if args.max_input_rows is not None:
        train_data_part = train_data.head(args.max_input_rows)
    else:
        train_data_part = train_data
        if args.active_learning_iterations > 0:
            print("Warning: using all data to train the model. No data left for active learning")

    train_data_part.reset_index(drop=True, inplace=True)

    tokenizer, model = initialize_model(args.student_model_name, args.use_t5_encoder, num_labels=len(labels))

    gold_labeled = pd.read_csv(args.validation_input_file, encoding="utf-8")
    gold_labeled['annotations'] = [json_repair.loads(x.replace('"', '\\"').replace("'", '"')) for x in
                                   gold_labeled['annotations']]
    valid_data = annotations_to_dataset(gold_labeled, tokenizer, labels)

    train_data_part_annotations = label_with_llm_concurrently(train_data_part, llm_args, aspect_map)
    print("Count of annotated rows: ", len(train_data_part_annotations))

    if os.path.exists("train_data_part_labeled-cache.pkl"):
        with open("train_data_part_labeled-cache.pkl", "rb") as f:
            train_data_part_labeled = pickle.load(f)
    else:
        train_data_part_labeled = annotations_to_dataset(train_data_part_annotations, tokenizer,
                                                         labels)
        with open("train_data_part_labeled-cache.pkl", "wb") as f:
            pickle.dump(train_data_part_labeled, f)

    if args.load_student_model is not None and os.path.exists(args.load_student_model):
        model = AutoModelForTokenClassification.from_pretrained(args.load_student_model)
    if not args.load_student_model or args.continue_training:
        model, best_metrics = train_model(train_data_part_labeled, model, tokenizer, labels, args)
        model.save_pretrained(f'./models/initial')
        tokenizer.save_pretrained(f'./models/initial')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    unlabeled_texts = \
        train_data[~train_data['text'].isin(gold_labeled['text']) & ~train_data['text'].isin(train_data_part['text'])][
            'text'].values.tolist()

    active_learning_loop(model=model, tokenizer=tokenizer, train_dataset=train_data_part_labeled,
                         unlabeled_texts=unlabeled_texts, labels=labels, aspect_map=aspect_map, llm_args=llm_args,
                         iterations=args.active_learning_iterations,
                         active_learning_strategy=args.active_learning_strategy,
                         samples_per_iteration=args.samples_per_iteration)
