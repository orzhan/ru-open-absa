from tqdm import tqdm
import torch
import argparse
import json
import os
import pickle
import random
import pandas as pd
from transformers import AutoModelForTokenClassification, AutoConfig, EarlyStoppingCallback, PreTrainedModel, \
    T5EncoderModel, TrainerCallback, pipeline
from transformers import AutoTokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Label reviews using encoder token-level classification model')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input CSV file containing reviews in column "text"')
    parser.add_argument('--output_file', type=str, required=True, help='Output CSV file for the labeled reviews')
    parser.add_argument('--aspect_map', type=str, required=True,
                        help='Input JSON file with aspect map: json array with {name, description} fields')
    parser.add_argument('--load_student_model', type=str, default=None,
                        help='Load a pre-trained student model')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')

    args, unknown = parser.parse_known_args()

    model = AutoModelForTokenClassification.from_pretrained(args.load_student_model)
    tokenizer = AutoTokenizer.from_pretrained(args.load_student_model)

    input_data = pd.read_csv(args.input_file, encoding="utf-8")
    aspect_map = json.load(open(args.aspect_map, "r", encoding="utf-8"))

    token_classification_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="max")

    annotations = []
    # process in batches
    for i in tqdm(range(0, len(input_data), args.batch_size)):
        batch = input_data["text"].iloc[i:i + args.batch_size].tolist()
        batch_annotations = token_classification_pipeline(batch)
        annotations.extend(batch_annotations)

    # Save the annotations
    input_data["annotations"] = annotations
    input_data.to_csv(args.output_file, index=False)














