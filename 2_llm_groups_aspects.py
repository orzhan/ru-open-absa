#!/usr/bin/env python
# coding: utf-8
import argparse
import json
from collections import defaultdict, Counter

import json_repair
import numpy as np
import pandas as pd
import torch
from openai import OpenAI
from scipy.cluster.hierarchy import linkage, fcluster
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from llm import llm_parser, query_llm_for_summaries, query_llm_for_group_aspects


def extract_aspects_with_examples(model_answers):
    aspects_with_examples = defaultdict(lambda: {"count": 0, "examples": {"positive": [], "negative": []}})
    for answer_str in model_answers:
        answer = json_repair.loads(answer_str.replace("'", "\""))
        for annotation in answer:
            aspect = annotation.get('aspect')
            sentiment = annotation.get('sentiment')
            if aspect:
                aspects_with_examples[aspect]["count"] += 1
                # Collect up to 5 examples, balancing positive and negative sentiments
                if annotation.get('text') not in aspects_with_examples[aspect]["examples"]["positive"] and \
                        annotation.get('text') not in aspects_with_examples[aspect]["examples"]["negative"]:
                    if sentiment in ["positive", "negative"]:
                        if len(aspects_with_examples[aspect]["examples"][sentiment]) < 5:
                            aspects_with_examples[aspect]["examples"][sentiment].append(annotation.get('text'))

    # Balance the examples to have a mix of positive and negative
    for aspect, data in aspects_with_examples.items():
        positive_examples = data["examples"]["positive"]
        negative_examples = data["examples"]["negative"]
        balanced_examples = []

        # Try to get an equal number of positive and negative examples, up to 5 total
        for _ in range(min(5, len(positive_examples) + len(negative_examples))):
            if positive_examples:
                balanced_examples.append(positive_examples.pop(0))
            if negative_examples:
                balanced_examples.append(negative_examples.pop(0))

        # Replace the examples with the balanced ones
        aspects_with_examples[aspect]["examples"] = balanced_examples

    return aspects_with_examples


def parse_gpt_response(response_text):
    # Parse the JSON response from GPT into a Python dictionary
    try:
        response_text = response_text[response_text.find("["):]
        response_text = response_text[:response_text.rfind("]") + 1]
        response_data = json_repair.loads(response_text.replace("'", '"'))
        return response_data
    except json.JSONDecodeError:
        print("Failed to decode GPT response")
        return []


# Function to extract aspects from model_answers
def extract_aspects(model_answers):
    aspects = []
    for answer in model_answers:
        parsed_answer = json_repair.loads(answer.replace("'", "\""))
        for annotation in parsed_answer:
            aspect = annotation.get('aspect') + ": " + annotation.get('text')
            if aspect:
                aspects.append(aspect)
    return aspects


def main(input_file, num_aspects, output_file, max_input_length, llm_args=None):
    # Load data
    df = pd.read_csv(input_file)
    annotations = df['annotations'].tolist()

    gpt_response = query_llm_for_group_aspects(annotations, num_aspects, max_input_length, llm_args)
    print(gpt_response)

    # Parse the GPT response
    result = parse_gpt_response(gpt_response)
    print(result)

    # result_str_keys = {str(k): v for k, v in result.items()}
    # Save to output file
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

    # Print result to stdout
    print(json.dumps(result, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Cluster review aspects into top-level aspects and optionally into sub-aspects.')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input CSV file containing review aspects in a column named "annotations"')
    parser.add_argument('--num_aspects', type=int, required=True, help='Number of top-level aspects to cluster')
    parser.add_argument('--max_input_length', type=int, default=2048, help='Maximum context length for LLM')

    parser.add_argument('--output_file', type=str, required=False, help='Output JSON file for the clustered aspects')

    args, unknown = parser.parse_known_args()
    llm_args = llm_parser().parse_args(unknown)
    main(args.input_file, args.num_aspects, args.output_file, args.max_input_length, llm_args)
