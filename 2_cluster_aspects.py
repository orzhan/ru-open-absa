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

from llm import llm_parser, query_llm_for_summaries


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
        response_text = response_text[response_text.find("{"):]
        response_text = response_text[:response_text.rfind("}") + 1]
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


# Function to get embeddings
def get_embedding(text, tokenizer, model):
    text = f"В отзывах часто упоминается характеристика: {text}, которая получает оценку за качество и удовлетворенность клиентов."

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def main(input_file, num_aspects, num_subaspects, output_file, summarize=False, llm_args=None):
    # Load data
    df = pd.read_csv(input_file)
    model_answers = df['annotations'].tolist()
    aspects = extract_aspects(model_answers)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
    model = AutoModel.from_pretrained("intfloat/multilingual-e5-small")

    # Generate embeddings with tqdm progress bar
    print("Generating embeddings...")
    embeddings = np.vstack([get_embedding(aspect, tokenizer, model) for aspect in tqdm(aspects)])

    # Perform hierarchical clustering for top-level aspects
    Z = linkage(embeddings, 'ward')
    top_clusters = fcluster(Z, t=num_aspects, criterion='maxclust')

    # Organize aspects into top-level clusters and remove duplicates
    top_cluster_mapping = defaultdict(set)
    for aspect, cluster_label in zip(aspects, top_clusters):
        top_cluster_mapping[cluster_label].add(aspect)

    # Function to get the majority vote aspect name
    def get_majority_aspect_name(aspects):
        aspect_counter = Counter(aspects)
        return aspect_counter.most_common(1)[0][0]

    result = {}

    # Perform hierarchical clustering for sub-aspects if needed
    if num_subaspects:
        for cluster_label, cluster_aspects in top_cluster_mapping.items():
            # Skip sub-clustering if there's only one aspect in the top-level cluster
            if len(cluster_aspects) <= 1:
                majority_name = get_majority_aspect_name(cluster_aspects)
                result[str(cluster_label)] = {majority_name: list(cluster_aspects)}
                continue

            cluster_aspects_list = list(cluster_aspects)
            sub_embeddings = np.vstack(
                [get_embedding(aspect, tokenizer, model) for aspect in tqdm(cluster_aspects_list)])

            # Proceed with sub-clustering if there are multiple aspects
            Z_sub = linkage(sub_embeddings, 'ward')
            sub_clusters = fcluster(Z_sub, t=num_subaspects, criterion='maxclust')
            sub_cluster_mapping = defaultdict(set)
            for sub_aspect, sub_cluster_label in zip(cluster_aspects_list, sub_clusters):
                sub_cluster_mapping[sub_cluster_label].add(sub_aspect)

            # Convert cluster labels to string and majority names for JSON compatibility
            majority_names = {str(sub_cl): get_majority_aspect_name(aspects) for sub_cl, aspects in
                              sub_cluster_mapping.items()}
            result[str(cluster_label)] = majority_names
    else:
        majority_names = {str(cl): get_majority_aspect_name(aspects) for cl, aspects in top_cluster_mapping.items()}
        result = {}
        for cl, aspects in top_cluster_mapping.items():
            result[str(cl)] = {'majority_name': majority_names[str(cl)], 'aspects': list(aspects)}

    # result = list(result.values())

    if summarize:
        aspects_with_examples = extract_aspects_with_examples(model_answers)
        cluster_data = {}
        for cluster_label, aspects in top_cluster_mapping.items():
            cluster_aspects_data = {}
            for aspect in aspects:
                aspect_data = aspects_with_examples.get(aspect, {})
                cluster_aspects_data[aspect] = {
                    "count": aspect_data.get("count", 0),
                    "examples": aspect_data.get("examples", [])
                }

            # Summarize cluster name based on the majority vote
            majority_name = get_majority_aspect_name(list(aspects))

            cluster_data[str(cluster_label)] = {
                "majority_name": majority_name,
                "aspects": cluster_aspects_data
            }

            gpt_response = query_llm_for_summaries(cluster_data[str(cluster_label)], llm_args)
            print(gpt_response)

            # Parse the GPT response
            summary = parse_gpt_response(gpt_response)
            print(summary)

            result[str(cluster_label)]['name'] = summary['name']
            result[str(cluster_label)]['description'] = summary['description']

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
    parser.add_argument('--num_subaspects', type=int, default=None,
                        help='Number of sub-aspects per top-level aspect to cluster')
    parser.add_argument('--output_file', type=str, required=False, help='Output JSON file for the clustered aspects')
    parser.add_argument('--summarize', type=bool, required=False, default=False,
                        help='Use LLM to generate name and description for clusters')

    args, unknown = parser.parse_known_args()
    llm_args = llm_parser().parse_args(unknown)
    main(args.input_file, args.num_aspects, args.num_subaspects, args.output_file, args.summarize, llm_args)
