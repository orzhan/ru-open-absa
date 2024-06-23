import argparse
import json

import pandas as pd

from llm import label_with_llm_concurrently, get_best_gold_example, llm_parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Label reviews using LLM')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input CSV file containing reviews in column "text"')
    parser.add_argument('--max_input_rows', type=int, default=None,
                        help='Maximum number of input rows to process')
    parser.add_argument('--max_text_length', type=int, default=512)
    parser.add_argument('--aspect_map', type=str, required=True,
                        help='Input JSON file with aspect map: json array with {name, description} fields')
    parser.add_argument('--output_file', type=str, required=True, help='Output CSV file for the labeled reviews')

    args, unknown = parser.parse_known_args()
    llm_args = llm_parser().parse_args(unknown)

    with open(args.aspect_map, "r", encoding="utf-8") as f:
        aspect_map = json.load(f)

    train_data = pd.read_csv(args.input_file, encoding="utf-8")
    if args.max_text_length is not None:
        print(f"Filtering out reviews longer than {args.max_text_length} characters. Original size: {len(train_data)}")
        train_data = train_data[train_data["text"].str.len() <= args.max_text_length]
        print(f"Filtered size: {len(train_data)}")
    if args.max_input_rows is not None:
        train_data = train_data.head(args.max_input_rows)

    train_data.reset_index(drop=True, inplace=True)
    train_data_annotated = label_with_llm_concurrently(train_data, llm_args, aspect_map)
    train_data_annotated.to_csv(args.output_file, index=False)
