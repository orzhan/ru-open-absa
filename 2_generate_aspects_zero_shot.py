import argparse
import json

from llm import llm_parser, query_llm_for_zero_shot_aspects

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate aspect list in zero shot manner using LLM')
    parser.add_argument('--entity_type', type=str, required=True,
                        help='For which type of objects we need aspect list (cars / restaurants / hotels)')
    parser.add_argument('--num_aspects', type=int, required=True, help='Number of top-level aspects to cluster')
    parser.add_argument('--output_file', type=str, required=False, help='Output JSON file for the clustered aspects')

    args, unknown = parser.parse_known_args()
    llm_args = llm_parser().parse_args(unknown)

    gpt_response = query_llm_for_zero_shot_aspects(args.entity_type, args.num_aspects, llm_args)
    aspect_map = json.load(gpt_response)['answer']

    if args.output_file is not None:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(aspect_map, f, ensure_ascii=False, indent=4)

    print(json.dumps(aspect_map, ensure_ascii=False, indent=4))
