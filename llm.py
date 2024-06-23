import pandas as pd
import argparse
import json
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import anthropic
import json_repair
from openai import OpenAI
import requests
from tqdm import tqdm
import tiktoken


def count_tokens_in_string(text):
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(enc.encode(text))


def llm_parser():
    parser = argparse.ArgumentParser(add_help=False)  # Disable help to prevent conflict with main parser
    parser.add_argument('--base_url', type=str, default='https://api.deepinfra.com/v1/inference/')
    parser.add_argument('--model_name', type=str, default='openchat/openchat_3.5')
    parser.add_argument('--llm_api_type', type=str,
                        default='deepinfra')  # types can be, openai, anthropic, and deepinfra
    parser.add_argument('--api_token', type=str)
    parser.add_argument('--api_delay', type=int, default=0, help='Delay between API calls in seconds')
    parser.add_argument('--llm_api_threads', type=int, default=1)
    parser.add_argument('--llm_max_tokens', type=int, default=1024)
    parser.add_argument('--llm_temperature', type=float, default=0.0)
    parser.add_argument('--few_shot_count', type=int, default=1, help='Number of shots for few-shot learning')
    parser.add_argument('--reset_cache', type=bool, default=False)
    parser.add_argument('--gold_labels_file', type=str, required=False,
                        help='Optional file with gold labels for 1-shot prompt')

    return parser


def get_prompt(review_text: str, aspect_map, gold_example_texts=None, gold_example_responses=None):
    filtered_aspect_map = [{'name': aspect['name'], 'description': aspect['description']} for aspect in
                           aspect_map.values()]
    if "other" not in [aspect['name'] for aspect in filtered_aspect_map]:
        filtered_aspect_map.append(
            {'name': 'other', 'description': 'Any aspect that does not fit in the provided list'})
    aspect_map_str = json.dumps(filtered_aspect_map, ensure_ascii=False)

    system_prompt = f'''Using the provided list of aspects, analyze the review text for explicit mentions of these aspects. 
            Extract direct mentions as text fragments (up to 7 words), accurately identifying the aspect and sentiment.
            Ignore comparisons to other locations, plans or expectations. 
            Format your response as JSON, with each object containing the following
        properties:
        - "text": A fragment of the review text mentioning the aspect, maximum 7 words, very important: written exactly as it appears in the review text
        - "aspect": The extracted aspect which must be from the provided list <aspects>
        - "sentiment": The sentiment associated with the aspect, either "positive" or "negative"

        Here is an example of the expected JSON format:
        {{"answer":[{{
        "text": "[fragment of review text]",
        "aspect": "[aspect name]",
        "sentiment": "[positive/negative]"
        }}, ...]}}

        <aspects>
        {aspect_map_str}
        </aspects>
        '''
    user_prompt = f'''<review_text>
        {review_text}
        </review_text>'''

    if gold_example_texts is not None and gold_example_responses is not None:
        one_shot_user_prompts = [f'''<review_text>
            {gold_example_text}
            </review_text>''' for gold_example_text in gold_example_texts]

        one_shot_assistant_responses = [json.dumps({"answer": gold_example_response}, ensure_ascii=False) for
                                        gold_example_response in gold_example_responses]
        return system_prompt, one_shot_user_prompts, one_shot_assistant_responses, user_prompt
    else:
        return system_prompt, None, None, user_prompt


def get_prompt_without_map(review_text: str):
    system_prompt = f'''Analyze the review text for explicit mentions of aspects. 
            Extract direct mentions as text fragments (up to 7 words), accurately identifying the aspect and sentiment.
            Ignore comparisons to other locations, plans or expectations. 
            Format your response as JSON, with each object containing the following
        properties:
        - "text": A fragment of the review text mentioning the aspect, maximum 7 words, very important: written exactly as it appears in the review text
        - "aspect": The extracted aspect, in Russian
        - "sentiment": The sentiment associated with the aspect, either "positive" or "negative"

        Here is an example of the expected JSON format:
        {{"answer":[{{
        "text": "[fragment of review text]",
        "aspect": "[aspect name in Russian]",
        "sentiment": "[positive/negative]"
        }}, ...]}}
        '''
    user_prompt = f'''<review_text>
        {review_text}
        </review_text>'''
    return system_prompt, user_prompt


def get_best_gold_example(gold_labeled, few_shot_count):
    gold_example_texts = []
    gold_example_responses = []
    gold_labeled_short = gold_labeled[gold_labeled['text'].str.len() < 200]
    # we will greedy collect the example that includes as many previously not-collected aspect-sentiment pairs as possible
    # take them from gold_labeled_short
    # and doing it few_shot_count times

    collected_aspect_sentiment_pairs = set()

    for _ in range(few_shot_count):
        gold_example_text = None
        gold_example_response = None
        best_aspect_sentiment_pairs = set()
        best_aspect_sentiment_pairs_count = 0

        for _, gold_row in gold_labeled_short.iterrows():
            gold = json.loads(gold_row['annotations'].replace('"', '\\"').replace("'", '"'))
            current_row_aspect_sentiment_pairs = set()
            for annotation in gold:
                aspect_sentiment_pair = (annotation['aspect'], annotation['sentiment'])
                current_row_aspect_sentiment_pairs.add(aspect_sentiment_pair)
            new_pairs = current_row_aspect_sentiment_pairs - collected_aspect_sentiment_pairs
            if len(new_pairs) > best_aspect_sentiment_pairs_count:
                best_aspect_sentiment_pairs = new_pairs
                best_aspect_sentiment_pairs_count = len(new_pairs)
                gold_example_text = gold_row['text']
                gold_example_response = gold
        collected_aspect_sentiment_pairs |= best_aspect_sentiment_pairs
        gold_example_texts.append(gold_example_text)
        gold_example_responses.append(gold_example_response)

    print(f"Found {len(collected_aspect_sentiment_pairs)} unique aspect-sentiment pairs in {few_shot_count} examples")
    for gold_example_text, gold_example_response in zip(gold_example_texts, gold_example_responses):
        print(f"Example text: {gold_example_text}")
        print(f"Example response: {gold_example_response}")

    return gold_example_texts, gold_example_responses


def label_review_with_llm(review_text: str, aspect_map, args, gold_example_texts: str = None,
                          gold_example_responses=None):
    global requests_cache
    if review_text in requests_cache:
        return requests_cache[review_text]

    if aspect_map is not None:
        system_prompt, example_prompts, example_responses, user_prompt = get_prompt(review_text, aspect_map,
                                                                                    gold_example_texts,
                                                                                    gold_example_responses)
    else:
        system_prompt, user_prompt = get_prompt_without_map(review_text)
        example_prompts = []
        example_responses = []
    t = query_llm(system_prompt, user_prompt, args, example_prompts, example_responses)
    with cache_lock:
        requests_cache[review_text] = t
        with open("requests_cache.pkl", "wb") as f:
            pickle.dump(requests_cache, f)
    time.sleep(args.api_delay)
    return t


def generate_prompt_for_cluster_descriptions(data):
    system_prompt = (
        "Summarize the information into name and description. "
        "The description should be about 20-30 words long and reflect the key aspects and their context. "
        "Note that all aspects can include positive and negative feedback, "
        "while some may have only positive or negative examples. "
        "Important: Please ensure the aspect name and description do not imply any sentiment, focusing purely on the thematic content."
        "The name and description will be used in aspect-based sentiment analysis; it is important to keep aspect name and description neutral in terms of sentiment "
        " (not 'Высокое качество', but 'Качество'). "
        "The description also need to be neutral: not 'положительный опыт', but 'положительный или отрицательный опыт' or just 'опыт'."
        "Avoid too general names ('оценка', 'впечатление', 'качество') and too broad descriptions: if too different aspects are supplied, focus on the most frequent."
        "\n\n"
    )
    prompt = ""
    j = 1
    # sorted_cluster_data = sorted(cluster_data.items(), key=lambda x: sum(aspect['count'] for aspect in x[1]['aspects'].values()))
    # sorted_cluster_data_dict = {cluster_id: data for cluster_id, data in sorted_cluster_data}

    prompt += f"Cluster includes the following aspects:\n"
    # Sort aspects by count and pick the top 5
    top_aspects = sorted(data['aspects'].items(), key=lambda item: item[1]['count'], reverse=True)[:10]
    for aspect, details in top_aspects:
        # Ensure there's at least one example before attempting to access it
        example_text = ('"' + '"; "'.join(details['examples'][0:5]) + '"') if details[
            'examples'] else "No example available"
        prompt += f"- {aspect} (Examples: {example_text}, Count: {details['count']}),\n"
        # prompt += f"- {aspect}, Count: {details['count']},\n"
    prompt += "\n"

    prompt += """Please provide improved name and description for the cluster, format your response as JSON: {"name": "", "description": ""}."""
    prompt += "Provide name and description in Russian."
    return system_prompt, prompt


def query_llm(system_prompt, user_prompt, args, example_prompts=None, example_responses=None):
    if args.llm_api_type == 'deepinfra':
        server_url = args.base_url + args.model_name
        payload = {
            "input": "",
            "temperature": args.llm_temperature,
            "max_new_tokens": args.llm_max_tokens,
            "truncate": 1023,
            "response_format": {"type": "json_object"},
            "stop": ["</s>", "[INST]", "[/INST]"]
        }
        headers = {'Authorization': 'Bearer ' + args.api_token}
        if example_prompts is not None:
            payload[
                "input"] = f"<s> [INST] {system_prompt} "
            for i in range(len(example_prompts)):
                payload["input"] += f"[INST]{example_prompts[i]}[/INST] {example_responses[i]} "
            payload["input"] += f"[INST]{user_prompt}[/INST]"
        else:
            payload["input"] = f"<s> [INST]{system_prompt}{user_prompt}[/INST]"

        response = requests.post(server_url, json=payload, headers=headers)
        t = response.json()['results'][0]['generated_text']
    elif args.llm_api_type == 'openai':
        chatgpt_client = OpenAI(api_key=args.api_token, base_url=args.base_url)
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        if example_prompts is not None:
            for i in range(len(example_prompts)):
                messages += [{"role": "user", "content": example_prompts[i]},
                             {"role": "assistant", "content": example_responses[i]}]
        messages += [{"role": "user", "content": user_prompt}]
        chat_completion = chatgpt_client.chat.completions.create(
            model=args.model_name,
            messages=messages,
            max_tokens=args.llm_max_tokens,
            temperature=args.llm_temperature
        )
        t = chat_completion.choices[0].message.content
    elif args.llm_api_type == 'anthropic':
        messages = []
        if example_prompts is not None:
            for i in range(len(example_prompts)):
                messages += [{"role": "user", "content": example_prompts[i]},
                             {"role": "assistant", "content": example_responses[i]}]
        messages += [{"role": "user", "content": user_prompt}]
        t = anthropic.Anthropic(api_key=args.api_token).messages.create(
            model=args.model_name,
            system=system_prompt,
            messages=messages,
            max_tokens=args.llm_max_tokens,
            temperature=args.llm_temperature
        ).content[0].text
    else:
        raise ValueError(f"Unknown LLM API type: {args.llm_api_type}")
    return t


def query_llm_for_summaries(data, args):
    system_prompt, prompt = generate_prompt_for_cluster_descriptions(data)
    t = query_llm(system_prompt, prompt, args)
    return t


def query_llm_for_entity_aspect_summary(aspect_fragments_text, aspect, description, args):
    system_prompt_v2 = f'''From the reviews provided, summarize key features relevant to the aspect "{aspect}", which focuses on "{description}", 
    in 3-7 bullet points (group when possible), about 5-10 words long. Each point should highlight a distinct characteristic. 
    Avoid repeating similar descriptions, such as synonymous adjectives or phrases, to ensure a diverse portrayal of the aspect.
    Ignore any information not directly related to "{aspect}".  
    Focus on both positive and negative attributes. Important: Respond only in Russian and start your response with the bullet points.
'''
    system_prompt = f'''From the reviews, there are several extracted fragments that are relevant to aspect {aspect}.
    Your task is to summarize these fragments into several concise sentences describing the aspect.
    Ignore any information not directly related to "{aspect}".  
    Pay attention both to positive and negative reviews (if any).
    Respond in Russian.'''

    prompt = f"<fragments>{aspect_fragments_text}</fragments>"
    t = query_llm(system_prompt, prompt, args)
    return t


def generate_prompt_for_group_aspects(annotations, num_aspects, max_input_length):
    # first, we have to select the most frequent aspects, and corresponding text fragments,
    # making sure they don't exceed max_input_length tokens, and then we can generate the prompt
    # leaving 1024 tokens for output
    # and targeting for num_aspects output aspects, with names and descriptions

    system_prompt = f'''Use the aspects extracted from reviews, to create groups of aspects.
For each extracted aspect, you are given name, count and three examples of text fragments.
Please create {num_aspects} groups, make sure they don't intersect and are based on the provided data.
Aspect groups that you create will be used for annotation of reviews. They can be linked to positive or negative sentiment,
so please make the names and descriptions neutral (for example "качество" instead of "высокое качество").
Try to cover most provided aspects with your groups. First, provide your analysis. 
Then, respond with json object: {{"answer": [{{"name": "[first group name (in Russian)]", "description": [short description about 20 words (in Russian)]}}, ...]}}'''
    system_prompt_length = count_tokens_in_string(system_prompt)
    expected_response_length = 1024

    aspects = {}
    for annotation in annotations:
        annotation = json_repair.loads(annotation.replace("'", "\""))
        for aspect in annotation:
            if not 'aspect' in aspect:
                continue
            if aspect['aspect'] not in aspects:
                aspects[aspect['aspect']] = []
            aspects[aspect['aspect']].append(aspect['text'])
    aspects = {k: v for k, v in sorted(aspects.items(), key=lambda item: len(item[1]), reverse=True)}
    # now convert aspects to strings: store name and up to 3 examples for each
    aspects = {k: json.dumps({"name": k, "count": len(v), "examples": v[:3]}, ensure_ascii=False) for k, v in
               aspects.items()}

    tokens_used = 0
    selected_aspects = []
    for aspect, texts in aspects.items():
        if aspect not in selected_aspects:
            selected_aspects.append(aspect)
            tokens_used += count_tokens_in_string(texts)
            if tokens_used >= max_input_length - system_prompt_length - expected_response_length:
                selected_aspects.pop()
                break
    user_prompt = json.dumps([aspects[aspect] for aspect in selected_aspects], ensure_ascii=False)
    return system_prompt, user_prompt


def query_llm_for_group_aspects(annotations, num_aspects, max_input_length, args):
    system_prompt, prompt = generate_prompt_for_group_aspects(annotations, num_aspects, max_input_length)
    t = query_llm(system_prompt, prompt, args)
    return t


def query_llm_for_zero_shot_aspects(entity_name, aspect_count, args):
    system_prompt = '''For aspect-based sentiment analysis of reviews, generate a list of aspects. Format your response
    as json object: {"answer": [{"name": "[aspect name in Russian]", "description": "[aspect description in Russian]"}, ...]}'''
    user_prompt = f'''The reviews will be about {entity_name}. Please provide a list of {aspect_count} aspects.'''
    t = query_llm(system_prompt, user_prompt, args)
    return t


def label_with_llm_concurrently(reviews_df, args, aspect_map=None):
    global requests_cache
    if args.reset_cache:
        requests_cache = {}

    if args.gold_labels_file is not None:
        gold_labeled = pd.read_csv(args.gold_labels_file, encoding="utf-8")
        gold_example_texts, gold_example_responses = get_best_gold_example(gold_labeled, args.few_shot_count)
    else:
        gold_example_texts = None
        gold_example_responses = None

    # Create a column for annotations in the dataframe
    if 'annotations' not in reviews_df.columns:
        reviews_df['annotations'] = pd.Series([[]] * len(reviews_df))

    reviews_df_to_label = reviews_df[reviews_df['annotations'].apply(lambda x: len(x) == 0)]
    print("Number of reviews to label: ", len(reviews_df_to_label), " out of ", len(reviews_df))

    # parse json in annotations where not empty
    reviews_df['annotations'] = reviews_df['annotations'].apply(lambda x: json_repair.loads(x.replace("'", '"'))['answer'] if len(x) > 0 else x)

    with ThreadPoolExecutor(max_workers=args.llm_api_threads) as executor:
        future_to_index = {executor.submit(label_review_with_llm, row['text'], aspect_map, args, gold_example_texts,
                                           gold_example_responses): index
                           for index, row in reviews_df_to_label.iterrows()}
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index)):
            index = future_to_index[future]
            try:
                t_raw = future.result()
                t = t_raw[t_raw.find("["):]
                if t.find("]") != -1:
                    t = t[:t.find("]") + 1]
                else:
                    t = t[:t.rfind("}") + 1]
                t = t.replace("\n", " ").replace("\r", " ").strip()
                t = json_repair.loads(t)
                # Update the corresponding row in the dataframe
                reviews_df.at[index, 'annotations'] = t if t != "" else []
                if len(reviews_df.at[index, 'annotations']) == 0:
                    print("Warning, empty annotation for review: ", reviews_df.iloc[index]['text'], " response: ",
                          t_raw)
            except Exception as exc:
                print(f"{reviews_df.iloc[index]['text']} generated an exception: {exc}")
                # Set empty annotation in case of an exception
                reviews_df.at[index, 'annotations'] = []

    print("Labeling reviews: input row: ", len(reviews_df))
    reviews_df = reviews_df[reviews_df['annotations'].apply(lambda x: len(x) > 0)]
    print("After filtering empty annotations: ", len(reviews_df))
    return reviews_df.reset_index(drop=True)


requests_cache = {}
if os.path.exists("requests_cache.pkl"):
    with open("requests_cache.pkl", "rb") as f:
        requests_cache = pickle.load(f)
requests_cache['found'] = 0
requests_cache['added'] = 0
cache_lock = Lock()
