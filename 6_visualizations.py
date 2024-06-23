import argparse
import json
import re

import numpy as np
import pandas as pd
import torch
from json_repair import json_repair
from scipy.cluster.hierarchy import linkage, fcluster
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from llm import llm_parser, query_llm_for_entity_aspect_summary
from utils import find_best_match_with_position

STYLES = '''<style>
.table {
  display: table;
  width: 100%;
  border-collapse: collapse;
}

.row {
  display: table-row;
}

.cell {
  width: 20%;
  display: table-cell;
  padding: 10px;
  border: 1px solid #ccc;
  vertical-align: top;
}

.cell h2 {
  margin-top: 0;
  font-size: 1em;
  font-weight: bold;
}

.cell ul {
  margin-bottom: 0;
  padding-left: 20px;
}

.cell details {
  min-height: 30px; 
}

.cell summary {
  cursor: pointer;
  outline: none;
}

</style>'''


# Function to get embeddings
def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def create_aspect_summary(aspect, aspect_map, group_data, max_text_fragments=10):
    description = [x for x in aspect_map.values() if x['name'] == aspect][0]['description']
    aspect_fragments = []
    # here, insert all text fragment from group_data relevant to aspect.
    for idx, row in group_data.iterrows():
        annotations = json_repair.loads(row["annotations"].replace("'", '"'))
        if 'answer' in annotations:
            annotations = annotations['answer']
        text = row["text"]
        # split text by "." or newline
        text_fragments = re.split(r'\.|\n', text)
        for entity in annotations:
            if 'aspect' not in entity or 'sentiment' not in entity:
                continue
            # find fragment containing entity['text']
            if entity['aspect'] == aspect:
                # extended_fragment = "; ".join([x for x in text_fragments if x.find(entity['text']) != -1])
                entity_start, entity_end, score = find_best_match_with_position(text, entity['text'])
                start = entity_start
                # look for ".", "\n" or a space but only if at least 20 characters were added
                nwords = 0
                while start > 0:
                    if text[start] == "." or text[start] == "\n" or text[start] == "!" or text[start] == ",":
                        start += 1
                        break
                    if text[start] == " ":
                        nwords += 1
                        if nwords > 2:
                            start += 1
                            break
                    start -= 1
                end = entity_end
                nwords = 0
                while end < len(text):
                    if text[end] == "." or text[end] == "\n" or text[end] == "!":
                        break
                    if text[end] == " ":
                        nwords += 1
                        if nwords > 2:
                            break
                    end += 1
                extended_fragment = text[start:end + 1]
                aspect_fragments.append({"text": extended_fragment, "sentiment": entity['sentiment']})

    if len(aspect_fragments) == 0:
        return ""

    if len(aspect_fragments) < max_text_fragments:
        selected_fragments = aspect_fragments
    else:
        # next step is deduplication, and selection of not too many fragments
        embeddings = np.vstack(
            [get_embedding(aspect_fragment['text'], tokenizer, model) for aspect_fragment in aspect_fragments])

        # Perform hierarchical clustering for top-level aspects
        Z = linkage(embeddings, 'ward')
        top_clusters = fcluster(Z, t=max_text_fragments, criterion='maxclust')
        # select one fragment per cluster and make a list of those fragments
        selected_fragments = []
        for cluster_label in np.unique(top_clusters):
            cluster_indices = np.where(top_clusters == cluster_label)
            selected_fragments.append(aspect_fragments[cluster_indices[0][0]])

    print("-" * 50)
    print(aspect)
    print(selected_fragments)
    # pass max 10 fragments to gpt and ask to summarize
    gpt_response = query_llm_for_entity_aspect_summary(json.dumps(selected_fragments, ensure_ascii=False), aspect,
                                                       description, llm_args)
    print(gpt_response)
    return gpt_response.replace("\n", "<br />\n")


def create_html_visualization(entities, text, full_text=None):
    """
    Creates an HTML visualization for entity groups extracted by a token classification pipeline.
    """
    colors = {
        "positive": "#009432",  # Darker green
        "negative": "#c23616",  # Darker red
        "neutral": "#718093"  # Darker gray for neutral
    }

    html = '<div style="font-family: Arial, sans-serif; line-height: 2; margin-bottom: 20px;">'
    last_end = 0
    for entity in entities:
        if not 'start' in entity:
            start, end, score = find_best_match_with_position(text, entity['text'])
            sentiment = entity['sentiment']
            aspect_name = entity['aspect']
            word = entity['text']
        else:
            start = entity['start']
            end = entity['end']
            score = entity['score']
            entity_group = entity['entity_group']
            sentiment = 'neutral'
            if entity_group != 'other':
                sentiment = entity_group.split("_")[1]
                aspect_name = entity_group.split("_")[0]
            word = text[start:end]
        if sentiment not in colors:
            continue
        if start > last_end:
            html += text[last_end:start]
            # Add the entity with aspect label and styled background
        entity_text = f"<span style='background-color: white; padding: 0.2em; border-radius: 4px;'>{word}</span>"
        aspect_label = f"<div style='color: #fff; background-color: {colors[sentiment]}; border-radius: 8px; width: 100%; display: inline-block; margin-top: 5px; font-size: 0.9em; text-align: center; font-weight: bold;'>{aspect_name}</div>"
        html += f"<div style='display: inline-block; vertical-align: top; background-color: {colors[sentiment]}; border-radius: 8px; padding: 0.2em;  margin-right: 10px; margin-bottom: 2px;' title='{full_text}'>{entity_text}<br>{aspect_label}</div>"
        last_end = end

    html += text[last_end:]
    html += '</div>'
    return html


# Display only aspect-related text fragments and group them by aspect, not by review
def create_aspects_visualization(group_data, aspect_stats, aspect_map, summarize_aspects=False):
    table = {}
    for aspect_data in aspect_map.values():
        table[aspect_data['name']] = {"head": '<h2>' + aspect_data['name'] + '</h2>', "stats": "", "summary": "",
                                      "texts": ""}
        # for stats, make gray bar by default
        table[aspect_data['name']]['stats'] += '<div style="height: 10px; background: #dcdcdc; border-radius: 5px;"></div>'
    for aspect, counts in aspect_stats.items():
        if aspect == 'other':
            continue
        positive = counts.get('positive', 0)
        negative = counts.get('negative', 0)
        total = positive + negative
        positive_percent = (positive / total * 100) if total else 0
        negative_percent = (negative / total * 100) if total else 0
        if negative_percent >= 95:
            table[aspect]['stats'] = '<div style="height: 10px; background: #ffbaba; border-radius: 5px;"></div>'
        else:
            table[aspect][
                'stats'] = f'<div style="height: 10px; background: linear-gradient(90deg, #b2fab4 {positive_percent}%, #ffbaba {negative_percent}%); border-radius: 5px;"></div>'

        if summarize_aspects:
            table[aspect]['summary'] += create_aspect_summary(aspect, aspect_map, group_data)

        # here, insert all text fragment from group_data relevant to aspect.
        for idx, row in group_data.iterrows():
            annotations = json_repair.loads(row["annotations"].replace("'", '"'))
            if 'answer' in annotations:
                annotations = annotations['answer']
            text = row["text"]
            # split text by "." or newline
            text_fragments = re.split(r'\.|\n', text)
            for entity in annotations:
                if 'aspect' not in entity or 'sentiment' not in entity:
                    continue
                # find fragment containing entity['text']
                extended_fragment = "; ".join([x for x in text_fragments if x.find(entity['text']) != -1])
                if entity['aspect'] == aspect:
                    table[aspect]['texts'] += create_html_visualization([entity], entity['text'], extended_fragment)

    return table


def aspect_visualization_html(table):
    html = '<div class="table">'
    html += '<div class="row">'
    for aspect, data in table.items():
        html += '<div class="cell">'
        html += data['head']
        html += data['stats']
        html += '</div>'
    html += '</div>'
    html += '<div class="row">'
    for aspect, data in table.items():
        html += '<div class="cell">'
        html += '<details><summary>' + data['summary'] + '</summary>'
        html += data['texts']
        html += '</div>'
    html += '</div>'
    html += '</div>'
    return html


def create_aspect_stats_visualization(aspect_stats, aspect_map):
    """
    Modifies the layout to put all aspects in one line with shorter bars.
    """
    html = '<div style="font-family: Arial, sans-serif; margin-top: 20px; display: flex; flex-wrap: wrap;">'
    for aspect, counts in aspect_stats.items():
        if aspect == 'other':
            continue
        positive = counts.get('positive', 0)
        negative = counts.get('negative', 0)
        total = positive + negative
        positive_percent = (positive / total * 100) if total else 0
        negative_percent = (negative / total * 100) if total else 0

        html += f'<div style="width: 17%; margin-right: 10px; margin-bottom: 10px;">'
        html += f'<strong>{aspect}</strong>'
        if negative_percent >= 95:
            html += '<div style="height: 10px; background: #ffbaba; border-radius: 5px;"></div>'
        else:
            html += f'<div style="height: 10px; background: linear-gradient(90deg, #b2fab4 {positive_percent}%, #ffbaba {negative_percent}%); border-radius: 5px;"></div>'
        html += '</div>'
    html += '</div>'
    return html


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Display annotations')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input CSV file containing reviews in column "text" and annotations in column "annotations"')
    parser.add_argument('--group_by_field', type=str, default="address", help='Field to group the annotations by')
    parser.add_argument('--display_style', type=str, default="reviews", help='Display style: reviews or aspects')
    parser.add_argument('--summarize_aspects', type=bool, default=False,
                        help='Summarize aspects (for aspect display style)')
    parser.add_argument('--aspect_map', type=str, required=True,
                        help='Input JSON file with aspect map: json array with {name, description} fields')
    parser.add_argument('--output_file', type=str, default="visualization.html",
                        help='Output HTML file for the visualizations')

    args, unknown = parser.parse_known_args()
    llm_args = llm_parser().parse_args(unknown)

    input_data = pd.read_csv(args.input_file, encoding="utf-8")
    aspect_map = json.load(open(args.aspect_map, "r", encoding="utf-8"))

    if args.summarize_aspects:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
        model = AutoModel.from_pretrained("intfloat/multilingual-e5-small")

    count = 0
    # Group the annotations by the specified field
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(STYLES)
        # Calculate the size of each group and sort them in descending order by size
        grouped_data = input_data.groupby(args.group_by_field).size().sort_values(ascending=False)
        # Iterate over sorted group names to access group data in the sorted order
        for group_name in tqdm(grouped_data.index):
            group_data = input_data[input_data[args.group_by_field] == group_name]
            f.write(f"{len(group_data)} reviews for entity: {group_name}")
            # calculate aspect stats: positive and negative counts for each aspect
            aspect_stats = {}
            for idx, row in group_data.iterrows():
                try:
                    annotations = json_repair.loads(row["annotations"].replace("'", '"'))
                    if 'answer' in annotations:
                        annotations = annotations['answer']
                    for entity in annotations:
                        if 'aspect' not in entity or 'sentiment' not in entity:
                            continue
                        aspect = entity['aspect']
                        sentiment = entity['sentiment']
                        if aspect not in aspect_stats:
                            aspect_stats[aspect] = {"positive": 0, "negative": 0}
                        if sentiment in ["positive", "negative"]:
                            aspect_stats[aspect][sentiment] += 1
                except Exception as ex:
                    print(row)
                    print(ex)
                    pass
            if args.display_style == "reviews":
                html = create_aspect_stats_visualization(aspect_stats, aspect_map)
                f.write(html)
                for idx, row in group_data.iterrows():
                    annotations = json_repair.loads(row["annotations"].replace('"', '\\"').replace("'", '"'))
                    text = row["text"]
                    html = create_html_visualization(annotations, text)
                    f.write(html)
                    f.write("<hr />")
            elif args.display_style == "aspects":
                table = create_aspects_visualization(group_data, aspect_stats, aspect_map, args.summarize_aspects)
                f.write(aspect_visualization_html(table))
            f.write("<hr style='height: 2px' />")
            count += 1
            if count >= 10:
                break
