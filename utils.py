from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from thefuzz import process
from tqdm import tqdm
import json

def text_overlap(text1, text2):
    """
    Calculate the overlap between two strings as the number of common words divided
    by the average number of words in both strings.
    """
    words1 = set(text1.split())
    words2 = set(text2.split())
    common_words = words1.intersection(words2)
    return len(common_words) / ((len(words1) + len(words2)) / 2)


def adjust_predictions(annotations, predictions):
    """
    Adjust predictions to account for splits, combining predictions where appropriate
    based on aspect and sentiment match and sufficient text overlap.
    """
    adjusted_predictions = []
    for pred in predictions:
        # Find any annotation that matches aspect and sentiment and has sufficient text overlap
        match = next((ann for ann in annotations if ann['aspect'] == pred['aspect'] and
                      ann['sentiment'] == pred['sentiment'] and
                      text_overlap(ann['text'], pred['text']) >= 0.5), None)
        if match:
            # If there's a match, adjust the prediction to exactly match the annotation's text
            adjusted_pred = pred.copy()
            adjusted_pred['text'] = match['text']
            adjusted_predictions.append(adjusted_pred)
        else:
            adjusted_predictions.append(pred)
    return adjusted_predictions


def calculate_f1(annotations, predictions):
    if not annotations and not predictions:
        return 1.0, 1.0, 1.0, 0, 0, 0, 0

    # Initialize counters
    true_positives = 0
    partial_positives = 0
    false_positives = 0
    false_negatives = 0

    # Separate main aspects and subaspects for comparison
    anns_set = {((ann['aspect'].split("/")[0] if "/" in ann['aspect'] else ann['aspect']), ann['sentiment']) for ann in
                annotations}
    preds_set = {((pred['aspect'].split("/")[0] if "/" in pred['aspect'] else pred['aspect']), pred['sentiment']) for
                 pred in predictions}

    for ann in annotations:
        aspect, subaspect = (ann['aspect'].split("/") + [None])[:2]  # Handle cases without subaspect
        matched_preds = [pred for pred in predictions if (
                pred['aspect'].split("/")[0] == aspect and pred['sentiment'] == ann['sentiment']) and text_overlap(
            ann['text'], pred['text']) >= 0.5]

        if matched_preds:
            exact_match = any(pred for pred in matched_preds if pred['aspect'] == ann['aspect'])
            if exact_match:
                true_positives += 1
            else:
                partial_positives += 1
        else:
            false_negatives += 1

    for pred in predictions:
        if not any(ann for ann in annotations if
                   ann['aspect'] == pred['aspect'] and ann['sentiment'] == pred['sentiment'] and text_overlap(
                       ann['text'], pred['text']) >= 0.5):
            false_positives += 1

    # Weight partial positives lower than true positives, e.g., half credit
    weighted_true_positives = true_positives + (partial_positives * 0.5)

    precision = weighted_true_positives / (
            weighted_true_positives + false_positives) if weighted_true_positives + false_positives > 0 else 0
    recall = weighted_true_positives / (
            weighted_true_positives + false_negatives) if weighted_true_positives + false_negatives > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return f1_score, precision, recall, true_positives, partial_positives, false_positives, false_negatives


def generate_aspect_sentiment_labels(annotations_lists):
    # Initialize a set to store unique aspect-sentiment combinations
    unique_aspects = {}
    aspect_names = []
    aspect_counts = {}
    label_counts = {}
    label_index = 1  # Start indexing from 1 since 0 will be reserved for "other"

    for annotation_strings in annotations_lists:
        annotations = json.loads(annotation_strings.replace("'", '"'), strict=False)
        for annotation in annotations:
            aspect = annotation['aspect']
            aspect_names.append(aspect)
            if not aspect in aspect_counts:
                aspect_counts[aspect] = 0
            aspect_counts[aspect] += 1
            for sentiment in ['positive', 'negative']:
                if aspect + "_" + sentiment not in unique_aspects:
                    unique_aspects[aspect + "_" + sentiment] = label_index
                    label_index += 1
                    label_counts[aspect + "_" + sentiment] = 0
            if sentiment in ['positive', 'negative']:
                label_counts[aspect + "_" + annotation['sentiment']] += 1

    # Add a generic "other" label
    unique_aspects['other'] = 0

    print(f"Labels retrieved from gold set: {unique_aspects}")

    return unique_aspects, list(set(aspect_names)), aspect_counts, label_counts


def find_best_match_with_position(text, substring):
    # Direct match check
    direct_start = text.find(substring)
    if direct_start != -1:
        direct_end = direct_start + len(substring)
        return direct_start, direct_end, 100

    # Fuzzy match as fallback
    best_score = 0
    best_pos = (0, 0)
    words = text.split()
    for start in range(len(words)):
        for end in range(start + 1, len(words) + 1):
            candidate = ' '.join(words[start:end])
            score = fuzz.ratio(candidate, substring)
            if score > best_score:
                best_score = score
                best_pos = (start, end)

    # Convert word positions to character positions
    start_char_pos = len(' '.join(words[:best_pos[0]])) + (1 if best_pos[0] > 0 else 0)
    end_char_pos = len(' '.join(words[:best_pos[1]]))
    return start_char_pos, end_char_pos, best_score


def align_tokens_and_labels(tokenized_output, text, annotations, aspect_sentiment_labels):
    tokens = tokenized_output["tokens"]  # Directly access tokens from the dictionary
    offset_mapping = tokenized_output["offset_mapping"]

    # Initialize labels for each token. Assuming 'other' for unlabelled tokens.
    labels = [aspect_sentiment_labels['other']] * len(tokens)
    worst_score = 100

    for annotation in annotations:
        if type(annotation) != dict or 'sentiment' not in annotation or 'aspect' not in annotation:
            continue
        if annotation['sentiment'] not in ['positive', 'negative']:
            continue  # Skip if sentiment is neither positive nor negative

        aspect_label_key = f"{annotation['aspect']}_{annotation['sentiment']}"
        label = aspect_sentiment_labels.get(aspect_label_key, aspect_sentiment_labels['other'])

        # Attempt to find the annotated text in the original text
        try:
            start_index, end_index, best_score = find_best_match_with_position(text, annotation['text'])
        except Exception as e:
            print(f"Matching not possible for: {annotation} -> {e}")
            continue
        # print(f"Annotation: '{annotation['text']}' -> Start: {start_index}, End: {end_index}, Label: {label}")

        worst_score = min(worst_score, best_score)

        # Align labels with tokens
        for i, (start, end) in enumerate(offset_mapping):
            # print(f"Token: '{tokens[i]}' ({start}, {end}) -> Label: {labels[i]}")
            if start >= start_index and end <= end_index:
                labels[i] = label

    return tokenized_output["input_ids"], labels, worst_score


def find_substring_location(original_text, annotated_text):
    match = process.extractOne(annotated_text, [original_text], scorer=fuzz.partial_ratio)
    start_index = original_text.find(match[0])
    end_index = start_index + len(match[0])
    return start_index, end_index
