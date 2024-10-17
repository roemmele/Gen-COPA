import json
import re
import random
import fire
import os
import pprint
from nltk.metrics.distance import edit_distance
from jinja2 import Environment, meta


def extract_pattern_labels(pattern):
    pattern_labels = meta.find_undeclared_variables(
        Environment().parse(pattern))  # Get labels as an unordered set
    # Find the order the labels appear in
    label_idxs = [pattern.index(label) for label in pattern_labels]
    pattern_labels = [label for _, label
                      in sorted(zip(label_idxs, pattern_labels))]  # Restore the order
    return pattern_labels


def convert_pattern_to_regex(pattern, item, placeholder="<<!!**PLACEHOLDER**!!>>"):
    label_regexs = {}
    pattern_labels = extract_pattern_labels(pattern)
    assert placeholder not in pattern
    for label in pattern_labels:
        if label in item:
            label_regexs[label] = item[label]
        else:
            label_regexs[label] = placeholder
    regex = Environment().from_string(pattern).render(**label_regexs)
    pattern_labels = [label for label in pattern_labels
                      if label_regexs[label] == placeholder]
    assert len(regex.split(placeholder)) == len(pattern_labels + [None])
    regex = "".join([span + ("(?P<{}>.+)".format(label) if label else "")
                     for span, label in zip(regex.split(placeholder), pattern_labels + [None])])
    return regex


def get_orig_copa_items(filepaths):
    items = []
    for filepath in filepaths:
        with open(filepath) as f:
            for item in f:
                item = json.loads(item)
                if int(item["more_plausible_alternative"]) == 1:
                    more_plausible_alternative = item["alternative_1"]
                    less_plausible_alternative = item["alternative_2"]
                else:
                    more_plausible_alternative = item["alternative_2"]
                    less_plausible_alternative = item["alternative_1"]

                items.append({"premise": item["premise"],
                              "more_plausible_alternative": more_plausible_alternative,
                              "less_plausible_alternative": less_plausible_alternative})
    return items


def convert_item_to_token_set(item):
    token_set = set("{} {} {}".format(item['premise'],
                                      item['more_plausible_alternative'],
                                      item['less_plausible_alternative']).lower().split())
    return token_set


def exceeds_overlap_threshold(item, token_sets, threshold=0.8):
    new_token_set = convert_item_to_token_set(item)
    for i, token_set in enumerate(token_sets):
        overlap_tokens = new_token_set.intersection(token_set)
        score = len(overlap_tokens) / len(new_token_set)
        if score >= threshold:
            return i, score
    return None, None


def has_loop(item):
    if (item["premise"] == item["more_plausible_alternative"]
        or item["premise"] == item["less_plausible_alternative"]
            or item["more_plausible_alternative"] == item["less_plausible_alternative"]):
        return True
    return False


def main(items,
         output_pattern="{{premise}}\n{% if asks_for == 'cause' %}What was the cause of this\?{% else %}What happened as a result\?{% endif %}\nMore Plaus[ai]ble Alternative: {{more_plausible_alternative}}\nLess Plaus[ai]ble Alternative: {{less_plausible_alternative}}",
         save_report_to_dir="./data/example_outputs/",
         save_passed_examples_to_dir="./data/example_outputs/pre-validation-passed_items/",
         plagiarism_threshold=1.0,
         repetition_threshold=1.0,
         orig_copa_dev="./data/orig-COPA/dev.jsonl",
         orig_copa_test="./data/orig-COPA/test.jsonl",
         legacy_examples_dir=None,
         add_new_exemplars="./data/orig-COPA/4-fixed-exemplars.json"):

    random.seed(123)
    items_file = items
    model_name = os.path.splitext(os.path.split(items_file)[-1])[0]
    if not os.path.isdir(save_report_to_dir):
        os.makedirs(save_report_to_dir)
    if not os.path.isdir(save_passed_examples_to_dir):
        os.makedirs(save_passed_examples_to_dir)

    report = {'criteria': {'output_pattern': output_pattern,
                           'repetition_threshold': repetition_threshold,
                           'plagiarism_threshold': plagiarism_threshold},
              **{asks_for: {'total': 0,
                            'malformed': 0,
                            'repeated': 0,
                            'plagiarized': 0,
                            'looped': 0} for asks_for in ('cause', 'effect')}}

    orig_copa_items = get_orig_copa_items(
        filepaths=[orig_copa_dev, orig_copa_test])
    orig_item_token_sets = [convert_item_to_token_set(item)
                            for item in orig_copa_items]
    new_item_token_sets = []
    examples = []
    passed_examples = []

    with open(items_file) as f:
        for i, item in enumerate(f):
            item = json.loads(item)
            asks_for = item['asks_for']

            example = {'item_id': item["item_id"],
                       'asks_for': asks_for,
                       'exemplars': item["exemplars"],
                       'auto_status': None,
                       'premise': None,
                       'more_plausible_alternative': None,
                       'less_plausible_alternative': None}
            report[asks_for]['total'] += 1

            regex = convert_pattern_to_regex(pattern=output_pattern, item=item)
            output_match = re.match(regex, item['output'])

            if not output_match:
                report[asks_for]['malformed'] += 1
                example["auto_status"] = 'fail-malformed'
                examples.append(example)
                continue

            new_item = {label: segment.strip()
                        for label, segment in output_match.groupdict().items()}

            example['premise'] = new_item["premise"]
            example['more_plausible_alternative'] = new_item["more_plausible_alternative"]
            example['less_plausible_alternative'] = new_item["less_plausible_alternative"]

            is_plagiarized, plagiarism_score = exceeds_overlap_threshold(item=new_item,
                                                                         token_sets=orig_item_token_sets,
                                                                         threshold=plagiarism_threshold)
            if is_plagiarized != None:
                print("PLAGIARISM ALERT! (SCORE = {})".format(plagiarism_score))
                print("   generated item:", {key: val for key, val in example.items()
                                             if key in ("premise", "less_plausible_alternative", "more_plausible_alternative")})
                print("    original item:", orig_copa_items[is_plagiarized])
                print()
                report[asks_for]["plagiarized"] += 1
                example["auto_status"] = "fail-plagiarized"
                examples.append(example)
                continue

            is_repeated, repetition_score = exceeds_overlap_threshold(item=new_item,
                                                                      token_sets=new_item_token_sets,
                                                                      threshold=repetition_threshold)
            if is_repeated != None:
                print("REPETITION ALERT! (SCORE = {})".format(repetition_score))
                print("   generated item:", new_item)
                print("    existing item:",
                      {key: val for key, val in passed_examples[is_repeated].items()
                       if key in ("premise", "less_plausible_alternative", "more_plausible_alternative")})
                print()
                report[asks_for]["repeated"] += 1
                example["auto_status"] = "fail-repeated"
                examples.append(example)
                continue

            is_looped = has_loop(new_item)
            if is_looped:
                print("LOOP ALERT!")
                print("   generated item:", new_item)
                print()
                report[asks_for]["looped"] += 1
                example["auto_status"] = "fail-looped"
                examples.append(example)
                continue

            example["auto_status"] = "pass"
            examples.append(example)

            new_item_token_sets.append(convert_item_to_token_set(new_item))
            passed_examples.append(example)

    statused_items_file = os.path.join(save_report_to_dir,
                                       (os.path.splitext(os.path.split(items_file)[-1])[0]
                                        + ".jsonl"))
    with open(statused_items_file, "w") as f:
        f.write("\n".join([json.dumps(example) for example in examples]))
    print("Saved {} items with automatically marked status to {}".format(
        len(examples), statused_items_file))

    report['cause_and_effect'] = {}
    for asks_for in ('cause', 'effect'):
        fail_categories = [cat for cat in report[asks_for] if cat != "total"]
        n_failed = sum([report[asks_for][cat] for cat in fail_categories])
        report[asks_for]["failed ({})".format(
            " + ".join(fail_categories))] = n_failed
        report[asks_for]["passed"] = report[asks_for]["total"] - n_failed
        for cat in report[asks_for]:
            if cat not in report['cause_and_effect']:
                report['cause_and_effect'][cat] = 0
            report['cause_and_effect'][cat] += report[asks_for][cat]

    pprint.pprint(report)
    report_file = os.path.join(save_report_to_dir,
                               os.path.splitext(os.path.split(items_file)[-1])[0] + ".report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=4)
    print("Saved validation status report to {}".format(report_file))

    if legacy_examples_dir:
        legacy_alt_labels = {}
        for legacy_file in os.listdir(legacy_examples_dir):
            if legacy_file.startswith(model_name):
                with open(os.path.join(legacy_examples_dir, legacy_file)) as f:
                    legacy_examples = [json.loads(item) for item in f]
                    for ex in legacy_examples:
                        ex_item_id = "{}:{}".format(model_name,
                                                    ex["origin_item_id"])
                        if ex["more_plausible_alternative"] == 1:
                            legacy_alt_labels[ex_item_id] = {'alternative_1': 'more_plausible_alternative',
                                                             'alternative_2': 'less_plausible_alternative'}
                        else:
                            legacy_alt_labels[ex_item_id] = {'alternative_1': 'less_plausible_alternative',
                                                             'alternative_2': 'more_plausible_alternative'}
                break
        if not legacy_alt_labels:
            print("Warning: no legacy alternative labels file was loaded")
    else:
        legacy_alt_labels = None

    # Filter non-passing examples
    examples = [example for example in examples
                if example["auto_status"].startswith("pass")]

    n_items_per_alt_order = int(len(examples) / 2)
    alt_labels = ([{'alternative_1': 'more_plausible_alternative',
                    'alternative_2': 'less_plausible_alternative'}] * n_items_per_alt_order
                  + [{'alternative_1': 'less_plausible_alternative',
                      'alternative_2': 'more_plausible_alternative'}] * n_items_per_alt_order)
    if len(examples) % 2:
        alt_labels.append(alt_labels[0])
    random.shuffle(alt_labels)

    if add_new_exemplars:
        with open(add_new_exemplars) as f:
            new_exemplars = json.load(f)
    else:
        new_exemplars = None

    for i, example in enumerate(examples):
        example["item_id"] = "{}:{}".format(model_name, example["item_id"])
        if legacy_alt_labels and example["item_id"] in legacy_alt_labels:
            labels = legacy_alt_labels[example["item_id"]]
        else:
            labels = alt_labels[i]
        more_plausible_alternative = 1 if labels['alternative_1'] == "more_plausible_alternative" else 2
        alternative1 = example.pop(labels['alternative_1'])
        alternative2 = example.pop(labels['alternative_2'])

        example['alternative_1'] = alternative1
        example['alternative_2'] = alternative2
        example['more_plausible_alternative'] = more_plausible_alternative
        if new_exemplars:
            example["evaluation_exemplars"] = new_exemplars

    examples_file = os.path.join(save_passed_examples_to_dir,
                                 (os.path.splitext(os.path.split(items_file)[-1])[0]
                                  + ".examples.jsonl"))
    with open(examples_file, "w") as f:
        f.write("\n".join([json.dumps(example) for example in examples]))
    print("Saved {} examples (equally balanced between ask type and alternatives order) to {}".format(
        len(examples), examples_file))


if __name__ == '__main__':
    fire.Fire(main)
