import fire
import json
import os
import pandas
import re
import pprint
from sklearn.metrics import cohen_kappa_score


def verify_question(question, item):
    match = re.match(".*\nALTERNATIVE 1: (.+)\nALTERNATIVE 2: (.+)\n",
                     question)
    assert match
    alt1, alt2 = match.groups()
    assert alt1 == item["alternative_1"] and alt2 == item["alternative_2"],\
        "MISMATCH!\nANNOTATOR QUESTION: {}\nJSON ITEM: {}".format(
            question, item)


def parse_choice(text):
    if text.startswith("*Alternative 1*"):
        return "1"
    elif text.startswith("*Alternative 2*"):
        return "2"
    elif text.startswith("*Unclear*"):
        return "0"
    else:
        assert False, "Can't parse choice text: {}".format(text)


def check_content_warning(text):
    if text.endswith("CW"):
        return True
    else:
        return False


def assess_item_validity(model_choice, human_choice):
    if str(model_choice) == str(human_choice):
        return True
    else:
        return False


def add_exemplars(items,
                  dir_with_exemplars):
    exemplars = {}
    for filename in os.listdir(dir_with_exemplars):
        if not filename.endswith(".jsonl"):
            continue
        filepath = os.path.join(dir_with_exemplars, filename)
        with open(filepath) as f:
            for line in f:
                item = json.loads(line)
                exemplars[item["item_id"]] = item["exemplars"]

    for item in items:
        if "exemplars" in item:
            continue
        exmpls = exemplars[item["item_id"]]
        item["exemplars"] = exmpls

    return items


def main(out_file,
         judgement_sets_dir="/Users/mroemmele/gen-copa-research/data/gen-COPA/3_shot_1000_items/stage1_validation_items/subsets",
         select_subsets_with_prefix=None,
         responses_dir="/Users/mroemmele/gen-copa-research/data/gen-COPA/3_shot_1000_items/stage1_validation_items/responses",
         responses_file_template=["synthetic_copa_judgement_sets_{}.form (Responses).xlsx",
                                  "synthetic_copa_judgement_sets_stage1_{}.form (Responses).xlsx"],
         ground_truth_annotator="MSR",
         generic_annotator_alias="CROWDSOURCER",
         exclude_items_with_prefix=["dev"],
         add_exemplars_from_dir="/Users/mroemmele/gen-copa-research/data/gen-COPA/3_shot_1000_items/pre_validation_items/items_with_auto_status",
         cw_items="/Users/mroemmele/gen-copa-research/data/gen-COPA/3_shot_1000_items/stage1_validation_items/cw_items.txt",
         max_autopass_items_per_group=300):

    if cw_items:
        cw_items_file = cw_items
        with open(cw_items_file) as f:
            cw_items = set([item_id.strip() for item_id in f])

    agreement_data = {ground_truth_annotator: [],
                      generic_annotator_alias: []}

    stats = {"n_items": 0,
             "n_cw_items": 0,
             "n_autopass_items": 0,
             "n_autopass_items_no_cw": 0,
             'n_cond_valid_items': 0,
             "n_cond_valid_items_no_cw": 0,
             "percent_autopass": None,
             "percent_cond_valid_among_autopass": None,
             "percent_cond_valid_among_all": None}

    report = {"all": {metric: val for metric, val in stats.items()}}
    report["all"]["agreement_sample_size"] = 0
    all_items = []

    for items_file in os.listdir(judgement_sets_dir):
        if select_subsets_with_prefix and not items_file.startswith(select_subsets_with_prefix):
            continue
        items_name = os.path.splitext(items_file)[0]
        items_file = os.path.join(judgement_sets_dir, items_file)
        if not os.path.splitext(items_file)[-1] == ".jsonl":
            continue

        with open(items_file) as f:
            items = [json.loads(item) for item in f]

        if not "autofail" in items_name:
            if type(responses_file_template) == str:
                responses_file_template = [responses_file_template]
            for template in responses_file_template:
                responses_file = os.path.join(responses_dir,
                                              template.format(items_name))
                if os.path.exists(responses_file):
                    break
            responses = pandas.read_excel(
                responses_file, header=None).T[1:].values
            annotator_ids, responses = responses[0][1:], responses[1:]
            assert len(items) == len(responses)
        else:
            responses = [None] * len(items)

        for item, response in zip(items, responses):
            item_id = item["item_id"]
            item["more_plausible_alternative"] = str(
                item["more_plausible_alternative"])

            group_name, idx = item_id.split(":")

            if group_name in exclude_items_with_prefix:
                continue

            if group_name not in report:
                report[group_name] = {metric: val for metric, val
                                      in stats.items()}

            if max_autopass_items_per_group != None and report[group_name]["n_autopass_items_no_cw"] == max_autopass_items_per_group:
                continue

            for category in ("all", group_name):
                report[category]["n_items"] += 1

            if response is None:
                item["semifinal_status"] = "invalid"
                item["content_warning"] = False
                all_items.append(item)
                continue

            question, choices = response[0], response[1:]
            verify_question(question, item)

            for category in ("all", group_name):
                report[category]["n_autopass_items"] += 1

            has_cw = True if cw_items and item_id in cw_items else False

            for annot_id, choice_text in zip(annotator_ids, choices):
                choice = parse_choice(choice_text)
                if annot_id == ground_truth_annotator:
                    is_valid = assess_item_validity(model_choice=item["more_plausible_alternative"],
                                                    human_choice=choice)
                    item["semifinal_status"] = "valid" if is_valid else "invalid"
                    has_cw = has_cw or check_content_warning(choice_text)
                    item["content_warning"] = has_cw
                    if has_cw:
                        for category in ("all", group_name):
                            report[category]["n_cw_items"] += 1
                    else:
                        for category in ("all", group_name):
                            report[category]["n_autopass_items_no_cw"] += 1
                    if is_valid:
                        for category in ("all", group_name):
                            report[category]["n_cond_valid_items"] += 1
                            if not has_cw:
                                report[category]["n_cond_valid_items_no_cw"] += 1

                if len(annotator_ids) > 1:
                    if annot_id == ground_truth_annotator:
                        report["all"]["agreement_sample_size"] += 1
                        agreement_data[ground_truth_annotator].append(choice)
                    else:
                        agreement_data[generic_annotator_alias].append(choice)

            assert "semifinal_status" in item
            all_items.append(item)

    if add_exemplars_from_dir and "exemplars" not in all_items[0]:
        all_items = add_exemplars(items=all_items,
                                  dir_with_exemplars=add_exemplars_from_dir)

    agree_score = cohen_kappa_score(y1=agreement_data[ground_truth_annotator],
                                    y2=agreement_data[generic_annotator_alias])
    report["all"]["cohen_kappa_agreement_score"] = agree_score

    for category in report:
        report[category]["percent_autopass"] = (report[category]["n_autopass_items"]
                                                / report[category]["n_items"])
        report[category]["percent_cond_valid_among_all"] = (report[category]["n_cond_valid_items"]
                                                            / report[category]["n_items"])
        report[category]["percent_cond_valid_among_autopass"] = (report[category]["n_cond_valid_items"]
                                                                 / report[category]["n_autopass_items"])
    pprint.pprint(report)

    with open(out_file, "w") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("Wrote all items to {}".format(out_file))

    report_file = "{}.report.json".format(os.path.splitext(out_file)[0])
    with open(report_file, "w") as f:
        json.dump(report, f, indent=4)
    print("Wrote items report to {}".format(report_file))


if __name__ == "__main__":
    fire.Fire(main)
