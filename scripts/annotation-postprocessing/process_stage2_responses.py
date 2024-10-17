import fire
import json
import os
import pandas
import re
import pprint
import random
import numpy
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa


def fleiss_kappa_score(labels):
    table = numpy.array([numpy.array(rater_labels)
                         for rater_labels in labels.values() if len(rater_labels)]).T
    cat_table, _ = aggregate_raters(table)
    score = fleiss_kappa(cat_table)
    return score


def verify_question(question, item):
    match = re.match(".*\nALTERNATIVE 1: (.+)\nALTERNATIVE 2: (.+)\n",
                     question)
    assert match
    alt1, alt2 = match.groups()
    assert alt1 == item["alternative_1"] and alt2 == item["alternative_2"],\
        "MISMATCH!\n FORM QUESTION: {}\nSAVED ITEM: {}".format(question, item)


def parse_choice(text):
    if text.startswith("*Alternative 1*"):
        return "1"
    elif text.startswith("*Alternative 2*"):
        return "2"
    else:
        assert False, "Can't parse choice text: {}".format(text)


def assess_item_validity(expert_choice,
                         novice1_choice,
                         novice2_choice,
                         model_choice):
    if (novice1_choice == novice2_choice
        and novice2_choice == expert_choice
            and novice2_choice == model_choice):
        return True
    else:
        return False


def reassign_plausibility_labels(items,
                                 legacy_more_plausible_alts=None):
    assert len(items) % 2 == 0
    n_items_per_alt_order = int(len(items) / 2)
    alt_labels = ([{'alternative_1': 'more_plausible_alternative',
                    'alternative_2': 'less_plausible_alternative'}] * n_items_per_alt_order
                  + [{'alternative_1': 'less_plausible_alternative',
                      'alternative_2': 'more_plausible_alternative'}] * n_items_per_alt_order)
    random.shuffle(alt_labels)

    for item, labels in zip(items, alt_labels):
        old_alt1 = item["alternative_1"]
        old_alt2 = item["alternative_2"]
        old_more_plausible_alt = item["more_plausible_alternative"]

        if legacy_more_plausible_alts and item["item_id"] in legacy_more_plausible_alts:
            # import pdb
            # pdb.set_trace()
            if legacy_more_plausible_alts[item["item_id"]] == "1":
                labels = {'alternative_1': 'more_plausible_alternative',
                          'alternative_2': 'less_plausible_alternative'}
            else:
                assert legacy_more_plausible_alts[item["item_id"]] == "2"
                labels = {'alternative_1': 'less_plausible_alternative',
                          'alternative_2': 'more_plausible_alternative'}

        legend = {"more_plausible_alternative": None,
                  "less_plausible_alternative": None}
        if item["more_plausible_alternative"] == "1":
            legend["more_plausible_alternative"] = item["alternative_1"]
            legend["less_plausible_alternative"] = item["alternative_2"]
        elif item["more_plausible_alternative"] == "2":
            legend["more_plausible_alternative"] = item["alternative_2"]
            legend["less_plausible_alternative"] = item["alternative_1"]
        else:
            assert False

        new_more_plausible_alternative = ("1" if labels['alternative_1'] == "more_plausible_alternative"
                                          else "2")
        new_alternative1 = legend.pop(labels['alternative_1'])
        new_alternative2 = legend.pop(labels['alternative_2'])

        item['alternative_1'] = new_alternative1
        item['alternative_2'] = new_alternative2
        item['more_plausible_alternative'] = new_more_plausible_alternative

        if new_more_plausible_alternative != old_more_plausible_alt:
            assert old_alt1 == item["alternative_2"]
            assert old_alt2 == item["alternative_1"]

    assert (sum(1 for item in items if item["more_plausible_alternative"] == "1")
            == sum(1 for item in items if item["more_plausible_alternative"] == "2"))

    return items


def create_balanced_eval_sets(items,
                              legacy_more_plausible_alts=None):
    item_sets = {}
    for item in items:
        item_id = item["item_id"]
        group_id = item["group_id"]
        item["eval_set_tags"] = []
        if group_id not in item_sets:
            item_sets[group_id] = {"invalid": [],
                                   "valid": []}
        status = item["status"]
        item_sets[group_id][status].append(item)

    first_k_valid = numpy.min([len(sets["valid"])
                               for sets in item_sets.values()])
    # Ensure first k is even number
    first_k_valid = first_k_valid - (first_k_valid % 2)
    k_per_alt_order = int(first_k_valid / 2)

    balanced_items = []
    for group_id in item_sets:
        valid_set = item_sets[group_id]["valid"]
        n_valid = len(valid_set)

        invalid_set = item_sets[group_id]["invalid"]
        n_invalid = len(invalid_set)

        # failed_set = item_sets[group_id]["fail"]
        # n_failed = len(failed_set)

        item_excluded_from_valid_set = None
        if n_valid % 2:  # Odd number of valid items - remove one
            rand_i = random.sample(range(n_valid), k=1)[0]
            item_excluded_from_valid_set = valid_set.pop(rand_i)

        valid_set = reassign_plausibility_labels(items=valid_set,
                                                 legacy_more_plausible_alts=legacy_more_plausible_alts)

        first_k_valid_set = []
        first_k_valid_set_distr = {"1": 0, "2": 0}

        for item in valid_set:
            if sum(first_k_valid_set_distr.values()) < first_k_valid:
                if first_k_valid_set_distr[item["more_plausible_alternative"]] < k_per_alt_order:
                    item["eval_set_tags"].append(
                        "valid-first-{}".format(first_k_valid))
                    first_k_valid_set_distr[item["more_plausible_alternative"]] += 1

            item["eval_set_tags"].extend(["valid"])

        item_excluded_from_total_set = None
        if n_invalid % 2:
            if item_excluded_from_valid_set:  # Odd number of invalid items, and one has already been excluded from the valid set, so add it to the total set
                invalid_set.append(item_excluded_from_valid_set)
            else:  # Odd number of invalid items, and there's no items already excluded from valid set -  just remove one item from invalid set
                rand_i = random.sample(range(n_invalid), k=1)[0]
                item_excluded_from_total_set = invalid_set.pop(rand_i)

        elif item_excluded_from_valid_set:  # Even number of invalid items but an item has been excluded from the valid set - so remove the same item from the total set
            item_excluded_from_total_set = item_excluded_from_valid_set

        invalid_set = reassign_plausibility_labels(items=invalid_set,
                                                   legacy_more_plausible_alts=legacy_more_plausible_alts)

        # for item in invalid_set:
        #     item["eval_set_tags"].append("pass")

        model_items = valid_set + invalid_set
        if item_excluded_from_total_set:
            model_items.append(item_excluded_from_total_set)
        # model_items.extend(failed_set)
        assert len(model_items) == n_valid + n_invalid  # + n_failed

        # verify_balanced(items=[item for item in model_items
        #                        if "pass" in item["eval_set_tags"]])
        verify_balanced(items=[item for item in model_items
                               if "valid" in item["eval_set_tags"]])
        verify_balanced(items=[item for item in model_items
                               if "valid-first-{}".format(first_k_valid) in item["eval_set_tags"]])

        balanced_items.extend(model_items)

    # verify_balanced(items=[item for item in balanced_items
    #                        if "pass" in item["eval_set_tags"]])
    verify_balanced(items=[item for item in balanced_items
                           if "valid" in item["eval_set_tags"]])
    verify_balanced(items=[item for item in balanced_items
                           if "valid-first-{}".format(first_k_valid) in item["eval_set_tags"]])

    return balanced_items


def verify_balanced(items):
    gold_alts = [int(item["more_plausible_alternative"]) for item in items]
    baseline_alts = [1 for item in items]
    is_correct = numpy.array(gold_alts) == numpy.array(baseline_alts)
    assert numpy.mean(is_correct) == 0.5


def main(out_file,
         stage1_items_file="/Users/mroemmele/gen-copa-research/data/gen-COPA/3_shot_1000_items/stage1_validation_items/items_mistral_phi_only_with_semifinal_status.jsonl",
         judgement_sets_dir="/Users/mroemmele/gen-copa-research/data/gen-COPA/3_shot_1000_items/stage2_validation_items/subsets",
         select_subsets_with_prefix="mistral_phi_only_set",
         responses_dir="/Users/mroemmele/gen-copa-research/data/gen-COPA/3_shot_1000_items/stage2_validation_items/responses",
         responses_file_template="synthetic_copa_judgement_sets_stage2_{}.form (Responses).xlsx",
         new_exemplars_file="/Users/mroemmele/gen-copa-research/data/orig-COPA/dev/exemplars/4-fixed-exemplars.json",
         expert_annotator="MSR",
         balanced_eval_sets=True,
         separate_valid_items=True,
         legacy_more_plausible_alts=None):

    if legacy_more_plausible_alts:
        legacy_more_plausible_alts_file = legacy_more_plausible_alts
        with open(legacy_more_plausible_alts) as f:
            legacy_more_plausible_alts = json.load(f)

    agreement_data = {"expert-annotator": [],
                      "novice-annotator1": [],
                      "novice-annotator2": []}

    stats = {"n_items": 0,
             "n_valid_items": 0,
             "percent_valid": None}

    report = {"all": {metric: val for metric, val in stats.items()}}
    report["all"]["agreement_sample_size"] = 0

    final_items = []
    with open(stage1_items_file) as f:
        for line in f:
            item = json.loads(line)

            group_id = item["item_id"].split(":")[0]
            item["group_id"] = group_id

            if group_id not in report:
                report[group_id] = {metric: val for metric, val
                                    in stats.items()}

            auto_status = item.pop("auto_status")
            item["status"] = auto_status

            if auto_status != "pass" or item["content_warning"]:
                continue

            item["status"] = "invalid"

            item.pop("semifinal_status")

            final_items.append(item)

            for category in ("all", group_id):
                report[category]["n_items"] += 1

    final_items = {item["item_id"]: item for item in final_items}

    observed_item_ids = set()
    for set_file in os.listdir(judgement_sets_dir):
        if select_subsets_with_prefix and not set_file.startswith(select_subsets_with_prefix):
            continue

        set_name = os.path.splitext(set_file)[0]
        set_file = os.path.join(judgement_sets_dir, set_file)
        if not set_file.endswith(".jsonl"):
            continue

        with open(set_file) as f:
            judgement_set = [json.loads(item) for item in f]

        responses_file = os.path.join(responses_dir,
                                      responses_file_template.format(set_name))
        if not os.path.exists(responses_file):
            continue
        print("Checking {}...".format(set_file))

        responses = pandas.read_excel(responses_file,
                                      header=None).T[1:].values
        annotator_ids, responses = responses[0][1:], responses[1:]
        assert len(judgement_set) == len(responses)

        for judgement_item, response in zip(judgement_set, responses):
            if judgement_item["item_id"] not in final_items:
                print("No judgement found for item {}, removing".format(
                    judgement_item["item_id"]))
                continue

            item = final_items[judgement_item["item_id"]]
            item_id = item["item_id"]
            group_id = item["group_id"]

            if item_id in observed_item_ids:
                print("Encountered duplicate judgement for item {}".format(item_id))
                continue

            observed_item_ids.add(item_id)

            question, choices = response[0], response[1:]
            verify_question(question, item)

            expert_choice = None
            novice_choices = []
            report["all"]["agreement_sample_size"] += 1
            annotator_num = 0
            for annot_id, choice_text in zip(annotator_ids, choices):
                choice = parse_choice(choice_text)
                if annot_id == expert_annotator:
                    expert_choice = choice
                    agreement_data["expert-annotator"].append(choice)
                else:
                    annotator_num += 1
                    novice_choices.append(choice)
                    agreement_data["novice-annotator{}".format(
                        annotator_num)].append(choice)
            if not expert_choice:
                expert_choice = item["more_plausible_alternative"]
                agreement_data["expert-annotator"].append(expert_choice)

            is_valid = assess_item_validity(expert_choice=expert_choice,
                                            novice1_choice=novice_choices[0],
                                            novice2_choice=novice_choices[1],
                                            model_choice=item["more_plausible_alternative"])

            if is_valid:
                final_items[item_id]["status"] = "valid"
                for category in ("all", group_id):
                    report[category]["n_valid_items"] += 1

    final_items = list(final_items.values())

    novice_agreement_score = cohen_kappa_score(y1=agreement_data["novice-annotator1"],
                                               y2=agreement_data["novice-annotator2"])
    report["all"]["cohen_kappa_between_novice_annotators"] = novice_agreement_score
    if len(agreement_data["expert-annotator"]):
        full_agreement_score = fleiss_kappa_score(labels=agreement_data)
        report["all"]["fleiss_kappa_between_all_annotators"] = full_agreement_score

    for category in report:
        report[category]["percent_valid"] = (report[category]["n_valid_items"]
                                             / report[category]["n_items"])
    pprint.pprint(report)

    if balanced_eval_sets:
        random.seed(123)
        final_items = create_balanced_eval_sets(final_items,
                                                legacy_more_plausible_alts=legacy_more_plausible_alts)

    if new_exemplars_file:
        with open(new_exemplars_file) as f:
            new_exemplars = json.load(f)
    else:
        new_exemplars = None

    with open(out_file, "w") as f:
        for item in final_items:
            gen_exemplars = item.pop("exemplars")
            item["generation_exemplars"] = gen_exemplars
            if "evaluation_exemplars" not in item:
                if new_exemplars:
                    item["evaluation_exemplars"] = new_exemplars
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("Wrote all items to {}".format(out_file))

    if separate_valid_items:
        valid_out_file = "{}.valid_only.jsonl".format(
            os.path.splitext(out_file)[0])
        with open(valid_out_file, "w") as f:
            for item in final_items:
                if "valid" in item["eval_set_tags"]:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print("Wrote valid items to {}".format(valid_out_file))

    report_file = "{}.report.json".format(os.path.splitext(out_file)[0])
    with open(report_file, "w") as f:
        json.dump(report, f, indent=4)
    print("Wrote items report to {}".format(report_file))


if __name__ == "__main__":
    fire.Fire(main)
