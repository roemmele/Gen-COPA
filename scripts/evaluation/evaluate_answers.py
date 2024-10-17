import json
import fire
import re
import os
import pprint


def main(items):
    items_file = items
    results = {}
    metrics = {"n_items": 0,
               "n_with_recognized_answer": 0,
               "n_open_answer_is_correct": 0,
               "n_forced_answer_is_correct": 0,
               "n_alternative1_is_correct": 0}
    updated_items = []
    with open(items_file) as f:
        for item in f:
            item = json.loads(item)

            if 'output' in item:
                pred_answer = item['output']
            elif 'pred' in item:
                pred_answer = item["pred"]
            else:
                assert False, "answer prediction not found"

            pred_answer = re.search("[1-2]", pred_answer)
            if pred_answer:
                pred_answer = pred_answer.group(0)
                forced_pred_answer = pred_answer
                answer_recognized = True
            else:
                answer_recognized = False
                forced_pred_answer = "1"

            item["answer"] = forced_pred_answer

            if "most_plausible_alternative" in item:
                item["more_plausible_alternative"] = item["most_plausible_alternative"]
                item.pop("most_plausible_alternative")

            gold_answer = str(item["more_plausible_alternative"])
            if gold_answer == "1":
                alt1_is_correct = True
            else:
                alt1_is_correct = False

            if str(pred_answer) == gold_answer:
                open_answer_is_correct = True
            else:
                open_answer_is_correct = False

            if str(forced_pred_answer) == gold_answer:
                forced_answer_is_correct = True
            else:
                forced_answer_is_correct = False

            item["answer_is_correct"] = forced_answer_is_correct

            group_id = item.get("group_id", "all")

            if "eval_set_tags" in item:
                tags = ([tag for tag in item["eval_set_tags"]]
                        + ["{}:{}".format(group_id, tag) for tag in item["eval_set_tags"]])
            else:
                tags = [group_id]

            for tag in tags:
                if tag not in results:
                    results[tag] = {key: val for key, val in metrics.items()}
                results[tag]["n_items"] += 1
                results[tag]["n_with_recognized_answer"] += int(
                    answer_recognized)
                results[tag]["n_open_answer_is_correct"] += int(
                    open_answer_is_correct)
                results[tag]["n_forced_answer_is_correct"] += int(
                    forced_answer_is_correct)
                results[tag]["n_alternative1_is_correct"] += int(
                    alt1_is_correct)

            updated_items.append(item)

    for tag, tag_results in results.items():
        results[tag]["percent_with_recognized_answer"] = (tag_results["n_with_recognized_answer"]
                                                          / tag_results["n_items"])
        results[tag]["open_answer_accuracy"] = (tag_results["n_open_answer_is_correct"]
                                                / tag_results["n_items"])
        results[tag]["forced_answer_accuracy"] = (tag_results["n_forced_answer_is_correct"]
                                                  / tag_results["n_items"])
        results[tag]["alternative1_baseline_accuracy"] = (tag_results["n_alternative1_is_correct"]
                                                          / tag_results["n_items"])

    pprint.pprint(results)

    updated_items_file = (os.path.splitext(items_file)[0]
                          + ".with_answers.jsonl")
    with open(updated_items_file, 'w') as f:
        f.write("\n".join([json.dumps(item, ensure_ascii=False)
                           for item in updated_items]))
    print("Saved items with processed answers to {}".format(updated_items_file))

    results_file = os.path.splitext(items_file)[0] + ".results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print("Saved score results to {}".format(results_file))


if __name__ == '__main__':
    fire.Fire(main)
