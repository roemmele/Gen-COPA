import fire
import json
import pprint
import os


def main(items="/Users/mroemmele/copa/model_outputs/synthetic/3_shot_1000_items/stage2_validation_items/items_for_annotation.invalid_only*.json"):

    items_file = items
    with open(items_file) as f:
        items = json.load(f)

    results = {"n_annotated_items": 0,
               "annotation_counts": {}}

    for item in items:
        if not item["annotations"]:
            continue

        results["n_annotated_items"] += 1

        if "group_id" in item:
            group_id = item["group_id"]
        else:
            group_id = item["item_id"].split(":")[0]

        values = item["annotations"]

        for category in ("all", group_id):
            if category not in results["annotation_counts"]:
                results["annotation_counts"][category] = {}
            for value in values:
                if value not in results["annotation_counts"][category]:
                    results["annotation_counts"][category][value] = 0
                results["annotation_counts"][category][value] += 1

    pprint.pprint(results)

    results_file = "{}.results.json".format(os.path.splitext(items_file)[0])
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    print("Saved annotation results to {}".format(results_file))


if __name__ == "__main__":
    fire.Fire(main)
