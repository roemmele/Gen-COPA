import fire
import json
import os


def main(items="/Users/mroemmele/gen-copa-research/data/gen-COPA/3_shot_1000_items/quality_judgement_annotation/all_valid_items.json",
         id_map="/Users/mroemmele/gen-copa-research/data/gen-COPA/3_shot_1000_items/quality_judgement_annotation/all_valid_items.id_map.json",
         annotation_key="high_quality",
         allowed_annotation_values=(0, 1,)):

    items_file = items
    with open(items_file) as f:
        items = json.load(f)

    id_map_file = id_map
    with open(id_map_file) as f:
        id_map = {int(num): label for num,
                  label in json.load(f).items()}

    results = {}
    for item in items:
        real_item_id = id_map[item["i"]]
        group_id = real_item_id.split(":")[0]
        assert item[annotation_key] in allowed_annotation_values
        if group_id not in results:
            results[group_id] = {"n_{}".format(annotation_key): 0,
                                 "n_total": 0}
        results[group_id]["n_{}".format(
            annotation_key)] += item[annotation_key]
        results[group_id]["n_total"] += 1

    for group_id in results:
        results[group_id]["percent_{}".format(annotation_key)] = results[group_id]["n_{}".format(
            annotation_key)] / results[group_id]["n_total"]

    print(json.dumps(results, indent=4))

    results_file = "{}.results.json".format(os.path.splitext(items_file)[0])
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    print("Saved results to {}".format(results_file))


if __name__ == "__main__":
    fire.Fire(main)
