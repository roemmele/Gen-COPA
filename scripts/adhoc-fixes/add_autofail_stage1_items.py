import fire
import os
import json
import random


def main(save_to,
         items_dir="/Users/mroemmele/copa/model_outputs/synthetic/3_shot_1000_items/initial_judgement_sets/items",
         examples_dir="/Users/mroemmele/copa/model_outputs/synthetic/3_shot_1000_items/items_by_model/examples",
         examples_file_template="{}.examples.jsonl"):

    n_items_per_model = {}
    for items_file in os.listdir(items_dir):
        if not os.path.splitext(items_file)[-1] == ".jsonl":
            continue
        if "autofail" in items_file:
            continue
        items_fp = os.path.join(items_dir, items_file)
        with open(items_fp) as f:
            items = [json.loads(item) for item in f]
        for item in items:
            item_id = item["item_id"]
            model_name = item_id.split(":")[0]
            if model_name not in n_items_per_model:
                n_items_per_model[model_name] = 0
            n_items_per_model[model_name] += 1

    random.seed(123)
    # Add a new set with the proportional number of auto-failed items
    failed_items = []
    for model_name, n_items in n_items_per_model.items():
        if model_name == "dev":
            continue
        with open(os.path.join(examples_dir, examples_file_template.format(model_name))) as f:
            items = [json.loads(item) for item in f]
        sample_items = random.sample(items, k=n_items)
        for item in sample_items:
            if item["auto_status"].startswith("fail"):
                item.pop("exemplars")
                failed_items.append(item)

    with open(save_to, "w") as f:
        f.write("\n".join([json.dumps(item) for item in failed_items]))
    print("Saved {} auto-failed items to {}".format(len(failed_items), save_to))


if __name__ == '__main__':
    fire.Fire(main)
