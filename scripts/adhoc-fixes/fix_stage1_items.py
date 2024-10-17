import fire
import os
import json
import random


def main(old_items_dir="/Users/mroemmele/copa/model_outputs/synthetic/3_shot_1000_items/human_judgement_sets/old_items",
         new_items_dir="/Users/mroemmele/copa/model_outputs/synthetic/3_shot_1000_items/human_judgement_sets/items",
         examples_dir="/Users/mroemmele/copa/model_outputs/synthetic/3_shot_1000_items/items_by_model/examples",
         examples_file_template="{}.examples.jsonl"):

    # Verify new examples match existing ones
    examples = {}
    for examples_file in os.listdir(examples_dir):
        if not examples_file.endswith(".jsonl"):
            continue
        with open(os.path.join(examples_dir, examples_file)) as f:
            for line in f:
                ex = json.loads(line)
                ex.pop("exemplars")
                examples[ex["item_id"]] = ex

    if not os.path.isdir(new_items_dir):
        os.makedirs(new_items_dir)
    n_items_per_model = {}
    for items_file in os.listdir(old_items_dir):
        if not os.path.splitext(items_file)[-1] == ".jsonl":
            continue
        old_items_fp = os.path.join(old_items_dir, items_file)
        with open(old_items_fp) as f:
            items = [json.loads(item) for item in f]
        for item in items:
            old_item_id = item["item_id"]
            elements = old_item_id.split("-")
            model_name = "-".join(elements[:-1])
            idx = elements[-1]
            if model_name not in n_items_per_model:
                n_items_per_model[model_name] = 0
            n_items_per_model[model_name] += 1
            if "origin_item_id" in item:
                new_item_id = "{}:{}".format(
                    model_name, item["origin_item_id"])
            else:
                new_item_id = "{}:{}".format(model_name, idx)
            item["item_id"] = new_item_id

            item["auto_status"] = "pass"
            item.pop("origin_item_id", None)

            if not new_item_id.startswith("dev"):
                assert item == examples[new_item_id]

        new_items_fp = os.path.join(new_items_dir, items_file)
        with open(new_items_fp, "w") as f:
            f.write("\n".join([json.dumps(item, ensure_ascii=False)
                               for item in items]))
        print("Saved amended human judgement items to {}".format(new_items_fp))

    # random.seed(123)
    # # Add a new set with the proportional number of auto-failed items
    # failed_items = []
    # for model_name, n_items in n_items_per_model.items():
    #     if model_name == "dev":
    #         continue
    #     with open(os.path.join(examples_dir, examples_file_template.format(model_name))) as f:
    #         items = [json.loads(item) for item in f]
    #     sample_items = random.sample(items, k=n_items)
    #     for item in sample_items:
    #         if item["auto_status"].startswith("fail"):
    #             item.pop("exemplars")
    #             failed_items.append(item)

    # import pdb
    # pdb.set_trace()
    # autofail_set_fp = os.path.join(new_items_dir, "autofail_set.jsonl")
    # with open(autofail_set_fp, "w") as f:
    #     f.write("\n".join([json.dumps(item) for item in failed_items]))
    # print("Saved {} auto-failed items to {}".format(len(failed_items), autofail_set_fp))


if __name__ == '__main__':
    fire.Fire(main)
