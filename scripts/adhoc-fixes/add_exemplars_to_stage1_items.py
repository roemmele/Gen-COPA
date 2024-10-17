import fire
import json
import os


def main(old_file="/Users/mroemmele/copa/model_outputs/synthetic/3_shot_1000_items/stage1_validation_items/old_items_with_semifinal_status.jsonl",
         new_file="/Users/mroemmele/copa/model_outputs/synthetic/3_shot_1000_items/stage1_validation_items/items_with_semifinal_status.jsonl",
         dir_with_exemplars="/Users/mroemmele/copa/model_outputs/synthetic/3_shot_1000_items/pre_validation_items/items_with_auto_status"):

    exemplars = {}

    for filename in os.listdir(dir_with_exemplars):
        if not filename.endswith(".jsonl"):
            continue
        filepath = os.path.join(dir_with_exemplars, filename)
        with open(filepath) as f:
            for line in f:
                item = json.loads(line)
                exemplars[item["item_id"]] = item["exemplars"]

    with open(old_file) as f:
        items = [json.loads(item) for item in f]

    for item in items:
        exmpls = exemplars[item["item_id"]]
        item["exemplars"] = exmpls

    with open(new_file, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    fire.Fire(main)
