import fire
import json
import os


def main(items,
         out_dir="/Users/mroemmele/copa/model_outputs/synthetic/3_shot_1000_items/pre_validation_items/passed_items",
         exemplars="/Users/mroemmele/copa/original_items/4-fixed-exemplars.json"):

    items_file = items
    with open(items_file) as f:
        items = [json.loads(item) for item in f]

    exemplars_file = exemplars
    with open(exemplars_file) as f:
        eval_exemplars = json.load(f)

    passed_items = []
    for item in items:
        if item["auto_status"].startswith("pass"):
            gen_exemplars = item.pop("exemplars")
            item["generation_exemplars"] = gen_exemplars
            item["evaluation_exemplars"] = eval_exemplars
            passed_items.append(item)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    out_file = os.path.join(out_dir, os.path.split(items_file)[-1])

    with open(out_file, "w") as f:
        for item in passed_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Saved {} items to {}".format(len(passed_items), out_file))


if __name__ == "__main__":
    fire.Fire(main)
