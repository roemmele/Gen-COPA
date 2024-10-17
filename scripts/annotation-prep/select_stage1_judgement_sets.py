import fire
import json
import random
import os


def main(out_file,
         items_dir="/Users/mroemmele/gen-copa-research/data/gen-COPA/3_shot_1000_items/pre_validation_items/passed_items/",
         select_files=["Mistral-7B-v0.3.examples.jsonl",
                       "phi-2.examples.jsonl"],
         n_items_per_file=50,
         exclude_items_in_dir="/Users/mroemmele/gen-copa-research/data/gen-COPA/3_shot_1000_items/stage1_validation_items/subsets/",
         selection_method="random"):

    assert selection_method in ("random", "linear")
    random.seed(123)

    excluded_item_ids = set()
    if exclude_items_in_dir:
        filepaths = []
        for path in os.listdir(exclude_items_in_dir):
            path = os.path.join(exclude_items_in_dir, path)
            if os.path.isdir(path):
                filepaths.extend([os.path.join(path, file)
                                  for file in os.listdir(path) if os.path.splitext(file)[-1] == ".jsonl"])
            elif path.endswith(".jsonl"):
                filepaths.append(path)
        for filepath in filepaths:
            with open(filepath) as f:
                excluded_item_ids.update(
                    [json.loads(item)["item_id"] for item in f])

    selected_passed_items = []
    selected_failed_items = []
    for items_file in os.listdir(items_dir):
        if not items_file.endswith(".jsonl"):
            continue
        if select_files and not items_file in select_files:
            continue
        filepath = os.path.join(items_dir, items_file)
        item_count = 0
        if type(n_items_per_file) == dict:
            if items_file in n_items_per_file:
                n_items = n_items_per_file[items_file]
            else:
                continue
        else:
            n_items = n_items_per_file
        print("Gathering {} items from file {}".format(n_items, filepath))
        if selection_method == "linear":
            with open(filepath) as f:
                for item in f:
                    item = json.loads(item)
                    if item["item_id"] in excluded_item_ids:
                        continue
                    #item.pop("exemplars", None)
                    if item["auto_status"] == "pass":
                        selected_passed_items.append(item)
                    else:
                        import pdb
                        pdb.set_trace()
                        selected_failed_items.append(item)
                    item_count += 1
                    if item_count == n_items:
                        break
        elif selection_method == "random":
            with open(filepath) as f:
                all_items = [json.loads(item) for item in f]
            random.shuffle(all_items)
            for item in all_items:
                if item["item_id"] in excluded_item_ids:
                    continue
                #item.pop("exemplars", None)
                if item["auto_status"] == "pass":
                    selected_passed_items.append(item)
                else:
                    selected_failed_items.append(item)
                item_count += 1
                if item_count == n_items:
                    break

    if selected_passed_items:
        random.shuffle(selected_passed_items)
        with open(out_file, "w") as f:
            f.write("\n".join([json.dumps(item, ensure_ascii=False)
                               for item in selected_passed_items]))
        print("Saved {} auto-passed items to {}".format(
            len(selected_passed_items), out_file))

    if selected_failed_items:
        failed_out_file = os.path.splitext(out_file)[0] + "_autofail.jsonl"
        with open(failed_out_file, "w") as f:
            f.write("\n".join([json.dumps(item, ensure_ascii=False)
                               for item in selected_failed_items]))
        print("Saved {} auto-failed items to {}".format(len(selected_failed_items), failed_out_file))


if __name__ == "__main__":
    fire.Fire(main)
