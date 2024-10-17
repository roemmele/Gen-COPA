import fire
import json
import os


def main(copy_from_dir="/Users/mroemmele/copa/model_outputs/synthetic/3_shot_1000_items/generated_answers/pre_validation_consistency",
         save_to_dir="/Users/mroemmele/copa/model_outputs/synthetic/3_shot_1000_items/generated_answers/post_validation_all_annotated",
         required_suffix="with_answers.jsonl",
         reference_items="/Users/mroemmele/copa/model_outputs/synthetic/3_shot_1000_items/stage2_validation_items/items_with_final_status.jsonl"):

    reference_items_file = reference_items
    with open(reference_items_file) as f:
        reference_items = [json.loads(item) for item in f]
        reference_items = {item["item_id"]: item for item in reference_items}

    for filename in os.listdir(copy_from_dir):
        if not filename.endswith(required_suffix):
            continue
        filepath = os.path.join(copy_from_dir, filename)
        new_items = []
        with open(filepath) as f:
            items = [json.loads(item) for item in f]
            for item in items:
                if item["item_id"] not in reference_items:
                    continue
                ref_item = reference_items[item["item_id"]]
                item["status"] = ref_item["status"]
                new_items.append(item)
        new_filepath = os.path.join(save_to_dir, filename)
        with open(new_filepath, "w") as f:
            f.write("\n".join([json.dumps(item, ensure_ascii=False)
                               for item in new_items]))
        print("Saved {} items to {}".format(len(new_items), new_filepath))


if __name__ == "__main__":
    fire.Fire(main)
