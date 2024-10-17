import fire
import json
import os


def main(old_dir="/Users/mroemmele/copa/model_outputs/synthetic/3_shot_1000_items/generated_answers/post_validation/old_4_shot_fixed_exemplars",
         new_dir="/Users/mroemmele/copa/model_outputs/synthetic/3_shot_1000_items/generated_answers/post_validation/4_shot_fixed_exemplars",
         items_file="/Users/mroemmele/copa/model_outputs/synthetic/3_shot_1000_items/stage2_validation_items/items_with_final_status.jsonl"):

    with open(items_file) as f:
        items = [json.loads(item) for item in f]

    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)

    for filename in os.listdir(old_dir):
        if filename.endswith("with_answers.jsonl") or not filename.endswith(".jsonl"):
            continue

        old_filepath = os.path.join(old_dir, filename)
        new_filepath = os.path.join(new_dir, filename)

        with open(old_filepath) as f:
            old_items = [json.loads(item) for item in f]
            old_items = {item["item_id"]: item for item in old_items}

        new_items = []
        for i, item in enumerate(items):
            item_id = item["item_id"]
            old_item = old_items[item_id]
            if not item["alternative_1"] == old_item["alternative_1"]:
                print("oh no")
                import pdb
                pdb.set_trace()
            #assert item["alternative_1"] == old_item["alternative_1"]
            assert item["alternative_2"] == old_item["alternative_2"]
            assert item["more_plausible_alternative"] == old_item["more_plausible_alternative"]

            new_item = {**item,
                        "prompt": old_item["prompt"],
                        "model": old_item["model"],
                        "generation_params": old_item["generation_params"],
                        "output": old_item["output"]}
            new_items.append(new_item)

        import pdb
        pdb.set_trace()
        with open(new_filepath, "w") as f:
            f.write("\n".join([json.dumps(item, ensure_ascii=False)
                               for item in new_items]))

        print("Wrote {} items to {}".format(len(new_items), new_filepath))


if __name__ == "__main__":
    fire.Fire(main)
