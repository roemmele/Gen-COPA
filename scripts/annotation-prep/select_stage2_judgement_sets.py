import fire
import json
import os


def main(out_dir="/Users/mroemmele/gen-copa-research/data/gen-COPA/3_shot_1000_items/stage2_validation_items/subsets/",
         add_subset_prefix="mistral_phi_only",
         items_file="/Users/mroemmele/gen-copa-research/data/gen-COPA/3_shot_1000_items/stage1_validation_items/items_mistral_phi_only_with_semifinal_status.jsonl",
         n_items_per_file=50,
         skip_items=["Llama-2-7b-hf_p0.9:779"]):

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    selected_items = []
    with open(items_file) as f:
        for line in f:
            item = json.loads(line)
            if (item["semifinal_status"] == "valid"
                and not item["content_warning"]
                    and item["item_id"] not in skip_items):
                selected_items.append(item)

    subset_num = 1
    for i in range(0, len(selected_items), n_items_per_file):
        subset = selected_items[i:i + n_items_per_file]
        subset_fp = os.path.join(out_dir, "{}set{}.jsonl".format(
            (add_subset_prefix + "_") if add_subset_prefix else "", subset_num))
        assert not os.path.exists(subset_fp)
        with open(subset_fp, "w") as f:
            f.write("\n".join([json.dumps(item) for item in subset]))
        print("Wrote {} items to {}".format(len(subset), subset_fp))
        subset_num += 1


if __name__ == "__main__":
    fire.Fire(main)
