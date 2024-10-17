import fire
import json
import random
import os


def main(out_file,
         items="/Users/mroemmele/gen-copa-research/data/gen-COPA/3_shot_1000_items/stage2_validation_items/items_incl_mistral_phi_with_final_status.valid_only.jsonl",
         select_items_with_status=None,
         shuffle=True,
         annotation_key="high_quality",
         default_label=2):

    items_file = items
    with open(items_file) as f:
        items = [json.loads(item) for item in f]

    if shuffle:
        random.seed(123)
        random.shuffle(items)

    masked_to_real_ids = {}
    out_items = []
    i = 0
    for item in items:
        if not select_items_with_status or item["status"] == select_items_with_status:
            if item["more_plausible_alternative"] == "1":
                labels = {'more_plausible_alternative': 'alternative_1',
                          'less_plausible_alternative': 'alternative_2'}
            else:
                assert item["more_plausible_alternative"] == "2"
                labels = {'less_plausible_alternative': 'alternative_1',
                          'more_plausible_alternative': 'alternative_2'}

            if item["asks_for"] == "effect":
                asks_for = "What happened as a result?"
            else:
                asks_for = "What was the cause of this?"

            out_item = {"i": i,
                        "premise": item["premise"],
                        "asks_for": asks_for,
                        "more_plausible_alternative": item[labels["more_plausible_alternative"]],
                        "less_plausible_alternative": item[labels["less_plausible_alternative"]],
                        annotation_key: default_label}
            out_items.append(out_item)
            masked_to_real_ids[i] = item["item_id"]
            i += 1

    with open(out_file, "w") as f:
        json.dump(out_items, f, indent=4)
    print("Saved {} items to {}".format(len(out_items), out_file))

    ids_file = "{}.id_map.json".format(os.path.splitext(out_file)[0])
    with open(ids_file, "w") as f:
        json.dump(masked_to_real_ids, f, indent=4)
    print("Saved id lookup for items to {}".format(ids_file))


if __name__ == "__main__":
    fire.Fire(main)
