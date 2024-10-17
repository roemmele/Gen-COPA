import fire
import json
import os


def load_composition_quality_annotations(folder, annotation_key="high_quality"):
    with open(os.path.join(folder, "all_valid_items.json")) as f:
        annotation_items = json.load(f)
    with open(os.path.join(folder, "all_valid_items.id_map.json")) as f:
        id_map = {int(idx): real_id for idx, real_id in json.load(f).items()}
    for item in annotation_items:
        item["item_id"] = id_map[item["i"]]
    annotation_items = {item["item_id"]: item[annotation_key]
                        for item in annotation_items}
    return annotation_items


def main(out_file,
         stage2_items="../../../data/gen-COPA/3_shot_1000_items/stage2_validation_items/for_reporting_only_items_incl_mistral_phi_with_final_status.jsonl",
         composition_quality_annotations_folder="../../../data/gen-COPA/3_shot_1000_items/quality_judgement_annotation"):

    quality_annotations = load_composition_quality_annotations(
        composition_quality_annotations_folder)

    with open(stage2_items) as f:
        items = [json.loads(line) for line in f]

    for item in items:
        annotation = quality_annotations.get(item["item_id"], None)
        if annotation == 1:
            label = "high-quality"
        elif annotation == 0:
            label = "not-high-quality"
        else:
            label = "n/a"
        item["composition_quality"] = label

    with open(out_file, "w") as f:
        f.write("\n".join([json.dumps(item, ensure_ascii=False)
                           for item in items]))
    print("Saved {} items to {}".format(len(items), out_file))


if __name__ == "__main__":
    fire.Fire(main)
