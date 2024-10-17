from xml.etree import ElementTree
import json
import random
import os


def main(save_dir="../original_items",
         fixed_exemplar_dev_item_ids=["1", "12", "3", "9"],
         n_random_exemplars=3):

    random.seed(123)

    exmp_idxs = {'cause': [], 'effect': []}
    json_items = {'dev': [], 'test': []}
    fixed_exmps = {item_id: None for item_id in fixed_exemplar_dev_item_ids}

    for split in ('dev', 'test',):
        xml_items = ElementTree.parse(
            "../COPA-resources/datasets/copa-{}.xml".format(split)).getroot()
        for i, xml_item in enumerate(xml_items):
            item_id = xml_item.get('id')
            premise, alt1, alt2 = list(xml_item)
            json_item = {'item_id': "{}-{}".format(split, item_id),
                         'asks_for': xml_item.get('asks-for'),
                         'premise': premise.text,
                         'alternative_1': alt1.text,
                         'alternative_2': alt2.text,
                         'more_plausible_alternative': xml_item.get('most-plausible-alternative')}
            json_items[split].append(json_item)
            if split == "dev":
                exmp_idxs[json_item['asks_for']].append(i)
                if item_id in fixed_exmps:
                    fixed_exmps[item_id] = json_item

    fixed_exmps = [fixed_exmps[item_id]
                   for item_id in fixed_exemplar_dev_item_ids]

    fixed_ex_set_filepath = os.path.join(save_dir,
                                         "{}-fixed-exemplars.json".format(len(fixed_exemplar_dev_item_ids)))
    with open(fixed_ex_set_filepath, "w") as f:
        json.dump(fixed_exmps, f, indent=4)
    print("Saved exemplar set to {}".format(fixed_ex_set_filepath))

    for split in ('dev', 'test',):

        filepath = os.path.join(save_dir, "{}.jsonl".format(split))
        with open(filepath, "w") as f:
            for item_i, item in enumerate(json_items[split]):
                f.write(json.dumps(item) + "\n")
        print("Saved {} items to {}".format(len(json_items[split]),
                                            filepath))

        fixed_ex_filepath = os.path.join(save_dir,
                                         "{}-with-{}-fixed-exemplars.jsonl".format(split, len(fixed_exemplar_dev_item_ids)))
        with open(fixed_ex_filepath, "w") as f:
            for item_i, item in enumerate(json_items[split]):
                updated_item = {**item,
                                "exemplars": fixed_exmps}
                f.write(json.dumps(updated_item) + "\n")
        print("Saved {} items to {}".format(len(json_items[split]),
                                            fixed_ex_filepath))

        rand_ex_filepath = os.path.join(save_dir,
                                        "{}-with-{}-random-exemplars.jsonl".format(split, n_random_exemplars))
        with open(rand_ex_filepath, "w") as f:
            for item_i, item in enumerate(json_items[split]):
                asks_for = item['asks_for']
                rand_idxs = random.sample([rand_i for rand_i in exmp_idxs[asks_for]
                                           if (split == "test" or rand_i != item_i)],
                                          k=n_random_exemplars)
                updated_item = {**item,
                                'exemplars': [json_items["dev"][rand_i] for rand_i in rand_idxs]}
                f.write(json.dumps(updated_item) + "\n")
        print("Saved {} items to {}".format(len(json_items[split]),
                                            rand_ex_filepath))


if __name__ == "__main__":
    main()
