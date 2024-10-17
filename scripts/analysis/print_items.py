import fire
import json
from jinja2 import Environment


def main(out_file="/Users/mroemmele/copa/model_outputs/synthetic/3_shot_1000_items/stage2_validation_items/items_with_final_status.txt",
         items="/Users/mroemmele/copa/model_outputs/synthetic/3_shot_1000_items/stage2_validation_items/items_with_final_status.jsonl",
         template="Premise: {{premise}}\n{% if asks_for == 'cause' %}What was the cause of this?{% else %}What happened as a result?{% endif %}\nMore Plausible Alternative: {% if more_plausible_alternative == '1' %}{{alternative_1}}{% else %}{{alternative_2}}{% endif %}\nLess Plausible Alternative: {% if more_plausible_alternative == '1' %}{{alternative_2}}{% else %}{{alternative_1}}{% endif %}"):

    items_file = items
    with open(items_file) as f:
        items = [json.loads(item) for item in f]

    text = ""
    for item in items:
        # if item["more_plausible_alternative"] == "1":
        #     labels = {'more_plausible_alternative': 'alternative_1',
        #               'less_plausible_alternative': 'alternative_2'}
        # else:
        #     assert item["more_plausible_alternative"] == "2"
        #     labels = {'less_plausible_alternative': 'alternative_1',
        #               'more_plausible_alternative': 'alternative_2'}

        text += "{}\n".format(item["item_id"])
        text += "Status: {}\n".format(item["status"])
        text += Environment().from_string(template).render(**dict(item.items())) + "\n\n"

    with open(out_file, "w") as f:
        f.write(text)

    print("Printed items to {}".format(out_file))


if __name__ == "__main__":
    fire.Fire(main)
