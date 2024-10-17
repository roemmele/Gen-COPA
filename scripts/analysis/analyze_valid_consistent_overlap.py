import fire
import json
import os


def main(items,
         selector_prefix=None):

    items_file = items
    with open(items_file) as f:
        items = [json.loads(item) for item in f]

    if selector_prefix:
        items = [item for item in items
                 if item["item_id"].startswith(selector_prefix)]

    results = {"n_consistent": 0,
               "n_valid": 0,
               "n_valid_and_consistent": 0}
    for item in items:
        if item["answer_is_correct"]:
            results["n_consistent"] += 1
        if item["status"] == "valid":
            results["n_valid"] += 1
            if item["answer_is_correct"]:
                results["n_valid_and_consistent"] += 1

    results["percent_valid_given_consistent"] = (results["n_valid_and_consistent"] /
                                                 results["n_consistent"])
    results["percent_consistent_given_valid"] = (results["n_valid_and_consistent"] /
                                                 results["n_valid"])

    results_file = "{}.valid_consistent_overlap.json".format(
        os.path.splitext(items_file)[0])
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    print(results)


if __name__ == "__main__":
    fire.Fire(main)
