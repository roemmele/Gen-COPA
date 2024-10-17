import json
import random
import fire
import itertools


def main(out_file,
         items="/Users/mroemmele/copa/original_items/dev.jsonl",
         extension_items=None,
         n_items_per_ask_type=500,
         n_exemplars=3):

    random.seed(123)

    exmp_idxs = {'cause': [],
                 'effect': []}

    items_file = items
    dev_items = []
    with open(items_file) as f:
        for item_i, item in enumerate(f):
            try:
                item = json.loads(item)
            except:
                print(item)
                import pdb
                pdb.set_trace()
            exmp_idxs[item['asks_for']].append(item_i)
            dev_items.append(item)

    if extension_items:
        ext_items_file = extension_items
        with open(ext_items_file) as f:
            extension_items = itertools.cycle([json.loads(item) for item in f])

    with open(out_file, "w") as f:
        for asks_for in exmp_idxs:
            for item_i in range(n_items_per_ask_type):
                rand_idxs = random.sample(exmp_idxs[asks_for], k=n_exemplars)
                if extension_items:
                    ext_item = next(extension_items)
                item = {**ext_item,
                        **{'asks_for': asks_for,
                           'exemplars': [dev_items[rand_i]
                                         for rand_i in rand_idxs]}}
                f.write(json.dumps(item) + "\n")
    print("Saved {} items ({} per ask type) to {}".format(
        n_items_per_ask_type * 2, n_items_per_ask_type, out_file))


if __name__ == '__main__':
    fire.Fire(main)
