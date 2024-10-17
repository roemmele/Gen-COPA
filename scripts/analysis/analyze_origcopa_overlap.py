import json
import fire
import os
import numpy
from tqdm import tqdm
from nltk import word_tokenize
# Note there is more than one library named rouge; this one comes from "pip install py-rouge"
from rouge import Rouge


def encode_segment_ngrams(segment, n=3):
    tokens = [token.lower() for token in word_tokenize(segment)]
    ngrams = set(zip(*[tokens[i:] for i in range(n)]))
    return ngrams


def encode_copa_item_ngrams(item, n=3):
    ngrams = set()
    for seg_label in ("premise", "more_plausible_alternative", "less_plausible_alternative",):
        seg_ngrams = encode_segment_ngrams(item[seg_label], n=n)
        ngrams.update(seg_ngrams)
    return ngrams


def get_max_sim_origcopa_item(gen_copa_items, orig_copa_items):
    maxsim_scores = []
    scorer = Rouge(['rouge-n'], max_n=3)
    orig_copa_texts = ["{} {} {}".format(item["premise"], item["more_plausible_alternative"], item["less_plausible_alternative"])
                       for item in orig_copa_items]
    gen_copa_texts = ["{} {} {}".format(item["premise"], item["more_plausible_alternative"], item["less_plausible_alternative"])
                      for item in gen_copa_items]
    for gen_copa_text in tqdm(gen_copa_texts, desc="Calculating ROUGE"):
        scores = [scorer.get_scores(gen_copa_text, orig_copa_text)["rouge-3"]["f"]
                  for orig_copa_text in orig_copa_texts]
        max_score = numpy.max(scores)
        maxsim_scores.append(max_score)
    # print(numpy.array(maxsim_scores))
    return maxsim_scores


def get_copa_items(filepaths):
    items = []
    for filepath in filepaths:
        with open(filepath) as f:
            for item in f:
                item = json.loads(item)
                if int(item["more_plausible_alternative"]) == 1:
                    more_plausible_alternative = item["alternative_1"]
                    less_plausible_alternative = item["alternative_2"]
                else:
                    more_plausible_alternative = item["alternative_2"]
                    less_plausible_alternative = item["alternative_1"]

                items.append({"premise": item["premise"],
                              "more_plausible_alternative": more_plausible_alternative,
                              "less_plausible_alternative": less_plausible_alternative})
    return items


def main(items,
         n=3,
         orig_copa_dev="/Users/mroemmele/gen-copa-research/data/orig-COPA/dev/input_items/dev.jsonl",
         orig_copa_test="/Users/mroemmele/gen-copa-research/data/orig-COPA/test/input_items/test.jsonl"):

    orig_copa_items = get_copa_items(
        filepaths=[orig_copa_dev, orig_copa_test])
    orig_copa_ngram_set = set([ngram for item in orig_copa_items
                               for ngram in encode_copa_item_ngrams(item, n=n)])

    items_file = items
    gen_copa_items = get_copa_items(filepaths=[items_file])

    gen_copa_ngrams = []
    for item in gen_copa_items:
        item_ngrams = encode_copa_item_ngrams(item, n=n)
        gen_copa_ngrams.append(item_ngrams)

    results = {}
    # for item_ngrams in gen_copa_ngrams:
    #     common_ngrams = orig_copa_ngram_set.intersection(item_ngrams)
    #     print(common_ngrams)
    #     percent_common = len(orig_copa_ngram_set.intersection(
    #         item_ngrams)) / len(item_ngrams)
    #     print(percent_common)
    #     percent_unique = 1 - percent_common
    #     import pdb
    #     pdb.set_trace()
    results["mean_unique_ngram_rate_per_item"] = numpy.mean([(1 - len(orig_copa_ngram_set.intersection(item_ngrams)) / len(item_ngrams))
                                                             for item_ngrams in gen_copa_ngrams])
    ngram_set = set([ngram for item_ngrams in gen_copa_ngrams
                     for ngram in item_ngrams])
    results["overall_rate_unique_ngrams"] = 1 - \
        len(orig_copa_ngram_set.intersection(ngram_set)) / len(ngram_set)

    results["mean_maxsim_score"] = numpy.mean(
        get_max_sim_origcopa_item(gen_copa_items, orig_copa_items))

    print(json.dumps(results, indent=4))

    results_file = "{}.origcopa_overlap.json".format(
        os.path.splitext(items_file)[0])
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    print("Saved analysis to {}".format(results_file))


if __name__ == "__main__":
    fire.Fire(main)
