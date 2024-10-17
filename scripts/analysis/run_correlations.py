import fire
import json
import numpy
import itertools
import os
from scipy.stats import spearmanr


def main(scores="/Users/mroemmele/gen-copa-research/scores/scores.json"):

    scores_file = scores
    with open(scores_file) as f:
        scores = json.load(f)

    results = {}

    metrics = scores.keys()
    for metric1, metric2 in itertools.combinations(metrics, r=2):
        metric1_groups = set(scores[metric1].keys())
        metric2_groups = set(scores[metric2].keys())
        groups = metric1_groups.intersection(metric2_groups)
        metric1_scores = numpy.array([scores[metric1][group]
                                      for group in groups])
        metric2_scores = numpy.array([scores[metric2][group]
                                      for group in groups])
        spearmanr_result = spearmanr(metric1_scores, metric2_scores)
        results["spearmanr_{}_{}".format(
            metric1, metric2)] = {"correlation": spearmanr_result.correlation,
                                  "p_value": spearmanr_result.pvalue}

    results_file = "{}.correlations.json".format(
        os.path.splitext(scores_file)[0])
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    print("Saved correlation results to {}".format(results_file))


if __name__ == "__main__":
    fire.Fire(main)
