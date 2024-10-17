import fire
import json
import numpy
import torch
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
from transformers import pipeline


@torch.no_grad()
def main(items,
         out_file,
         model):

    items_file = items
    with open(items_file) as f:
        items = [json.loads(item) for item in f]

    device = (torch.device("cuda" if torch.cuda.is_available()
                           else "mps" if torch.backends.mps.is_available()
                           else "cpu"))
    featurizer = pipeline(model=model,
                          task="feature-extraction",
                          device=device)

    scores = {"labels": [], "scores": []}
    for item in tqdm(items, desc="Predicting answers"):

        if not item["premise"]:
            item["output"] = ""
            continue

        scores = {"labels": [],
                  "scores": []}

        premise_vector = featurizer(item["premise"],
                                    return_tensors=True).sum(axis=1)
        for label in ("1", "2"):
            alternative_vector = featurizer(item["alternative_{}".format(label)],
                                            return_tensors=True).sum(axis=1)
            score = cosine_similarity(premise_vector,
                                      alternative_vector).item()
            scores["labels"].append(label)
            scores["scores"].append(score)

        pred_label = scores["labels"][numpy.argmax(scores["scores"])]
        item["model"] = model
        item["scores"] = str(scores)
        item["output"] = pred_label

    with open(out_file, 'w') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("Saved {} items with answer predictions to {}".format(len(items), out_file))


if __name__ == "__main__":
    fire.Fire(main)
