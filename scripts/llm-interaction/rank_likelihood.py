import fire
import json
import torch
import numpy
import requests
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from jinja2 import Environment

headers = {"Authorization": "Bearer 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"}


@torch.no_grad()
def main(items,
         out_file,
         model,
         endpoint=None,
         answer_patterns={"1": "Premise: {{premise}}\n{% if asks_for == 'cause' %}What was the cause of this?{% else %}What happened as a result?{% endif %}\nMore Plausible Alternative: {{alternative_1}}\nLess Plausible Alternative: {{alternative_2}}",
                          "2": "Premise: {{premise}}\n{% if asks_for == 'cause' %}What was the cause of this?{% else %}What happened as a result?{% endif %}\nMore Plausible Alternative: {{alternative_2}}\nLess Plausible Alternative: {{alternative_1}}"}
         ):

    items_file = items
    with open(items_file) as f:
        items = [json.loads(item) for item in f]

    model_name = model
    if not endpoint:
        device = (torch.device("cuda" if torch.cuda.is_available()
                               else "mps" if torch.backends.mps.is_available()
                               else "cpu"))
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     trust_remote_code=True).to(device)
        print("Loaded model {} on local device {}".format(model_name, device))
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        device, model, tokenizer = None, None, None

    print("Answer patterns for first item:")
    for label, pattern in answer_patterns.items():
        print("LABEL = {}".format(label))
        print(Environment().from_string(pattern).render(**items[0]))
        print("--------------------------")

    for item in tqdm(items, desc="Predicting answers"):
        if not item["premise"]:
            item["output"] = ""
            continue
        scores = {"labels": [],
                  "scores": []}
        for label, pattern in answer_patterns.items():
            input_text = Environment().from_string(pattern).render(**item)
            if endpoint:
                response = requests.post(endpoint,
                                         headers=headers,
                                         json={"inputs": input_text})
                score = response.json()
            else:
                inputs = tokenizer(input_text,
                                   return_tensors="pt")
                input_ids = inputs.input_ids.to(device)
                attention_mask = inputs.attention_mask.to(device)
                score = model(input_ids=input_ids,
                              attention_mask=attention_mask,
                              labels=input_ids).loss.item()
            scores["labels"].append(label)
            scores["scores"].append(score)

        pred_label = scores["labels"][numpy.argmin(scores["scores"])]
        item["pattern"] = answer_patterns[pred_label]
        item["model"] = model_name
        item["output"] = str(pred_label)

    with open(out_file, 'w') as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    print("Saved {} items with answer predictions to {}".format(len(items), out_file))


if __name__ == "__main__":
    fire.Fire(main)
