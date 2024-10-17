# Gen-COPA

This repo contains code and data associated with the paper "From Test-Taking to Test-Making: Examining LLM Authoring of Commonsense Assessment Items", published in the Findings track of EMNLP 2024.

This work explores using LLMs to generate items in the style of the [Choice of Plausible Alternatives (COPA) evaluation benchmark](https://asgordon.github.io/copa.html). The main finding of the paper is that LLMs that obtain high performance in answering COPA are also better at writing their own COPA items. See the paper for details.

## Final Dataset of Gen-COPA items

In this work, we used an assortment of open-weight LLMs to generate new COPA items, referred to as "Gen-COPA" to distinguish them from items in the original benchmark. All Gen-COPA items with their annotated validity and composition quality labels are here in ./data/gen-COPA/all.jsonl. They are also available on the Huggingface Hub at [huggingface.co/datasets/roemmele/Gen-COPA](https://huggingface.co/datasets/roemmele/Gen-COPA).

## Example commands to reproduce experiments/analyses in paper

Install library dependencies: `pip install -r requirements.txt`

You'll need a Huggingface API key (https://huggingface.co/settings/tokens) to run these examples. Make sure you add this key to each of the configs used by the scripts below (i.e. under "credentials" > "huggingface" > "api_key"). 

The examples use the free phi-2 endpoint on HuggingFace (https://huggingface.co/microsoft/phi-2). Note you may need to wait for the endpoint to initialize, and there's no guarantee of its future availability. Adjust the "endpoint" attribute in the config with your own endpoint to ensure you can run the examples. 

### Orig-COPA answering performance:

#### Apply LLM to answer orig-COPA items:

```python scripts/llm-interaction/interact.py -config configs/answer_orig_copa_4_shot.json```

#### Evaluate LLM answers to orig-COPA items:

```python scripts/evaluation/evaluate_answers.py -items ./data/example_outputs/phi-2_orig-COPA-test-with-4-fixed-exemplars.jsonl```

### Synthesizing Gen-COPA set:

#### Apply LLM to synthesize Gen-COPA items:

```python scripts/llm-interaction/interact.py -config configs/synthesize_gen_copa.json```

#### Post-process LLM outputs for Gen-COPA task to assign automatic pass/fail status and gather passed items into a single file:

```python scripts/llm-output-processing/process_model_outputs.py -items ./data/example_outputs/phi-2_gen-copa_items.jsonl```

### Gen-COPA consistency:

#### For consistency analysis, apply LLM to answer passed Gen-COPA items:

```python scripts/llm-interaction/interact.py -config configs/consistency_gen_copa_4_shot.json```

#### Compute consistency by evaluating LLM answers to Gen-COPA items:

```python scripts/evaluation/evaluate_answers.py -items ./data/example_outputs/pre_validation_consistency/phi-2_gen-copa_items.jsonl```

### Gen-COPA validity and composition quality:

The validity and compositition quality annotation tasks were done manually using both Google Forms and local text editors. 

### Answering performance on valid Gen-COPA:

After obtaining the validity annotations, the same scripts cited above were used to compute the model accuracy performance on the Gen-COPA items that were annotated as valid.

