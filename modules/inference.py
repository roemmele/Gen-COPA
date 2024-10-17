import json
import re
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer
from transformers.pipelines import check_task
from .api import init_api, call_api
from .utils import instantiate_template


def load_tokenizer(name_or_path, token=None):
    tokenizer = AutoTokenizer.from_pretrained(name_or_path,
                                              token=token,
                                              trust_remote_code=True)
    print("Loaded tokenizer '{}'".format(name_or_path))
    if not tokenizer.pad_token:
        print("Setting tokenizer.pad_token to tokenizer.eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(name_or_path,
               task,
               token=None,
               quantization_params={}):
    _, task_defaults, _ = check_task(task)
    model = None
    if quantization_params:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(**quantization_params)
    else:
        quant_config = None
    for model_class in task_defaults["pt"]:
        try:
            model = model_class.from_pretrained(name_or_path,
                                                quantization_config=quant_config,
                                                token=token,
                                                trust_remote_code=True)
            print("Loaded model '{}' for task '{}'".format(name_or_path, task))
            break
        except (OSError, ValueError):
            continue
    assert model, "Failed to load model '{}' for task '{}'".format(
        name_or_path, task)
    return model


def load_pipeline(task,
                  model,
                  device=None,
                  api_key=None,
                  quantization_params={},
                  **kwargs):
    model_name = model
    tokenizer = load_tokenizer(name_or_path=model_name,
                               token=api_key)
    model = load_model(name_or_path=model_name,
                       task=task,
                       token=api_key,
                       quantization_params=quantization_params)
    model_pipeline = pipeline(task=task,
                              model=model,
                              tokenizer=tokenizer,
                              device=device,
                              **kwargs)
    return model_pipeline


def run_pipeline(text,
                 pipeline,
                 generation_params={}):
    output = pipeline(text,
                      return_full_text=False,
                      **generation_params)[0]
    if "generated_text" in output:
        return output["generated_text"].strip()
    elif "translation_text" in output:
        return output["translation_text"].strip()
    else:
        assert False, "Failed to parse HuggingFace pipeline output: {}".format(
            output)


def infer(credentials,
          data,
          model,
          out_file,
          api=None,
          task=None,
          prompt=None,
          endpoint=None,
          device=None,
          output_pattern=None,
          generation_params={},
          quantization_params={},
          on_error=None,
          max_attempts=1,
          timeout=120,
          save_mode='w'):

    if api:
        init_api(credentials, api)
        pipeline = None
    else:
        pipeline = load_pipeline(task=task,
                                 model=model,
                                 device=device,
                                 api_key=credentials.get(
                                     "huggingface", {}).get("api_key", None),
                                 quantization_params=quantization_params)

    n_outputs = 0
    with open(out_file, save_mode) as f:
        for item in tqdm(data, desc="Running model '{}'".format(model)):
            output = infer_item(item=item,
                                model=model,
                                pipeline=pipeline,
                                api=api,
                                task=task,
                                prompt=prompt,
                                endpoint=endpoint,
                                output_pattern=output_pattern,
                                generation_params=generation_params,
                                max_attempts=max_attempts)
            if output["output"] == None:
                if on_error == "skip":
                    print("Failed at item = {}. Setting output to null and skipping.".format(
                        item))
                else:
                    print("Failed at item = {}. Quitting.".format(item))
                    break
            f.write(json.dumps(output, ensure_ascii=False) + "\n")
            n_outputs += 1

    print("Saved {} outputs to {}".format(n_outputs, out_file))


def infer_item(item,
               model,
               pipeline=None,
               api=None,
               task=None,
               prompt=None,
               endpoint=None,
               output_pattern=None,
               generation_params={},
               max_attempts=1,
               timeout=120):

    if 'prompt' in item:
        prompt = item['prompt']

    assert prompt is not None, "No prompt defined in config or data files"

    text = instantiate_template(item=item, template=prompt)

    try:
        if api:
            out = call_api(text=text,
                           model=model,
                           api=api,
                           endpoint=endpoint,
                           generation_params=generation_params,
                           max_attempts=max_attempts,
                           timeout=timeout)
        else:
            out = run_pipeline(text=text,
                               pipeline=pipeline,
                               generation_params=generation_params)
        if output_pattern:
            pattern_match = re.match(pattern=output_pattern, string=out)
            if pattern_match:
                out = pattern_match.groups()
                if len(out) == 1:
                    out = out[0]
    except Exception as e:
        print(e)
        out = None

    output = {}
    if task:
        output['task'] = task
    output.update({**item,
                   'prompt': prompt,
                   'model': model,
                   'generation_params': str(generation_params),
                   'output_pattern': output_pattern,
                   'output': out})

    return output
