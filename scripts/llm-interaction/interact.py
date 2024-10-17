import fire
import os
import pprint
import commentjson
import sys
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.realpath(__file__)))))
from modules.inference import infer
from modules.data import read_data
from modules.utils import instantiate_template, get_template_args


def validate_config(config):
    # Required parameters
    required = ['model', 'out_file', 'prompt', 'data_file']
    for arg in required:
        assert arg in config, "missing parameter in config: {}".format(arg)

    # Optional parameters
    config["task"] = config.get("task", None)
    config["api"] = config.get("api", None)
    config["endpoint"] = config.get("endpoint", None)
    config["output_pattern"] = config.get("output_pattern", None)
    config["device"] = config.get("device", None)
    config["on_error"] = config.get("on_error", None)
    config["max_attempts"] = config.get("max_attempts", 1)
    config["timeout"] = config.get("timeout", 120)
    config["credentials"] = config.get("credentials", {})
    config["generation_params"] = config.get("generation_params", {})
    config["quantization_params"] = config.get("quantization_params", {})

    return config


def preview_data(data, config):
    item = data[0]

    print("\nEXAMPLE ITEM:")
    pprint.pprint(item)

    if 'prompt' not in item:
        print("\nPROMPT FOR EXAMPLE ITEM:")
        print(instantiate_template(item=item,
                                   template=config['prompt']))


def main(config):
    config_file = config
    with open(config_file) as f:
        config = commentjson.load(f)

    config = validate_config(config)

    pprint.pprint({key: val for key, val in config.items()
                   if key != 'credentials'})

    if type(config["prompt"]) == list:
        config["prompt"] = "".join(config["prompt"])

    prompt_template_args = get_template_args(config['prompt'])

    data_params = {arg: val for arg, val in config.items()
                   if ((arg.startswith('data') or arg in prompt_template_args) and val != None)}

    data = read_data(params=data_params)
    preview_data(data, config)

    out_dir = os.path.dirname(config['out_file'])
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    infer(credentials=config['credentials'],
          data=data,
          model=config['model'],
          api=config['api'],
          out_file=config['out_file'],
          task=config['task'],
          prompt=config['prompt'],
          endpoint=config['endpoint'],
          device=config["device"],
          output_pattern=config["output_pattern"],
          generation_params=config['generation_params'],
          quantization_params=config["quantization_params"],
          on_error=config['on_error'],
          max_attempts=config['max_attempts'],
          timeout=config['timeout'])


if __name__ == '__main__':
    fire.Fire(main)
