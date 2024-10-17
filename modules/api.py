import os
import json
import requests
import time
import re


def init_api(credentials, api):
    if api == 'openai':
        global openai
        import openai
        init_openai(organization=credentials['openai']['organization'],
                    api_key=credentials['openai']['api_key'])
    elif api == 'huggingface':
        init_huggingface(api_key=credentials['huggingface']['api_key'])
    elif api == 'cohere':
        global cohere
        import cohere
        init_cohere(api_key=credentials['cohere']['api_key'])
    elif api == 'anthropic':
        global anthropic
        import anthropic
        init_anthropic(api_key=credentials['anthropic']['api_key'])
    elif api == 'sagemaker-huggingface':
        global sagemaker
        import sagemaker
        pass
    else:
        assert False, "API type '{}' not recognized".format(api)


def init_openai(organization, api_key):
    openai.organization = organization
    openai.api_key = api_key


def init_huggingface(api_key):
    global authorization_headers
    authorization_headers = {"Authorization": "Bearer {}".format(api_key)}


def init_cohere(api_key):
    global api_client
    api_client = cohere.Client(api_key)


def init_anthropic(api_key):
    global api_client
    api_client = anthropic.Client(api_key)


def init_sagemaker_huggingface(endpoint_name):
    global api_client
    session = sagemaker.Session()
    api_client = sagemaker.huggingface.model.HuggingFacePredictor(endpoint_name=endpoint_name,
                                                                  sagemaker_session=session)


def call_api(text,
             model,
             api,
             endpoint,
             generation_params={},
             max_attempts=1,
             timeout=120,
             wait_secs=10):
    for attempt in range(1, max_attempts + 1):
        try:
            if api == "openai":
                output = call_openai(text=text,
                                     model=model,
                                     generation_params=generation_params)
            elif api == "huggingface":
                output = call_huggingface(text=text,
                                          model=model,
                                          endpoint_url=endpoint,
                                          generation_params=generation_params,
                                          timeout=timeout)
            elif api == "cohere":
                output = call_cohere(text=text,
                                     model=model,
                                     generation_params=generation_params)
            elif api == "anthropic":
                output = call_anthropic(text=text,
                                        model=model,
                                        generation_params=generation_params)
            elif api == "sagemaker-huggingface":
                output = call_sagemaker_huggingface(text=text,
                                                    model=model,
                                                    generation_params=generation_params,
                                                    endpoint_name=endpoint)
            return output
        except Exception as e:
            print("ERROR: {}".format(e))
            if attempt == max_attempts:
                print("API call (attempt {}/{}) failed. Quitting.".format(
                    attempt, max_attempts))
                return None
            print("API call (attempt {}/{}) failed. Retrying in {} seconds...".format(
                attempt, max_attempts, wait_secs))
            time.sleep(wait_secs)


def call_openai(text,
                model,
                generation_params={}):
    try:
        if model.startswith('gpt-'):
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{'content': text, 'role': 'user'}],
                **generation_params)
            output = response['choices'][0]['message']['content'].strip()
        else:
            response = openai.Completion.create(
                model=model,
                prompt=text,
                **generation_params)
            output = response['choices'][0]['text'].strip()
        return output
    except Exception as e:
        raise Exception(str(e))


def call_cohere(text,
                model,
                generation_params={}):
    try:
        response = api_client.generate(
            model=model,
            prompt=text,
            **generation_params)
        output = response.generations[0].text.strip()
        return output
    except Exception as e:
        if hasattr(e, "http_status") and e.http_status == 498:  # Blocked input
            output = ""
            return output
        raise Exception(str(e))


def call_anthropic(text,
                   model,
                   generation_params={}):
    if 'max_tokens' in generation_params:
        generation_params['max_tokens_to_sample'] = generation_params.pop(
            'max_tokens')
    try:
        response = api_client.completion(
            model=model,
            prompt="{} {}{}".format(
                anthropic.HUMAN_PROMPT, text, anthropic.AI_PROMPT),
            stop_sequences=[anthropic.HUMAN_PROMPT],
            **generation_params)
        output = response['completion'].strip()
        return output
    except Exception as e:
        raise Exception(str(e))


def call_sagemaker_huggingface(text,
                               model,
                               endpoint_name,
                               generation_params={}):
    try:
        if not api_client:
            init_sagemaker_huggingface(endpoint_name)
        response = api_client.predict({"inputs": text,
                                       "parameters": {"return_full_text": False,
                                                      "max_length": None,
                                                      **generation_params}})
        output = parse_huggingface_response(response=response)
        return output
    except Exception as e:
        raise Exception(str(e))


def call_huggingface(text,
                     model,
                     endpoint_url,
                     generation_params={},
                     timeout=120):
    if 'temperature' in generation_params:
        print("WARNING: The HuggingFace Inference API may not be correctly applying the temperature parameter. "
              "Unexpected output has observed when temperature is set close to 0 (the API requires positive values > 0). "
              "This particular setting should be equivalent to greedy decoding, "
              "but it yields different results compared to when greedy decoding is used explicitly (i.e. when do_sample=False). "
              "Use this parameter at your own risk.")
    try:
        response = requests.post(endpoint_url,
                                 headers=authorization_headers,
                                 json={"inputs": text,
                                       "parameters": {"return_full_text": False,
                                                      **generation_params}},
                                 timeout=timeout)
        output = parse_huggingface_response(response=response.json())
        return output
    except Exception as e:
        raise Exception(str(e))


def parse_huggingface_response(response):
    if 'error' in response:
        raise Exception(str(response))
    item = response[0]
    if 'generated_text' in item:
        return item['generated_text'].strip()
    elif 'translation_text' in item:
        return item['translation_text'].strip()
    else:
        assert False, "Failed to parse HuggingFace API response: {}".format(
            item)
