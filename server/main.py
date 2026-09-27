from fastapi import FastAPI
import ast
from datetime import datetime
import json
import os, yaml
from pathlib import Path
from pydantic import BaseModel
import time
from typing import Union
import uvicorn

from openai import OpenAI
from utils import standardize, change_name
from qos_simulator import QoSSimulator, parse_bool

from slowapi.errors import RateLimitExceeded
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

config_file='config.yml'
CONFIG = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)
CACHE_FOLDER = CONFIG['cache_folder']
GENERATED_CACHE_FOLDER = CONFIG.get(
    "generated_cache_folder", "../data/generated_cache/solvable_v1/responses"
)
LOG_FILE = os.getenv("SERVER_LOG_FILE") or CONFIG['log_file']
SERVER_PORT = int(os.getenv("SERVER_PORT") or CONFIG["port"])
# The API simulator is configured independently from the inference agent.
# Environment variables take precedence; config.yml is a legacy fallback.
SIMULATOR_API_KEY = os.getenv("SIMULATOR_API_KEY") or CONFIG.get("api_key")
SIMULATOR_API_BASE = os.getenv("SIMULATOR_API_BASE") or CONFIG.get("api_base") or "https://api.openai.com/v1"
SIMULATOR_MODEL = os.getenv("SIMULATOR_MODEL") or CONFIG.get("model", "gpt-4-turbo")
SIMULATOR_SEED = int(os.getenv("SIMULATOR_SEED") or CONFIG.get("simulator_seed", 42))
SIMULATOR_TEMPERATURE = float(
    os.getenv("SIMULATOR_TEMPERATURE") or CONFIG.get("temperature", 0)
)

SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
QOS_PROFILES_FILE = os.getenv("QOS_PROFILES_FILE") or CONFIG.get(
    "qos_profiles_file", "../data/qos/v5/api_qos_profiles.jsonl"
)
if not os.path.isabs(QOS_PROFILES_FILE):
    QOS_PROFILES_FILE = os.path.normpath(os.path.join(SERVER_DIR, QOS_PROFILES_FILE))
QOS_SIMULATOR = QoSSimulator.from_file(
    path=Path(QOS_PROFILES_FILE),
    enabled=parse_bool(os.getenv("QOS_ENABLED"), default=CONFIG.get("qos_enabled", False)),
    scenario=os.getenv("QOS_PROFILE") or CONFIG.get("qos_profile", "normal"),
    seed=int(os.getenv("QOS_SEED") or CONFIG.get("qos_seed", 42)),
    sleep_enabled=parse_bool(
        os.getenv("QOS_SLEEP_ENABLED"),
        default=CONFIG.get("qos_sleep_enabled", True),
    ),
)

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class Info(BaseModel):
    category: str
    tool_name: str
    api_name: str
    tool_input: Union[str, dict]
    strip: str
    toolbench_key: str

def prepare_tool_identifiers(info):
    category = info.category
    standard_category = category.replace(" ", "_").replace(",", "_").replace("/", "_")
    while " " in standard_category or "," in standard_category:
        standard_category = standard_category.replace(" ", "_").replace(",", "_")
    standard_category = standard_category.replace("__", "_")
    
    tool_name = info.tool_name
    api_name = change_name(standardize(info.api_name)).split(f"_for_{tool_name}")[0]
    if not tool_name.endswith(f"_for_{standard_category}"):
        tool_name = standardize(info.tool_name)
        tool_name += f"_for_{standard_category}"
    return tool_name, standard_category, api_name


def qos_api_id(standard_category, tool_name, api_name):
    tool_id = standardize(tool_name.split("_for_")[0])
    return f"{standard_category}/{tool_id}/{api_name}"


def finish_tool_call(info, response, response_type, qos):
    if isinstance(response, str):
        response = json.loads(response)
    response = QOS_SIMULATOR.resolve(response, qos)
    write_log(request=info, response=response, type=response_type)
    return response


def load_cache_examples(*cache_paths):
    examples = {}
    for cache_path in cache_paths:
        if not os.path.isfile(cache_path):
            continue
        try:
            with open(cache_path, encoding="utf-8") as source:
                loaded = json.load(source)
            if isinstance(loaded, dict):
                examples.update(loaded)
        except Exception as error:
            print(f"Loading cache examples error from {cache_path}: {error}")
    return examples


def canonical_tool_input(value):
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            try:
                value = json.loads(value)
            except (TypeError, json.JSONDecodeError):
                return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def write_log(request, response, type):
    log = """\
>>>>>>>>>>>>>>>>>>>>>>>
TIME: {curr_time}
TYPE: {type}
REQUEST: {request}
RESPONSE: {response}
<<<<<<<<<<<<<<<<<<<<<<<
"""
    try:
        with open(LOG_FILE, "a") as f:
            curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(log.format(curr_time=curr_time, type=type, request=request, response=response))
    except OSError as error:
        print(f"Unable to write server log {LOG_FILE}: {error}")


@app.post('/virtual')
def get_virtual_response(info: Info):
    tool_name, standard_category, api_name = prepare_tool_identifiers(info)
    tool_input = info.tool_input
    tool_name_original = info.tool_name

    if api_name == "chat_with_user":
        response_dict = {"error": "", "response": "Chat with user."}
        write_log(request=info, response=response_dict, type="chat_with_user")
        return response_dict

    api_id = qos_api_id(standard_category, tool_name_original, api_name)
    
    try:
        tool_input = json.loads(tool_input)
    except Exception as e:
        if tool_input == "":
            tool_input = {}
        elif isinstance(tool_input, dict):
            tool_input = tool_input
        else:
            print(f"Can not parse tool input into json: {tool_input}")
            print(type(tool_input))
            print(tool_input)
            response_dict = {"error": f"Tool input parse error...\n", "response": ""}
            write_log(request=info, response=response_dict, type="tool_input_parse_error")
            return response_dict

    qos = QOS_SIMULATOR.prepare_call(api_id)
    if not QOS_SIMULATOR.should_generate_response(qos):
        QOS_SIMULATOR.wait_for_latency(qos)
        return finish_tool_call(info, {}, "qos_rejected_before_llm", qos)

    # Cached responses are optional few-shot examples. They are never returned
    # directly: every virtual response is generated by the LLM simulator.
    official_cache_path = os.path.join(
        CACHE_FOLDER, standard_category, tool_name, api_name + ".json"
    )
    generated_cache_path = os.path.join(
        GENERATED_CACHE_FOLDER, standard_category, tool_name, api_name + ".json"
    )
    cache = load_cache_examples(official_cache_path, generated_cache_path)

    # parse api_doc
    tool_name_original = standardize(tool_name_original)
    api_name = standardize(api_name)
    api_doc = {
        'tool_description': "",
        'api_info': "",
    }
    try:
        if os.path.exists(os.path.join(CONFIG['tools_folder'], standard_category)):
            if os.path.exists(os.path.join(CONFIG['tools_folder'], standard_category, tool_name_original.split("_for_")[0]+".json")):
                # read json
                api_intro = json.load(open(os.path.join(CONFIG['tools_folder'], standard_category, tool_name_original.split("_for_")[0]+".json"), "r"))
                # get tool_dexcription and api_info
                tool_description = api_intro['tool_description']
                api_info = []
                for api in api_intro['api_list']:
                    if api_name == standardize(api['name']):
                        api_info.append({
                            'name': api['name'],
                            'description': api['description']
                        })
                # check invalid api name
                if len(api_info) == 0:
                    print("cant match api name")
                api_doc = {
                    'tool_description': tool_description,
                    'api_info': api_info
                }
            else:
                print(f"cant get {tool_name_original}")
    except Exception as e:
        print(f"Loading api_doc error: {e}")

    # Use only examples for other inputs. Passing an exact cache match would
    # let the simulator copy the stored response instead of generating one.
    example_num = 5
    tool_input_key = canonical_tool_input(tool_input)
    api_example = [
        item
        for item in cache.items()
        if canonical_tool_input(item[0]) != tool_input_key
    ][:example_num]
    while len(str(api_example)) > 2048 and example_num > 1:
        example_num -= 1
        api_example = list(cache.items())[:example_num]

    print(f"generating virtual response for {api_id}; examples={len(api_example)}")
        
    generation_started = time.monotonic()
    result = fake_response_function_chat(api_example,tool_input,api_doc)
    generation_elapsed = time.monotonic() - generation_started
    QOS_SIMULATOR.wait_for_latency(qos, elapsed_seconds=generation_elapsed)

    return finish_tool_call(info, result, "llm_virtual_response", qos)
    
def is_valid_json(result):
    """
    Checks if the given string is valid JSON.

    Args:
      data: The string to be checked.

    Returns:
      True if the string is valid JSON, False otherwise.
    """
    # check json format
    try:
        result = json.loads(result)
        return True
    except Exception as e:
        print(f"Can not parse result into json: {result}")
        return False

def fake_response_function_chat(api_example, tool_input, api_doc):
    '''
    api_example: list of tuple, [(input, output), ...]
    tool_input: dict, input of the tool
    api_doc: dict, api document
    '''
    system_prompt = '''
Imagine you are an API Server operating within a specialized tool, which contains a collection of distinct APIs. Your role is to deeply understand the function of each API based on their descriptions in the API documentation. As you receive specific inputs for individual API calls within this tool, analyze these inputs to determine their intended purpose. Your task is to craft a JSON formatted response that aligns with the expected output of the API, guided by the provided examples.\n
Your responses must adhere to a specific JSON structure, which is as follows:\n
{
    "error": "",
    "response": "<Your_Response>"
}\n
The error field should remain empty, indicating no errors in processing. The response field should contain the content you formulate based on the API's functionality and the input provided. Ensure that your responses are meaningful, directly addressing the API's intended functionality. If the provided examples are mostly error messages or lack substantial content, use your judgment to create relevant and accurate responses. The key is to maintain the JSON format's integrity while ensuring that your response is an accurate reflection of the API's intended output within the tool.\n
Please note that your answer should not contain anything other than a json format object, which should be parsable directly to json.
Note that:
- your response should be around 100 to 200 words, containing rich information given the api input parameters. Keep Your answer short and simple.
- your response must be effective and have practical content.
- if the api response example if null or ineffective, ignore the example and give your independent response.

API calls may fail for various reasons, such as invalid input parameters, authentication issues, or server errors. Your goal is to generate a response that accurately reflects the API's intended functionality, even if the input parameters are incorrect. Your response should be informative and relevant to the API's purpose, providing a clear and concise explanation of the expected output based on the input provided.
Here is an example:
API doc
{
    "name": "properties/get-broadband",
    "url": "https://zoopla.p.rapidapi.com/properties/get-broadband",
    "description": "Get broadband information",
    "method": "GET",
    "required_parameters": [
        {
            "name": "listing_id",
            "type": "NUMBER",
            "description": "The value of listing_id field returned in .../properties/list endpoint",
            "default": "56354192"
        }
    ],
    "optional_parameters": [],
    "code": "import requests\n\nurl = \"https://zoopla.p.rapidapi.com/properties/get-broadband\"\nquerystring = {\"listing_id\": \"56354192\"}\n\nheaders = {\n            \"X-RapidAPI-Key\": \"SIGN-UP-FOR-KEY\",\n            \"X-RapidAPI-Host\": \"zoopla.p.rapidapi.com\"\n        }\n\nresponse = requests.get(url, headers=headers, params=querystring)\nprint(response.json())\n",
    "convert_code": "import requests\n\nurl = \"https://zoopla.p.rapidapi.com/properties/get-broadband\"\nquerystring = {\"listing_id\": \"56354192\"}\n\nheaders = {\n            \"X-RapidAPI-Key\": \"SIGN-UP-FOR-KEY\",\n            \"X-RapidAPI-Host\": \"zoopla.p.rapidapi.com\"\n        }\n\nresponse = requests.get(url, headers=headers, params=querystring)\nprint(response.json())\n",
    "test_endpoint": "",
    "statuscode": 200,
    "schema": {}
}
Request:
    data = {
        "category": "Business",
        "tool_name": "zoopla_v2",
        "api_name": "properties_get_broadband",
        "tool_input": {'listing_id': '456789', "abdc": 11123},
        "strip": "",
        "toolbench_key": "xxx"
    }
Response:
    {"error":"Function executing from toolenv.tools.Business.zoopla_v2.api import properties_get_broadband error...\nproperties_get_broadband() got an unexpected keyword argument 'abdc'","response":""}


Your will also be given successful examples of API calls and their expected outputs, based on which you will generate the response for the given input.
    '''
    system_prompt = {"role": "system", "content": system_prompt}
    # user prompt, truncated to 2048 characters if too long
    user_prompt = "API Documentation:"+str(api_doc)+"\n"+"API Examples:"+str(api_example)[:2048]+"\n"+"API Input:"+str(tool_input)+"\n"
    user_prompt = {"role": "user", "content": user_prompt}

    client = OpenAI(
        api_key = SIMULATOR_API_KEY,
        base_url = SIMULATOR_API_BASE,
    )
    max_retries = 3 
    flag = False
    for attempt in range(max_retries):
        response = client.chat.completions.create(
            model = SIMULATOR_MODEL,
            messages=[system_prompt, user_prompt],
            max_tokens = 1024,
            temperature=SIMULATOR_TEMPERATURE,
            seed=SIMULATOR_SEED,
            response_format={"type": "json_object"},
        )
        result = response.choices[0].message.content
        if "```json" in result:
            result = result.replace("```json", "").replace("```", "").strip()
        if is_valid_json(result):
            flag = True
            break
        print(f"Invalid JSON response on attempt {attempt + 1}. Retrying...")
        time.sleep(1)  # Optional delay between retries

    if flag:
        return result
    else:
        fake_error = {
            "error": "Failed to generate fake response",
            "response": "",
        }
        return json.dumps(fake_error)

if __name__ == "__main__":
    uvicorn.run(app="main:app", host="0.0.0.0", port=SERVER_PORT)
