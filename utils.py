import json
import numpy as np


def parse_json_output(output):
    json_text = output.strip("```").strip("json").strip("\n")
    return json.loads(json_text)


def get_uncertainty(logprobs):
    logprobs = [token.logprob for token in logprobs.content]
    avg_nll = -np.mean(logprobs)
    return avg_nll


if __name__ == "__main__":
    from base_agent import get_response
    formulation_prompt = "You must output JSON data that can be parsed by json.loads()."
    instance_prompt = "Give a list of pairs of 10 random countries and their capitals."
    output, logprobs, usage = get_response(formulation_prompt, instance_prompt)
    print(output)
    print(parse_json_output(output))
    print(get_uncertainty(logprobs))
