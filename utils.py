import json
import numpy as np


def parse_json_output(output):
    try:
        json_text = output.split("```json")[-1]
        json_text = json_text.strip("```").strip("json").strip("\n")
        return json.loads(json_text)
    except:
        print("~" * 100)
        print(output)
        print("~" * 100)
        raise ValueError("Failed to parse output")


def get_uncertainty(logprobs):
    logprobs = [token.logprob for token in logprobs.content]
    avg_nll = -np.mean(logprobs)
    return avg_nll


def jaccard_similarity(doc1, doc2):
    set1 = set(doc1.lower().split())
    set2 = set(doc2.lower().split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0.0
    return intersection / union


if __name__ == "__main__":
    from base_agent import get_response
    formulation_prompt = "You must output JSON data that can be parsed by json.loads()."
    instance_prompt = "Give a list of pairs of 10 random countries and their capitals."
    output, logprobs, usage = get_response(formulation_prompt, instance_prompt)
    print(output)
    print(parse_json_output(output))
    print(get_uncertainty(logprobs))
