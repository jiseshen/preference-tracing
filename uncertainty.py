import numpy as np


def uncertainty(logprobs):
    logprobs = [token.logprob for token in logprobs.content]
    avg_nll = -np.mean(logprobs)
    return avg_nll


if __name__ == "__main__":
    from base_agent import get_response
    formulation_prompt = "You are a naughty assistant. You are given a task to avoid answering the question but generate a question that is difficult to answer."
    instance_prompt = "What is the capital of the moon?"
    print(uncertainty(get_response(formulation_prompt, instance_prompt)[1]))
