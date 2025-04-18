import numpy as np


def uncertainty(logprobs):
    logprobs = [token.logprob for token in logprobs.content]
    avg_nll = -np.mean(logprobs)
    return avg_nll



if __name__ == "__main__":
    from base_agent import get_response
    messages = [
        {"role": "developer", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of the moon?"},
    ]
    print(uncertainty(get_response(messages)[1]))