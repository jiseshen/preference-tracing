from openai import OpenAI
import dotenv
import os

dotenv.load_dotenv(override=True)

model_id = 'gpt-4.1'

client = OpenAI()

def get_response(messages, model_id=model_id, return_top_logprobs=None):
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0.2,
        logprobs=True,
        top_logprobs=return_top_logprobs,
    )
    output = response.choices[0].message.content
    logprobs = response.choices[0].logprobs
    usage = response.usage
    return output, logprobs, usage


if __name__ == "__main__":
    messages = [
        {"role": "developer", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of the moon?"},
    ]
    output, logprobs, usage = get_response(messages)
    print(output)
    print(logprobs)
    print(usage)