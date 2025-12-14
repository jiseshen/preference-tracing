from openai import OpenAI
import dotenv
import os

dotenv.load_dotenv(override=True)

model_id = 'gpt-4.1'

client = OpenAI()

def get_response(formulation_prompt, instance_prompt, model_id=model_id, temperature=0.2, return_top_logprobs=None):
    messages = [
        {"role": "developer", "content": formulation_prompt},
        {"role": "user", "content": instance_prompt}
    ]
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
    formulation_prompt = "You are a naughty assistant. You are given a task to avoid answering the question but generate a question that is difficult to answer."
    instance_prompt = "What is the capital of the moon?"
    output, logprobs, usage = get_response(formulation_prompt, instance_prompt)
    print(output)
    print(logprobs)
    print(usage)