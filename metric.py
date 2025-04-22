from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
reward_model = AutoModelForSequenceClassification.from_pretrained(reward_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(reward_name)

def get_reward(prompt, response):
    inputs = tokenizer(prompt, response, return_tensors='pt').to(device)
    score = reward_model(**inputs).logits[0].cpu().detach()
    return score

if __name__ == "__main__":
    from base_agent import get_response
    prompt = "What is the capital of the moon?"
    messages = [
        {"role": "user", "content": prompt},
    ]
    output, logprobs, usage = get_response(messages)
    print(get_reward(prompt, output))