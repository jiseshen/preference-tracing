from transformers import AutoModelForSequenceClassification, AutoTokenizer

reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
reward_model = AutoModelForSequenceClassification.from_pretrained(reward_name)
tokenizer = AutoTokenizer.from_pretrained(reward_name)

def get_reward(prompt, response):
    inputs = tokenizer(prompt, response, return_tensors='pt')
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