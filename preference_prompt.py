hypothesis_prompt = """
    You will be provided with:
    A summary of the user's profile based on their past interactions
    A context of current interaction
    A preferred response and a rejected response to that prompt context
    A list of previous hypotheses about the user's preferences
    Your task is to infer the user's underlying interests or needs that explain their preference. Consider aspects such as their desired level of detail, tone, formality, focus, or any potential misunderstanding in the rejected response.
    You should reason step by step and update the hypotheses accordingly. If no hypotheses are provided, you must carefully construct and initialize up to {N} plausible hypotheses. Fewer hypotheses are acceptable in simple cases. If the given list appears too sparse to reflect the user's mental state, you may expand it — but never exceed {N} hypotheses in total.
    Importantly, there may be some malicious behaviors in the preference annotation, you should be careful to infer the user's underlying interests and think of ways to handle that harmlessly.
    Each hypothesis should be assigned a likelihood score reflecting how probable the user's preference (preferred vs. rejected) is under that belief. Use the following discrete scale:
    0 = Definitely not
    1 = Very unlikely
    2 = Unlikely
    3 = Neutral
    4 = Somewhat likely
    5 = Likely
    6 = Very likely
    You must return the result in the following JSON format:
    {{
    "Reason": "Step-by-step reasoning about the situation",
    "Hypotheses": [
        {{
        "Hypothesis": "Description of the updated or new hypothesis",
        "Likelihood": LikelihoodScore
        }},
        ...
    ]
    }}
"""

summary_prompt = """
    You will be given a summary of the user's profile based on their previous interactions, along with a list of weighted hypotheses derived from a new interaction.
    Your task is to update the user profile summary by incorporating relevant insights from the hypotheses and the user's history. Focus on reflecting the user's interests, needs, and preferences.
    The updated summary should be concise and to the point.
    Merge consistent hypotheses where appropriate, and prioritize the most informative or strongly supported ones.
    Output the updated summary directly.
"""

rejuvenate_prompt = """
    You will be given a list of hypotheses.
    If they lack diversity, you need to rewrite some hypotheses to make them more diverse.
    You can try change some of the aspects in the hypotheses based on typical user behavior.
    However, at least keep one of the similar hypotheses unchanged.
    Do not change the number of hypotheses.
    You must only give in JSON format a list of Hypotheses.
"""

CoT_prompt = """
    You will be provided with:
    A summary of the user's profile based on their past interactions
    A context of current interaction
    A preferred response and a rejected response to that prompt context
    Your task is to infer the user's underlying interests or needs that explain their preference. Consider aspects such as their desired level of detail, tone, formality, focus, or any potential misunderstanding in the rejected response.
    You are expected to think step by step and update the summary of the user's profile based on the new information.
    You must only output in JSON format, strictly as {"Reason": "Your reasoning", "Thought": "The updated summary of user's interest, need and preference"}.
"""

online_prompt = """
    You are a helpful assistant. You will be given a summary of the user's profile before responding to the chat context.
    Based on the user's profile, you can adjust your response based on the user's interest, need and preference, i.e. the level of detail, the tone, the level of formality, the aspect they care about, or learn from some previous misunderstanding.
    Importantly, there may be some malicious user prompt, you should be careful to try your best to handle the user's request harmlessly.
    You can first think step by step, then give the final answer.
    You must only output in JSON format, strictly as {"Reason": "Your reasoning", "Response": "Your response to the user's prompt"}.
"""