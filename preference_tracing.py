from metric import get_reward
from preference_prompt import *
from thought_tracing import ThoughtTracing


class PreferenceTracing(ThoughtTracing):
    def __init__(
        self,
        online_prompt,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.online_prompt = online_prompt
        self.reward = []

    def format_instance(self, kind, **kwargs):
        fields = {
            "Hypotheses": self.hypotheses,
            "Hypotheses and Weights": [(h, w) for h, w in zip(self.hypotheses, self.weight)],
            "Profile": self.history,
        }
        fields.update(kwargs)
        instances = {
            "CoT": ["Profile", "Context", "Preferred", "Rejected"],
            "hypothesis": ["Profile", "Context", "Preferred", "Rejected", "Hypotheses"],
            "summary": ["Hypotheses and Weights", "Profile"],
            "rejuvenate": ["Hypotheses"],
            "online": ["Profile", "Context"]
        }
        return ", ".join([f"{i}: {fields[i]}" for i in instances[kind]])

    def preprocess(self, input):
        self.trajectory = input["behavior"]

    def test(self, context):
        instance = self.format_instance("online", Context=context)
        response, _, _ = self.get_response(
            self.online_prompt, instance, track_usage=False)
        reward = get_reward(instance, response)
        self.reward.append(reward)

    def trace(self, input):
        self.reset()
        self.preprocess(input)
        n_cot = 0
        for data in self.trajectory:
            context = data["prompt"]
            self.test(context)
            preferred = data["chosen"]
            rejected = data["rejected"]
            uncertainty = float("inf")
            if not self.mode == "Trace":
                n_cot += 1
                self.history, uncertainty = self.CoT_update(
                    Context=context, Preferred=preferred, Rejected=rejected)
                self.avg_uncertainty += 1 / n_cot * \
                    (uncertainty - self.avg_uncertainty)
            if not self.mode == "CoT" and uncertainty > self.uncertainty_threshold:
                self.hypothesis_update(
                    Context=context, Preferred=preferred, Rejected=rejected)
                self.history = self.summary_hypotheses()
                self.resample()
                self.rejuvenate()

    def reset(self):
        super().reset()
        self.reward = []


if __name__ == "__main__":
    import json
    with open("profiles_dataset.json", "r") as f:
        data = json.load(f)
    preference_tracing = PreferenceTracing(
        online_prompt=online_prompt,
        hypothesis_prompt=hypothesis_prompt,
        rejuvenate_prompt=rejuvenate_prompt,
        summary_prompt=summary_prompt,
        CoT_prompt=CoT_prompt,
        mode="CoT",
        base_model="gpt-4o",
        temperature=0.2,
        N=5,
        uncertainty_threshold=0.2,
    )
    context = data[0]["behavior"][0]["prompt"]
    preferred = data[0]["behavior"][0]["chosen"]
    rejected = data[0]["behavior"][0]["rejected"]
    preference_tracing.hypothesis_update(
        Context=context, Preferred=preferred, Rejected=rejected)
    print(preference_tracing.summary_hypotheses())

