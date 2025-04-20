from base_agent import get_response
from utils import parse_json_output, get_uncertainty
import numpy as np


class ThoughtTracing:
    def __init__(
        self,
        mode,
        hypothesis_prompt,
        reweight_prompt,
        rejuvenate_prompt,
        summary_prompt,
        CoT_prompt,
        conclusion_prompt,
        parse_prompt=None,
        perception_prompt=None,
        base_model='gpt-4.1',
        temperature=0.2,
        uncertainty_threshold=0.01,
        N=5,
    ):
        self.mode = mode
        self.model_id = base_model
        self.temperature = temperature
        self.hypothesis_prompt = hypothesis_prompt
        self.reweight_prompt = reweight_prompt
        self.rejuvenate_prompt = rejuvenate_prompt
        self.summary_prompt = summary_prompt
        self.CoT_prompt = CoT_prompt
        self.conclusion_prompt = conclusion_prompt
        self.parse_prompt = parse_prompt
        self.perception_prompt = perception_prompt
        self.history = ""
        self.output_token = 0
        self.input_token = 0
        self.hypotheses = []
        self.weight = []
        self.get_response = self._wrap_get_response()
        self.uncertainty_threshold = uncertainty_threshold
        self.N = N

    def _wrap_get_response(self):
        def get_response_tracked(**kwargs):
            response, logprobs, usage = get_response(**kwargs)
            self.input_token += usage["input_tokens"]
            self.output_token += usage["output_tokens"]
            return response, logprobs, usage
        return get_response_tracked

    def parse_state_action(self, input):
        output, _, _ = self.get_response(self.parse_prompt, input)
        return parse_json_output(output)

    def CoT_update(self, p, a):
        instance = f"History: {self.history}, Perception: {p}, Action: {a}"
        summary, logprobs, _ = self.get_response(self.CoT_prompt, instance)
        uncertainty = get_uncertainty(logprobs)
        return summary, uncertainty

    def hypothesis_update(self, p, a):
        instance = f"History: {self.history}, Perception: {p}, Action: {a}, Hypotheses: {self.hypotheses}"
        hypothesis, _, _ = self.get_response(self.hypothesis_prompt, instance)
        self.hypotheses = parse_json_output(hypothesis)
        if not self.weight:
            self.weight = np.ones(len(self.hypotheses)) / len(self.hypotheses)

    def reweight(self, p, a):
        instance = f"History, {self.history}, Perception: {p}, Action: {a}, Hypotheses: {self.hypotheses}"
        likelihood, _, _ = self.get_response(self.reweight_prompt, instance)
        likelihood = parse_json_output(likelihood)
        self.weight = np.array(likelihood)
        self.weight = self.weight / np.sum(self.weight)

    def summary_hypotheses(self):
        instance = f"Hypotheses and Weights: {[(h, i) for h, i in zip(self.hypotheses, self.weight)]}, History: {self.history}"
        summary, _, _ = self.get_response(self.summary_prompt, instance)
        return summary

    def resample(self):
        self.weight = np.random.dirichlet(self.weight)
        self.hypotheses = np.random.choice(
            self.hypotheses, size=len(self.hypotheses), p=self.weight)
        self.weight = np.ones(len(self.hypotheses)) / len(self.hypotheses)

    def rejuvenate(self):
        instance = f"Hypotheses: {self.hypotheses}"
        new_hypothesis, _, _ = self.get_response(
            self.rejuvenate_prompt, instance)
        self.hypotheses = parse_json_output(new_hypothesis)

    def update_history(self, i, p, a, summary):
        self.history += f"<step {i}><perception>{p}</perception><action>{a}</action><thought>{summary}</thought></step {i}>"

    def trace(self, input):
        trajectory = self.parse_state_action(
            input) if self.parse_prompt else input
        for i, step in enumerate(trajectory):
            p = step["Perception"]
            a = step["Action"]
            uncertainty = float("inf")
            if not self.mode == "Trace":
                summary, uncertainty = self.CoT_update(p, a)
            if not self.mode == "CoT" and uncertainty > self.uncertainty_threshold:
                self.hypothesis_update(p, a)
                self.reweight(p, a)
                summary = self.summary_hypotheses()
                self.resample()
                self.rejuvenate()
            self.update_history(i, p, a, summary)

    def reset(self):
        self.history = ""
        self.hypotheses = []
        self.weight = []
        self.output_token = 0
        self.input_token = 0

    def infer(self, input, query, answer):
        self.reset()
        self.trace(input)
        instance = f"History: {self.history}, Query: {query}"
        conclusion, _, _ = self.get_response(self.conclusion_prompt, instance)
        conclusion = parse_json_output(conclusion)
        acc = conclusion["Answer"] == answer
        history = self.history
        return acc, history
