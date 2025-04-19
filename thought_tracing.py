from base_agent import get_response
from utils import parse_json_output
import numpy as np


class ThoughtTracing:
    def __init__(
        self,
        hypothesis_prompt,
        reweight_prompt,
        rejuvenate_prompt,
        summary_prompt,
        CoT_prompt,
        parse_prompt=None,
        perception_prompt=None,
        base_model='gpt-4.1',
        temperature=0.2
    ):
        self.model_id = base_model
        self.temperature = temperature
        self.hypothesis_prompt = hypothesis_prompt
        self.reweight_prompt = reweight_prompt
        self.rejuvenate_prompt = rejuvenate_prompt
        self.summary_prompt = summary_prompt
        self.CoT_prompt = CoT_prompt
        self.parse_prompt = parse_prompt
        self.perception_prompt = perception_prompt
        self.history = ""
        self.hypotheses = []
        self.weight = []

    def parse_state_action(self, input):
        output, _, _ = get_response(self.parse_prompt, input)
        return parse_json_output(output)

    def infer_perception(self, s):
        perception, _, _ = get_response(self.perception_prompt, f"Current State: {s}")
        return perception
    
    def hypothesis_update(self, s, a, p=None):
        if p:
            instance = f"History: {self.history}, Current State: {s}, Agent Perception: {self.perception_history}, Action: {a}, Hypotheses: {self.hypotheses}"
        else:
            instance = f"Current State: {s}, Action: {a}, Hypotheses: {self.hypotheses}"
        hypothesis, _, _ = get_response(self.hypothesis_prompt, instance)
        self.hypothesis = parse_json_output(hypothesis)
        if not self.weight:
            self.weight = np.ones(len(self.hypotheses)) / len(self.hypotheses)

    def reweight(self, s, a, p=None):
        if p:
            instance = f"History, {self.history}, Current State: {s}, Current Perception: {p}, Action: {a}, Hypotheses: {self.hypotheses}"
        else:
            instance = f"History, {self.history}, Current State: {s}, Action: {a}, Hypotheses: {self.hypotheses}"
        likelihood, _, _ = get_response(self.reweight_prompt, instance)
        likelihood = parse_json_output(likelihood)
        self.weight = self.weight * likelihood
        self.weight = self.weight / sum(self.weight)

    def resample(self):
        self.weight = np.random.dirichlet(self.weight)
        self.hypotheses = np.random.choice(self.hypotheses, size=len(self.hypotheses), p=self.weight)
        self.weight = np.ones(len(self.hypotheses)) / len(self.hypotheses)
        
    def rejuvenate(self):
        instance = f"Hypotheses: {self.hypotheses}"
        new_hypothesis, _, _ = get_response(self.rejuvenate_prompt, instance)
        self.hypotheses = parse_json_output(new_hypothesis)
        
    def summary_hypotheses(self):
        instance = f"Hypotheses and Weights: {[(h, i) for h, i in zip(self.hypotheses, self.weight)]}, History: {self.history}"
        summary, _, _ = get_response(self.summary_prompt, instance)
        return summary
        
    def update_history(self, i, summary, s, a, p=None):
        self.history += f"<step {i}><state>{s}</state><perception>{p}</perception><action>{a}</action><thought>{summary}</thought></step {i}>"
        
    def trace(self, input):
        trajectory = self.parse_state_action(input) if self.parse_prompt else input
        for i, step in enumerate(trajectory):
            s = step["State"]
            a = step["Action"]
            p = self.infer_perception(s) if self.perception_prompt else None
            self.hypothesis_update(s, a, p)
            self.reweight(s, a, p)
            self.resample()
            self.rejuvenate()
            summary = self.summary_hypotheses()
            self.update_history(i, summary, s, a, p, summary)