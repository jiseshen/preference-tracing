from base_agent import get_response
from utils import parse_json_output, get_uncertainty, jaccard_similarity
from thought_prompt import *
import numpy as np


class ThoughtTracing:
    def __init__(
        self,
        mode,
        hypothesis_prompt,
        rejuvenate_prompt,
        summary_prompt,
        CoT_prompt,
        conclusion_prompt=None,
        parse_prompt=None,
        base_model='gpt-4.1',
        temperature=0.2,
        uncertainty_threshold=0.2,
        N=4,
        max_steps=10,
    ):
        self.mode = mode
        self.model_id = base_model
        self.temperature = temperature
        self.hypothesis_prompt = hypothesis_prompt.format(N=N)
        self.rejuvenate_prompt = rejuvenate_prompt
        self.summary_prompt = summary_prompt
        self.CoT_prompt = CoT_prompt
        self.conclusion_prompt = conclusion_prompt
        self.parse_prompt = parse_prompt
        self.history = ""
        self.trajectory = []
        self.knowledge = ""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.hypotheses = []
        self.weight = []
        self.get_response = self._wrap_get_response()
        self.uncertainty_threshold = uncertainty_threshold
        self.N = N
        self.max_steps = max_steps
        self.avg_uncertainty = 0
        self.trace_count = 0

    def _wrap_get_response(self):
        def get_response_tracked(formulation_prompt, instance_prompt, track_usage=True, **kwargs):
            response, logprobs, usage = get_response(
                formulation_prompt, instance_prompt, model_id=self.model_id, **kwargs)
            if track_usage:
                self.prompt_tokens += usage.prompt_tokens
                self.completion_tokens += usage.completion_tokens
            return response, logprobs, usage
        return get_response_tracked

    def preprocess(self, input):
        if self.parse_prompt:
            output, _, _ = self.get_response(self.parse_prompt, input)
            output = parse_json_output(output)
            self.trajectory = output["Trajectory"]
            self.knowledge = output["Knowledge"]
        else:
            self.trajectory = input
            self.knowledge = ""

    def format_instance(self, kind, **kwargs):
        fields = {
            "Knowledge": self.knowledge,
            "History": self.history,
            "Hypotheses": self.hypotheses,
            "Hypotheses and Weights": [(h, w) for h, w in zip(self.hypotheses, self.weight)],
        }
        fields.update(kwargs)
        instances = {
            "CoT": ["Knowledge", "History", "Perception", "Action"],
            "hypothesis": ["Knowledge", "History", "Perception", "Action", "Hypotheses"],
            "summary": ["Hypotheses and Weights", "History"],
            "rejuvenate": ["Knowledge", "Hypotheses"],
            "conclusion": ["Knowledge", "History", "Query"],
        }
        return ", ".join([f"{i}: {fields[i]}" for i in instances[kind]])

    def check_ess(self):
        ess = 1.0 / np.sum(self.weight ** 2)
        return ess < len(self.hypotheses) / 2
    
    def check_similarity(self):
        for i in range(len(self.hypotheses)):
            for j in range(i + 1, len(self.hypotheses)):
                if jaccard_similarity(self.hypotheses[i], self.hypotheses[j]) > 0.8:
                    return True
        return False
    
    def CoT_update(self, **kwargs):
        instance = self.format_instance("CoT", **kwargs)
        summary, logprobs, _ = self.get_response(self.CoT_prompt, instance)
        uncertainty = get_uncertainty(logprobs)
        summary = parse_json_output(summary)
        return summary["Thought"], uncertainty

    def hypothesis_update(self, **kwargs):
        self.trace_count += 1
        instance = self.format_instance("hypothesis", **kwargs)
        hypotheses, _, _ = self.get_response(self.hypothesis_prompt, instance)
        hypotheses = parse_json_output(hypotheses)["Hypotheses"]
        self.hypotheses = [i["Hypothesis"] for i in hypotheses]
        if len(self.weight) == 0:
            self.weight = [i["Likelihood"] for i in hypotheses]
        for i, h in enumerate(hypotheses):
            self.weight[i] *= h["Likelihood"]
        self.weight = np.array(self.weight) / np.sum(self.weight)

    def summary_hypotheses(self):
        instance = self.format_instance("summary")
        summary, _, _ = self.get_response(self.summary_prompt, instance)
        return summary

    def resample(self):
        eps = 1e-3
        self.weight = np.random.dirichlet(self.weight + eps)
        self.hypotheses = np.random.choice(
            self.hypotheses, size=len(self.hypotheses), p=self.weight)
        self.weight = np.ones(len(self.hypotheses)) / len(self.hypotheses)

    def rejuvenate(self):
        instance = self.format_instance("rejuvenate")
        new_hypothesis, _, _ = self.get_response(
            self.rejuvenate_prompt, instance)
        self.hypotheses = parse_json_output(new_hypothesis)

    def update_history(self, i, p, a, summary):
        self.history += f"<step {i}><new_perception>{p}</new_perception><action>{a}</action><thought>{summary}</thought></step {i}>"

    def trace(self, input):
        n_cot = 0
        self.preprocess(input)
        if len(self.trajectory) > self.max_steps:
            return False
        for i, step in enumerate(self.trajectory):
            p = step["Perception"]
            a = step["Action"]
            uncertainty = float("inf")
            if not self.mode == "Trace":
                n_cot += 1
                summary, uncertainty = self.CoT_update(Perception=p, Action=a)
                self.avg_uncertainty += 1 / n_cot * \
                    (uncertainty - self.avg_uncertainty)
            if not self.mode == "CoT" and uncertainty > self.uncertainty_threshold:
                self.hypothesis_update(Perception=p, Action=a)
                summary = self.summary_hypotheses()
                if self.check_ess():
                    self.resample()
                if self.check_similarity():
                    self.rejuvenate()
            self.update_history(i, p, a, summary)
        return True

    def reset(self):
        self.history = ""
        self.hypotheses = []
        self.weight = []
        self.knowledge = ""
        self.trajectory = []
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.avg_uncertainty = 0
        self.trace_count = 0
        
    def infer(self, input, query, answer):
        self.reset()
        try:
            success = self.trace(input)
        except Exception as e:
            print("~" * 100)
            print(e)
            print("~" * 100)
            return -2, None, None, None, None
        if not success:
            return -1, None, None, None, None
        instance = f"Knowledge: {self.knowledge}, History: {self.history}, Query: {query}"
        conclusion, _, _ = self.get_response(self.conclusion_prompt, instance)
        conclusion = parse_json_output(conclusion)
        acc = conclusion["Answer"] == answer
        reasoning = conclusion["Reason"]
        history = self.history
        return acc, conclusion["Answer"], reasoning, history, (self.prompt_tokens, self.completion_tokens)


if __name__ == "__main__":
    thought_tracing = ThoughtTracing(
        parse_prompt=parse_prompt,
        hypothesis_prompt=hypothesis_prompt,
        rejuvenate_prompt=rejuvenate_prompt,
        summary_prompt=summary_prompt,
        CoT_prompt=CoT_prompt,
        conclusion_prompt=conclusion_prompt,
        mode="Trace",
        base_model="gpt-4o-mini",
        temperature=0.2,
    )
    test_input = test_input.split("Question:")
    input = "".join(test_input[:-1])
    question = test_input[-1]
    acc, answer, history, tokens = thought_tracing.infer(input, question, "a")
    print(acc)
    print(answer)
    print(history)
    print(tokens)
    print("~~~~~~~~~~~~~")
    print(thought_tracing.trajectory)
    print("~~~~~~~~~~~~~")
    print(thought_tracing.knowledge)
