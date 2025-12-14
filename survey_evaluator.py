import os
import sys
import json
import numpy as np
from typing import Dict, List
from datasets import load_dataset

sys.path.append(os.path.join(os.path.dirname(__file__), "thought-tracing"))
from agents.load_model import load_model

class SurveyEvaluator:
    def __init__(self, args):
        self.eval_model = load_model(args.eval_model, **args.__dict__)
        self.args = args

    def load_survey_data(self):
        dataset = load_dataset("HannahRoseKirk/prism-alignment", "survey")
        survey_data = {}
        for item in dataset['train']:
            user_id = item.get('user_id')
            if user_id:
                survey_data[user_id] = item
        return survey_data

    def extract_survey_profile(self, survey_data: Dict) -> str:
        profile_aspects = []
        key_fields = [
            'user_id',
            'self_description',
            'system_string',
            'age',
            'gender',
            'religion',
            'ethnicity',
        ]
        for field in key_fields:
            if field in survey_data and survey_data[field]:
                value = survey_data[field]
                profile_aspects.append(f"{field.replace('_', ' ').title()}: {value}")
        return "\n".join(profile_aspects)

    def evaluate_profile_alignment(self, inferred_profile: str, survey_profile: str) -> Dict:
        prompt = f"""<inferred user profile from conversation behavior>
{inferred_profile}
</inferred user profile from conversation behavior>

<actual user survey responses>
{survey_profile}
</actual user survey responses>

Evaluate how well the inferred profile aligns with the actual survey data across the following dimensions:
1. Communication style (1-10)
2. Value alignment (1-10)
3. Preference consistency (1-10)
4. Overall accuracy (1-10)

For each dimension, provide a score and brief explanation.

Format your response as:
Communication Style: [score] - [explanation]
Value Alignment: [score] - [explanation]
Preference Consistency: [score] - [explanation]
Overall Accuracy: [score] - [explanation]
"""
        
        response = self.eval_model.interact(prompt, temperature=0, max_tokens=512)
        
        scores = self.parse_evaluation_response(response)
        
        return scores

    def parse_evaluation_response(self, response: str) -> Dict:
        scores = {
            'communication_style': 5.0,
            'value_alignment': 5.0,
            'preference_consistency': 5.0,
            'overall_accuracy': 5.0,
            'explanations': {}
        }
        
        lines = response.split('\n')
        for line in lines:
            for key, field in [
                ('Communication Style:', 'communication_style'),
                ('Value Alignment:', 'value_alignment'),
                ('Preference Consistency:', 'preference_consistency'),
                ('Overall Accuracy:', 'overall_accuracy')
            ]:
                if key in line:
                    try:
                        parts = line.split(key)[1].strip()
                        score_part = parts.split('-')[0].strip()
                        score = float(score_part)
                        scores[field] = max(1.0, min(10.0, score))
                        
                        if '-' in parts:
                            explanation = '-'.join(parts.split('-')[1:]).strip()
                            scores['explanations'][field] = explanation
                    except:
                        pass
        
        return scores

    def evaluate_all_profiles(self, tracing_results: List[Dict], survey_data: Dict) -> Dict:
        evaluation_results = []
        
        for user_result in tracing_results:
            user_id = user_result['user_id']
            
            if user_id not in survey_data:
                continue
            
            inferred_profile = user_result['final_profile']
            survey_profile = self.extract_survey_profile(survey_data[user_id])
            
            if not survey_profile.strip():
                continue
            
            alignment_scores = self.evaluate_profile_alignment(inferred_profile, survey_profile)
            
            evaluation_results.append({
                'user_id': user_id,
                'inferred_profile': inferred_profile,
                'survey_profile': survey_profile,
                'alignment_scores': alignment_scores
            })
        
        summary = self.compute_summary_statistics(evaluation_results)
        
        return {
            'individual_evaluations': evaluation_results,
            'summary_statistics': summary
        }

    def compute_summary_statistics(self, evaluation_results: List[Dict]) -> Dict:
        if not evaluation_results:
            return {}
        
        metrics = ['communication_style', 'value_alignment', 'preference_consistency', 'overall_accuracy']
        
        summary = {}
        for metric in metrics:
            scores = [r['alignment_scores'][metric] for r in evaluation_results]
            summary[metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'ci': 1.96 * np.std(scores) / np.sqrt(len(scores)),
                'min': np.min(scores),
                'max': np.max(scores)
            }
        
        return summary

def run_survey_evaluation(args):
    with open(args.tracing_results, 'r') as f:
        tracing_results = json.load(f)
    
    evaluator = SurveyEvaluator(args)
    survey_data = evaluator.load_survey_data()
    
    evaluation_results = evaluator.evaluate_all_profiles(tracing_results, survey_data)
    
    output_file = os.path.join(args.output_dir, f"survey_evaluation_{args.run_id}.json")
    with open(output_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print("\n=== Survey Evaluation Summary ===")
    for metric, stats in evaluation_results['summary_statistics'].items():
        print(f"{metric.replace('_', ' ').title()}: {stats['mean']:.2f} ± {stats['ci']:.2f} (range: {stats['min']:.1f}-{stats['max']:.1f})")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-model', type=str, default='gpt-4', help='Model for evaluation')
    parser.add_argument('--tracing-results', type=str, required=True, help='Path to preference tracing results')
    parser.add_argument('--output-dir', type=str, default='preference_results', help='Output directory')
    parser.add_argument('--run-id', type=str, default='default', help='Run ID')
    
    args = parser.parse_args()
    run_survey_evaluation(args)
