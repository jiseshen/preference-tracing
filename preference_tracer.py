import os
import sys
import json
import re
import numpy as np
import random
from copy import deepcopy
from typing import List, Dict, Union
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset

sys.path.append(os.path.join(os.path.dirname(__file__), "thought-tracing"))
from agents.load_model import load_model
from hypothesis import HypothesesSetV3, compute_ess, resample_hypotheses_with_other_info
from utils import softmax, prompting_for_ordered_list, overall_jaccard_similarity

from rich import print, box
from rich.panel import Panel
from rich.console import Console

from checkpoint_manager import CheckpointManager
from data_manager import DataManager

console = Console()

class PreferenceTracer:
    def __init__(self, args, data_manager: DataManager = None):
        self.tracer_model = load_model(args.tracing_model, **args.__dict__)
        self.eval_model = load_model(args.eval_model, **args.__dict__)
        self.args = args
        self.data_manager = data_manager
        random.seed(args.seed)
        np.random.seed(args.seed)

    def reset_hypotheses_for_new_conversation(self, hypotheses: HypothesesSetV3) -> HypothesesSetV3:
        """
        Reset contexts and perceptions for a new conversation while keeping hypothesis texts and weights.
        This avoids leaking previous conversation history while preserving belief state.
        """
        if hypotheses is None:
            return None
        new_contexts = []
        new_perceptions = [{'perception': 'conversation_reset'}]
        # Preserve parent chain to allow backtracking if needed
        parents = hypotheses.hypotheses
        reset = HypothesesSetV3(
            target_agent=hypotheses.target_agent,
            contexts=new_contexts,
            perceptions=new_perceptions,
            texts=hypotheses.texts,
            weights=hypotheses.weights,
            parent_hypotheses=parents,
            previous_ess=hypotheses.previous_ess,
            weight_details=hypotheses.weight_details
        )
        return reset

    def initialize_hypotheses(self, user_id: str, conversation_history: List[Dict], candidates: List[Union[Dict, str]]) -> HypothesesSetV3:
        history_str = self.format_conversation_history(conversation_history)
        candidates_str = self.format_candidates(candidates)
        
        context_input = f"<conversation history>\n{history_str}\n</conversation history>\n\n<current responses>\n{candidates_str}\n</current responses>"
        
        prompt = f"{context_input}\n\nGenerate a numbered list of {self.args.n_hypotheses} hypotheses about the user's preferences, values, and communication style that would explain their response choice. Focus on what matters to this user in AI conversations."
        
        should_print = self.args.print if hasattr(self.args, 'print') else False
        
        if should_print:
            console.print(Panel(f"[bold cyan]Initializing hypotheses for user {user_id}[/bold cyan]", style="cyan"))
        
        if self.data_manager:
            self.data_manager.log(f"=== Initializing Hypotheses for User {user_id} ===")
        
        hypotheses_list = prompting_for_ordered_list(self.tracer_model, prompt=prompt, n=self.args.n_hypotheses)
        hypotheses_list = [h.strip() for h in hypotheses_list]
        
        weights = np.ones(len(hypotheses_list)) / len(hypotheses_list)
        
        if should_print:
            console.print(Panel("\n".join([f"{i+1}. {h}" for i, h in enumerate(hypotheses_list)]), 
                       title="Initial Hypotheses", style="green"))
        
        if self.data_manager:
            for i, h in enumerate(hypotheses_list):
                self.data_manager.log(f"  {i+1}. {h}")
        
        return HypothesesSetV3(
            target_agent=user_id,
            contexts=[{'candidates': candidates}],
            perceptions=[{'perception': 'initial', 'input_candidates': candidates_str}],
            texts=hypotheses_list,
            weights=weights
        )

    def propagate_hypotheses(self, existing_hypotheses: HypothesesSetV3, conversation_history: List[Dict], candidates: List[Union[Dict, str]]) -> HypothesesSetV3:
        history_str = self.format_conversation_history(conversation_history)
        candidates_str = self.format_candidates(candidates)
        
        new_context = f"<conversation history>\n{history_str}\n</conversation history>\n\n<current responses>\n{candidates_str}\n</current responses>"
        
        if self.args.print if hasattr(self.args, 'print') else False:
            print(Panel("[bold yellow]Propagating hypotheses...[/bold yellow]", style="yellow"))
        
        propagation_prompts = [
            f"<previous preference>\n{hypothesis}\n</previous preference>\n\n<new context>\n{new_context}\n</new context>\n\nQuestion: Based on the previous hypothesis and new conversation context, what are the user's updated preferences and values? Provide a refined hypothesis."
            for hypothesis in existing_hypotheses.texts
        ]
        
        propagated_texts = self.tracer_model.batch_interact(propagation_prompts, temperature=0.3, max_tokens=256)
        
        new_contexts = existing_hypotheses.contexts + [{'candidates': candidates}]
        new_perceptions = existing_hypotheses.perceptions + [{'perception': f'turn_{len(existing_hypotheses.contexts)}', 'input_candidates': candidates_str}]
        
        return HypothesesSetV3(
            target_agent=existing_hypotheses.target_agent,
            contexts=new_contexts,
            perceptions=new_perceptions,
            texts=propagated_texts,
            weights=existing_hypotheses.weights,
            parent_hypotheses=existing_hypotheses.hypotheses,
            propagation_prompts=propagation_prompts,
            propagated_outputs=propagated_texts
        )

    def weigh_hypotheses(self, hypotheses: HypothesesSetV3, chosen_response: Union[Dict, str], candidates: List[Union[Dict, str]]) -> Dict:
        chosen_content = chosen_response if isinstance(chosen_response, str) else chosen_response.get('content', '')
        candidates_str = self.format_candidates(candidates, include_chosen=True, chosen_text=chosen_content)
        
        if self.args.print if hasattr(self.args, 'print') else False:
            print(Panel("[bold magenta]Weighing hypotheses based on user choice...[/bold magenta]", style="magenta"))
        
        system_prompt = (
            "You evaluate how likely a user would choose a specific response given their preference. "
            "Format your answer as brief reasoning followed by a final line exactly: 'Answer: <the answer letter>'. "
            "Only one final Answer line; do not include any other 'Answer:' occurrences."
        )
        
        word_mapping = {
            'a': "Very Likely (90%)", 
            'b': "Likely (70%)", 
            'c': "Somewhat Likely (50%)", 
            'd': "Somewhat Unlikely (30%)", 
            'e': "Unlikely (10%)", 
            'f': "Very Unlikely (<5%)"
        }
        score_mapping = {'a': 3, 'b': 2.5, 'c': 2, 'd': 1, 'e': 0.5, 'f': 0.001}
        
        options_str = "\n".join([f"({k}) {v}" for k, v in word_mapping.items()])
        
        likelihood_prompts = []
        for hypothesis in hypotheses.texts:
            prompt = (
                f"<preference>\n{hypothesis}\n</preference>\n\n"
                f"<available responses>\n{candidates_str}\n</available responses>\n\n"
                f"<chosen response>\n{chosen_content}\n</chosen response>\n\n"
                f"Question: Given this hypothesis, analyze briefly (<=3 sentences) how well the chosen response matches. "
                f"Then output a single final line with one of: {options_str}.\n"
            )
            likelihood_prompts.append(prompt)
        
        raw_predictions = self.tracer_model.batch_interact(likelihood_prompts, temperature=0, system_prompts=system_prompt, max_tokens=256)
        
        reasonings = []
        answers = []
        for response in raw_predictions:
            reasoning = response.strip()
            option_letter = 'c'
            if 'Answer:' in response or 'answer:' in response:
                parts = re.split(r'(?i)answer:', response, maxsplit=1)
                reasoning = parts[0].strip()
                answer_segment = parts[1].strip()
                match = re.search(r'\(([a-f])\)', answer_segment.lower())
                if match:
                    option_letter = match.group(1)
            reasonings.append(reasoning)
            answers.append(option_letter)
        
        raw_scores = np.array([self.map_response_to_score(a, score_mapping) for a in answers])
        # Cumulative update: combine previous weights as prior with new likelihood scores
        prior = hypotheses.weights if hasattr(hypotheses, 'weights') and hypotheses.weights is not None else np.ones(len(raw_scores)) / len(raw_scores)
        eps = 1e-12
        log_prior = np.log(prior + eps)
        combined = log_prior + raw_scores  # log posterior up to additive constant
        weights = softmax(combined)
        
        if self.args.print if hasattr(self.args, 'print') else False:
            print("\n[bold]Hypothesis Weights:[/bold]")
            for i, (h, w, score) in enumerate(zip(hypotheses.texts, weights, raw_scores)):
                print(Panel(f"[cyan]{h}[/cyan]\n\n[bold yellow]Raw Score: {score:.3f} | Weight: {w:.3f}[/bold yellow]", 
                           title=f"Hypothesis {i+1}", style="blue"))
        
        return {
            'prompts': likelihood_prompts,
            'raw_predictions': raw_predictions,
            'reasonings': reasonings,
            'raw_scores': raw_scores,
            'weights': weights
        }

    def map_response_to_score(self, response: str, mapping: dict) -> float:
        for option in mapping.keys():
            if response.startswith(f"({option})") or response.startswith(f"{option})") or response.startswith(f"{option}.") or option in response[:3]:
                return mapping[option]
        return 0.5

    def summarize_hypotheses(self, hypotheses: HypothesesSetV3) -> str:
        weighted_hypotheses = "\n".join([f"- {text} (weight: {weight:.3f})" for text, weight in zip(hypotheses.texts, hypotheses.weights)])
        
        prompt = f"<weighted hypotheses>\n{weighted_hypotheses}\n</weighted hypotheses>\n\nSummarize these hypotheses into a coherent user preference profile. Focus on the most important preferences (higher weights) and synthesize overlapping themes. Provide a concise profile."
        
        summary = self.tracer_model.interact(prompt, temperature=0, max_tokens=512)
        
        if self.args.print if hasattr(self.args, 'print') else False:
            print(Panel(summary, title="[bold green]User Preference Profile Summary[/bold green]", 
                       style="green", box=box.DOUBLE))
        
        return summary

    def rejuvenate_hypotheses(self, existing_hypotheses: HypothesesSetV3) -> HypothesesSetV3:
        """
        Rejuvenate hypotheses by paraphrasing them to increase diversity while maintaining meaning.
        Called when text diversity is too low (< 0.25) but ESS is still acceptable.
        """
        if self.args.print if hasattr(self.args, 'print') else False:
            for h in existing_hypotheses.texts:
                print(Panel(h, title="Low Variance Hypotheses", style="red", box=box.SIMPLE_HEAD))
        
        system_prompt = "Your task is to paraphrase the following user preference hypothesis. Keep the meaning intact while rephrasing it. Do not add any additional comments."
        revision_prompts = [f"{hypothesis}" for hypothesis in existing_hypotheses.texts]
        revised_texts = self.tracer_model.batch_interact(revision_prompts, system_prompts=system_prompt, temperature=1, max_tokens=256)
        existing_hypotheses.update_texts(revised_texts)
        
        overall_text_diversity = 1 - overall_jaccard_similarity(existing_hypotheses.texts)
        if self.args.print if hasattr(self.args, 'print') else False:
            print(Panel(f"Text diversity: {overall_text_diversity}", title="Diversity of Rejuvenated Hypotheses", style="blue", box=box.SIMPLE_HEAD))
            print(Panel("\n".join(existing_hypotheses.texts), title="Rejuvenated hypotheses", style="blue", box=box.SIMPLE_HEAD))
        
        return existing_hypotheses

    def evaluate_generation(self, user_profile: str, conversation_history: List[Dict], candidates: List[Union[Dict, str]], chosen_response: Union[Dict, str]) -> float:
        history_str = self.format_conversation_history(conversation_history)
        user_query = conversation_history[-1]['content']
        
        gen_prompt = f"<conversation history>\n{history_str}\n</conversation history>\n\n<user preference profile>\n{user_profile}\n</user preference profile>\n\nBased on the user's preferences, generate an appropriate response to their last message."
        
        generated_response = self.tracer_model.interact(gen_prompt, temperature=0.7, max_tokens=256)
        
        chosen_content = chosen_response if isinstance(chosen_response, str) else chosen_response.get('content', '')
        eval_prompt = f"<generated response>\n{generated_response}\n</generated response>\n\n<actual chosen response>\n{chosen_content}\n</actual chosen response>\n\nRate how similar these responses are in style, content, and alignment with user preferences on a scale of 1-10. Output only the number without any explanation.\n\nRating:"
        
        rating_response = self.eval_model.interact(eval_prompt, temperature=0, max_tokens=10)
        
        match = re.search(r'\d+\.?\d*', rating_response.strip())
        rating = float(match.group(0)) if match else 5.0
        rating = max(1.0, min(10.0, rating))
        
        return rating / 10.0

    def predict_choice(self, user_profile: str, conversation_history: List[Dict], candidates: List[Union[Dict, str]]) -> int:
        history_str = self.format_conversation_history(conversation_history)
        candidates_str = self.format_candidates(candidates, numbered=True)
        
        prompt = f"<conversation history>\n{history_str}\n</conversation history>\n\n<response candidates>\n{candidates_str}\n</response candidates>\n\n<user preference profile>\n{user_profile}\n</user preference profile>\n\nBased on the user's preferences, which response would they most likely choose? Answer with just the number (1-{len(candidates)}).\n\nAnswer:"
        
        prediction = self.tracer_model.interact(prompt, temperature=0, max_tokens=10)
        
        match = re.search(r'\d+', prediction.strip())
        predicted_idx = int(match.group(0)) - 1 if match else 0
        predicted_idx = max(0, min(len(candidates) - 1, predicted_idx))
        
        return predicted_idx

    def format_conversation_history(self, history: List[Dict]) -> str:
        formatted = []
        for msg in history:
            role = "User" if msg['role'] == 'user' else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted)

    def format_candidates(self, candidates: List[Union[Dict, str]], include_chosen: bool = False, numbered: bool = False, chosen_text: str = None) -> str:
        formatted = []
        for idx, cand in enumerate(candidates):
            text = cand if isinstance(cand, str) else cand.get('content', '')
            prefix = f"{idx + 1}. " if numbered else "- "
            chosen_mark = ""
            if include_chosen:
                if isinstance(cand, dict) and cand.get('if_chosen', False):
                    chosen_mark = " [CHOSEN]"
                elif chosen_text is not None and text == chosen_text:
                    chosen_mark = " [CHOSEN]"
            formatted.append(f"{prefix}{text}{chosen_mark}")
        return "\n".join(formatted)

    def trace_user_preferences(self, user_bundle: Dict) -> Dict:
        """
        Trace preferences for a user across multiple conversations.
        The user_bundle format is produced by prism_adapter.load_prism_users():
        {
            'user_id': str,
            'conversations': [ { 'conversation_id': str, 'turns': [ ... ] }, ... ]
        }
        For each new conversation, we reset history (contexts/perceptions) but keep hypotheses.
        """
        user_id = user_bundle['user_id']
        conversations = user_bundle['conversations']

        hypotheses = None
        turn_results = []
        hypotheses_list = []
        
        should_print = self.args.print if hasattr(self.args, 'print') else False
        
        if self.data_manager:
            self.data_manager.start_user_logging(user_id)
            self.data_manager.log(f"{'='*70}")
            self.data_manager.log(f"Tracing preferences for user: {user_id}")
            self.data_manager.log(f"{'='*70}")
        
        if should_print:
            console.print("\n" + "="*70)
            console.print(Panel(f"[bold blue]>>> Tracing preferences for user: {user_id}[/bold blue]", 
                       style="blue", box=box.DOUBLE))
            console.print("="*70 + "\n")
        
        global_turn_idx = 0
        for conv_idx, conv in enumerate(conversations):
            conv_turns = conv['turns']
            # New conversation boundary: reset contexts/perceptions but keep beliefs
            if hypotheses is not None:
                hypotheses = self.reset_hypotheses_for_new_conversation(hypotheses)
                if self.data_manager:
                    self.data_manager.log(f"\n=== New conversation: {conv.get('conversation_id', conv_idx)} (contexts reset) ===")
            for t_idx_within, turn_data in enumerate(conv_turns):
                user_msg = turn_data['user_message']
                candidates = turn_data['candidates']
                chosen_response = turn_data['chosen']
                
                if not user_msg or not candidates or not chosen_response:
                    continue
                
                user_msg_text = user_msg['content'] if isinstance(user_msg, dict) else str(user_msg)
                chosen_text = chosen_response if isinstance(chosen_response, str) else chosen_response.get('content', '')
                
                if should_print:
                    console.print(f"\n[bold cyan]{'='*70}[/bold cyan]")
                    console.print(f"[bold cyan]Turn {global_turn_idx} (conv {conv_idx}, t{t_idx_within}): Processing[/bold cyan]")
                    console.print(f"[bold cyan]{'='*70}[/bold cyan]\n")
                    console.print(Panel(f"[white]{user_msg_text}[/white]", title="User Message", style="cyan"))
                    console.print(Panel(f"[white]{chosen_text}[/white]", title="Chosen Response", style="green"))
            
                if self.data_manager:
                    self.data_manager.log(f"\n--- Turn {global_turn_idx} (conv {conv_idx}, t{t_idx_within}) ---")
                    self.data_manager.log(f"User Message: {user_msg_text[:100]}...")
                    self.data_manager.log(f"Chosen Response: {chosen_text[:100]}...")
            
                # Build history restricted to current conversation up to this turn
                history = self.build_history(conv_turns[:t_idx_within + 1])
            
                # Initialize or propagate hypotheses
                if hypotheses is None:
                    hypotheses = self.initialize_hypotheses(user_id, history, candidates)
                else:
                    hypotheses = self.propagate_hypotheses(hypotheses, history, candidates)
            
                # Weigh hypotheses based on user's choice
                weight_results = self.weigh_hypotheses(hypotheses, chosen_response, candidates)
                hypotheses.update_weights(weight_results['weights'])
                hypotheses.weight_details = weight_results
                
                # Resample or rejuvenate hypotheses to maintain diversity
                ess = None
                overall_text_diversity = None
                resampled = False
                rejuvenated = False
                
                if self.args.n_hypotheses > 1:
                    ess = compute_ess(hypotheses)
                    overall_text_diversity = 1 - overall_jaccard_similarity(hypotheses.texts)
                    # Persist current ESS for dumping
                    hypotheses.update_ess(ess)
                    
                    # Resample if effective sample size is too low (hypothesis degeneracy)
                    if ess < self.args.n_hypotheses / 2:
                        if should_print:
                            console.print(Panel(f"ESS: {ess:.2f}", title="Resampling Hypotheses", style="yellow"))
                        if self.data_manager:
                            self.data_manager.log(f"Resampling (ESS: {ess:.2f})")
                        hypotheses = resample_hypotheses_with_other_info(hypotheses, ess)
                        resampled = True
                    # Rejuvenate if text diversity is too low (hypotheses too similar)
                    elif overall_text_diversity < 0.25:
                        if should_print:
                            console.print(Panel(f"Text diversity: {overall_text_diversity:.3f}", title="Low Variance Hypotheses", style="red"))
                        if self.data_manager:
                            self.data_manager.log(f"Rejuvenating (Diversity: {overall_text_diversity:.3f})")
                        hypotheses = self.rejuvenate_hypotheses(hypotheses)
                        rejuvenated = True
                
                hypotheses_list.append(hypotheses)
                
                # Summarize current user profile
                user_profile = self.summarize_hypotheses(hypotheses)
                
                # Evaluate generation quality
                gen_score = self.evaluate_generation(user_profile, history, candidates, chosen_response)
                
                # Predict user choice
                predicted_idx = self.predict_choice(user_profile, history, candidates)
                # Determine actual index depending on data shape
                if candidates and isinstance(candidates[0], dict):
                    actual_idx = next((i for i, c in enumerate(candidates) if c.get('if_chosen', False)), 0)
                else:
                    # candidates are strings; match by equality
                    try:
                        actual_idx = candidates.index(chosen_text)
                    except ValueError:
                        actual_idx = 0
                prediction_correct = (predicted_idx == actual_idx)

                if should_print:
                    console.print(f"\n[bold]Turn {global_turn_idx} Results:[/bold]")
                    console.print(f"  Gen Score: [cyan]{gen_score:.3f}[/cyan]")
                    console.print(f"  Prediction: [{'green' if prediction_correct else 'red'}]{'✓' if prediction_correct else '✗'}[/{'green' if prediction_correct else 'red'}] (predicted: {predicted_idx}, actual: {actual_idx})")
                    if self.args.n_hypotheses > 1 and ess is not None:
                        console.print(f"  ESS: [yellow]{ess:.2f}[/yellow]")
                        console.print(f"  Text Diversity: [yellow]{overall_text_diversity:.3f}[/yellow]")

                turn_metrics = {
                    'gen_score': gen_score,
                    'prediction_correct': prediction_correct,
                    'predicted_idx': predicted_idx,
                    'actual_idx': actual_idx,
                    'ess': ess if ess is not None else float('nan'),
                    'text_diversity': overall_text_diversity if overall_text_diversity is not None else float('nan'),
                    'resampled': resampled,
                    'rejuvenated': rejuvenated,
                    'weights': hypotheses.weights.tolist(),
                    'hypotheses': hypotheses.texts[:3]  # 只保存前3个假设
                }

                if self.data_manager:
                    self.data_manager.save_turn_metrics(user_id, global_turn_idx, turn_metrics)
                    self.data_manager.log(f"Gen Score: {gen_score:.3f}, Prediction: {'✓' if prediction_correct else '✗'}")

                turn_results.append({
                    'turn': global_turn_idx,
                    'user_profile': user_profile,
                    'gen_score': gen_score,
                    'prediction_correct': prediction_correct,
                    'predicted_idx': predicted_idx,
                    'actual_idx': actual_idx,
                    'hypotheses': hypotheses.texts,
                    'weights': hypotheses.weights.tolist(),
                    'ess': ess if ess is not None else None,
                    'text_diversity': overall_text_diversity if overall_text_diversity is not None else None
                })
                global_turn_idx += 1
        
        result = {
            'user_id': user_id,
            'turn_results': turn_results,
            'final_profile': turn_results[-1]['user_profile'] if turn_results else ""
        }
        
        if should_print:
            console.print("\n" + "="*70)
            console.print(Panel(f"[bold green]✓ Completed tracing for user {user_id}[/bold green]\n[cyan]Total turns: {len(turn_results)}[/cyan]", 
                       style="green", box=box.DOUBLE))
            console.print("="*70 + "\n")
        
        if self.data_manager:
            self.data_manager.log(f"\nCompleted tracing for user {user_id}")
            self.data_manager.log(f"Total turns: {len(turn_results)}")
            self.data_manager.save_user_log()
        
        # Dump detailed trace
        result = self.dump(result, hypotheses_list)
        
        return result

    def group_by_turns(self, conversations: List[Dict]) -> List[Dict]:
        turns = {}
        for msg in conversations:
            turn = msg['turn']
            if turn not in turns:
                turns[turn] = {'user_message': None, 'candidates': [], 'chosen': None}
            
            if msg['role'] == 'user':
                turns[turn]['user_message'] = msg
            else:
                turns[turn]['candidates'].append(msg)
                if msg.get('if_chosen', False):
                    turns[turn]['chosen'] = msg
        
        return [turns[i] for i in sorted(turns.keys())]

    def build_history(self, turns: List[Dict]) -> List[Dict]:
        history = []
        for turn in turns:
            if turn['user_message']:
                if isinstance(turn['user_message'], str):
                    history.append({'role': 'user', 'content': turn['user_message']})
                else:
                    history.append(turn['user_message'])
            if turn['chosen']:
                if isinstance(turn['chosen'], str):
                    history.append({'role': 'assistant', 'content': turn['chosen']})
                else:
                    history.append(turn['chosen'])
        return history

    def dump(self, user_result: Dict, hypotheses_list: List[HypothesesSetV3]):
        trace_steps = []
        for idx, h in enumerate(hypotheses_list):
            weight_details = None
            if hasattr(h, 'weight_details') and h.weight_details:
                weight_details = {
                    'prompts': h.weight_details.get('prompts'),
                    'raw_predictions': h.weight_details.get('raw_predictions'),
                    'reasonings': h.weight_details.get('reasonings'),
                    'raw_scores': h.weight_details['raw_scores'].tolist() if isinstance(h.weight_details.get('raw_scores'), np.ndarray) else h.weight_details.get('raw_scores'),
                    'weights': h.weight_details['weights'].tolist() if isinstance(h.weight_details.get('weights'), np.ndarray) else h.weight_details.get('weights')
                }
            
            step_data = {
                'step': idx,
                'hypotheses': h.texts,
                'weights': h.weights.tolist(),
                'ess': h.previous_ess,
                'input_candidates': h.perceptions[-1].get('input_candidates', '<history_placeholder>') if h.perceptions else None,
                'weight_details': weight_details,
                'propagation_info': {
                    'prompts': h.kwargs.get('propagation_prompts', None),
                    'outputs': h.kwargs.get('propagated_outputs', None)
                } if idx > 0 else None
            }
            trace_steps.append(step_data)
        
        user_result['trace_steps'] = trace_steps
        user_result.pop('detailed_hypotheses', None)
        
        return user_result

def run_preference_tracing(args):

    run_dir = os.path.join(args.output_dir, args.run_id)
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    checkpoint_manager = CheckpointManager(checkpoint_dir, args.run_id)
    
    progress = checkpoint_manager.get_progress_summary()
    if progress['completed'] > 0:
        print(f"\n📌 Found existing checkpoint for run '{args.run_id}':")
        print(f"   - Completed: {progress['completed']} users")
        print(f"   - Failed: {progress['failed']} users")
        print(f"   - Last updated: {progress['last_updated']}")
        
        if hasattr(args, 'resume') and not args.resume:
            print("   ⚠️  Use --resume to continue from checkpoint or --reset-checkpoint to start fresh")
            return
 
    save_logs = hasattr(args, 'print') and args.print
    data_manager = DataManager(args.output_dir, args.run_id, save_logs=save_logs)

    from prism_adapter import load_prism_users
    users = load_prism_users(n_users=args.n_users)

    tracer = PreferenceTracer(args, data_manager=data_manager)

    all_user_ids = [u['user_id'] for u in users]
    pending_users = checkpoint_manager.get_pending_users(all_user_ids)
    
    if not pending_users:
        print(f"\n✅ All {len(all_user_ids)} users already completed!")
        print(f"   Use --reset-checkpoint to reprocess all users")
        return
    
    print(f"\n📊 Processing Status:")
    print(f"   Total users: {len(all_user_ids)}")
    print(f"   Completed: {len(checkpoint_manager.get_completed_users())}")
    print(f"   Remaining: {len(pending_users)}")
    print(f"   Failed: {len(checkpoint_manager.get_failed_users())}\n")
    
    all_results = []
    
    # Process users (only pending ones)
    user_map = {u['user_id']: u for u in users}

    for user_id in tqdm(pending_users, desc="Tracing users"):
        if user_id not in user_map:
            continue
        user_bundle = user_map[user_id]
        
        checkpoint_manager.mark_user_started(user_id)
        
        result = tracer.trace_user_preferences(user_bundle)
        all_results.append(result)

        data_manager.save_user_trace(user_id, result)

        turns_completed = len(result.get('turn_results', []))
        checkpoint_manager.mark_user_completed(user_id, turns_completed)
        
        # except Exception as e:
        #     error_msg = str(e)
        #     print(f"❌ Error processing user {user_id}: {error_msg}")
        #     checkpoint_manager.mark_user_failed(user_id, error_msg)
        #     continue

    run_dir = os.path.join(args.output_dir, args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    output_file = os.path.join(run_dir, "results.json")
    
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            existing_results = json.load(f)
        existing_user_ids = {r['user_id'] for r in existing_results}
        new_results = [r for r in all_results if r['user_id'] not in existing_user_ids]
        all_results = existing_results + new_results
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    summary_file = data_manager.export_summary()
    
    print(f"\n✅ Results saved:")
    print(f"   📄 Tracing results: {output_file}")
    print(f"   📊 Metrics: {os.path.join(data_manager.metrics_dir, 'all_metrics.json')}")
    print(f"   📝 Summary: {summary_file}")
    if save_logs:
        print(f"   📋 Logs: {data_manager.logs_dir}/")

    analyze_results(all_results, args)

def analyze_results(results: List[Dict], args):
    turn_gen_scores = {}
    turn_pred_accuracy = {}
    
    for user_result in results:
        for turn_result in user_result['turn_results']:
            turn = turn_result['turn']
            
            if turn not in turn_gen_scores:
                turn_gen_scores[turn] = []
            if turn not in turn_pred_accuracy:
                turn_pred_accuracy[turn] = []
            
            turn_gen_scores[turn].append(turn_result['gen_score'])
            turn_pred_accuracy[turn].append(1.0 if turn_result['prediction_correct'] else 0.0)
    
    summary = {
        'turn_gen_scores': {t: {'mean': np.mean(scores), 'std': np.std(scores), 'ci': 1.96 * np.std(scores) / np.sqrt(len(scores))} for t, scores in turn_gen_scores.items()},
        'turn_pred_accuracy': {t: {'mean': np.mean(acc), 'std': np.std(acc), 'ci': 1.96 * np.std(acc) / np.sqrt(len(acc))} for t, acc in turn_pred_accuracy.items()}
    }
    
    run_dir = os.path.join(args.output_dir, args.run_id)
    summary_file = os.path.join(run_dir, "analysis_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n=== Generation Score by Turn ===")
    for turn in sorted(turn_gen_scores.keys()):
        stats = summary['turn_gen_scores'][turn]
        print(f"Turn {turn}: {stats['mean']:.3f} ± {stats['ci']:.3f}")
    
    print("\n=== Prediction Accuracy by Turn ===")
    for turn in sorted(turn_pred_accuracy.keys()):
        stats = summary['turn_pred_accuracy'][turn]
        print(f"Turn {turn}: {stats['mean']:.3f} ± {stats['ci']:.3f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracing-model', type=str, default='gpt-4o-mini', help='Model for tracing')
    parser.add_argument('--eval-model', type=str, default='gpt-4o-mini', help='Model for evaluation')
    parser.add_argument('--n-hypotheses', type=int, default=5, help='Number of hypotheses')
    parser.add_argument('--n-users', type=int, default=10, help='Number of users to process')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='preference_results', help='Output directory')
    parser.add_argument('--run-id', type=str, default='default', help='Run ID')
    parser.add_argument('--print', action='store_true', help='Print debug information')
    
    args = parser.parse_args()
    run_preference_tracing(args)
