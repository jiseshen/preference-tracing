# PRISM Preference Tracing

Online learning framework for tracing user preferences in pluralistic alignment using hypothesis-based particle filtering.

## Overview

This system adapts the thought-tracing framework to learn user preferences from conversational data. It maintains multiple hypotheses about user preferences and updates them through:

1. **Initialization**: Generate initial hypotheses about user preferences from first interaction
2. **Propagation**: Update hypotheses as new conversation turns are observed
3. **Weighting**: Evaluate likelihood of user's choice under each hypothesis
4. **Resampling**: Resample hypotheses when effective sample size drops
5. **Summarization**: Aggregate weighted hypotheses into coherent user profile

## Architecture

### Core Components

- **PreferenceTracer**: Main tracing engine that maintains and updates preference hypotheses
- **SurveyEvaluator**: Evaluates inferred profiles against ground-truth survey data
- **Visualizer**: Generates learning curves and analysis plots

### Key Differences from Original Tracer

- **Target**: User preferences instead of agent mental states
- **Perception**: Conversation history + response candidates
- **Action**: User's choice among response candidates
- **Hypothesis**: User preference profile (values, communication style, priorities)

## Installation

```bash
pip install datasets transformers openai numpy pandas matplotlib seaborn tqdm
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-key-here"
```

## Usage

### Quick Start - Full Pipeline

```bash
python run_prism_pipeline.py \
    --stage all \
    --tracing-model gpt-4o-mini \
    --eval-model gpt-4o-mini \
    --n-hypotheses 4 \
    --n-users 10 \
    --run-id experiment_1
```

### Stage-by-Stage Execution

**1. Preference Tracing**
```bash
python run_prism_pipeline.py \
    --stage trace \
    --tracing-model gpt-4o-mini \
    --n-hypotheses 4 \
    --n-users 20 \
    --run-id run1
```

**2. Survey Evaluation**
```bash
python run_prism_pipeline.py \
    --stage evaluate \
    --eval-model gpt-4o-mini \
    --run-id run1
```

**3. Visualization**
```bash
python run_prism_pipeline.py \
    --stage visualize \
    --run-id run1
```

### Direct Script Usage

```bash
# Preference tracing
python preference_tracer.py \
    --tracing-model gpt-4o-mini \
    --eval-model gpt-4o-mini \
    --n-hypotheses 4 \
    --n-users 10 \
    --output-dir results

# Survey evaluation
python survey_evaluator.py \
    --eval-model gpt-4o-mini \
    --tracing-results results/preference_tracing_results_default.json \
    --output-dir results

# Visualization
python visualize_results.py \
    --results-file results/preference_tracing_results_default.json \
    --summary-file results/preference_tracing_summary_default.json \
    --survey-eval-file results/survey_evaluation_default.json \
    --output-dir results/plots
```

## Outputs

### Preference Tracing Results
- `preference_tracing_results_{run_id}.json`: Per-user, per-turn detailed results
  - User profiles at each turn
  - Hypothesis weights
  - Generation scores
  - Prediction accuracy

### Summary Statistics
- `preference_tracing_summary_{run_id}.json`: Aggregated metrics
  - Generation quality by turn (mean, std, CI)
  - Prediction accuracy by turn (mean, std, CI)

### Survey Evaluation
- `survey_evaluation_{run_id}.json`: Profile alignment with surveys
  - Communication style alignment
  - Value alignment
  - Preference consistency
  - Overall accuracy

### Visualizations
- `plots/learning_curves.png`: Generation quality and prediction accuracy over turns
- `plots/user_trajectories.png`: Individual user learning trajectories
- `plots/survey_alignment.png`: Profile-survey alignment scores

## Evaluation Metrics

### Generation Quality Score
- Range: 0-1
- Measures: How well generated responses match user's chosen response
- Evaluator: Separate LLM judges similarity in style, content, and alignment

### Prediction Accuracy
- Range: 0-1 (binary)
- Measures: Whether system correctly predicts user's choice among candidates
- Based on: Inferred user profile applied to candidate ranking

### Survey Alignment Scores
- Range: 1-10 per dimension
- Dimensions:
  - Communication style
  - Value alignment
  - Preference consistency
  - Overall accuracy

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--tracing-model` | gpt-4o-mini | Model for hypothesis generation and propagation |
| `--eval-model` | gpt-4o-mini | Model for evaluation and scoring |
| `--n-hypotheses` | 4 | Number of preference hypotheses to maintain |
| `--n-users` | 10 | Number of users to process |
| `--seed` | 42 | Random seed for reproducibility |
| `--output-dir` | preference_results | Directory for outputs |
| `--run-id` | default | Identifier for experiment run |

## Algorithm Details

### Hypothesis Update Flow

```
For each user:
  For each conversation turn:
    1. Observe: conversation_history + response_candidates
    2. If first turn:
         hypotheses = initialize(observation)
       Else:
         hypotheses = propagate(previous_hypotheses, observation)
    3. weights = weigh(hypotheses, user_choice)
    4. If ESS < threshold:
         hypotheses = resample(hypotheses, weights)
    5. profile = summarize(hypotheses)
    6. Evaluate:
         - gen_score = evaluate_generation(profile, chosen_response)
         - prediction = predict_choice(profile, candidates)
```

### Weight Calculation

Uses likelihood estimation via LLM prompting:
- Prompt: Given hypothesis H and candidates C, how likely is choice c?
- Response: 6-point scale (Very Likely to Very Unlikely)
- Scoring: Map to numerical values, apply softmax

### Resampling Trigger

Effective Sample Size (ESS): `ESS = 1 / Σ(w_i²)`
- Resample when: `ESS < n_hypotheses / 2`
- Prevents: Hypothesis degeneracy

## Data Format

### PRISM Dataset Structure
```json
{
  "conversation_id": "...",
  "user_id": "...",
  "turn": 0,
  "role": "user|model",
  "content": "...",
  "model_provider": "...",
  "model_name": "...",
  "score": 1-100,
  "if_chosen": true|false,
  "within_turn_id": 0-3
}
```

## Extending the System

### Custom Hypothesis Generators
Override `initialize_hypotheses()` in `PreferenceTracer`

### Alternative Weighting Schemes
Modify `weigh_hypotheses()` method

### Additional Evaluation Metrics
Extend `evaluate_generation()` or add new evaluators

## Citation

Based on thought-tracing framework adapted for preference learning in pluralistic alignment.

## License

See LICENSE file for details.
