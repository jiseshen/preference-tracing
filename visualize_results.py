import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def plot_learning_curves(summary_file: str, output_dir: str):
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    def _extract_curve(stats_dict):
        turns = sorted([int(t) for t in stats_dict.keys()])
        means = []
        cis = []
        for t in turns:
            entry = stats_dict.get(str(t), {}) if isinstance(list(stats_dict.keys())[0], str) else stats_dict.get(t, {})
            means.append(entry.get('mean', np.nan))
            cis.append(entry.get('ci', np.nan))
        # Filter nan
        turns_clean = []
        means_clean = []
        cis_clean = []
        for t, m, c in zip(turns, means, cis):
            if np.isfinite(m):
                turns_clean.append(t)
                means_clean.append(m)
                cis_clean.append(c if np.isfinite(c) else 0.0)
        return turns_clean, means_clean, cis_clean

    # Generation quality
    gen_turns, gen_means, gen_cis = _extract_curve(summary['turn_gen_scores'])
    ax1 = axes[0]
    ax1.plot(gen_turns, gen_means, marker='o', linewidth=2, markersize=8, label='Generation Score')
    if gen_turns:
        ax1.fill_between(gen_turns,
                         np.array(gen_means) - np.array(gen_cis),
                         np.array(gen_means) + np.array(gen_cis),
                         alpha=0.3)
    ax1.set_xlabel('Turn', fontsize=12)
    ax1.set_ylabel('Generation Quality Score', fontsize=12)
    ax1.set_title('Generation Quality Over Turns', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Prediction accuracy
    pred_turns, pred_means, pred_cis = _extract_curve(summary['turn_pred_accuracy'])
    ax2 = axes[1]
    ax2.plot(pred_turns, pred_means, marker='s', linewidth=2, markersize=8, color='coral', label='Prediction Accuracy')
    if pred_turns:
        ax2.fill_between(pred_turns,
                         np.array(pred_means) - np.array(pred_cis),
                         np.array(pred_means) + np.array(pred_cis),
                         alpha=0.3, color='coral')
    ax2.set_xlabel('Turn', fontsize=12)
    ax2.set_ylabel('Prediction Accuracy', fontsize=12)
    ax2.set_title('Choice Prediction Accuracy Over Turns', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Soft loss
    soft_turns, soft_means, soft_cis = _extract_curve(summary.get('turn_soft_loss', {}))
    ax3 = axes[2]
    ax3.plot(soft_turns, soft_means, marker='^', linewidth=2, markersize=8, color='slateblue', label='Soft CE Loss')
    if soft_turns:
        ax3.fill_between(soft_turns,
                         np.array(soft_means) - np.array(soft_cis),
                         np.array(soft_means) + np.array(soft_cis),
                         alpha=0.3, color='slateblue')
    ax3.set_xlabel('Turn', fontsize=12)
    ax3.set_ylabel('Soft Cross-Entropy Loss', fontsize=12)
    ax3.set_title('Soft Loss Over Turns', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/learning_curves.png', dpi=300, bbox_inches='tight')
    print(f"Learning curves saved to {output_dir}/learning_curves.png")
    plt.close()

def plot_survey_alignment(survey_eval_file: str, output_dir: str):
    with open(survey_eval_file, 'r') as f:
        survey_results = json.load(f)
    
    if not survey_results['summary_statistics']:
        print("No survey evaluation data available")
        return
    
    metrics = ['communication_style', 'value_alignment', 'preference_consistency', 'overall_accuracy']
    metric_labels = [m.replace('_', ' ').title() for m in metrics]
    means = [survey_results['summary_statistics'][m]['mean'] for m in metrics]
    cis = [survey_results['summary_statistics'][m]['ci'] for m in metrics]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = sns.color_palette("husl", len(metrics))
    bars = ax.bar(metric_labels, means, yerr=cis, capsize=5, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Alignment Score (1-10)', fontsize=12)
    ax.set_title('User Profile Alignment with Survey Data', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 10])
    ax.grid(True, axis='y', alpha=0.3)
    
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/survey_alignment.png', dpi=300, bbox_inches='tight')
    print(f"Survey alignment plot saved to {output_dir}/survey_alignment.png")
    plt.close()

def plot_user_trajectories(results_file: str, output_dir: str, n_users: int = 5):
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    for idx, user_result in enumerate(results[:n_users]):
        user_id = user_result['user_id']
        turns = [r['turn'] for r in user_result['turn_results']]
        gen_scores = [r['gen_score'] for r in user_result['turn_results']]
        pred_correct = [1 if r['prediction_correct'] else 0 for r in user_result['turn_results']]
        
        axes[0].plot(turns, gen_scores, marker='o', linewidth=2, label=f'User {user_id[:8]}', alpha=0.7)
        axes[1].plot(turns, pred_correct, marker='s', linewidth=2, label=f'User {user_id[:8]}', alpha=0.7)
    
    axes[0].set_xlabel('Turn', fontsize=12)
    axes[0].set_ylabel('Generation Score', fontsize=12)
    axes[0].set_title('Individual User Generation Quality Trajectories', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Turn', fontsize=12)
    axes[1].set_ylabel('Prediction Correct (0/1)', fontsize=12)
    axes[1].set_title('Individual User Prediction Accuracy Trajectories', fontsize=14, fontweight='bold')
    axes[1].legend(loc='best', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/user_trajectories.png', dpi=300, bbox_inches='tight')
    print(f"User trajectories saved to {output_dir}/user_trajectories.png")
    plt.close()

def generate_all_plots(results_file: str, summary_file: str, survey_eval_file: str, output_dir: str):
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating learning curves...")
    plot_learning_curves(summary_file, output_dir)
    
    print("Generating user trajectories...")
    plot_user_trajectories(results_file, output_dir)
    
    if survey_eval_file and os.path.exists(survey_eval_file):
        print("Generating survey alignment plot...")
        plot_survey_alignment(survey_eval_file, output_dir)
    
    print(f"\nAll plots saved to {output_dir}/")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-file', type=str, required=True, help='Path to preference tracing results')
    parser.add_argument('--summary-file', type=str, required=True, help='Path to summary statistics')
    parser.add_argument('--survey-eval-file', type=str, default=None, help='Path to survey evaluation results')
    parser.add_argument('--output-dir', type=str, default='preference_results/plots', help='Output directory for plots')
    
    args = parser.parse_args()
    
    generate_all_plots(args.results_file, args.summary_file, args.survey_eval_file, args.output_dir)
