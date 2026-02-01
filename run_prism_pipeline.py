#!/usr/bin/env python3
import os
import sys
import argparse
import shutil
import dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "thought-tracing"))

def main():
    parser = argparse.ArgumentParser(description='PRISM Preference Tracing Pipeline')
    parser.add_argument('--stage', type=str, required=True, 
                       choices=['trace', 'evaluate', 'visualize', 'all'],
                       help='Pipeline stage to run')
    parser.add_argument('--tracing-model', type=str, default='gpt-4o-mini', 
                       help='Model for preference tracing')
    parser.add_argument('--eval-model', type=str, default='gpt-4o-mini', 
                       help='Model for evaluation')
    parser.add_argument('--n-hypotheses', type=int, default=5, 
                       help='Number of hypotheses to maintain')
    parser.add_argument('--n-users', type=int, default=10, 
                       help='Number of users to process')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed')
    parser.add_argument('--output-dir', type=str, default='preference_results', 
                       help='Output directory')
    parser.add_argument('--run-id', type=str, default='default', 
                       help='Run identifier')
    parser.add_argument('--resume', action='store_true', help='Resume from existing run directory/checkpoints if present')
    parser.add_argument('--print', action='store_true', help='Enable verbose console logging during tracing')
    
    args = parser.parse_args()
    dotenv.load_dotenv(override=True)
    print(f"Using OpenAI API Key: ...{os.getenv('OPENAI_API_KEY')[-4:]}")
    os.makedirs(args.output_dir, exist_ok=True)
    run_dir = os.path.join(args.output_dir, args.run_id)
    results_file = os.path.join(run_dir, "results.json")
    analysis_summary_file = os.path.join(run_dir, "analysis_summary.json")
    survey_eval_file = os.path.join(run_dir, "survey_evaluation.json")

    # Fresh start behavior: if not resuming and run dir exists, remove it
    if args.stage in ['trace', 'all']:
        if os.path.exists(run_dir) and not args.resume:
            print(f"\n[Reset] Removing existing run directory: {run_dir}")
            shutil.rmtree(run_dir)
        os.makedirs(run_dir, exist_ok=True)
    
    if args.stage in ['trace', 'all']:
        print("\n" + "="*60)
        print("STAGE 1: Running Preference Tracing")
        print("="*60)
        
        from preference_tracer import run_preference_tracing
        run_preference_tracing(args)
    
    if args.stage in ['evaluate', 'all']:
        print("\n" + "="*60)
        print("STAGE 2: Evaluating Against Survey Data")
        print("="*60)
        
        if not os.path.exists(results_file):
            print(f"Error: Results file not found at {results_file}")
            print("Please run the 'trace' stage first.")
            sys.exit(1)
        
        from survey_evaluator import run_survey_evaluation
        
        eval_args = argparse.Namespace(
            eval_model=args.eval_model,
            tracing_results=results_file,
            output_dir=run_dir,
            run_id=args.run_id
        )
        run_survey_evaluation(eval_args)
    
    if args.stage in ['visualize', 'all']:
        print("\n" + "="*60)
        print("STAGE 3: Generating Visualizations")
        print("="*60)
        
        if not os.path.exists(analysis_summary_file):
            print(f"Error: Summary file not found at {analysis_summary_file}")
            print("Please run the 'trace' stage first.")
            sys.exit(1)
        
        from visualize_results import generate_all_plots
        
        plot_dir = os.path.join(run_dir, 'plots')
        generate_all_plots(
            results_file=results_file,
            summary_file=analysis_summary_file,
            survey_eval_file=survey_eval_file if os.path.exists(survey_eval_file) else None,
            output_dir=plot_dir
        )
    
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)
    print(f"Run directory: {run_dir}")

if __name__ == "__main__":
    main()
