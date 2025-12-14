#!/usr/bin/env python3
"""
Demo script to show the detailed logging functionality of PRISM preference tracer.
This will trace 2 users with verbose output showing all steps.
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='Demo: PRISM Preference Tracing with Detailed Logging')
    parser.add_argument('--n-users', type=int, default=2, help='Number of users to trace (default: 2)')
    parser.add_argument('--n-hypotheses', type=int, default=3, help='Number of hypotheses (default: 3)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("PRISM Preference Tracing - Logging Demo")
    print("="*70)
    print(f"\nThis demo will trace {args.n_users} user(s) with detailed logging enabled.")
    print("You will see:")
    print("  • Initial hypothesis generation")
    print("  • Hypothesis propagation at each turn")
    print("  • Weighting based on user choices")
    print("  • User profile summaries")
    print("  • Evaluation metrics")
    print("\nMake sure OPENAI_API_KEY is set in your environment.\n")
    
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Import and run
    sys.path.append(os.path.dirname(__file__))
    from preference_tracer import run_preference_tracing
    
    # Create args namespace
    tracer_args = argparse.Namespace(
        tracing_model='gpt-4o-mini',
        eval_model='gpt-4o-mini',
        n_hypotheses=args.n_hypotheses,
        n_users=args.n_users,
        seed=42,
        output_dir='demo_results',
        run_id='demo',
        print=True,  # Enable detailed logging
        save_detailed_traces=True  # Save detailed traces per user
    )
    
    print("\n" + "="*70)
    print("Starting preference tracing...")
    print("="*70 + "\n")
    
    run_preference_tracing(tracer_args)
    
    print("\n" + "="*70)
    print("Demo Complete!")
    print("="*70)
    print(f"\nResults saved to: demo_results/")
    print(f"Detailed traces: demo_results/detailed_trace_*_demo.json")
    print(f"Summary: demo_results/preference_tracing_summary_demo.json")

if __name__ == "__main__":
    main()
