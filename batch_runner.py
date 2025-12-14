#!/usr/bin/env python3
"""
Batch experiment runner for PRISM preference tracing
Supports running multiple configurations from configs.yaml
"""

import os
import sys
import yaml
import subprocess
from datetime import datetime

def load_configs(config_file='configs.yaml'):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def run_experiment(config_name, config_params):
    print(f"\n{'='*70}")
    print(f"Running Experiment: {config_name}")
    print(f"{'='*70}")
    print(f"Config: {config_params}")
    
    cmd = ['python', 'run_prism_pipeline.py', '--stage', 'all']
    
    for key, value in config_params.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key.replace("_", "-")}')
        else:
            cmd.extend([f'--{key.replace("_", "-")}', str(value)])
    
    print(f"\nCommand: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✓ {config_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {config_name} failed with error: {e}")
        return False

def run_ablation_study(study_name, study_config):
    print(f"\n{'='*70}")
    print(f"Running Ablation Study: {study_name}")
    print(f"{'='*70}")
    
    base_params = study_config['base']
    configs = study_config['configs']
    
    results = []
    for config in configs:
        # Merge base params with specific config
        params = {**base_params, **config}
        success = run_experiment(f"{study_name}_{config['run_id']}", params)
        results.append((config['run_id'], success))
    
    print(f"\n{'='*70}")
    print(f"Ablation Study '{study_name}' Summary:")
    print(f"{'='*70}")
    for run_id, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{run_id}: {status}")
    
    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch experiment runner')
    parser.add_argument('--config-file', type=str, default='configs.yaml',
                       help='Path to configuration file')
    parser.add_argument('--experiments', nargs='+', default=None,
                       help='Specific experiments to run (default: all)')
    parser.add_argument('--ablation-studies', nargs='+', default=None,
                       help='Specific ablation studies to run')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print commands without executing')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config_file):
        print(f"Error: Config file '{args.config_file}' not found")
        sys.exit(1)
    
    configs = load_configs(args.config_file)
    
    # Identify single experiments vs ablation studies
    single_experiments = {}
    ablation_studies = {}
    
    for name, config in configs.items():
        if 'configs' in config and 'base' in config:
            ablation_studies[name] = config
        else:
            single_experiments[name] = config
    
    print(f"{'='*70}")
    print(f"PRISM Preference Tracing - Batch Runner")
    print(f"{'='*70}")
    print(f"Config file: {args.config_file}")
    print(f"Single experiments: {list(single_experiments.keys())}")
    print(f"Ablation studies: {list(ablation_studies.keys())}")
    print(f"Dry run: {args.dry_run}")
    
    if args.dry_run:
        print("\nDry run mode - commands will be printed but not executed")
    
    start_time = datetime.now()
    
    # Run single experiments
    if args.experiments:
        experiments_to_run = {k: v for k, v in single_experiments.items() 
                            if k in args.experiments}
    else:
        experiments_to_run = single_experiments
    
    exp_results = []
    for exp_name, exp_config in experiments_to_run.items():
        if args.dry_run:
            print(f"\nWould run: {exp_name} with config: {exp_config}")
        else:
            success = run_experiment(exp_name, exp_config)
            exp_results.append((exp_name, success))
    
    # Run ablation studies
    if args.ablation_studies:
        studies_to_run = {k: v for k, v in ablation_studies.items() 
                         if k in args.ablation_studies}
    else:
        studies_to_run = ablation_studies
    
    ablation_results = []
    for study_name, study_config in studies_to_run.items():
        if args.dry_run:
            print(f"\nWould run ablation study: {study_name}")
            for config in study_config['configs']:
                print(f"  - {config['run_id']}: {config}")
        else:
            results = run_ablation_study(study_name, study_config)
            ablation_results.extend([(f"{study_name}_{r[0]}", r[1]) for r in results])
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"BATCH RUN COMPLETE")
    print(f"{'='*70}")
    print(f"Duration: {duration}")
    
    if not args.dry_run:
        print(f"\nExperiment Results:")
        all_results = exp_results + ablation_results
        for name, success in all_results:
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"  {name}: {status}")
        
        total = len(all_results)
        passed = sum(1 for _, s in all_results if s)
        print(f"\nTotal: {passed}/{total} passed")

if __name__ == "__main__":
    main()
