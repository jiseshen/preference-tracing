#!/usr/bin/env python3
"""
Quick test script to verify PRISM preference tracing setup
"""
import os
import sys

def test_imports():
    print("Testing imports...")
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), "thought-tracing"))
        
        from agents.load_model import load_model
        print("✓ agents.load_model")
        
        from hypothesis import HypothesesSetV3, compute_ess
        print("✓ hypothesis")
        
        from utils import softmax, prompting_for_ordered_list
        print("✓ utils")
        
        from datasets import load_dataset
        print("✓ datasets")
        
        import numpy as np
        print("✓ numpy")
        
        print("\n✓ All imports successful!")
        return True
    except Exception as e:
        print(f"\n✗ Import failed: {e}")
        return False

def test_dataset_loading():
    print("\nTesting PRISM dataset loading...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("HannahRoseKirk/prism-alignment", "conversations")
        print(f"✓ Dataset loaded (conversations)")
        print(f"  - Available splits: {list(dataset.keys())}")
        train_data = dataset['train']
        print(f"  - Train size: {len(train_data)}")
        sample = train_data[0]
        print(f"  - Sample keys: {list(sample.keys())}")
        print(f"  - Sample conversation_id: {sample.get('conversation_id', 'N/A')}")
        print(f"  - Sample user_id: {sample.get('user_id', 'N/A')}")
        print("\n✓ Dataset loading successful!")
        return True
    except Exception as e:
        print(f"\n✗ Dataset loading failed: {e}")
        return False

def test_hypothesis_creation():
    print("\nTesting hypothesis creation...")
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), "thought-tracing"))
        from hypothesis import HypothesesSetV3
        import numpy as np
        
        texts = [
            "User prefers concise responses",
            "User values technical accuracy",
            "User appreciates friendly tone"
        ]
        weights = np.array([0.5, 0.3, 0.2])
        
        hypotheses = HypothesesSetV3(
            target_agent="test_user",
            contexts=[{"test": "context"}],
            perceptions=[{"perception": "test"}],
            texts=texts,
            weights=weights
        )
        
        print(f"✓ Created hypotheses set with {len(hypotheses.texts)} hypotheses")
        print(f"  - Weights sum: {sum(hypotheses.weights):.3f}")
        
        from hypothesis import compute_ess
        ess = compute_ess(hypotheses)
        print(f"  - ESS: {ess:.2f}")
        
        print("\n✓ Hypothesis creation successful!")
        return True
    except Exception as e:
        print(f"\n✗ Hypothesis creation failed: {e}")
        return False

def main():
    print("="*60)
    print("PRISM Preference Tracing - Setup Verification")
    print("="*60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    print()
    
    results.append(("Dataset Loading", test_dataset_loading()))
    print()
    
    results.append(("Hypothesis Creation", test_hypothesis_creation()))
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n✓ All tests passed! System is ready.")
        print("\nNext steps:")
        print("  1. Set OPENAI_API_KEY environment variable")
        print("  2. Run: python run_prism_pipeline.py --stage all --n-users 5")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
