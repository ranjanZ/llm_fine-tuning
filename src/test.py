#!/usr/bin/env python3
"""
Testing and inference script for the addition GPT model.
"""

import torch
from dataset import AdditionDataset
from model import AdditionDecoder
from train import GPTTrainer
import random

def interactive_demo(model: AdditionDecoder, dataset: AdditionDataset):
    """Interactive demo to test the model."""
    model.eval()
    
    print("\n" + "="*60)
    print("Interactive Addition Demo")
    print("="*60)
    print("Enter addition problems like '12+34=' or 'q' to quit")
    print("="*60)
    
    while True:
        user_input = input("\nEnter problem: ").strip()
        
        if user_input.lower() == 'q':
            break
        
        # Validate input
        if '+' not in user_input:
            print("Please include '+' (e.g., '12+34')")
            continue
        
        # Add '=' if not present
        if '=' not in user_input:
            user_input = user_input + '='
        
        try:
            # Encode prompt
            prompt_tokens = dataset.encode(user_input, add_eos=False)
            prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long)
            
            # Generate
            with torch.no_grad():
                generated = model.generate(
                    prompt_tensor,
                    max_new_tokens=10,
                    temperature=0.8,
                    eos_token_id=dataset.eos_token_id
                )
            
            # Decode
            generated_tokens = generated[0].tolist()
            generated_text = dataset.decode(generated_tokens)
            
            # Extract numbers and calculate expected
            parts = user_input.rstrip('=').split('+')
            if len(parts) == 2:
                try:
                    a = int(parts[0])
                    b = int(parts[1])
                    expected = str(a + b)
                    
                    print(f"\n{a} + {b} = ?")
                    print(f"Model says: {generated_text}")
                    print(f"Correct answer: {expected}")
                    
                except:
                    print(f"\nModel output: {generated_text}")
            else:
                print(f"\nModel output: {generated_text}")
                
        except Exception as e:
            print(f"Error: {e}")
            print("Make sure to use only digits 0-9 and '+'")

def batch_test(model: AdditionDecoder, dataset: AdditionDataset, num_tests: int = 20):
    """Test model on a batch of problems."""
    model.eval()
    
    print(f"\nBatch Testing ({num_tests} problems)")
    print("="*60)
    
    correct = 0
    total = 0
    
    for i in range(num_tests):
        # Generate random problem
        problem, expected, full_problem = dataset.generate_random_problem()


        try:
            # Encode prompt
            prompt_tokens = dataset.encode(problem, add_eos=False)
            prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long)
            
            # Generate
            with torch.no_grad():
                generated = model.generate(
                    prompt_tensor,
                    max_new_tokens=10,
                    temperature=0.8,
                    eos_token_id=dataset.eos_token_id
                )
            
            # Decode
            generated_tokens = generated[0].tolist()
            generated_text = dataset.decode(generated_tokens)
            
            # Check correctness
            if generated_text == expected:
                correct += 1
                symbol = "✓"
            else:
                symbol = "✗"
            
            total += 1
            
            if i < 5:  # Show first 5 examples
                print(f"{symbol} {problem} → Expected: {expected}, Got: {generated_text}")
            
        except Exception as e:
            print(f"Error testing {problem}: {e}")
            continue
    
    accuracy = correct / total * 100
    print(f"\nResults: {correct}/{total} correct ({accuracy:.1f}%)")
    
    return accuracy

def load_model_from_checkpoint(checkpoint_path: str, dataset: AdditionDataset) -> AdditionDecoder:
    """Load a trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model with same architecture
    model = AdditionDecoder(
        vocab_size=checkpoint.get('vocab_size', dataset.vocab_size),
        embed_dim=checkpoint.get('embed_dim', 96),
        num_layers=checkpoint.get('num_layers', 3),
        num_heads=checkpoint.get('num_heads', 4),
        max_seq_len=checkpoint.get('max_seq_len', 15)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Model loaded successfully!")
    return model

def main():
    """Main testing function."""
    print("Testing Script")
    print("=" * 50)
    
    print("Choose an option:")
    print("1. Test with a new model")
    print("2. Load and test a trained model")
    print("3. Interactive demo with new model")
    print("4. Interactive demo with trained model")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = AdditionDataset(max_digits=2, num_samples=100)
    
    if choice in ['1', '3']:
        # Create new model
        print("\nCreating new model...")
        from model import create_model
        model = create_model(
            vocab_size=dataset.vocab_size,
            embed_dim=64,
            num_layers=2,
            num_heads=2,
            max_seq_len=10
        )
        
        print("\nNote: Model is untrained - expect random answers!")
        
        if choice == '3':
            interactive_demo(model, dataset)
        else:
            accuracy = batch_test(model, dataset, num_tests=10)
    
    elif choice in ['2', '4']:
        # Load trained model
        checkpoint_path = input("Enter checkpoint path (or press Enter for 'best_model.pth'): ").strip()
        if not checkpoint_path:
            checkpoint_path = "model/best_model.pth"
        
        try:
            model = load_model_from_checkpoint(checkpoint_path, dataset)
            
            if choice == '4':
                interactive_demo(model, dataset)
            else:
                accuracy = batch_test(model, dataset, num_tests=20)
        except FileNotFoundError:
            print(f"Checkpoint file '{checkpoint_path}' not found!")
            print("Please train a model first using train.py")
            return
    
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()

