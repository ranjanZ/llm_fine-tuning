#!/usr/bin/env python3
"""
Direct Preference Optimization (DPO) for Addition GPT.
Fixed implementation with proper padding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Dict, List, Optional, Tuple
from dataset import AdditionDataset
from model import AdditionDecoder

class DPODataset:
    """Dataset for DPO training with proper padding."""
    
    def __init__(self, max_digits: int = 2, num_pairs: int = 1000):
        self.max_digits = max_digits
        self.num_pairs = num_pairs
        
        # We need a tokenizer - create a temporary dataset to use its tokenizer
        temp_ds = AdditionDataset(max_digits=1, num_samples=1)
        self.tokenizer = temp_ds.tokenizer
        
        # Generate preference pairs
        self.data = self._generate_pairs()
        
        print(f"DPO Dataset: {len(self.data)} preference pairs")
    
    def _generate_pairs(self) -> List[Dict]:
        """Generate preference pairs (chosen vs rejected)."""
        data = []
        
        for _ in range(self.num_pairs):
            # Generate problem
            a = random.randint(1, 10**self.max_digits - 1)
            b = random.randint(1, 10**self.max_digits - 1)
            correct = a + b
            
            # Create correct answer (chosen)
            chosen_answer = str(correct)
            
            # Create incorrect answer (rejected) - ensure same length when possible
            if len(str(correct)) > 1 and random.random() < 0.5:
                # Try to create incorrect answer of same length
                digits = list(str(correct))
                # Swap two random digits
                i, j = random.sample(range(len(digits)), 2)
                digits[i], digits[j] = digits[j], digits[i]
                rejected_answer = ''.join(digits)
                
                # If by chance it's the same, add 1
                if rejected_answer == chosen_answer:
                    rejected_answer = str(correct + 1)
            else:
                # Off by one (keeps similar length)
                offset = random.choice([-1, 1])
                rejected_answer = str(max(1, correct + offset))
            
            # Ensure they're different
            while rejected_answer == chosen_answer:
                rejected_answer = str(int(rejected_answer) + random.randint(-2, 2))
                rejected_answer = str(max(1, int(rejected_answer)))  # Ensure positive
            
            data.append({
                'problem': f"{a}+{b}=",
                'chosen': chosen_answer,
                'rejected': rejected_answer,
            })
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize sequences
        chosen_sequence = item['problem'] + item['chosen']
        rejected_sequence = item['problem'] + item['rejected']
        
        chosen_tokens = self.tokenizer.encode(chosen_sequence, add_eos=True)
        rejected_tokens = self.tokenizer.encode(rejected_sequence, add_eos=True)
        prompt_tokens = self.tokenizer.encode(item['problem'], add_eos=False)
        
        return {
            'chosen_tokens': chosen_tokens,
            'rejected_tokens': rejected_tokens,
            'prompt_tokens': prompt_tokens,
            'problem': item['problem'],
            'chosen_answer': item['chosen'],
            'rejected_answer': item['rejected']
        }
    
    def collate_fn(self, batch):
        """Collate function with padding."""
        # Find max length for chosen and rejected tokens
        max_chosen_len = max(len(item['chosen_tokens']) for item in batch)
        max_rejected_len = max(len(item['rejected_tokens']) for item in batch)
        max_len = max(max_chosen_len, max_rejected_len)
        
        # Pad sequences
        padded_chosen = []
        padded_rejected = []
        prompts = []
        problems = []
        chosen_answers = []
        rejected_answers = []
        
        for item in batch:
            # Pad chosen tokens
            chosen_pad_len = max_len - len(item['chosen_tokens'])
            padded_chosen.append(
                item['chosen_tokens'] + [self.tokenizer.pad_token_id] * chosen_pad_len
            )
            
            # Pad rejected tokens
            rejected_pad_len = max_len - len(item['rejected_tokens'])
            padded_rejected.append(
                item['rejected_tokens'] + [self.tokenizer.pad_token_id] * rejected_pad_len
            )
            
            prompts.append(item['prompt_tokens'])
            problems.append(item['problem'])
            chosen_answers.append(item['chosen_answer'])
            rejected_answers.append(item['rejected_answer'])
        
        return {
            'chosen_tokens': torch.tensor(padded_chosen, dtype=torch.long),
            'rejected_tokens': torch.tensor(padded_rejected, dtype=torch.long),
            'prompt_tokens': prompts,
            'problems': problems,
            'chosen_answers': chosen_answers,
            'rejected_answers': rejected_answers
        }

class DPOTrainer:
    """DPO trainer with proper padding handling."""
    
    def __init__(self, policy_model: AdditionDecoder, 
                 reference_model: AdditionDecoder, 
                 beta: float = 0.1, lr: float = 1e-5):
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.beta = beta
        
        self.device = torch.device('cpu')
        self.policy_model.to(self.device)
        self.reference_model.to(self.device)
        
        # Freeze reference model
        for param in self.reference_model.parameters():
            param.requires_grad = False
        
        # Optimizer for policy model only
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=lr,
            weight_decay=0.01
        )
        
        self.history = {
            'loss': [],
            'accuracy': [],
            'chosen_reward': [],
            'rejected_reward': []
        }
    
    def compute_logprobs(self, model: AdditionDecoder, tokens: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities of a sequence."""
        # Input is all tokens except last, target is all tokens except first
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]
        
        # Get model output
        logits, _ = model(inputs)
        
        # Compute log probs for target tokens
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather log probs for actual target tokens
        target_log_probs = torch.gather(
            log_probs, 
            dim=-1, 
            index=targets.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask out padding (pad_token_id = 0)
        mask = (targets != 0).float()
        masked_log_probs = target_log_probs * mask
        
        # Sum over sequence (excluding padding)
        sequence_logprobs = masked_log_probs.sum(dim=-1)
        
        return sequence_logprobs
    
    def compute_dpo_loss(self, policy_chosen_logp: torch.Tensor,
                        policy_rejected_logp: torch.Tensor,
                        ref_chosen_logp: torch.Tensor,
                        ref_rejected_logp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute DPO loss and rewards."""
        # Log ratios
        policy_log_ratio = policy_chosen_logp - policy_rejected_logp
        ref_log_ratio = ref_chosen_logp - ref_rejected_logp
        
        # Loss
        losses = -F.logsigmoid(self.beta * (policy_log_ratio - ref_log_ratio))
        
        # Rewards for monitoring
        chosen_rewards = self.beta * (policy_chosen_logp - ref_chosen_logp).detach()
        rejected_rewards = self.beta * (policy_rejected_logp - ref_rejected_logp).detach()
        
        return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()
    
    def train_step(self, chosen_tokens: torch.Tensor, rejected_tokens: torch.Tensor):
        """Single training step."""
        self.policy_model.train()
        
        chosen_tokens = chosen_tokens.to(self.device)
        rejected_tokens = rejected_tokens.to(self.device)
        
        # Get log probabilities from policy model
        policy_chosen_logp = self.compute_logprobs(self.policy_model, chosen_tokens)
        policy_rejected_logp = self.compute_logprobs(self.policy_model, rejected_tokens)
        
        # Get log probabilities from reference model (no grad)
        with torch.no_grad():
            ref_chosen_logp = self.compute_logprobs(self.reference_model, chosen_tokens)
            ref_rejected_logp = self.compute_logprobs(self.reference_model, rejected_tokens)
        
        # Compute DPO loss and rewards
        loss, chosen_reward, rejected_reward = self.compute_dpo_loss(
            policy_chosen_logp, policy_rejected_logp,
            ref_chosen_logp, ref_rejected_logp
        )
        
        # Compute accuracy (policy should prefer chosen over rejected)
        accuracy = (policy_chosen_logp > policy_rejected_logp).float().mean().item()
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item(), accuracy, chosen_reward.item(), rejected_reward.item()
    
    def train(self, dataloader, epochs: int = 5):
        """DPO training loop."""
        print(f"\nStarting DPO training for {epochs} epochs...")
        print("=" * 60)
        
        for epoch in range(epochs):
            total_loss = 0
            total_acc = 0
            total_chosen_reward = 0
            total_rejected_reward = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(dataloader):
                loss, acc, chosen_reward, rejected_reward = self.train_step(
                    batch['chosen_tokens'], batch['rejected_tokens']
                )
                
                total_loss += loss
                total_acc += acc
                total_chosen_reward += chosen_reward
                total_rejected_reward += rejected_reward
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx:3d}: loss={loss:.4f}, acc={acc:.4f}")
            
            avg_loss = total_loss / num_batches
            avg_acc = total_acc / num_batches
            avg_chosen_reward = total_chosen_reward / num_batches
            avg_rejected_reward = total_rejected_reward / num_batches
            
            self.history['loss'].append(avg_loss)
            self.history['accuracy'].append(avg_acc)
            self.history['chosen_reward'].append(avg_chosen_reward)
            self.history['rejected_reward'].append(avg_rejected_reward)
            
            print(f"\nEpoch {epoch + 1:3d}/{epochs}")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Accuracy: {avg_acc:.4f}")
            print(f"  Chosen reward: {avg_chosen_reward:.4f}")
            print(f"  Rejected reward: {avg_rejected_reward:.4f}")
            
            # Test on some examples
            if (epoch + 1) % 2 == 0:
                print("\n  Test examples:")
                self.test_examples()
    
    def test_examples(self):
        """Test model on example problems."""
        self.policy_model.eval()
        
        examples = [
            ("12+34=", "46", "47"),  # Correct vs off-by-one
            ("5+7=", "12", "13"),    # Correct vs off-by-one  
            ("99+1=", "100", "101"), # Correct vs off-by-one
        ]
        
        # Get tokenizer from a temporary dataset
        temp_ds = AdditionDataset(max_digits=2, num_samples=1)
        tokenizer = temp_ds.tokenizer
        
        print("")
        for problem, correct, incorrect in examples:
            # Create token sequences
            correct_tokens = tokenizer.encode(problem + correct, add_eos=True)
            incorrect_tokens = tokenizer.encode(problem + incorrect, add_eos=True)
            
            correct_tensor = torch.tensor([correct_tokens], dtype=torch.long).to(self.device)
            incorrect_tensor = torch.tensor([incorrect_tokens], dtype=torch.long).to(self.device)
            
            # Get log probabilities
            with torch.no_grad():
                policy_correct_logp = self.compute_logprobs(self.policy_model, correct_tensor)
                policy_incorrect_logp = self.compute_logprobs(self.policy_model, incorrect_tensor)
                
                ref_correct_logp = self.compute_logprobs(self.reference_model, correct_tensor)
                ref_incorrect_logp = self.compute_logprobs(self.reference_model, incorrect_tensor)
            
            # Compute rewards
            correct_reward = self.beta * (policy_correct_logp - ref_correct_logp)
            incorrect_reward = self.beta * (policy_incorrect_logp - ref_incorrect_logp)
            
            prefers_correct = (policy_correct_logp > policy_incorrect_logp).item()
            
            print(f"    {problem}")
            print(f"      Correct ({correct}): logp={policy_correct_logp.item():.2f}, reward={correct_reward.item():.4f}")
            print(f"      Incorrect ({incorrect}): logp={policy_incorrect_logp.item():.2f}, reward={incorrect_reward.item():.4f}")
            print(f"      Prefers correct: {'✓' if prefers_correct else '✗'}")
            print("")
    
    def save_checkpoint(self, path: str):
        """Save DPO-trained model."""
        torch.save({
            'policy_model_state_dict': self.policy_model.state_dict(),
            'history': self.history,
            'beta': self.beta
        }, path)
        print(f"Saved DPO model to {path}")

def main():
    """Main DPO function."""
    print("=" * 60)
    print("Direct Preference Optimization (DPO)")
    print("=" * 60)
    
    # Default parameters
    model_path = "pretrained_model.pth"
    beta = 0.1
    max_digits = 2
    num_pairs = 500
    batch_size = 8
    epochs = 5
    lr = 1e-5
    output_path = "dpo_model.pth"
    
    print(f"\nUsing parameters:")
    print(f"  Model path: {model_path}")
    print(f"  Beta: {beta}")
    print(f"  Max digits: {max_digits}")
    print(f"  Num pairs: {num_pairs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Output: {output_path}")
    
    change = input("\nPress Enter to continue, or 'c' to change: ").strip().lower()
    if change == 'c':
        model_path = input(f"Model path [{model_path}]: ").strip() or model_path
        beta = float(input(f"Beta [{beta}]: ").strip() or beta)
        max_digits = int(input(f"Max digits [{max_digits}]: ").strip() or max_digits)
        num_pairs = int(input(f"Num pairs [{num_pairs}]: ").strip() or num_pairs)
        batch_size = int(input(f"Batch size [{batch_size}]: ").strip() or batch_size)
        epochs = int(input(f"Epochs [{epochs}]: ").strip() or epochs)
        lr = float(input(f"Learning rate [{lr}]: ").strip() or lr)
        output_path = input(f"Output path [{output_path}]: ").strip() or output_path
    
    # Load base model (two copies: policy and reference)
    print(f"\n1. Loading base model...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    policy_model = AdditionDecoder(
        vocab_size=checkpoint.get('vocab_size', 14),
        embed_dim=checkpoint.get('embed_dim', 96),
        num_layers=checkpoint.get('num_layers', 3),
        num_heads=checkpoint.get('num_heads', 4),
        max_seq_len=checkpoint.get('max_seq_len', 15)
    )
    policy_model.load_state_dict(checkpoint['model_state_dict'])
    
    reference_model = AdditionDecoder(
        vocab_size=checkpoint.get('vocab_size', 14),
        embed_dim=checkpoint.get('embed_dim', 96),
        num_layers=checkpoint.get('num_layers', 3),
        num_heads=checkpoint.get('num_heads', 4),
        max_seq_len=checkpoint.get('max_seq_len', 15)
    )
    reference_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create DPO dataset
    print(f"\n2. Creating DPO dataset...")
    dpo_dataset = DPODataset(max_digits=max_digits, num_pairs=num_pairs)
    
    # Create dataloader with custom collate function
    print(f"\n3. Creating dataloader...")
    dataloader = torch.utils.data.DataLoader(
        dpo_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dpo_dataset.collate_fn
    )
    
    # Create trainer
    print(f"\n4. Creating DPO trainer...")
    trainer = DPOTrainer(policy_model, reference_model, beta=beta, lr=lr)
    
    # Train
    print(f"\n5. Training DPO...")
    trainer.train(dataloader, epochs=epochs)
    
    # Save
    trainer.save_checkpoint(output_path)
    
    print(f"\nDPO training complete!")
    
    # Test the trained model
    print("\n" + "="*60)
    print("Testing DPO-trained model:")
    print("="*60)
    
    # Generate some examples
    policy_model.eval()
    test_problems = ["12+34=", "5+7=", "99+1="]
    temp_ds = AdditionDataset(max_digits=2, num_samples=1)
    
    for problem in test_problems:
        prompt_tokens = temp_ds.encode(problem, add_eos=False)
        prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long)
        
        with torch.no_grad():
            generated = policy_model.generate(
                prompt_tensor,
                max_new_tokens=10,
                temperature=0.8,
                eos_token_id=temp_ds.eos_token_id
            )
        
        generated_tokens = generated[0].tolist()
        generated_text = temp_ds.decode(generated_tokens)
        print(f"  {problem} → {generated_text}")
    
    return trainer

if __name__ == "__main__":
    main()
