#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) for the Addition GPT model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import time
from dataset import AdditionDataset
from model import AdditionDecoder

class SFTTrainer:
    """Trainer for Supervised Fine-Tuning."""
    
    def __init__(self, model: AdditionDecoder, dataset: AdditionDataset):
        self.model = model
        self.dataset = dataset
        self.device = torch.device('cpu')
        self.model.to(self.device)
        
        # Optimizer for full fine-tuning
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-5,  # Lower learning rate for fine-tuning
            weight_decay=0.01,
            betas=(0.9, 0.95)
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )
        
        # Loss function (ignore padding)
        self.criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_token_id)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
        """Single training step."""
        self.model.train()
        
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        logits, _ = self.model(inputs, targets)
        
        # Calculate loss (shifted for next token prediction)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = targets[:, 1:].contiguous()
        
        loss = self.criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1)
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Optimizer step
        self.optimizer.step()
        
        # Calculate accuracy
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=-1)
            pred_shift = predictions[:, :-1]
            target_shift = targets[:, 1:]
            
            # Mask out padding
            mask = target_shift != self.dataset.pad_token_id
            correct = (pred_shift == target_shift) & mask
            accuracy = correct.sum().item() / mask.sum().item() if mask.sum().item() > 0 else 0
        
        return loss.item(), accuracy
    
    def train_epoch(self, dataloader, epoch: int) -> Tuple[float, float]:
        """Train for one epoch."""
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            loss, accuracy = self.train_step(inputs, targets)
            
            total_loss += loss
            total_accuracy += accuracy
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx:3d}: loss = {loss:.4f}, acc = {accuracy:.4f}")
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        self.history['train_loss'].append(avg_loss)
        self.history['train_acc'].append(avg_accuracy)
        
        return avg_loss, avg_accuracy
    
    def evaluate(self, dataloader) -> Tuple[float, float]:
        """Evaluate model."""
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                logits, _ = self.model(inputs, targets)
                
                # Calculate loss
                shift_logits = logits[:, :-1, :].contiguous()
                shift_targets = targets[:, 1:].contiguous()
                
                loss = self.criterion(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_targets.view(-1)
                )
                
                # Calculate accuracy
                predictions = torch.argmax(logits, dim=-1)
                pred_shift = predictions[:, :-1]
                target_shift = targets[:, 1:]
                
                mask = target_shift != self.dataset.pad_token_id
                correct = (pred_shift == target_shift) & mask
                accuracy = correct.sum().item() / mask.sum().item() if mask.sum().item() > 0 else 0
                
                total_loss += loss.item()
                total_accuracy += accuracy
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        self.history['val_loss'].append(avg_loss)
        self.history['val_acc'].append(avg_accuracy)
        
        return avg_loss, avg_accuracy
    
    def train(self, train_loader, val_loader, epochs: int = 10):
        """Full fine-tuning loop."""
        print(f"\nStarting SFT for {epochs} epochs...")
        print("=" * 60)
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss, val_acc = self.evaluate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            epoch_time = time.time() - start_time
            
            print(f"\nEpoch {epoch + 1:3d}/{epochs} | Time: {epoch_time:.1f}s")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
    
    def save_checkpoint(self, path: str):
        """Save fine-tuned model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }, path)
        print(f"Fine-tuned model saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load fine-tuned model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']

def create_sft_dataset(num_samples: int = 1000, max_digits: int = 3) -> AdditionDataset:
    """Create a specialized dataset for fine-tuning."""
    # You can modify this to create different types of problems
    # For example: only even numbers, specific ranges, etc.
    return AdditionDataset(max_digits=max_digits, num_samples=num_samples)




def main():
    """Main SFT function with default parameters."""
    print("=" * 60)
    print("Supervised Fine-Tuning (SFT)")
    print("=" * 60)
    
    # Default parameters
    model_path = "model/best_model.pth"  # Change this to your model path
    max_digits = 3
    num_samples = 1000
    batch_size = 16
    epochs = 5
    output_path = "model/sft_model.pth"
    
    print(f"\nUsing parameters:")
    print(f"  Model path: {model_path}")
    print(f"  Max digits: {max_digits}")
    print(f"  Num samples: {num_samples}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Output: {output_path}")
    
    # Ask for user confirmation or changes
    change = input("\nPress Enter to continue with defaults, or 'c' to change: ").strip().lower()
    
    if change == 'c':
        model_path = input(f"Model path [{model_path}]: ").strip() or model_path
        max_digits = int(input(f"Max digits [{max_digits}]: ").strip() or max_digits)
        num_samples = int(input(f"Num samples [{num_samples}]: ").strip() or num_samples)
        batch_size = int(input(f"Batch size [{batch_size}]: ").strip() or batch_size)
        epochs = int(input(f"Epochs [{epochs}]: ").strip() or epochs)
        output_path = input(f"Output path [{output_path}]: ").strip() or output_path
    
    # Load pretrained model
    print(f"\n1. Loading pretrained model from {model_path}...")
    from model import AdditionDecoder
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = AdditionDecoder(
        vocab_size=checkpoint.get('vocab_size', 14),
        embed_dim=checkpoint.get('embed_dim', 96),
        num_layers=checkpoint.get('num_layers', 3),
        num_heads=checkpoint.get('num_heads', 4),
        max_seq_len=checkpoint.get('max_seq_len', 15)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create fine-tuning dataset
    print(f"\n2. Creating fine-tuning dataset...")
    dataset = create_sft_dataset(
        num_samples=num_samples,
        max_digits=max_digits
    )
    
    # Create dataloaders
    print(f"\n3. Creating dataloaders...")
    train_loader, val_loader = dataset.create_dataloader(
        batch_size=batch_size,
        split=0.9
    )
    
    # Create trainer and fine-tune
    print(f"\n4. Starting fine-tuning...")
    trainer = SFTTrainer(model, dataset)
    trainer.train(train_loader, val_loader, epochs=epochs)
    
    # Save fine-tuned model
    trainer.save_checkpoint(output_path)
    
    print(f"\nSFT completed! Model saved to {output_path}")
    
    # Test the fine-tuned model
    print("\n" + "="*60)
    print("Testing fine-tuned model:")
    print("="*60)
    
    # Create a simple test
    model.eval()
    test_problems = ["12+34=", "5+7=", "99+1="]
    
    for problem in test_problems:
        prompt_tokens = dataset.encode(problem, add_eos=False)
        prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long)
        
        with torch.no_grad():
            generated = model.generate(
                prompt_tensor,
                max_new_tokens=10,
                temperature=0.8,
                eos_token_id=dataset.eos_token_id
            )
        
        generated_tokens = generated[0].tolist()
        generated_text = dataset.decode(generated_tokens)
        print(f"  {problem} → {generated_text}")
    
    return trainer

if __name__ == "__main__":
    main()
