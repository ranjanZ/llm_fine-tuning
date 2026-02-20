#!/usr/bin/env python3
"""
Training script for the addition GPT model.
"""

import torch
import time
from typing import Tuple
from dataset import AdditionDataset
from model import AdditionDecoder

class GPTTrainer:
    """Trainer for GPT-style addition model with EOS."""
    
    def __init__(self, model: AdditionDecoder, dataset: AdditionDataset):
        self.model = model
        self.dataset = dataset
        self.device = torch.device('cpu')
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=3e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95)
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-5
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        print(f"Training on: {self.device}")
    
    def train_epoch(self, dataloader, epoch: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_tokens = 0
        total_correct = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, loss = self.model(inputs, targets)
            print(DBG)

            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Calculate accuracy
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=-1)
                
                # Shift for next-token prediction
                pred_shift = predictions[:, :-1]
                target_shift = targets[:, 1:]
                
                # Mask out padding
                mask = target_shift != self.dataset.pad_token_id
                correct = (pred_shift == target_shift) & mask
                
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0 and len(dataloader) > 50:
                print(f"  Batch {batch_idx:3d}: loss = {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        
        self.history['train_loss'].append(avg_loss)
        self.history['train_acc'].append(accuracy)
        
        return avg_loss, accuracy
    
    def evaluate(self, dataloader) -> Tuple[float, float]:
        """Evaluate model."""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        total_correct = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                logits, loss = self.model(inputs, targets)
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = torch.argmax(logits, dim=-1)
                pred_shift = predictions[:, :-1]
                target_shift = targets[:, 1:]
                
                mask = target_shift != self.dataset.pad_token_id
                correct = (pred_shift == target_shift) & mask
                
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        
        self.history['val_loss'].append(avg_loss)
        self.history['val_acc'].append(accuracy)
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs: int = 20):
        """Full training loop."""
        print(f"\nTraining for {epochs} epochs...")
        print("=" * 60)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss, val_acc = self.evaluate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Time tracking
            epoch_time = time.time() - start_time
            
            # Print progress
            print(f"\nEpoch {epoch + 1:3d}/{epochs} | Time: {epoch_time:.1f}s")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint("model/best_model.pth")
                print(f"  ✓ Saved best model")
            
            # Generate examples every few epochs
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print("\n  Example generations:")
                self.generate_examples(num_examples=2)
                print()
    
    def generate_examples(self, num_examples: int = 2):
        """Generate example predictions."""
        self.model.eval()
        
        correct_count = 0
        
        for i in range(num_examples):
            # Generate random problem
            problem, expected, full_problem = self.dataset.generate_random_problem()
            
            # Encode problem (without answer, just the prompt)
            prompt = f"{problem}"
            prompt_tokens = self.dataset.encode(prompt, add_eos=False)
            prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long).to(self.device)
            
            # Generate
            with torch.no_grad():
                generated = self.model.generate(
                    prompt_tensor,
                    max_new_tokens=10,
                    temperature=0.8,
                    eos_token_id=self.dataset.eos_token_id
                )
            
            # Decode
            generated_tokens = generated[0].tolist()
            generated_text = self.dataset.decode(generated_tokens)
            
            print(f"    Example {i+1}:")
            print(f"      Problem:    {problem}")
            print(f"      Expected:   {expected}")
            print(f"      Generated:  {generated_text}")
            
            # Check if correct
            if generated_text == expected:
                print(f"      ✓ CORRECT")
                correct_count += 1
            else:
                print(f"      ✗ WRONG")
        
        print(f"\n    Accuracy: {correct_count}/{num_examples}")
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'vocab_size': self.model.vocab_size,
            'embed_dim': self.model.embed_dim,
            'num_layers': len(self.model.blocks),
            'num_heads': self.model.blocks[0].attn.num_heads if len(self.model.blocks) > 0 else 4
        }, path)
        print(f"  Model saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']

def main():
    """Main training function."""
    print("Training Script")
    print("=" * 50)
    
    # Import here to avoid circular imports
    from dataset import AdditionDataset
    from model import create_model
    
    # Configuration
    config = {
        'max_digits': 2,
        'num_samples': 5000,
        'batch_size': 32,
        'epochs': 10,
        'embed_dim': 96,
        'num_layers': 2,
        'num_heads': 4,
        'max_seq_len': 15
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create dataset
    print("\n1. Creating dataset...")
    dataset = AdditionDataset(
        max_digits=config['max_digits'],
        num_samples=config['num_samples']
    )
    
    # Create model
    print("\n2. Creating model...")
    model = create_model(
        vocab_size=dataset.vocab_size,
        embed_dim=config['embed_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        max_seq_len=config['max_seq_len']
    )
    
    # Create dataloaders
    print("\n3. Creating dataloaders...")
    train_loader, val_loader = dataset.create_dataloader(
        batch_size=config['batch_size'],
        split=0.8
    )
    
    # Create trainer
    print("\n4. Creating trainer...")
    trainer = GPTTrainer(model, dataset)
    
    # Train
    print("\n5. Starting training...")
    trainer.train(train_loader, val_loader, epochs=config['epochs'])
    
    print("\nTraining completed!")
    return trainer

if __name__ == "__main__":
    trainer = main()
