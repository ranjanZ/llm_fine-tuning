#!/usr/bin/env python3
"""
Parameter-Efficient Fine-Tuning (PEFT) with LoRA and DoRA for Addition GPT.
Complete rewrite with proper implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataset import AdditionDataset
from model import AdditionDecoder

class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation."""
    
    def __init__(self, linear_layer: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.linear = linear_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Freeze original weights
        for param in self.linear.parameters():
            param.requires_grad = False
        
        # LoRA weights
        self.lora_A = nn.Parameter(torch.zeros(rank, linear_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear_layer.out_features, rank))
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with LoRA: x @ (W + BA/scaling).T"""
        # Original linear
        base_output = self.linear(x)
        
        # LoRA adaptation
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        
        return base_output + lora_output

class LoRAModel(nn.Module):
    """Complete model with LoRA adapters - no patching."""
    
    def __init__(self, base_model: AdditionDecoder, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.base_model = base_model
        self.rank = rank
        self.alpha = alpha
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Replace linear layers in attention blocks with LoRALinear
        self._replace_with_lora()
        
        print(f"LoRA trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def _replace_with_lora(self):
        """Replace linear layers in attention blocks with LoRA versions."""
        for block in self.base_model.blocks:
            attn = block.attn
            
            # Replace Q, K, V, and output projections
            attn.q_proj = LoRALinear(attn.q_proj, self.rank, self.alpha)
            attn.k_proj = LoRALinear(attn.k_proj, self.rank, self.alpha)
            attn.v_proj = LoRALinear(attn.v_proj, self.rank, self.alpha)
            attn.out_proj = LoRALinear(attn.out_proj, self.rank, self.alpha)
    
    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass - uses base model with LoRA layers."""
        return self.base_model(x, targets)
    
    def generate(self, *args, **kwargs):
        """Generate using base model."""
        return self.base_model.generate(*args, **kwargs)

class DoRALinear(nn.Module):
    """Linear layer with DoRA adaptation."""
    
    def __init__(self, linear_layer: nn.Linear, rank: int = 4):
        super().__init__()
        self.linear = linear_layer
        self.rank = rank
        
        # Freeze original weights
        for param in self.linear.parameters():
            param.requires_grad = False
        
        # Magnitude vector
        self.magnitude = nn.Parameter(torch.ones(linear_layer.out_features))
        
        # LoRA components for direction
        self.lora_A = nn.Parameter(torch.zeros(rank, linear_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear_layer.out_features, rank))
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with DoRA: decomposed weight adaptation."""
        # Get adapted weight: W + BA
        base_weight = self.linear.weight
        adapted_weight = base_weight + self.lora_B @ self.lora_A
        
        # Decompose: direction * magnitude
        direction = adapted_weight / (torch.norm(adapted_weight, dim=1, keepdim=True) + 1e-8)
        final_weight = direction * self.magnitude.unsqueeze(1)
        
        # Apply bias if exists
        if self.linear.bias is not None:
            return F.linear(x, final_weight, self.linear.bias)
        return F.linear(x, final_weight)

class DoRAModel(nn.Module):
    """Complete model with DoRA adapters."""
    
    def __init__(self, base_model: AdditionDecoder, rank: int = 4):
        super().__init__()
        self.base_model = base_model
        self.rank = rank
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Replace linear layers in attention blocks with DoRALinear
        self._replace_with_dora()
        
        print(f"DoRA trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def _replace_with_dora(self):
        """Replace linear layers in attention blocks with DoRA versions."""
        for block in self.base_model.blocks:
            attn = block.attn
            
            # Replace Q, K, V, and output projections
            attn.q_proj = DoRALinear(attn.q_proj, self.rank)
            attn.k_proj = DoRALinear(attn.k_proj, self.rank)
            attn.v_proj = DoRALinear(attn.v_proj, self.rank)
            attn.out_proj = DoRALinear(attn.out_proj, self.rank)
    
    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass - uses base model with DoRA layers."""
        return self.base_model(x, targets)
    
    def generate(self, *args, **kwargs):
        """Generate using base model."""
        return self.base_model.generate(*args, **kwargs)

class PEFTTrainer:
    """Trainer for PEFT methods."""
    
    def __init__(self, model: nn.Module, dataset: AdditionDataset, method: str = 'lora'):
        self.model = model
        self.dataset = dataset
        self.method = method
        self.device = torch.device('cpu')
        self.model.to(self.device)
        
        # Only train parameters that require grad
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=1e-3,
            weight_decay=0.01
        )
        
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
    
    def train_epoch(self, dataloader, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            logits, loss = self.model(inputs, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Calculate accuracy
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=-1)
                pred_shift = predictions[:, :-1]
                target_shift = targets[:, 1:]
                mask = target_shift != self.dataset.pad_token_id
                correct = (pred_shift == target_shift) & mask
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx:3d}: loss = {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        
        return avg_loss, accuracy
    
    def evaluate(self, dataloader):
        """Evaluate model."""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                logits, loss = self.model(inputs, targets)
                total_loss += loss.item()
                
                predictions = torch.argmax(logits, dim=-1)
                pred_shift = predictions[:, :-1]
                target_shift = targets[:, 1:]
                mask = target_shift != self.dataset.pad_token_id
                correct = (pred_shift == target_shift) & mask
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs: int = 5):
        """Train PEFT adapters."""
        print(f"\nTraining {self.method.upper()} for {epochs} epochs...")
        print("=" * 60)
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, val_acc = self.evaluate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"\nEpoch {epoch + 1:3d}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            
            # Test a few examples
            if (epoch + 1) % 2 == 0:
                print("\n  Test examples:")
                self.test_examples()
    
    def test_examples(self):
        """Test model on example problems."""
        self.model.eval()
        examples = ["12+34=", "5+7=", "99+1="]
        
        for example in examples:
            prompt_tokens = self.dataset.encode(example, add_eos=False)
            prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                generated = self.model.generate(
                    prompt_tensor,
                    max_new_tokens=10,
                    temperature=0.8,
                    eos_token_id=self.dataset.eos_token_id
                )
            
            generated_tokens = generated[0].tolist()
            generated_text = self.dataset.decode(generated_tokens)
            print(f"    {example} → {generated_text}")
    
    def save_checkpoint(self, path: str):
        """Save PEFT adapters."""
        # Save only the adapter weights
        adapter_weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                adapter_weights[name] = param
        
        torch.save({
            'method': self.method,
            'adapter_weights': adapter_weights,
            'history': self.history
        }, path)
        print(f"Saved {self.method.upper()} adapters to {path}")

def main():
    """Main PEFT function."""
    print("=" * 60)
    print("Parameter-Efficient Fine-Tuning (PEFT)")
    print("=" * 60)
    
    # Default parameters
    model_path = "pretrained_model.pth"
    method = "lora"
    rank = 4
    alpha = 1.0
    max_digits = 2
    num_samples = 500
    batch_size = 16
    epochs = 5
    output_path = "peft_model.pth"
    
    print(f"\nUsing parameters:")
    print(f"  Model path: {model_path}")
    print(f"  Method: {method}")
    print(f"  Rank: {rank}")
    print(f"  Alpha: {alpha}")
    print(f"  Max digits: {max_digits}")
    print(f"  Num samples: {num_samples}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Output: {output_path}")
    
    change = input("\nPress Enter to continue, or 'c' to change: ").strip().lower()
    if change == 'c':
        model_path = input(f"Model path [{model_path}]: ").strip() or model_path
        method = input(f"Method (lora/dora) [{method}]: ").strip().lower() or method
        rank = int(input(f"Rank [{rank}]: ").strip() or rank)
        alpha = float(input(f"Alpha [{alpha}]: ").strip() or alpha)
        max_digits = int(input(f"Max digits [{max_digits}]: ").strip() or max_digits)
        num_samples = int(input(f"Num samples [{num_samples}]: ").strip() or num_samples)
        batch_size = int(input(f"Batch size [{batch_size}]: ").strip() or batch_size)
        epochs = int(input(f"Epochs [{epochs}]: ").strip() or epochs)
        output_path = input(f"Output path [{output_path}]: ").strip() or output_path
    
    # Load base model
    print(f"\n1. Loading base model...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    base_model = AdditionDecoder(
        vocab_size=checkpoint.get('vocab_size', 14),
        embed_dim=checkpoint.get('embed_dim', 96),
        num_layers=checkpoint.get('num_layers', 3),
        num_heads=checkpoint.get('num_heads', 4),
        max_seq_len=checkpoint.get('max_seq_len', 15)
    )
    base_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create PEFT model
    print(f"\n2. Creating {method.upper()} model...")
    if method == 'lora':
        model = LoRAModel(base_model, rank=rank, alpha=alpha)
    else:
        model = DoRAModel(base_model, rank=rank)
    
    # Create dataset
    print(f"\n3. Creating dataset...")
    dataset = AdditionDataset(max_digits=max_digits, num_samples=num_samples)
    
    # Create dataloaders
    print(f"\n4. Creating dataloaders...")
    train_loader, val_loader = dataset.create_dataloader(
        batch_size=batch_size,
        split=0.9
    )
    
    # Train
    print(f"\n5. Training...")
    trainer = PEFTTrainer(model, dataset, method=method)
    trainer.train(train_loader, val_loader, epochs=epochs)
    
    # Save
    trainer.save_checkpoint(output_path)
    
    # Test final model
    print("\n" + "="*60)
    print("Final model test:")
    print("="*60)
    
    model.eval()
    test_problems = ["12+34=", "5+7=", "99+1=", "23+77=", "8+92="]
    
    correct = 0
    total = 0
    
    for problem in test_problems:
        prompt_tokens = dataset.encode(problem, add_eos=False)
        prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long).to(trainer.device)
        
        with torch.no_grad():
            generated = model.generate(
                prompt_tensor,
                max_new_tokens=10,
                temperature=0.8,
                eos_token_id=dataset.eos_token_id
            )
        
        generated_tokens = generated[0].tolist()
        generated_text = dataset.decode(generated_tokens)
        
        # Calculate expected answer
        parts = problem.rstrip('=').split('+')
        if len(parts) == 2:
            a, b = int(parts[0]), int(parts[1])
            expected = str(a + b)
            
            is_correct = generated_text == expected
            if is_correct:
                correct += 1
                symbol = "✓"
            else:
                symbol = "✗"
            
            total += 1
            print(f"  {problem} → {generated_text} (expected: {expected}) {symbol}")
        else:
            print(f"  {problem} → {generated_text}")
    
    if total > 0:
        accuracy = correct / total * 100
        print(f"\n  Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    
    print(f"\nPEFT training complete!")
    return trainer

if __name__ == "__main__":
    main()
