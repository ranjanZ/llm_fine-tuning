#!/usr/bin/env python3
"""
Dataset generation for addition problems.
"""

import torch
import torch.nn.functional as F
import random
from typing import List, Tuple
from tokenizer import Tokenizer

class AdditionDataset:
    """Dataset for addition problems with proper tokenization."""
    
    def __init__(self, max_digits: int = 2, num_samples: int = 10000):
        self.max_digits = max_digits
        self.num_samples = num_samples
        
        # Initialize tokenizer
        self.tokenizer = Tokenizer()
        
        # Generate data
        self.data = self._generate_dataset()
        
        print(f"Dataset created: {len(self.data)} samples")
        print(f"Vocabulary size: {self.tokenizer.vocab_size}")
        print(f"Max digits: {max_digits}")
    
    def _generate_dataset(self) -> List[List[int]]:
        """Generate addition problems."""
        data = []
        
        for _ in range(self.num_samples):
            # Generate two numbers
            a = random.randint(1, 10**self.max_digits - 1)
            b = random.randint(1, 10**self.max_digits - 1)
            
            # Get tokens using tokenizer
            tokens = self.tokenizer.tokenize_problem(a, b)
            data.append(tokens)
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens = self.data[idx]
        input_tokens = torch.tensor(tokens, dtype=torch.long)
        target_tokens = torch.tensor(tokens, dtype=torch.long)
        return input_tokens, target_tokens
    
    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size
    
    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id
    
    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id
    
    def encode(self, text: str, add_eos: bool = True) -> List[int]:
        """Convert text to tokens."""
        return self.tokenizer.encode(text, add_eos)
    
    def decode(self, tokens: List[int]) -> str:
        """Convert tokens to string."""
        return self.tokenizer.decode(tokens)
    
    def generate_random_problem(self) -> Tuple[str, str, str]:
        """Generate a random problem for testing."""
        a = random.randint(1, 10**self.max_digits - 1)
        b = random.randint(1, 10**self.max_digits - 1)
        
        problem = f"{a}+{b}="
        expected = str(a + b)
        full_problem = f"{problem}{expected}"
        
        return problem, expected, full_problem
    
    def create_dataloader(self, batch_size: int = 32, split: float = 0.8):
        """Create train/val dataloaders."""
        split_idx = int(len(self) * split)
        
        train_data = self.data[:split_idx]
        val_data = self.data[split_idx:]
        
        train_dataset = SubsetDataset(train_data, self)
        val_dataset = SubsetDataset(val_data, self)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )
        
        return train_loader, val_loader
    
    def collate_fn(self, batch):
        """Pad sequences to same length."""
        inputs, targets = zip(*batch)
        
        max_len = max(len(x) for x in inputs)
        
        padded_inputs = []
        padded_targets = []
        
        for inp, tgt in zip(inputs, targets):
            pad_len = max_len - len(inp)
            
            padded_inp = F.pad(inp, (0, pad_len), value=self.pad_token_id)
            padded_tgt = F.pad(tgt, (0, pad_len), value=self.pad_token_id)
            
            padded_inputs.append(padded_inp)
            padded_targets.append(padded_tgt)
        
        return (
            torch.stack(padded_inputs),
            torch.stack(padded_targets)
        )

class SubsetDataset(torch.utils.data.Dataset):
    """Wrapper for dataset subset."""
    def __init__(self, data, parent_dataset):
        self.data = data
        self.parent = parent_dataset
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens = self.data[idx]
        input_tokens = torch.tensor(tokens, dtype=torch.long)
        target_tokens = torch.tensor(tokens, dtype=torch.long)
        return input_tokens, target_tokens

def main():
    """Test the dataset."""
    print("Dataset Test")
    print("=" * 50)
    
    # Create a small dataset
    dataset = AdditionDataset(max_digits=1, num_samples=10)
    
    # Show dataset info
    print(f"Dataset size: {len(dataset)}")
    print(f"Vocabulary size: {dataset.vocab_size}")
    
    # Show some examples
    print("\nSample sequences:")
    for i in range(3):
        input_tokens, target_tokens = dataset[i]
        text = dataset.decode(input_tokens.tolist())
        print(f"  Example {i+1}:")
        print(f"    Input tokens: {input_tokens.tolist()}")
        print(f"    Text: {text}")
    
    # Test random problem generation
    print("\nRandom problem test:")
    problem, expected, full = dataset.generate_random_problem()
    print(f"  Problem: {problem}")
    print(f"  Expected: {expected}")
    print(f"  Full: {full}")
    
    # Test encoding/decoding
    print("\nEncoding/Decoding test:")
    test_text = "12+34=46"
    tokens = dataset.encode(test_text, add_eos=True)
    decoded = dataset.decode(tokens)
    print(f"  Text: {test_text}")
    print(f"  Tokens: {tokens}")
    print(f"  Decoded: {decoded}")

if __name__ == "__main__":
    main()



