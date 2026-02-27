#!/usr/bin/env python3
"""
GPT-style decoder-only transformer model for addition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class AdditionDecoder(nn.Module):
    """
    Pure GPT-style decoder-only transformer for learning addition.
    Uses <EOS> token to properly terminate generation.
    """
    
    def __init__(
        self,
        vocab_size: int = 14,      # 0-9, +, =, <PAD>, <EOS>
        embed_dim: int = 96,
        num_layers: int = 3,
        num_heads: int = 4,
        max_seq_len: int = 15,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Position embeddings
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Transformer decoder blocks
        self.blocks = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer norm
        self.ln_f = nn.LayerNorm(embed_dim)
        
        # Output projection (tied to input embeddings)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # Weight tying
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Calculate total parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"AdditionDecoder Model: {total_params:,} parameters")
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create causal mask for decoder."""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len = x.shape
        device = x.device
        
        # Token embeddings
        token_embeds = self.token_embedding(x)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_embeds = self.position_embedding(positions)
        
        # Combine embeddings
        x = token_embeds + pos_embeds
        
        # Causal mask
        causal_mask = self.create_causal_mask(seq_len).to(device)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, causal_mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language model head
        logits = self.lm_head(x)
        
        #print(DBG)
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=0  # 0 is padding token
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self, 
        prompt: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: int = 13  # <EOS> token
    ) -> torch.Tensor:
        """Generate tokens autoregressively with EOS stopping."""
        self.eval()
        
        generated = prompt.clone()
        eos_generated = torch.zeros(prompt.size(0), dtype=torch.bool, device=prompt.device)
        
        for _ in range(max_new_tokens):
            # Stop if all sequences have generated EOS
            if eos_generated.all():
                break
            
            # Crop sequence if too long
            if generated.shape[1] > self.max_seq_len:
                generated = generated[:, -self.max_seq_len:]
            
            # Get model predictions
            logits, _ = self(generated)
            logits = logits[:, -1, :] / temperature
            
            # Optionally apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Apply softmax
            probs = F.softmax(logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Check for EOS token
            eos_generated = eos_generated | (next_token.squeeze(-1) == eos_token_id)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated

class DecoderBlock(nn.Module):
    """Single decoder block."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.attn(self.ln1(x), mask)
        x = x + self.dropout(attn_out)
        
        # FFN with residual
        ffn_out = self.mlp(self.ln2(x))
        x = x + self.dropout(ffn_out)
        
        return x

class MultiHeadAttention(nn.Module):
    """Multi-head attention."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_scores = attn_scores + mask.unsqueeze(0).unsqueeze(0)
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        return self.out_proj(attn_output)

class MLP(nn.Module):
    """Feed-forward network."""
    
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.fc2 = nn.Linear(4 * embed_dim, embed_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def create_model(
    vocab_size: int = 14,
    embed_dim: int = 96,
    num_layers: int = 3,
    num_heads: int = 4,
    max_seq_len: int = 15,
    dropout: float = 0.1
) -> AdditionDecoder:
    """Factory function to create a model."""
    return AdditionDecoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        dropout=dropout
    )

def main():
    """Test the model."""
    print("Model Test")
    print("=" * 50)
    
    # Create a small model
    model = create_model(
        vocab_size=14,
        embed_dim=32,
        num_layers=1,
        num_heads=2,
        max_seq_len=10
    )
    
    print(f"Model created successfully")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 5
    
    # Create dummy input (avoid padding token 0)
    input_tokens = torch.randint(1, 14, (batch_size, seq_len))
    
    print(f"\nInput shape: {input_tokens.shape}")
    
    # Test without targets
    logits, loss = model(input_tokens)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss (no targets): {loss}")
    
    # Test with targets
    target_tokens = torch.randint(1, 14, (batch_size, seq_len))
    logits, loss = model(input_tokens, target_tokens)
    print(f"Loss (with targets): {loss.item():.4f}")
    
    # Test generation
    print("\nGeneration test:")
    prompt = torch.tensor([[11, 2, 3, 12]], dtype=torch.long)  # "+23="
    generated = model.generate(prompt, max_new_tokens=5)
    print(f"Prompt tokens: {prompt[0].tolist()}")
    print(f"Generated tokens: {generated[0].tolist()}")
    
    return model

if __name__ == "__main__":
    model = main()
