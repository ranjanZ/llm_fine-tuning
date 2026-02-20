#!/usr/bin/env python3
"""
Tokenizer for addition problems.
Handles conversion between text and token IDs.
"""

class Tokenizer:
    """Simple tokenizer for addition problems."""
    
    def __init__(self):
        # Define tokens
        self.tokens = [
            '<PAD>',    # 0
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # 1-10
            '+',        # 11
            '=',        # 12
            '<EOS>'     # 13
        ]
        
        # Create mappings
        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.tokens)}
        
        # Special tokens
        self.pad_token_id = self.token_to_id['<PAD>']
        self.eos_token_id = self.token_to_id['<EOS>']
        self.vocab_size = len(self.tokens)
    
    def encode(self, text: str, add_eos: bool = True) -> list:
        """Convert text to token IDs."""
        tokens = []
        
        for char in text:
            if char in self.token_to_id:
                tokens.append(self.token_to_id[char])
            else:
                # Skip unknown characters
                continue
        
        if add_eos:
            tokens.append(self.eos_token_id)
        
        return tokens
    
    def decode(self, token_ids: list) -> str:
        """Convert token IDs back to text."""
        tokens = []
        for token_id in token_ids:
            if token_id == self.eos_token_id:
                break  # Stop at EOS
            if token_id != self.pad_token_id:
                token = self.id_to_token.get(token_id, '?')
                tokens.append(token)
        return ''.join(tokens)
    
    def tokenize_problem(self, a: int, b: int) -> list:
        """Create token sequence for an addition problem."""
        problem = f"{a}+{b}="
        answer = str(a + b)
        sequence = problem + answer
        return self.encode(sequence, add_eos=True)

def main():
    """Test the tokenizer."""
    tokenizer = Tokenizer()
    
    print("Tokenizer Test")
    print("=" * 50)
    
    # Test encoding
    test_cases = [
        "1+2=3",
        "12+34=46",
        "99+1=100"
    ]
    
    for test in test_cases:
        tokens = tokenizer.encode(test, add_eos=True)
        decoded = tokenizer.decode(tokens)
        print(f"Text: {test}")
        print(f"Tokens: {tokens}")
        print(f"Decoded: {decoded}")
        print()
    
    # Test special case
    print("Special tokens:")
    print(f"PAD token ID: {tokenizer.pad_token_id}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")

if __name__ == "__main__":
    main()


