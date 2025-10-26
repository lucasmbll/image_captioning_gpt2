import torch
import torch.nn.functional as F
import tiktoken
import argparse
from gpt2_model import GPT, GPTConfig 

# Hyperparameters (modify these directly)
CHECKPOINT_PATH = "gpt2_checkpoints/latest_checkpoint.pt"
NUM_SAMPLES = 4
MAX_LENGTH = 100
TEMPERATURE = 1.0
TOP_K = 50
SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(checkpoint_path, device='cuda'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model']
    state_dict = {k: v for k, v in state_dict.items() if not k.endswith('.attn.bias')}
    model = GPT(checkpoint['config'])
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"✓ Loaded checkpoint from step {checkpoint['step']}")
    print(f"✓ Validation loss: {checkpoint['val_loss']:.4f}")
    return model

def generate_samples(model, prompt, num_samples=4, max_length=100, temperature=1.0, top_k=50, device='cuda', seed=42):
    """Generate text samples from a prompt"""
    enc = tiktoken.get_encoding("gpt2")
    
    # Encode the prompt
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).repeat(num_samples, 1)
    xgen = tokens.to(device)
    
    # Set up random number generator
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(seed)
    
    print(f"\n{'='*80}")
    print(f"Prompt: {prompt}")
    print(f"{'='*80}\n")
    
    # Generate tokens
    while xgen.size(1) < max_length:
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, _ = model(xgen)
            
            # Get logits for the last token and apply temperature
            logits = logits[:, -1, :] / temperature
            
            # Apply softmax and top-k sampling
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
            
            # Sample from top-k
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
            xcol = torch.gather(topk_indices, -1, ix)
            
            # Append to sequence
            xgen = torch.cat((xgen, xcol), dim=1)
    
    # Decode and print samples
    for i in range(num_samples):
        tokens_list = xgen[i].tolist()
        decoded = enc.decode(tokens_list)
        print(f"Sample {i+1}:")
        print(decoded)
        print(f"{'-'*80}\n")

def interactive_mode(model):
    """Run interactive generation loop"""
    print("\n" + "="*80)
    print("INTERACTIVE GENERATION MODE")
    print("="*80)
    print(f"Settings:")
    print(f"  - Number of samples: {NUM_SAMPLES}")
    print(f"  - Max length: {MAX_LENGTH}")
    print(f"  - Temperature: {TEMPERATURE}")
    print(f"  - Top-k: {TOP_K}")
    print(f"  - Device: {DEVICE}")
    print("="*80)
    print("Enter a prompt and press Enter to generate samples.")
    print("Press Ctrl+C to exit.")
    print("="*80 + "\n")
    
    generation_count = 0
    
    try:
        while True:
            # Get user input
            prompt = input("Enter your prompt: ").strip()
            
            if not prompt:
                print("⚠ Empty prompt, please enter some text.\n")
                continue
            
            # Generate samples
            generate_samples(
                model=model,
                prompt=prompt,
                num_samples=NUM_SAMPLES,
                max_length=MAX_LENGTH,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                device=DEVICE,
                seed=SEED + generation_count  # Different seed each time
            )
            
            generation_count += 1
            
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print(f"Exiting... Generated {generation_count} prompts total.")
        print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Interactive text generation with trained GPT-2 model')
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_PATH, help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, device=DEVICE)
    
    # Start interactive mode
    interactive_mode(model)

if __name__ == "__main__":
    main()