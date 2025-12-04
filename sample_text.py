import torch
import torch.nn.functional as F
import tiktoken
import argparse
from old_code.gpt2_model import GPT, GPTConfig 

# Hyperparameters (modify these directly)
CHECKPOINT_PATH = "gpt2_checkpoints/model_19073.pt"
NUM_SAMPLES = 4
MAX_LENGTH = 100
TEMPERATURE = 1.0
TOP_K = 10
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
    print(f"Loaded checkpoint from step {checkpoint['step']}")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}")
    return model

def generate_samples(model, prompt, num_samples=4, max_length=100, temperature=1.0, top_k=50, device='cuda', seed=42):
    enc = tiktoken.get_encoding("gpt2")
    
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).repeat(num_samples, 1)
    xgen = tokens.to(device)
    
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(seed)

    # EOT token id (allow special)
    eot_token = enc.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]
    finished = torch.zeros(num_samples, dtype=torch.bool, device=device)

    print(f"Prompt: {prompt}")
    
    # Generate tokens
    while xgen.size(1) < max_length and not torch.all(finished):
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, _ = model(xgen)

            # last-token logits
            logits = logits[:, -1, :] / max(1e-6, temperature)

            # mask out EOT for unfinished sampling? (optional)
            # If you want to allow natural early stop, don't penalize EOT here.

            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, k=top_k, dim=-1)

            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
            xcol = torch.gather(topk_indices, -1, ix)  # shape [B, 1]

            # For sequences already finished, force EOT token to keep length consistent
            xcol[finished] = eot_token

            # Append next token
            xgen = torch.cat((xgen, xcol), dim=1)

            # Update finished mask (newly finished if just emitted EOT)
            newly_finished = (xcol.squeeze(1) == eot_token)
            finished = finished | newly_finished
    
    # Decode (strip any trailing EOT)
    for i in range(num_samples):
        tokens_list = xgen[i].tolist()
        decoded = enc.decode(tokens_list).replace('<|endoftext|>', '').strip()
        print(f"\n Sample {i+1}:\n")
        print(decoded)

def interactive_mode(model):
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
            prompt = input("Enter your prompt: \n").strip()
            
            if not prompt:
                print("Empty prompt, please enter some text.\n")
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
    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, device=DEVICE)
    
    interactive_mode(model)

if __name__ == "__main__":
    main()