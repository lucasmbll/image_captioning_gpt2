import torch
from torch.nn import functional as F
import math
import os
import numpy as np
from hellaswag import render_example, iterate_examples
from torch.utils.tensorboard import SummaryWriter

from gpt2_model import GPT, GPTConfig

##### Dataloader #####
import tiktoken

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, split):
        self.B = B
        self.T = T
        assert split in {'train', 'val'}
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"

        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        self.current_position += B * T 
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T
        return x, y

### Utility Functions ###
def get_most_likely_row(tokens, mask, logits):
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    shift_mask = (mask[..., 1:]).contiguous() 
    masked_shift_losses = shift_losses * shift_mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    pred_norm = avg_loss.argmin().item()
    return pred_norm

### Training setup ###
import time 

device = 'cpu'
torch.manual_seed(420) #seed for model init
if torch.cuda.is_available():
    device = "cuda"
    torch.cuda.manual_seed(420)
print(f"using device: {device}")

enc = tiktoken.get_encoding("gpt2")

total_batch_size = 524288 #2**19 tokens => GPT3 small 
B = 8 #micro batch size
T = 1024
assert total_batch_size % (B*T) == 0 
grad_accum_steps = total_batch_size // (B*T)
print('using grad accumulation steps:', grad_accum_steps) 

train_loader = DataLoaderLite(B, T, split='train') 
val_loader = DataLoaderLite(B, T, split='val')

torch.set_float32_matmul_precision('high')  #for using TF32 on Ampere GPUs 

model = GPT(GPTConfig(vocab_size=50304)) 
model.to(device) 
use_compile = False 
if use_compile:
    model = torch.compile(model)

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715 #GPT3 paper
num_epochs = 1
max_steps = 19073 * num_epochs #10B tokens / total_batch_size

def get_lr(step): #lr schedule with linear warmup and cosine decay
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0.0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) #cosine decay
    return min_lr + coeff * (max_lr - min_lr)

def load_checkpoint(checkpoint_path, model, optimizer):
    """Load checkpoint and return the step to resume from"""
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_step = checkpoint['step'] + 1  # Resume from next step
        print(f"Resuming from step {start_step}")
        return start_step
    else:
        print("No checkpoint found. Starting training from scratch.")
        return 0

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device) #using AdamW with weight decay

# Log directory
log_dir = "/Data/lucas.mebille/gpt2_training_logs"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

# Checkpoint directory
checkpoint_dir = "/Data/lucas.mebille/gpt2_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

resume_checkpoint = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
start_step = load_checkpoint(resume_checkpoint, model, optimizer)

### Training Loop ###
for step in range(start_step, max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # Validation and checkpointing
    if (step % 250 == 0) or last_step: 
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x = x.to(device)
                y = y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):  #mixed precision for validation
                    logits, val_loss = model(x, y)
                val_loss = val_loss / val_loss_steps
                val_loss_accum += val_loss.detach()
            print(f"step {step}: val loss {val_loss_accum.item():.4f}")
            writer.add_scalar("Loss/val", val_loss_accum.item(), step)
            if (step % 2500 == 0 or last_step):
                checkpoint_path = os.path.join(checkpoint_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': model.state_dict(),
                    'config': model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item(),
                    'optimizer': optimizer.state_dict() 
                }
                torch.save(checkpoint, checkpoint_path)
            # Save latest checkpoint for resuming training
            latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
            checkpoint = {
                'model': model.state_dict(),
                'config': model.config,
                'step': step,
                'val_loss': val_loss_accum.item(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(checkpoint, latest_checkpoint_path)
    
    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        acc_norm = num_correct_norm / num_total
        print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
        writer.add_scalar("HellaSwag Accuracy", acc_norm, step)
    
    if ((step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + step) #different seed at each generation step
        while xgen.size(1) < max_length:
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                logits = logits[:, -1, :] # (B, vocab_size)
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                xgen = torch.cat((xgen, xcol), dim=1)
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"Sample {i}: {decoded}")

    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x = x.to(device)
        y = y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):  #mixed precision for training
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps #normalize loss because of gradient accumulation 
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #gradient clipping to avoid exploding gradients
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize() 
    t1 = time.time()
    d = (t1 - t0)
    tokens_per_sec = train_loader.B * train_loader.T * grad_accum_steps / d
    writer.add_scalar("Loss/train", loss_accum.item(), step)
    writer.add_scalar("Learning Rate", lr, step)
    writer.add_scalar("Gradient Norm", norm, step)
    writer.add_scalar("Tokens Per Second", tokens_per_sec, step)
    print(f"step {step+1}, lr: {lr:.5e}, loss: {loss_accum.item():.4f}, norm: {norm:.4f}, time/batch: {d:.2f}s, tokens/sec: {tokens_per_sec:.2f}")

writer.close()
