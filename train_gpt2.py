# implementation of GPT2 from scratch with PyTorch, inspired by Andrej Karpathy's nanoGPT
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect
import transformers
import os
import numpy as np
from hellaswag import render_example, iterate_examples
from torch.utils.tensorboard import SummaryWriter

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 #custom attribute to scale init later
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k  = k.view(B, T, self.n_head, C//self.n_head).transpose(1, 2) # (B,n_head,T,head_size)
        q  = q.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        v  = v.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)

        #att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        #att = F.softmax(att, dim=-1)
        #y = att @ v

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) #Flash Attention

        y = y.transpose(1, 2).contiguous().view(B,T,C) #concatenate heads
        y = self.c_proj(y)
        return y 
ln -s /Data/lucas.mebille/gpt2_checkpoints checkpoints
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) #pre-norm
        x = x + self.mlp(self.ln_2(x))
        return x 
    
@dataclass
class GPTConfig:
    block_size: int = 1024 #max sequence lenght
    vocab_size: int = 50257 #number of tokens: 50000 BPE + 256 Bytes tokens + end of text token
    n_layer: int = 6
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), #ModuleList can be indexed with integer
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        #weight sharing between token embedding and lm head
        self.transformer.wte.weight = self.lm_head.weight #tie weights: this is a common trick to improve performance and reduce number of parameters
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer)** -0.5 #scale init according to number of residual connections (2 per block)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std) #motivated by 1/sqrt(fan_in = n_embd) 
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)  
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T<= self.config.block_size, f"Sequence is too long, block size is {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # (T) 
        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        tok_emb = self.transformer.wte(idx) # (B, T, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) #(B,T,vocab_size)
        loss=None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

### the pretrained method is from Andrej Karapathy code
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad} #candidate params for decay
        no_decay = [p for n,p in param_dict.items() if p.dim() < 2] #bias and LayerNorm weights 
        decay = [p for n,p in param_dict.items() if p.dim() >= 2] #all other weights (embeddings, matmul)
        optim_groups = [
            {'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay)
        num_no_decay_params = sum(p.numel() for p in no_decay)
        print(f"num decay params: {num_decay_params}, num no decay params: {num_no_decay_params}")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters #new fusion option for AdamW to avoid iteration over param groups in optimizer step
        use_fused = device == 'cuda' and fused_available
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

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
        # advance the position in the tensor
        self.current_position += B * T 
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T
        return x, y
    
def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

#############
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
assert total_batch_size % (B*T) == 0 #total batch size must be multiple of B*T
grad_accum_steps = total_batch_size // (B*T)
print('using grad accumulation steps:', grad_accum_steps) #Only one update every grad_accum_steps batches to simulate larger batch size
# so the new time per update is roughly time per micro batch * grad_accum_steps

#get data_batch
train_loader = DataLoaderLite(B, T, split='train') #for efficient computation: use power of 2 batch sizes
# for token batch size : B*T, for small GPT3 : 0.5 million
val_loader = DataLoaderLite(B, T, split='val')

torch.set_float32_matmul_precision('high')  #for using TF32 on Ampere GPUs => This will impact only matrix multiplications and do not change the storage format of the tensors => need to optimize memory bandwidth separately (with mixed precision training for example)

model = GPT(GPTConfig(vocab_size=50304)) # we increase vocab size to 50304 to optimize for tensor cores (multiple of 256). Avoid ugly number in practice (CUDA kernels often used powers of 2 block tiles  ).
model.to(device) 
use_compile = False 
if use_compile:
    model = torch.compile(model)
# Add compilation time but increase training speed by fusing kernels (use always in production, you can skip it during research phase to have faster debugging iterations)

# Precision => TFLOPS specs, But expect in best case 60% of the theoretical max because of memory bandwidth bottlenecks. Less precision => less memory for storing data => less memory bandwidth bottleneck => closer to theoretical max TFLOPS  
# Default precision in PyTorch is float32 (FP32) => we will use mixed precision training to reduce memory bandwidth bottleneck and increase tokens per second

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

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device) #using AdamW with weight decay

# Log directory
log_dir = "/Data/lucas.mebille/gpt2_training_logs"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

# Checkpoint directory
checkpoint_dir = "/Data/lucas.mebille/gpt2_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # Validation and checkpointing
    if (step % 250 == 0 and step > 0) or last_step: #here only one epoch, but otherwise could have been used for avoiding overfitting and do early stopping
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
            if (step % 5000 == 0 or last_step): # step > 0 and 
                checkpoint_path = os.path.join(checkpoint_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': model.state_dict(),
                    'config': model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                torch.save(checkpoint, checkpoint_path)
    
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
    
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
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
        with torch.autocast(device_type=device, dtype=torch.bfloat16):  #mixed precision for training, bfloat16 is preferred on recent GPUs (Ampere and later) to avoid gradient scaler
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps #normalize loss because of gradient accumulation (check in loss the reduction method, here is mean)
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #gradient norm clipping to avoid exploding gradients
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
