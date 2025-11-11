"""GPT-2 model architecture"""
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect

@dataclass
class GPTConfig:
    block_size: int = 1024 #max sequence length
    vocab_size: int = 50304 #50257 #number of tokens: 50000 BPE + 256 Bytes tokens + end of text token
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    cross_attn_every: int = 2               # Add cross-attn every N layers


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 #custom attribute to scale init later
        self.n_head = config.n_head
        self.n_embd = config.n_embd
    
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k  = k.view(B, T, self.n_head, C//self.n_head).transpose(1, 2) # (B,n_head,T,head_size)
        q  = q.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        v  = v.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) #Flash Attention

        y = y.transpose(1, 2).contiguous().view(B,T,C) #concatenate heads
        y = self.c_proj(y)
        return y 

class CrossAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.q_proj = nn.Linear(config.n_embd, config.n_embd) # query from text
        self.kv_proj = nn.Linear(config.n_embd, 2 * config.n_embd) # key and value from image
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)  # output projection
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
    
    def forward(self, x, context):
        B, T, C = x.size() # x is text input
        _, T_c, _ = context.size() # context is image input, T_c is number of image tokens
        
        q = self.q_proj(x)  # queries from text (B, T, n_embd)
        kv = self.kv_proj(context)  # keys and values from image context (B, T_c, 2*n_embd)
        k, v = kv.split(self.n_embd, dim=2)  # Each (B, T_c, n_embd)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.view(B, T_c, self.n_head, head_dim).transpose(1, 2)  # (B, n_head, T_c, head_dim)
        v = v.view(B, T_c, self.n_head, head_dim).transpose(1, 2)  # (B, n_head, T_c, head_dim)

        # Cross-attention (no causal mask needed - can attend to all image tokens)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False) # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        
        return y

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
    
class BlockWithCrossAttention(nn.Module):    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_cross = nn.LayerNorm(config.n_embd)
        self.cross_attn = CrossAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x, context=None):
        x = x + self.attn(self.ln_1(x))
        if context is not None:
            x = x + self.cross_attn(self.ln_cross(x), context)
        x = x + self.mlp(self.ln_2(x))
        return x
    

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        blocks = []
        for i in range(config.n_layer):
            if (i % config.cross_attn_every == 0):
                blocks.append(BlockWithCrossAttention(config))
            else:
                blocks.append(Block(config))

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList(blocks),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer)** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)  
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, image_context=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence is too long, block size is {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        
        for block in self.transformer.h:
            if isinstance(block, BlockWithCrossAttention):
                x = block(x, context=image_context)
            else:
                x = block(x)
                
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss=None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    
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

    def load_pretrained_gpt2(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        pretrained_state = checkpoint['model']
        decoder = self.load_state_dict(pretrained_state, strict=False)
        return decoder
        

