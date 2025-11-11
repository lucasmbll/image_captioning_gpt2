import torch
import torch.nn as nn
from decoder import GPT, GPTConfig
from encoder import VisionEncoder, VisionEncoderConfig

class captionGPT2(nn.Module):
    def __init__(
        self,
        gpt_config: GPTConfig,
        vision_config: VisionEncoderConfig=None,
        freeze_gpt_base: bool = True
    ):
        super().__init__()
        
        self.gpt_config = gpt_config
        
        if vision_config is None:
            vision_config = VisionEncoderConfig(
                projection_dim=gpt_config.n_embd,  # Match GPT embedding dimension
                freeze_encoder=True
            )
        else:
            # Ensure projection_dim matches GPT n_embd
            assert vision_config.projection_dim == gpt_config.n_embd, \
                f"Vision projection_dim ({vision_config.projection_dim}) must match GPT n_embd ({gpt_config.n_embd})"
        
        self.vision_config = vision_config
        
        self.vision_encoder = VisionEncoder(vision_config)
        
        self.gpt = GPT(gpt_config)
        
        if freeze_gpt_base:
            self._freeze_gpt_base()
        
        # Print model info
        #self._print_model_info()
    
    def _freeze_gpt_base(self):
        for name, param in self.gpt.named_parameters():
            if 'cross_attn' not in name and 'ln_cross' not in name: # keep cross-attention layers trainable
                param.requires_grad = False
    
    def _print_model_info(self):
        """Print model architecture summary"""
        print(f"\n{'='*70}")
        print("captionGPT2 Model Summary")
        print(f"{'='*70}")
        print(f"Vision Encoder: {self.vision_config.model_name}")
        print(f"  - Num patches: {self.vision_encoder.get_num_patches()}")
        print(f"  - Output dim: {self.vision_config.projection_dim}")
        print(f"  - Frozen: {self.vision_config.freeze_encoder}")
        print(f"\nGPT-2 Decoder:")
        print(f"  - Layers: {self.gpt_config.n_layer}")
        print(f"  - Heads: {self.gpt_config.n_head}")
        print(f"  - Embedding: {self.gpt_config.n_embd}")
        print(f"  - Cross-attn every {self.gpt_config.cross_attn_every} layers")
        
        cross_attn_layers = [
            i for i, block in enumerate(self.gpt.transformer.h)
            if hasattr(block, 'cross_attn')
        ]
        print(f"  - Cross-attn layers: {cross_attn_layers}")
        print(f"{'='*70}\n")
    
    def forward(self, pixel_values, input_ids, targets=None):
        image_context = self.vision_encoder(pixel_values) # Encode image to visual features (B, num_patches, projection_dim)
        logits, loss = self.gpt(input_ids, targets=targets, image_context=image_context) # Go through the decoder (GPT2) logits: (B, T, vocab_size)
        return logits, loss
    
    def load_pretrained_gpt2(self, checkpoint_path):
        print("Loading Pretrained GPT-2 Weights")
        decoder = self.gpt.load_pretrained_gpt2(checkpoint_path)
        return decoder
    
    @torch.no_grad()
    def generate(
        self,
        pixel_values,
        tokenizer,
        max_length=50,
        temperature=1.0,
        top_k=50,
        start_token_id=None,
        eos_token_id=50256
    ):
        self.eval()
        
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        
        device = pixel_values.device
        batch_size = pixel_values.size(0)
        
        image_context = self.vision_encoder(pixel_values)
        
        if start_token_id is not None:
            generated = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)
        else:
            generated = torch.zeros((batch_size, 0), dtype=torch.long, device=device)
        
        for _ in range(max_length):
            logits, _ = self.gpt(generated, image_context=image_context)
            logits = logits[:, -1, :] / temperature
            
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            if (next_token == eos_token_id).all():
                break
        
        captions = []
        for seq in generated:
            caption = tokenizer.decode(seq.tolist())
            captions.append(caption)
        
        return captions
    
    def get_trainable_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        vision_trainable = sum(p.numel() for p in self.vision_encoder.parameters() if p.requires_grad)
        gpt_trainable = sum(p.numel() for p in self.gpt.parameters() if p.requires_grad)
        
        print(f"\n{'='*60}")
        print("Trainable Parameters")
        print(f"{'='*60}")
        print(f"Vision Encoder:  {vision_trainable:>12,} trainable")
        print(f"GPT-2 Decoder:   {gpt_trainable:>12,} trainable")
        print(f"{'─'*60}")
        print(f"Total:           {trainable:>12,} / {total:>12,} ({trainable/total*100:.1f}%)")
        print(f"{'='*60}\n")
        
        return {'total': total, 'trainable': trainable}