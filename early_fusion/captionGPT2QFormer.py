import torch
import torch.nn as nn
from early_fusion.decoder_qformer import GPT, GPTConfig, Block
from early_fusion.encoder_qformer import VisionEncoder, VisionEncoderConfig
from early_fusion.qformer import QFormer, QFormerConfig

class CaptionGPT2QFormer(nn.Module):
    def __init__(self, gpt_cfg: GPTConfig, vision_cfg: VisionEncoderConfig, qformer_cfg: QFormerConfig):
        super().__init__()
        self.gpt_config = gpt_cfg
        self.gpt = GPT(gpt_cfg)
        self.vision_config = vision_cfg
        self.vision_encoder = VisionEncoder(vision_cfg)
        self.qformer_config = qformer_cfg
        self.qformer = QFormer(qformer_cfg, out_dim=gpt_cfg.n_embd)

    def _build_prefix(self, pixel_values=None, image_features=None):
        if image_features is not None:
            # image_features: (B, 50, 768) raw CLIP features
            return self.qformer(image_features.to(self.qformer.to_gpt.weight.device))
        with torch.no_grad():
            vi_raw = self.vision_encoder(pixel_values)
        return self.qformer(vi_raw)

    def forward(self, pixel_values=None, input_ids=None, targets=None, image_features=None):
        prefix = self._build_prefix(pixel_values=pixel_values, image_features=image_features)
        logits, loss = self.gpt(input_ids, targets=targets, prefix_emb=prefix)
        return logits, loss, prefix

    @torch.no_grad()
    def generate(self, pixel_values=None, tokenizer=None, max_length=50, temperature=1.0,
                 top_k=50, start_token_id=50256, eos_token_id=50256, prefix=None, image_features=None):
        if prefix is None:
            if image_features is not None:
                prefix = self._build_prefix(image_features=image_features)
            else:
                if pixel_values is not None and pixel_values.dim() == 3:
                    pixel_values = pixel_values.unsqueeze(0)
                prefix = self._build_prefix(pixel_values=pixel_values)
        gen = torch.full((prefix.size(0), 1), start_token_id, dtype=torch.long, device=prefix.device)
        finished = torch.zeros(prefix.size(0), dtype=torch.bool, device=prefix.device)
        for _ in range(max_length):
            logits, _ = self.gpt(gen, prefix_emb=prefix)  # returns (B,T,V)
            next_logits = logits[:, -1, :] / temperature
            if top_k > 0:
                cutoff = torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[next_logits < cutoff] = float('-inf')
            next_tok = torch.multinomial(torch.softmax(next_logits, dim=-1), 1)
            next_tok[finished] = eos_token_id
            gen = torch.cat([gen, next_tok], dim=1)
            finished |= (next_tok.squeeze(-1) == eos_token_id)
            if finished.all():
                break

        decoded = []
        for seq in gen:
            toks = seq.tolist()
            if toks and toks[0] == start_token_id:
                toks = toks[1:]
            decoded.append(tokenizer.decode(toks, errors='replace'))
        return decoded
    

    
    def load_pretrained_gpt2(self, checkpoint_path):
        print("Loading Pretrained GPT-2 Weights")
        decoder = self.gpt.load_pretrained_gpt2(checkpoint_path)
        return decoder