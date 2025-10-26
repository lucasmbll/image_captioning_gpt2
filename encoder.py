"""Vision encoder for image captioning"""
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor

@dataclass
class VisionEncoderConfig:
    """Configuration for vision encoder"""
    model_name: str = "openai/clip-vit-base-patch32"
    projection_dim: int = 768  # Should match GPT n_embd
    freeze_encoder: bool = True
    dropout: float = 0.1
    projection_std: float = 0.02


class VisionEncoder(nn.Module):
    
    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        self.config = config
        self.model_name = config.model_name
        self.projection_dim = config.projection_dim

        self.vision_model = CLIPVisionModel.from_pretrained(config.model_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(config.model_name)

        # Get vision hidden size
        self.vision_hidden_size = self.vision_model.config.hidden_size
        
        # Freeze encoder if specified
        if config.freeze_encoder:
            for param in self.vision_model.parameters():
                param.requires_grad = False
            self.vision_model.eval()
            print(f"✓ Vision encoder frozen")
        
        # Projection layer to match GPT-2 embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(self.vision_hidden_size, config.projection_dim),
            nn.LayerNorm(config.projection_dim),
            nn.Dropout(config.dropout)
        )
        # Initialize projection
        nn.init.normal_(self.projection[0].weight, std=config.projection_std)
        nn.init.zeros_(self.projection[0].bias)
        
        print(f"✓ VisionEncoder initialized")
        print(f"  Model: {config.model_name}")
        print(f"  Hidden size: {self.vision_hidden_size}")
        print(f"  Output dim: {config.projection_dim}")
    
    def forward(self, pixel_values):
        # Extract features from vision model
        with torch.set_grad_enabled(self.vision_model.training):
            outputs = self.vision_model(
                pixel_values=pixel_values, #(B, 3, H, W)
                output_hidden_states=True,
                return_dict=True
            )
            
            # Use last hidden state (all patch embeddings)
            # For CLIP ViT-B/32: (B, 50, 768) - 49 patches + 1 CLS token
            image_features = outputs.last_hidden_state #(B, num_patches, projection_dim)
        
        # Project to GPT-2 embedding space
        image_features = self.projection(image_features)
        
        return image_features
    
    def preprocess_images(self, images):
        """
        Preprocess PIL images or tensors for the vision model
        
        Args:
            images: list of PIL Images or tensor (B, 3, H, W)
        
        Returns:
            pixel_values: (B, 3, 224, 224) preprocessed tensor
        """
        if isinstance(images, torch.Tensor):
            return images #(B, 3, H, W)

        # Use image processor
        processed = self.image_processor(images=images, return_tensors="pt")
        return processed['pixel_values']
    
    def get_num_patches(self):
        """Get number of visual tokens (patches) produced by encoder"""
        # For CLIP ViT: depends on patch size and image resolution
        # CLIP ViT-B/32: 224x224 image, 32x32 patches = 7x7 = 49 patches + 1 CLS = 50 tokens
        if 'clip' in self.model_name.lower():
            config = self.vision_model.config
            image_size = config.image_size
            patch_size = config.patch_size
            num_patches = (image_size // patch_size) ** 2 + 1  # +1 for CLS token
            return num_patches
        return None

