import torch
from captionGPT2 import captionGPT2
from decoder import GPTConfig
from encoder import VisionEncoderConfig
from torchvision import transforms
from PIL import Image
import tiktoken

# Paths
CHECKPOINT_PATH = "caption_checkpoints/latest_checkpoint.pt"  # Path to your trained checkpoint
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model configuration
GPT_CONFIG = GPTConfig(
    block_size=1024,
    vocab_size=50304,
    n_layer=12,
    n_head=12,
    n_embd=768,
    cross_attn_every=2
)

VISION_CONFIG = VisionEncoderConfig(
    model_name="openai/clip-vit-base-patch32",
    projection_dim=768,
    freeze_encoder=True,
    dropout=0.1
)

# Initialize model
model = captionGPT2(
    gpt_config=GPT_CONFIG,
    vision_config=VISION_CONFIG,
    freeze_gpt_base=True
)

# Load checkpoint
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model'])
model.to(DEVICE)
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load and preprocess the image
image_path = "cyclistes.png"  
image = Image.open(image_path).convert("RGB")
pixel_values = transform(image).unsqueeze(0).to(DEVICE)  # Add batch dimension

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Generate captions
captions = model.generate(
    pixel_values=pixel_values,
    tokenizer=tokenizer,
    max_length=75,  # Maximum caption length
    temperature=0.1,
    top_k=10,
    start_token_id=tokenizer._special_tokens['<|endoftext|>'],
    eos_token_id=tokenizer._special_tokens['<|endoftext|>']
)

# Print generated captions
for i, caption in enumerate(captions):
    print(f"Caption {i+1}: {caption}")