import os
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tiktoken

from captionGPT2 import captionGPT2
from decoder import GPTConfig
from encoder import VisionEncoderConfig
from vision_dataset import COCOCaptionDataset, collate_fn


# Configuration 

COCO_DATA_DIR = "COCO" 
CHECKPOINT_DIR = "caption_checkpoints"
LOG_DIR = "caption_logs"
GPT2_CHECKPOINT = "gpt2_checkpoints/old_check/model_19073.pt"

BATCH_SIZE = 32
NUM_EPOCHS = 1
MAX_CAPTION_LENGTH = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
NUM_WORKERS = 4

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

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

#os.makedirs(CHECKPOINT_DIR, exist_ok=True)
#os.makedirs(LOG_DIR, exist_ok=True)

torch.manual_seed(420)
if torch.cuda.is_available():
    torch.cuda.manual_seed(420)

torch.set_float32_matmul_precision('high')

# Initialization

print("Initializing captionGPT2 model")

model = captionGPT2(
    gpt_config=GPT_CONFIG,
    vision_config=VISION_CONFIG,
    freeze_gpt_base=True
)

# Load pretrained GPT-2 weights
if os.path.exists(GPT2_CHECKPOINT):
    model.load_pretrained_gpt2(GPT2_CHECKPOINT)
else:
    raise FileNotFoundError(f"GPT-2 checkpoint not found at {GPT2_CHECKPOINT}. Aborting.")

model.to(device)
model.get_trainable_params()

# Dataset

print("Loading COCO datasets")

train_dataset = COCOCaptionDataset(
    root_dir=COCO_DATA_DIR,
    split='train',
    image_processor=model.vision_encoder.image_processor,
    max_length=MAX_CAPTION_LENGTH
)

val_dataset = COCOCaptionDataset(
    root_dir=COCO_DATA_DIR,
    split='val',
    image_processor=model.vision_encoder.image_processor,
    max_length=MAX_CAPTION_LENGTH
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn,
    pin_memory=True
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# Optimizer and learning rate scheduler

optimizer = model.gpt.configure_optimizers(
    weight_decay=WEIGHT_DECAY,
    learning_rate=LEARNING_RATE,
    device=device
)

total_steps = len(train_loader) * NUM_EPOCHS

def get_lr(step): #cosine decay with warmup
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / WARMUP_STEPS
    if step > total_steps:
        return LEARNING_RATE * 0.1
    decay_ratio = (step - WARMUP_STEPS) / (total_steps - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return LEARNING_RATE * 0.1 + coeff * (LEARNING_RATE * 0.9)

# Log

writer = SummaryWriter(log_dir=LOG_DIR)
tokenizer = tiktoken.get_encoding("gpt2")

def save_checkpoint(step, epoch, val_loss, filename):
    checkpoint = {
        'model': model.state_dict(),
        'gpt_config': model.gpt_config,
        'vision_config': model.vision_config,
        'optimizer': optimizer.state_dict(),
        'step': step,
        'epoch': epoch,
        'val_loss': val_loss
    }
    path = os.path.join(CHECKPOINT_DIR, filename)
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")

def load_checkpoint(filename):
    path = os.path.join(CHECKPOINT_DIR, filename)
    if os.path.exists(path):
        print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['step'], checkpoint['epoch']
    return 0, 0

def validate(epoch, step):
    model.eval()
    total_loss = 0
    num_batches = min(50, len(val_loader))  # Validate on subset
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_batches:
                break
            
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)
            
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(pixel_values, input_ids, targets)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch}, Step {step}: val_loss = {avg_loss:.4f}")
    writer.add_scalar("Loss/val", avg_loss, step)
    
    return avg_loss

# Training loop

print("Starting training")

# Try to resume from checkpoint
start_step, start_epoch = load_checkpoint("latest_checkpoint.pt")
global_step = start_step

for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    
    for batch_idx, batch in enumerate(train_loader):
        t0 = time.time()
        
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)
        
        optimizer.zero_grad()
        
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(pixel_values, input_ids, targets)
        
        loss.backward()
        
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update weights
        lr = get_lr(global_step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        
        # Logging
        epoch_loss += loss.item()
        t1 = time.time()
        
        if global_step % 10 == 0:
            writer.add_scalar("Loss/train", loss.item(), global_step)
            writer.add_scalar("Learning_Rate", lr, global_step)
            writer.add_scalar("Gradient_Norm", norm, global_step)
            
            print(f"Epoch {epoch}, Step {global_step} [{batch_idx}/{len(train_loader)}]: "
                  f"loss={loss.item():.4f}, lr={lr:.6f}, norm={norm:.4f}, "
                  f"time={t1-t0:.2f}s")
        
        if global_step % 500 == 0 and global_step > 0: # Validate and save
            val_loss = validate(epoch, global_step)
            save_checkpoint(global_step, epoch, val_loss, "latest_checkpoint.pt")
            model.train()
        
        if global_step % 2000 == 0 and global_step > 0: # Validate
            val_loss = validate(epoch, global_step)
            save_checkpoint(global_step, epoch, val_loss, f"checkpoint_step_{global_step}.pt")
            model.train()
        
        global_step += 1
    
    # End of epoch
    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch} completed: avg_loss = {avg_epoch_loss:.4f}")
    
    val_loss = validate(epoch, global_step)
    save_checkpoint(global_step, epoch, val_loss, f"checkpoint_epoch_{epoch}.pt")

writer.close()
print("Training complete")