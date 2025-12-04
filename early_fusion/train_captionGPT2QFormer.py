import os
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tiktoken

from early_fusion.captionGPT2QFormer import CaptionGPT2QFormer
from early_fusion.qformer import QFormerConfig
from early_fusion.decoder_qformer import GPTConfig
from early_fusion.encoder_qformer import VisionEncoderConfig
from data.vision_dataset import COCOCaptionDataset, collate_fn

from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu


# Configuration 

COCO_DATA_DIR = "COCO" 
CHECKPOINT_DIR = "caption_checkpoints/run_qformer"
LOG_DIR = "caption_logs/run_qformer"
GPT2_CHECKPOINT = "gpt2_checkpoints/model_19073.pt"

MAX_LEARNING_RATE = 1e-4  
MIN_LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.05       
NUM_EPOCHS = 1  
BATCH_SIZE = 256
VAL_BATCH_SIZE = min(64, BATCH_SIZE)   
MAX_CAPTION_LENGTH = 100
NUM_WORKERS = 16

LOG_EVERY_CAPT  = 3200
EVAL_EVERY_CAPT = 32000
CKPT_EVERY_CAPT = 160000

LOG_EVERY  = max(1, LOG_EVERY_CAPT  // BATCH_SIZE)
EVAL_EVERY = max(1, EVAL_EVERY_CAPT // BATCH_SIZE)
CKPT_EVERY = max(1, CKPT_EVERY_CAPT // BATCH_SIZE)

GPT_CONFIG = GPTConfig(
    block_size=1024,
    vocab_size=50304,
    n_layer=12,
    n_head=12,
    n_embd=768,
    use_q_former_prefix=True
)

VISION_CONFIG = VisionEncoderConfig(
    model_name="openai/clip-vit-base-patch32",
    freeze_encoder=True,
    dropout=0.1
)

QFORMER_CONFIG = QFormerConfig(
    n_queries=32,
    n_layer=2,
    n_head=8,
    n_embd=768,
    vi_embd=768,
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

model = CaptionGPT2QFormer(
    gpt_cfg=GPT_CONFIG,
    vision_cfg=VISION_CONFIG,
    qformer_cfg=QFORMER_CONFIG
)

# Load pretrained GPT-2 weights
if os.path.exists(GPT2_CHECKPOINT):
    model.load_pretrained_gpt2(GPT2_CHECKPOINT)
else:
    raise FileNotFoundError(f"GPT-2 checkpoint not found at {GPT2_CHECKPOINT}. Aborting.")

model.to(device)

for p in model.gpt.parameters(): p.requires_grad = False
for p in model.qformer.parameters(): p.requires_grad = True
trainable = [p for p in model.parameters() if p.requires_grad]
print(f"Name of trainable parameters:")
for n, p in model.named_parameters():
    if p.requires_grad:
        print(f"  {n} - {p.shape}")
optimizer = torch.optim.AdamW(trainable, lr=MAX_LEARNING_RATE, betas=(0.9, 0.98), eps=1e-8, weight_decay=WEIGHT_DECAY, fused=True)

model = torch.compile(model)

# Dataset

print("Loading COCO datasets")
train_dataset = COCOCaptionDataset(
    root_dir=COCO_DATA_DIR,
    split='train',
    image_processor=model.vision_encoder.image_processor,
    max_length=MAX_CAPTION_LENGTH,
    feature_dir=os.path.join(COCO_DATA_DIR, "features_clip_vit_b32/train")  # use precomputed features
)
val_dataset = COCOCaptionDataset(
    root_dir=COCO_DATA_DIR,
    split='val',
    image_processor=model.vision_encoder.image_processor,
    max_length=MAX_CAPTION_LENGTH,
    feature_dir=os.path.join(COCO_DATA_DIR, "features_clip_vit_b32/val")
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# Optimizer and learning rate scheduler

first_epoch_steps = len(train_loader)
WARMUP_FRAC = 0.03
WARMUP_STEPS = int(first_epoch_steps * WARMUP_FRAC)
def lr_epoch1(step):
    if step < WARMUP_STEPS:
        return MAX_LEARNING_RATE * (step + 1) / WARMUP_STEPS
    decay_ratio = (step - WARMUP_STEPS) / max(1, (first_epoch_steps - WARMUP_STEPS))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LEARNING_RATE + coeff * (MAX_LEARNING_RATE - MIN_LEARNING_RATE)

FIXED_LR_AFTER = MIN_LEARNING_RATE

# Log

writer = SummaryWriter(log_dir=LOG_DIR)
tokenizer = tiktoken.get_encoding("gpt2")

def save_checkpoint(step, epoch, val_loss, filename):
    checkpoint = {
        'model': model.state_dict(),
        'gpt_config': model.gpt_config,
        'vision_config': model.vision_config,
        'qformer_config': model.qformer_config,
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
    num_batches = min(50, len(val_loader))
    predictions, references = {}, {}
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_batches: break
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)
            image_features = batch['image_features'].to(device)

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss, prefix = model(pixel_values=None, input_ids=input_ids, targets=targets, image_features=image_features)

            generated = model.generate(pixel_values=None, tokenizer=tokenizer, max_length=MAX_CAPTION_LENGTH,
                                       prefix=prefix, temperature=0.7, top_k=40, image_features=image_features)
            for j, caption in enumerate(generated):
                img_id = i * VAL_BATCH_SIZE + j
                ref_tokens = targets[j][targets[j] != -100].tolist()
                predictions[img_id] = caption.replace('<|endoftext|>', '').strip()
                references[img_id] = [tokenizer.decode(ref_tokens).replace('<|endoftext|>', '').strip()]
            
            total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
        
    writer.add_scalar("Loss/val", avg_loss, step)
    
    preds = {k: [v] for k, v in predictions.items()}
    print("Calculating Cider and BLEU scores...")
    cider = Cider()
    bleu = Bleu(4)  
    cider_score, _ = cider.compute_score(references, preds)
    bleu_score, _ = bleu.compute_score(references, preds)
    writer.add_scalar("Cider/val", cider_score, step)
    writer.add_scalar("BLEU-1/val", bleu_score[0], step)
    writer.add_scalar("BLEU-2/val", bleu_score[1], step)
    writer.add_scalar("BLEU-3/val", bleu_score[2], step)
    writer.add_scalar("BLEU-4/val", bleu_score[3], step)

    if len(predictions) > 0:
        sample_text = ""
        for j in range(min(3, len(predictions))):
            sample_text += f"**Pred {j}:** {predictions[j]}\n\n**Ref {j}:** {references[j][0]}\n\n---\n\n"
        writer.add_text("Samples/predictions", sample_text, step)
        
    print(f"[Val] Epoch {epoch} Step {step} - Loss: {avg_loss:.4f} Cider: {cider_score:.4f} BLEU-4: {bleu_score[3]:.4f}")
    return avg_loss

# Training loop
print("Starting training")

# Try to resume from checkpoint
start_step, start_epoch = load_checkpoint("latest_checkpoint.pt")
global_step = start_step
best_val_loss = float('inf')

for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        t0 = time.time()
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)
        image_features = batch['image_features'].to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss, _ = model(pixel_values=None, input_ids=input_ids, targets=targets, image_features=image_features)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = lr_epoch1(global_step if epoch == 0 and start_epoch == 0 else (batch_idx if epoch == 0 else FIXED_LR_AFTER))
        if epoch > 0: lr = FIXED_LR_AFTER
        for pg in optimizer.param_groups: pg['lr'] = lr
        optimizer.step()
        
        # Logging
        epoch_loss += loss.item()
        t1 = time.time()
        dt = t1 - t0
        
        if global_step % LOG_EVERY == 0:
            # Calculate throughput
            tokens_per_sec = (BATCH_SIZE * MAX_CAPTION_LENGTH) / dt
            images_per_sec = BATCH_SIZE / dt
            
            writer.add_scalar("Loss/train", loss.item(), global_step)
            writer.add_scalar("Perplexity/train", math.exp(loss.item()), global_step)
            writer.add_scalar("Learning_Rate", lr, global_step)
            writer.add_scalar("Gradient_Norm", norm, global_step)
            writer.add_scalar("Throughput/tokens_per_sec", tokens_per_sec, global_step)
            writer.add_scalar("Throughput/images_per_sec", images_per_sec, global_step)
            
            print(f"Epoch {epoch}, Step {global_step} [{(epoch+1)*batch_idx}/{NUM_EPOCHS*len(train_loader)}]: "
                  f"loss={loss.item():.4f}, lr={lr:.6f}, norm={norm:.4f}, "
                  f"time={dt:.2f}s, {images_per_sec:.1f} img/s, {tokens_per_sec:.0f} tok/s")
        
        if global_step % EVAL_EVERY == 0 and global_step > 0: # Validate and save
            val_loss = validate(epoch, global_step)
            save_checkpoint(global_step, epoch, val_loss, "latest_checkpoint.pt")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(global_step, epoch, val_loss, "best_model.pt")
                print(f"New best model saved with val_loss={val_loss:.4f}")
            
            model.train()
        
        if global_step % CKPT_EVERY == 0 and global_step > 0:
            save_checkpoint(global_step, epoch, val_loss, f"checkpoint_step_{global_step}.pt")
            model.train()
        
        global_step += 1
    
    # End of epoch
    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch} completed: avg_loss = {avg_epoch_loss:.4f}")
    
    val_loss = validate(epoch, global_step)
    save_checkpoint(global_step, epoch, val_loss, f"checkpoint_epoch_{epoch}.pt")
    
    # Save best model at epoch end
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(global_step, epoch, val_loss, "best_model.pt")
        print(f"New best model saved with val_loss={val_loss:.4f}")

writer.close()
print(f"Training complete. Best val_loss: {best_val_loss:.4f}")