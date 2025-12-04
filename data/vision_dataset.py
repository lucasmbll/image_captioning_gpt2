import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO
import tiktoken
import random

class COCOCaptionDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split='train',
        image_processor=None,
        max_length=77,
        transform=None,
        feature_dir=None,
        sample_one_caption=True  
    ):
        self.root_dir = root_dir
        self.split = split
        self.image_processor = image_processor
        self.max_length = max_length
        self.transform = transform
        self.feature_dir = feature_dir
        self.use_precomputed = feature_dir is not None
        self.sample_one_caption = sample_one_caption
        
        # Load COCO annotations
        ann_file = os.path.join(root_dir, 'annotations', f'captions_{split}2017.json')
        print(f"Loading COCO {split} annotations...")
        self.coco = COCO(ann_file)
        
        if self.use_precomputed:
            print(f"Using precomputed features from: {feature_dir}")
        
        # Tokenizer
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.eos_token = 50256
        
        if self.sample_one_caption:
            # Build image-based index: one entry per image
            self.img_ids = list(self.coco.imgs.keys())
            # Build mapping: img_id -> list of annotation ids
            self.img_to_anns = {}
            for img_id in self.img_ids:
                self.img_to_anns[img_id] = self.coco.getAnnIds(imgIds=img_id)
            print(f"COCO {split} dataset loaded: {len(self.img_ids)} images (sampling 1 caption per image)")
        else:
            # Original behavior: one entry per caption
            self.ann_ids = list(self.coco.anns.keys())
            print(f"COCO {split} dataset loaded: {len(self.ann_ids)} captions")
    
    def __len__(self):
        if self.sample_one_caption:
            return len(self.img_ids)
        else:
            return len(self.ann_ids)
    
    def __getitem__(self, idx):
        if self.sample_one_caption: # Training with one random caption per image
            img_id = self.img_ids[idx]
            ann_ids = self.img_to_anns[img_id]
            ann_id = random.choice(ann_ids)
            ann = self.coco.anns[ann_id]
        else:
            # Training with all captions
            ann_id = self.ann_ids[idx]
            ann = self.coco.anns[ann_id]
            img_id = ann['image_id']
        
        caption = ann['caption']
        
        # Load image features
        if self.use_precomputed:
            feature_path = os.path.join(self.feature_dir, f"{img_id}.pt")
            image_features = torch.load(feature_path, map_location='cpu', weights_only=True)  # (50, 768)
            if not isinstance(image_features, torch.Tensor):
                raise TypeError(f"Expected Tensor at {feature_path}, got {type(image_features)}")
            image_features = image_features.float()
        else:
            # Load and process image
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.root_dir, f'{self.split}2017', img_info['file_name'])
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            if self.image_processor is not None:
                processed = self.image_processor(images=image, return_tensors="pt")
                image_features = processed['pixel_values'].squeeze(0)
            else:
                raise ValueError("image_processor required when not using precomputed features")
        
        # Tokenize caption
        tokens = self.tokenizer.encode(caption)
        tokens = tokens[:self.max_length - 1]  # Leave room for EOS
        tokens.append(self.eos_token)
        
        input_ids = tokens[:-1]  # Input: all except last
        targets = tokens[1:]     # Target: all except first (shifted)
        
        return {
            'image_features': image_features,
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'targets': torch.tensor(targets, dtype=torch.long),
            'img_id': img_id
        }


def collate_fn(batch, pad_token_id=50256):
    image_features = torch.stack([item['image_features'] for item in batch])
    input_ids_list = [item['input_ids'] for item in batch]
    targets_list = [item['targets'] for item in batch]
    
    max_len = max(x.size(0) for x in input_ids_list)
    B = len(batch)
    
    input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
    targets = torch.full((B, max_len), -100, dtype=torch.long)
    
    for i, (ids, tgt) in enumerate(zip(input_ids_list, targets_list)):
        L = ids.size(0)
        input_ids[i, :L] = ids
        targets[i, :L] = tgt
    
    return {
        'image_features': image_features,
        'input_ids': input_ids,
        'targets': targets
    }