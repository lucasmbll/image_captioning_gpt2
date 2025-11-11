"""COCO Dataset for image captioning"""
import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import tiktoken
from pycocotools.coco import COCO

class COCOCaptionDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split='train',
        image_processor=None,
        max_length=77,
        transform=None
    ):
  
        self.root_dir = root_dir #coco root directory
        self.split = split
        self.image_processor = image_processor
        self.max_length = max_length
        self.transform = transform
        
        if split == 'train':
            self.img_dir = os.path.join(root_dir, 'train2017')
            ann_file = os.path.join(root_dir, 'annotations', 'captions_train2017.json')
        else:
            self.img_dir = os.path.join(root_dir, 'val2017')
            ann_file = os.path.join(root_dir, 'annotations', 'captions_val2017.json')
        
        # Load COCO annotations
        print(f"Loading COCO {split} annotations...")
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.anns.keys())
        
        # Initialize tokenizer
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.eot_token = self.tokenizer._special_tokens['<|endoftext|>']
        
        print(f"COCO {split} dataset loaded: {len(self.ids)} captions")
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
 
        ann_id = self.ids[idx]
        ann = self.coco.anns[ann_id]
        caption = ann['caption']
        img_id = ann['image_id']
        
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        
        # Preprocess image
        if self.image_processor is not None:
            pixel_values = self.image_processor(images=image, return_tensors="pt")['pixel_values'][0] # (3, 224, 224) preprocessed image
        else:
            pixel_values = self.transform(image) if self.transform else image 
        
        # Tokenize caption: <|endoftext|> + caption + <|endoftext|>
        tokens = [self.eot_token] + self.tokenizer.encode(caption) + [self.eot_token]
        
        # Truncate or pad to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Create input_ids and targets
        input_ids = tokens[:-1] #(max_length,) tokenized caption
        targets = tokens[1:] #(max_length,) target tokens for loss
        
        # Pad sequences
        pad_length = self.max_length - 1  # -1 because we removed one token
        input_ids = input_ids + [self.eot_token] * (pad_length - len(input_ids))
        targets = targets + [-100] * (pad_length - len(targets))  # -100 is ignore index
        
        return {
            'pixel_values': pixel_values,
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'targets': torch.tensor(targets, dtype=torch.long)
        }


def collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    targets = torch.stack([item['targets'] for item in batch])
    
    return {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'targets': targets
    }