"""Download COCO dataset"""
import os
import urllib.request
import zipfile
from tqdm import tqdm

def download_file(url, filename):
    """Download file with progress bar"""
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
        urllib.request.urlretrieve(url, filename=filename, reporthook=t.update_to)

def download_coco(data_dir='/Data/lucas.mebille/COCO'):
    """Download COCO 2017 dataset"""
    os.makedirs(data_dir, exist_ok=True)
    
    urls = {
        'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
        'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    }
    
    for name, url in urls.items():
        filename = os.path.join(data_dir, url.split('/')[-1])
        
        if os.path.exists(filename):
            print(f"✓ {filename} already exists, skipping download")
            continue
        
        print(f"Downloading {name} from {url}")
        download_file(url, filename)
        
        print(f"Extracting {filename}...")
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        print(f"✓ {name} downloaded and extracted")
    
    print("\n✓ COCO dataset setup complete!")
    print(f"Dataset location: {data_dir}")
    print("\nDirectory structure:")
    print(f"  {data_dir}/train2017/")
    print(f"  {data_dir}/val2017/")
    print(f"  {data_dir}/annotations/")

if __name__ == "__main__":
    download_coco()