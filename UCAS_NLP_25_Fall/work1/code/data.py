from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class TrainDataset(Dataset):
    def __init__(self, transform=None):
        self.data = []
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        data_dir = Path(__file__).resolve().parent.parent / "data" / "train"
        
        if not data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")
        
        for file in data_dir.iterdir():
            if file.suffix in {".jpg", ".png", ".jpeg"}:
                img_path = str(data_dir / file)
                if "cat" in file.name:
                    label = 0
                elif "dog" in file.name:
                    label = 1
                else:
                    continue
                
                self.data.append((label, img_path))
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        label, img_path = self.data[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
    

class TestDataset(Dataset):
    def __init__(self, transform=None):
        self.data = []
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        data_dir = Path(__file__).resolve().parent.parent / "data" / "val"
        
        if not data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")
        
        for file in data_dir.iterdir():
            if file.suffix in {".jpg", ".png", ".jpeg"}:
                img_path = str(data_dir / file)
                if "cat" in file.name:
                    label = 0
                elif "dog" in file.name:
                    label = 1
                else:
                    continue
                
                self.data.append((label, img_path))
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        label, img_path = self.data[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
    
def test():
    ds = TrainDataset()
    print(len(ds))
    img, label = ds[0]
    print(img.shape, label)