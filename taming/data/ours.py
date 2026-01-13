import os
import albumentations
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class ImagePaths(Dataset):
    def __init__(self, path, size=None):
        self.size = size
        self.images = sorted([os.path.join(path, folder, file) 
               for folder in os.listdir(path) 
               if os.path.isdir(os.path.join(path, folder))  
               for file in os.listdir(os.path.join(path, folder))])
        self._length = len(self.images)

        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image

    def __getitem__(self, i):
        example = self.preprocess_image(self.images[i])
        return example


def load_data(dataset_path=None,image_size=256,batch_size=32,num_workers=24,pin_memory=True,state=False):
    data = ImagePaths(dataset_path, size=image_size)
    loader = DataLoader(data, batch_size=batch_size, shuffle=state,num_workers=num_workers,pin_memory=pin_memory)
    return loader