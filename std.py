import os
from PIL import Image
import torch
from torch.utils.data import DataLoader,Dataset
from torch.utils.data import SequentialSampler
from torchvision import transforms
root = "/tn/FSRDiff/Result/residual001/sigma"
file_list = sorted(os.listdir(root))
class CustomDataset(Dataset):
    def __init__(self,root,file_list):
        self.root = root
        self.file_list = file_list
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, index):
        file_name = self.file_list[index]
        img_path = os.path.join(self.root,file_name)
        img = Image.open(img_path)
        image = self.transform(img)
        return image*255
val_dataset = CustomDataset(root, file_list)
sampler = SequentialSampler(val_dataset)
val_loader = DataLoader(val_dataset,batch_size=100,sampler=sampler)

all_pixture = torch.zeros(1)
all_pixture_ac = torch.zeros(1)
i=0
for val_data in val_loader:
    pixel_std = torch.zeros(3)
    i+=1
    B,C,H,W = val_data.size()
    std_per_image = torch.std(val_data,axis=(0))
    std_per_image_sum = torch.sum(std_per_image,axis=(1,2))
    std_per_image_sum_ave = std_per_image_sum/(H*W)
    pixel_std += std_per_image_sum_ave
    pixel_std_he = torch.sum(pixel_std)/3
#     pixel_std_he_ac = torch.sum(pixel_std)
#     print(torch.sum(pixel_std))
    all_pixture += pixel_std_he
#     all_pixture_ac += pixel_std_he_ac
    print(pixel_std_he)
print(all_pixture/i)
# print(all_pixture_ac/i)