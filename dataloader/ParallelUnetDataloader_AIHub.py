from PIL import Image
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np

class ParallelUnetDataloader_AIHub(Dataset):
    def __init__(self, json_path, transform=None):
        with open(json_path, 'r') as f:
            self.data_list = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        while True:
            try:
                data = self.data_list[idx]
                ia_img_path = os.path.join("trainexample/trainexample/Ia", data["wearing"])
                org_img_path = os.path.join("trainexample/trainexample/resized_org_model", data["wearing"])  # Original person image


                target_top = data['inner_top'] if data['inner_top'] is not None else data['main_top']

                jg_json_path = os.path.join("trainexample/trainexample/Jg", f"{target_top}_F.json")
                ic_img_path = os.path.join("trainexample/trainexample/Ic", f"{target_top}_F.jpg")

                jp_json_path = os.path.join("trainexample/trainexample/Jp", data["wearing"].replace(".jpg", ".json"))

                ia_img = Image.open(ia_img_path).convert('RGB')
                ic_img = Image.open(ic_img_path).convert('RGB')

                with open(jp_json_path, 'r') as f:
                    jp_data = json.load(f)
                    person_pose = torch.tensor(jp_data['landmarks'])

                with open(jg_json_path, 'r') as f:
                    jg_data = json.load(f)
                    garment_pose = torch.tensor(jg_data['landmarks'])

                org_img = Image.open(org_img_path).convert('RGB')  # Load original person image

                if self.transform:
                    org_img_t = self.transform(org_img)  # Transform original person image (person without clothes)
                    ic_img_t = self.transform(ic_img)  # Transform cloth image
                    # Concatenate person and cloth images to create 6-channel input
                    combined_img = torch.cat([org_img_t, ic_img_t], dim=0)  # Shape: [6, H, W]
                    ic_img = ic_img_t  # 3-channel garment image for garment_unet
                    org_img = org_img_t  # 3-channel target image
                else:
                    # If no transform, convert to tensors and concatenate
                    to_tensor = transforms.ToTensor()
                    org_img_t = to_tensor(org_img)
                    ic_img_t = to_tensor(ic_img)
                    combined_img = torch.cat([org_img_t, ic_img_t], dim=0)
                    ic_img = ic_img_t
                    org_img = org_img_t

                # Ensure pose tensors are 1D and float32
                person_pose = person_pose.flatten().float()  # Shape: [51]
                garment_pose = garment_pose.flatten().float()  # Shape: [51]

                return combined_img, person_pose, garment_pose, ic_img, org_img

            except FileNotFoundError:
                idx = (idx + 1) % len(self.data_list)  # Move to the next item

