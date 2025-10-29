"""Test script to load and test a trained model checkpoint"""
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from models import ParallelUNet
from dataloader import ParallelUnetDataloader_AIHub
import os

# Check for available checkpoints
checkpoint_files = [f for f in os.listdir('.') if f.endswith('.pt')]
if not checkpoint_files:
    print("No checkpoint files found. Please wait for training to save checkpoints (every 2 epochs).")
    print("Available checkpoint pattern: lightweight_parallel_unet_model1_epoch_X.pt")
    exit(1)

print("Available checkpoints:")
for i, f in enumerate(checkpoint_files):
    print(f"  {i+1}. {f}")

# Use the latest checkpoint (assuming naming convention)
latest_model1 = sorted([f for f in checkpoint_files if 'model1' in f])[-1] if any('model1' in f for f in checkpoint_files) else None
latest_model2 = sorted([f for f in checkpoint_files if 'model2' in f])[-1] if any('model2' in f for f in checkpoint_files) else None

if not latest_model1 or not latest_model2:
    print("Could not find both model checkpoints. Please wait for training to save them.")
    exit(1)

print(f"\nLoading checkpoints:")
print(f"  Model 1: {latest_model1}")
print(f"  Model 2: {latest_model2}")

# Setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset
json_file = "trainexample/trainexample/exampled_json_file.json"
dataset = ParallelUnetDataloader_AIHub(json_file, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Initialize models
EMB_DIM = 51
parallel_config_128 = {
    'garment_unet': {
        'dstack': {
            'blocks': [
                {'channels': 128, 'repeat': 3},
                {'channels': 256, 'repeat': 4},
                {'channels': 512, 'repeat': 6},
                {'channels': 1024, 'repeat': 7}]
        },
        'ustack': {
            'blocks': [
                {'channels': 1024, 'repeat': 7},
                {'channels': 512, 'repeat': 6}]
        }
    },
    'person_unet': {
        'dstack': {
            'blocks': [
                {'channels': 128, 'repeat': 3},
                {'channels': 256, 'repeat': 4},
                {'block_type': 'FiLM_ResBlk_Self_Cross', 'channels': 512, 'repeat': 6},
                {'block_type': 'FiLM_ResBlk_Self_Cross', 'channels': 1024, 'repeat': 7}]
        },
        'ustack': {
            'blocks': [
                {'block_type': 'FiLM_ResBlk_Self_Cross', 'channels': 1024, 'repeat': 7},
                {'block_type': 'FiLM_ResBlk_Self_Cross', 'channels': 512, 'repeat': 6},
                {'channels': 256, 'repeat': 4},
                {'channels': 128, 'repeat': 3}]
        }
    }
}

parallel_config_256 = parallel_config_128.copy()  # Same config for simplicity

model1 = ParallelUNet(EMB_DIM, parallel_config_128)
model2 = ParallelUNet(EMB_DIM, parallel_config_256)

# Load checkpoints
model1.load_state_dict(torch.load(latest_model1, map_location=device))
model2.load_state_dict(torch.load(latest_model2, map_location=device))

model1.to(device)
model2.to(device)
model1.eval()
model2.eval()

print("\nTesting on sample images...")
os.makedirs("test_results", exist_ok=True)

# Test on first 3 samples
with torch.no_grad():
    for i, (combined_img, person_pose, garment_pose, ic_img, org_img) in enumerate(dataloader):
        if i >= 3:  # Test only first 3
            break
            
        combined_img = combined_img.to(device)
        person_pose = person_pose.to(device)
        garment_pose = garment_pose.to(device)
        ic_img = ic_img.to(device)
        org_img = org_img.to(device)
        
        # Forward pass
        output1 = model1(combined_img, garment_pose, person_pose, ic_img)
        output1_concat = torch.cat([output1, ic_img], dim=1)
        output2 = model2(output1_concat, garment_pose, person_pose, ic_img)
        
        # Denormalize for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        
        org_img_vis = org_img * std + mean
        output2_vis = output2 * std + mean
        ic_img_vis = ic_img * std + mean
        
        # Save visualization
        combined_vis = torch.cat([
            ic_img_vis[0:1],   # garment
            org_img_vis[0:1],  # target
            output2_vis[0:1]   # prediction
        ], dim=0)
        
        save_path = f"test_results/sample_{i+1}.png"
        save_image(combined_vis, save_path, nrow=3, normalize=False)
        print(f"  Saved test result {i+1} to {save_path}")

print("\n✓ Testing complete! Check 'test_results/' folder for outputs.")

