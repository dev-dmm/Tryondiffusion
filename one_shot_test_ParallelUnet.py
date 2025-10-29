import zipfile
import os

# Check if the 'trainexample.zip' exists in the current directory
if os.path.exists('trainexample.zip'):
    # Unzip the file
    with zipfile.ZipFile('trainexample.zip', 'r') as zip_ref:
        zip_ref.extractall('./trainexample')
    print("ZIP file extracted.")
else:
    print("trainexample.zip does not exist in the current directory.")

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from models import LightweightParallelUNet, ParallelUNet, EfficientUNet
from dataloader import ParallelUnetDataloader_AIHub  
from torch.cuda.amp import autocast, GradScaler
import time
import os

torch.cuda.empty_cache()
# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize dataset and dataloader
json_file = "trainexample/trainexample/exampled_json_file.json"
dataset = ParallelUnetDataloader_AIHub(json_file, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)

# Initialize model and optimizer
IMG_CHANNEL = 3

EMB_DIM = 51

parallel_config_128 = {
    'garment_unet': {
        'dstack': {
            'blocks': [
                {
                    'channels': 128,
                    'repeat': 3
                },
                {
                    'channels': 256,
                    'repeat': 4
                },
                {
                    'channels': 512,
                    'repeat': 6
                },
                {
                    'channels': 1024,
                    'repeat': 7
                }]
        },
        'ustack': {
            'blocks': [
                {
                    'channels': 1024,
                    'repeat': 7
                },
                {
                    'channels': 512,
                    'repeat': 6
                }]
        }
    },
    'person_unet': {
        'dstack': {
            'blocks': [
                {
                    'channels': 128,
                    'repeat': 3
                },
                {
                    'channels': 256,
                    'repeat': 4
                },
                {
                    'block_type': 'FiLM_ResBlk_Self_Cross',
                    'channels': 512,
                    'repeat': 6
                },
                {

                    'block_type': 'FiLM_ResBlk_Self_Cross',
                    'channels': 1024,
                    'repeat': 7
                }]
        },
        'ustack': {
            'blocks': [
                {
                    'block_type': 'FiLM_ResBlk_Self_Cross',
                    'channels': 1024,
                    'repeat': 7
                },
                {
                    'block_type': 'FiLM_ResBlk_Self_Cross',
                    'channels': 512,
                    'repeat': 6
                },
                {
                    'channels': 256,
                    'repeat': 4
                },
                {
                    'channels': 128,
                    'repeat': 3
                }]
        }
    }
}

parallel_config_256 = {
    'garment_unet': {
        'dstack': {
            'blocks': [
                {
                    'channels': 128,
                    'repeat': 3
                },
                {
                    'channels': 256,
                    'repeat': 4
                },
                {
                    'channels': 512,
                    'repeat': 6
                },
                {
                    'channels': 1024,
                    'repeat': 7
                }]
        },
        'ustack': {
            'blocks': [
                {
                    'channels': 1024,
                    'repeat': 7
                },
                {
                    'channels': 512,
                    'repeat': 6
                }]
        }
    },
    'person_unet': {
        'dstack': {
            'blocks': [
                {
                    'channels': 128,
                    'repeat': 3
                },
                {
                    'channels': 256,
                    'repeat': 4
                },
                {
                    'block_type': 'FiLM_ResBlk_Self_Cross',
                    'channels': 512,
                    'repeat': 6
                },
                {

                    'block_type': 'FiLM_ResBlk_Self_Cross',
                    'channels': 1024,
                    'repeat': 7
                }]
        },
        'ustack': {
            'blocks': [
                {
                    'block_type': 'FiLM_ResBlk_Self_Cross',
                    'channels': 1024,
                    'repeat': 7
                },
                {
                    'block_type': 'FiLM_ResBlk_Self_Cross',
                    'channels': 512,
                    'repeat': 6
                },
                {
                    'channels': 256,
                    'repeat': 4
                },
                {
                    'channels': 128,
                    'repeat': 3
                }]
        }
    }
}

# Initialize both models and their optimizers
model1 = ParallelUNet(EMB_DIM, parallel_config_128)
model2 = ParallelUNet(EMB_DIM, parallel_config_256)  # Assuming you have a ParallelUNet class

optimizer1 = optim.AdamW(model1.parameters(), lr=0.0001)
optimizer2 = optim.AdamW(model2.parameters(), lr=0.0001)

criterion = torch.nn.MSELoss()  # Mean Squared Error Loss

# Move both models to the GPU
model1.to(device)
model2.to(device)

# Initialize the gradient scaler for fp16
scaler = GradScaler()

# Define the number of steps for gradient accumulation
accumulation_steps = 4  # Reduced for faster feedback on CPU


# Training loop
print("Starting training...")
print(f"Dataset size: {len(dataset)}")
print(f"Device: {device}")
for epoch in range(10):  # 10 epochs
    epoch_start_time = time.time()
    print(f"\n=== Starting Epoch {epoch + 1}/10 ===")
    epoch_loss = 0.0
    num_batches = 0

    optimizer1.zero_grad(set_to_none=True)
    optimizer2.zero_grad(set_to_none=True)

    # Wrap your dataloader with tqdm for a progress bar
    for i, (combined_img, person_pose, garment_pose, ic_img, org_img) in enumerate(dataloader):
        if i == 0:
            print(f"Batch shapes - combined_img: {combined_img.shape}, person_pose: {person_pose.shape}, garment_pose: {garment_pose.shape}, ic_img: {ic_img.shape}, org_img: {org_img.shape}")

        # Move data to the GPU and convert to fp16
        combined_img = combined_img.to(device)
        person_pose = person_pose.to(device)
        garment_pose = garment_pose.to(device)
        ic_img = ic_img.to(device)
        org_img = org_img.to(device)

        # Enable autocast for mixed-precision training
        with autocast():
            # Model 1's forward pass and loss calculation
            # Model signature: forward(x, gar_emb, pose_emb, seg_garment)
            # x: 6-channel concatenated image [person, cloth]
            # gar_emb: garment embedding (garment_pose landmarks)
            # pose_emb: pose embedding (person_pose landmarks)
            # seg_garment: 3-channel garment image
            output1 = model1(combined_img, garment_pose, person_pose, ic_img)
            loss1 = criterion(output1, org_img)  # Use original person image as the target

            # Upsample output1 from 128x128 to 256x256
            output1_upsampled = F.interpolate(output1, size=(256, 256), mode='bilinear', align_corners=True)

            # Model 2's forward pass and loss calculation
            # Concatenate output1 with ic_img to maintain 6-channel input for model2
            output1_concat = torch.cat([output1, ic_img], dim=1)
            output2 = model2(output1_concat, garment_pose, person_pose, ic_img)  # Using output1 as input to model2
            loss2 = criterion(output2, org_img)

            # Combine the losses if needed
            loss = loss1 + loss2
            loss = loss / accumulation_steps  # Normalize the loss because it is accumulated

        # Backpropagation using gradient accumulation
        scaler.scale(loss).backward()

        # Print progress more frequently (every batch for first few, then every accumulation step)
        if i < 3 or (i + 1) % accumulation_steps == 0:
            print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {loss.item() * accumulation_steps:.6f}")
            
            # Save visualization images periodically
            if (i + 1) % 50 == 0:  # Save every 50 batches
                os.makedirs("results", exist_ok=True)
                # Denormalize for visualization (reverse ImageNet normalization)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
                
                # Denormalize images
                org_img_vis = org_img * std + mean
                output2_vis = output2 * std + mean
                ic_img_vis = ic_img * std + mean
                
                # Combine images for visualization (input, garment, output, target)
                combined_vis = torch.cat([
                    ic_img_vis[0:1],  # garment
                    org_img_vis[0:1],  # target
                    output2_vis[0:1]   # prediction
                ], dim=0)
                
                save_path = f"results/epoch_{epoch + 1}_batch_{i + 1}.png"
                save_image(combined_vis, save_path, nrow=3, normalize=False)
                print(f"  -> Saved visualization to {save_path}")

        if (i + 1) % accumulation_steps == 0:  # Wait for several backward steps
            scaler.step(optimizer1)  # Performs the optimizer step for model1
            scaler.step(optimizer2)  # Performs the optimizer step for model2

            scaler.update()  # Updates the scale for next iteration
            optimizer1.zero_grad()  # Reset gradients tensors for model1
            optimizer2.zero_grad()  # Reset gradients tensors for model2


        epoch_loss += loss.item() * accumulation_steps  # Accumulate the true loss
        num_batches += 1

    avg_loss = epoch_loss / num_batches
    print(f"Epoch {epoch + 1}, Average Loss: {avg_loss}")

    # After the epoch ends, calculate the elapsed time
    epoch_end_time = time.time()
    elapsed_time = epoch_end_time - epoch_start_time
    print(f"Time taken for Epoch {epoch + 1}: {elapsed_time:.2f} seconds")

    # Save the models
    if epoch % 2 == 0:
        torch.save(model1.state_dict(), f"lightweight_parallel_unet_model1_epoch_{epoch + 1}.pt")
        torch.save(model2.state_dict(), f"parallel_unet_model2_epoch_{epoch + 1}.pt")
        print("Models saved.")


print("Training complete.")


