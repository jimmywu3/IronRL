import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

# --- CONFIGURATION ---
DATA_DIR = "minerl_data/basalt_caves"  # Where you downloaded the videos
IMG_SIZE = 64  # Resize to 64x64 (Critical for MineRL v0.4 compatibility)
BATCH_SIZE = 32
EPOCHS = 5
FRAMES_PER_VIDEO = 50  # How many frames to grab from each video (balance speed vs accuracy)

# --- 1. THE DATASET LOADER ---
class CaveDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = []
        self.labels = []
        self.video_files = [f for f in os.listdir(data_dir) if f.endswith('.mp4')]
        
        print(f"Found {len(self.video_files)} videos. Processing...")
        
        for idx, video_file in enumerate(self.video_files):
            path = os.path.join(data_dir, video_file)
            self._process_video(path)
            
            # Print progress every 10 videos
            if (idx + 1) % 10 == 0:
                print(f"   Processed {idx + 1}/{len(self.video_files)} videos...")

        print(f"   Dataset Loaded: {len(self.data)} images total.")
        print(f"   Cave Samples: {sum(self.labels)}")
        print(f"   Surface Samples: {len(self.labels) - sum(self.labels)}")

    def _process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            # Resize immediately
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            # Convert BGR (OpenCV) to RGB (MineRL/PyTorch)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        total_frames = len(frames)
        if total_frames < 100: return

        # --- IMPROVED LABELING STRATEGY ---
        
        # 1. SURFACE (Negative Samples)
        # Take from the first 80% (more variety of trees/sky)
        surface_limit = int(total_frames * 0.80)
        for _ in range(FRAMES_PER_VIDEO // 2):
            idx = random.randint(0, surface_limit)
            self.data.append(frames[idx])
            self.labels.append(0) # 0 = Surface

        # 2. CAVE (Positive Samples)
        # Only take from the last 3%, AND ONLY IF IT IS DARK
        cave_start = int(total_frames * 0.97)
        
        valid_cave_frames = []
        for i in range(cave_start, total_frames):
            # Calculate average brightness (0-255)
            brightness = np.mean(frames[i])
            
            # HEURISTIC: Caves are dark. Sky/Surface is bright (> 90).
            # Only keep frames that are darker than 90/255
            if brightness < 90:
                valid_cave_frames.append(frames[i])
        
        # If we found valid dark frames, pick random ones from them
        if len(valid_cave_frames) > 0:
            # We might not find enough dark frames, so sample with replacement if needed
            for _ in range(FRAMES_PER_VIDEO // 2):
                frame = random.choice(valid_cave_frames)
                self.data.append(frame)
                self.labels.append(1) # 1 = Cave

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert to PyTorch format: (Channel, Height, Width) and Float 0-1
        img = self.data[idx]
        img = torch.from_numpy(img).float() / 255.0
        img = img.permute(2, 0, 1) # HWC -> CHW
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label

# --- 2. THE NEURAL NETWORK (CNN) ---
class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        
        # Input: 3 channels (RGB), 64x64
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 64 -> 32
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 32 -> 16
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 16 -> 8
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 2) # Output: [prob_surface, prob_cave]
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- 3. TRAINING LOOP ---
def main():
    # Setup Device (Use MPS for M1/M2 Mac, or CPU if fails)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on: {device}")

    # Load Data
    dataset = CaveDataset(DATA_DIR)
    if len(dataset) == 0:
        print("Error: No images loaded. Did you run the download script?")
        return
        
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize Model
    model = SimpleClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("--- Starting Training ---")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate Accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.4f} | Acc: {100.*correct/total:.2f}%")

    # Save the model
    print("Saving Model...")
    torch.save(model.state_dict(), "cave_classifier.pth")
    print("Done! 'cave_classifier.pth' is ready for your agent.")

if __name__ == "__main__":
    main()