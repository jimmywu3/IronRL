import minerl
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

# CONFIG
DATA_DIR = "minerl_data"
EPOCHS = 2
BATCH_SIZE = 32
LR = 0.0001

# --- 1. THE BRAIN ---
class CloneAgent(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 64x64 RGB
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate size after CNN
        # 64 -> 31 -> 14 -> 6. 128 channels * 6 * 6 = 4608
        self.linear = nn.Linear(4608, 512)
        
        # --- OUTPUT HEADS (Muscles) ---
        # 1. Camera (Pitch, Yaw) -> Continuous numbers
        self.head_camera = nn.Linear(512, 2)
        
        # 2. Attack (0 or 1) -> Classification
        self.head_attack = nn.Linear(512, 2) 
        
        # 3. Forward (0 or 1)
        self.head_forward = nn.Linear(512, 2)
        
        # 4. Jump (0 or 1)
        self.head_jump = nn.Linear(512, 2)

    def forward(self, x):
        x = self.cnn(x)
        x = torch.relu(self.linear(x))
        
        camera = self.head_camera(x)
        attack = self.head_attack(x)
        forward = self.head_forward(x)
        jump = self.head_jump(x)
        
        return camera, attack, forward, jump
    
    def get_action(self, obs_pov):
        img = torch.from_numpy(obs_pov).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img = img.to(next(self.parameters()).device)
        
        with torch.no_grad():
            cam, atk, fwd, jmp = self.forward(img)
            
        action = {
            'camera': cam[0].cpu().numpy() * 100.0,
            'attack': torch.argmax(atk, dim=1).item(),
            'forward': torch.argmax(fwd, dim=1).item(),
            'jump': torch.argmax(jmp, dim=1).item(),
            'back': 0, 'left': 0, 'right': 0, 'sneak': 0, 'sprint': 0,
            'craft': 'none', 'nearbyCraft': 'none', 'nearbySmelt': 'none',
            'place': 'none', 'equip': 'none'
        }
        return action

# --- 2. TRAIN ---
def main():
    # Force CPU if MPS fails (MineRL data loading can be weird with MPS)
    device = torch.device("cpu") 
    print(f"Training on: {device}")

    # Load Data Pipeline
    os.environ["MINERL_DATA_ROOT"] = DATA_DIR
    data = minerl.data.make("MineRLObtainIronPickaxe-v0", data_dir=DATA_DIR, num_workers=2)
    
    agent = CloneAgent().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LR)
    
    # Loss functions
    mse_loss = nn.MSELoss() # For camera numbers
    ce_loss = nn.CrossEntropyLoss() # For buttons (yes/no)

    print("Starting Training Loop...")
    iter_count = 0
    
    # Iterate through the dataset
    for state, action, _, next_state, _ in data.batch_iter(batch_size=BATCH_SIZE, num_epochs=EPOCHS, seq_len=1):
        
        # PREPARE INPUT
        # Pov is [Batch, 1, 64, 64, 3] -> We need [Batch, 3, 64, 64]
        img = state['pov'].squeeze(1) # Remove sequence dim
        img = torch.from_numpy(img).float().permute(0, 3, 1, 2) / 255.0
        img = img.to(device)
        
        # PREPARE TARGETS
        # Camera: [Batch, 1, 2] -> [Batch, 2]
        cam_target = torch.from_numpy(action['camera']).squeeze(1).float().to(device) / 100.0
        
        # Buttons: MineRL gives 0/1. PyTorch needs LongTensor
        attack_target = torch.from_numpy(action['attack']).squeeze(1).long().to(device)
        forward_target = torch.from_numpy(action['forward']).squeeze(1).long().to(device)
        jump_target = torch.from_numpy(action['jump']).squeeze(1).long().to(device)

        # FORWARD PASS
        optimizer.zero_grad()
        pred_cam, pred_atk, pred_fwd, pred_jmp = agent(img)
        
        # CALCULATE LOSS (Weighted sum)
        loss_cam = mse_loss(pred_cam, cam_target)
        loss_atk = ce_loss(pred_atk, attack_target)
        loss_fwd = ce_loss(pred_fwd, forward_target)
        loss_jmp = ce_loss(pred_jmp, jump_target)
        
        total_loss = loss_cam + loss_atk + loss_fwd + loss_jmp
        
        # BACKWARD PASS
        total_loss.backward()
        optimizer.step()
        
        iter_count += 1
        if iter_count % 10 == 0:
            print(f"Step {iter_count} | Loss: {total_loss.item():.4f}")
            
        if iter_count % 500 == 0:
            print("Saving Checkpoint...")
            torch.save(agent.state_dict(), "behavioral_cloning.pth")

    print("Training Finished!")
    torch.save(agent.state_dict(), "behavioral_cloning.pth")

if __name__ == "__main__":
    main()