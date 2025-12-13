import minerl
import gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
import os
import cv2 

# --- CONFIGURATION ---
ENV_NAME = "MineRLObtainIronPickaxe-v0" 
MODEL_PATH = "ppo_agent_latest.pth" # The model you want to test
NUM_EPISODES = 5                    # How many times to run the agent
MAX_STEPS = 6000                    # Max steps per episode
DEVICE = torch.device("cpu")        # Keep it on CPU for stability during eval

# --- REUSING YOUR CLASSES ---
# We must define the exact same Neural Network architecture to load the weights.

class CloneAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Flatten()
        )
        self.linear = nn.Linear(4608, 512)
        
        self.head_camera = nn.Linear(512, 2) 
        self.head_attack = nn.Linear(512, 2) 
        self.head_forward = nn.Linear(512, 2)
        self.head_jump = nn.Linear(512, 2)
        self.head_craft = nn.Linear(512, 8) 
        
        self.head_value = nn.Linear(512, 1)
        self.camera_log_std = nn.Parameter(torch.zeros(2) - 1.0)

    def forward(self, x):
        x = self.cnn(x)
        x = torch.relu(self.linear(x))
        return (self.head_camera(x), self.head_attack(x), 
                self.head_forward(x), self.head_jump(x), 
                self.head_craft(x), self.head_value(x))

    def get_action(self, x):
        """
        Simplified for Eval: We just want the action, no values/entropy.
        """
        cam_mean, atk_logits, fwd_logits, jmp_logits, crf_logits, _ = self.forward(x)
        
        # For Eval, we can either sample (stochastic) or take argmax (deterministic).
        # PPO usually works best if we keep sampling but maybe reduce std dev for camera.
        
        cam_std = self.camera_log_std.exp().expand_as(cam_mean)
        dist_cam = Normal(cam_mean, cam_std)
        dist_atk = Categorical(logits=atk_logits)
        dist_fwd = Categorical(logits=fwd_logits)
        dist_jmp = Categorical(logits=jmp_logits)
        dist_crf = Categorical(logits=crf_logits)
        
        cam = dist_cam.sample()
        atk = dist_atk.sample()
        fwd = dist_fwd.sample()
        jmp = dist_jmp.sample()
        crf = dist_crf.sample()
        
        return {'camera': cam, 'attack': atk, 'forward': fwd, 'jump': jmp, 'craft': crf}

# --- WRAPPERS ---

class VideoOverlayWrapper(gym.Wrapper):
    def __init__(self, env, directory='eval_videos'):
        super().__init__(env)
        self.directory = directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            
        self.messages = [] 
        self.last_inv = None
        self.notify_items = [
            'log', 'planks', 'stick', 'crafting_table', 'wooden_pickaxe', 
            'cobblestone', 'stone_pickaxe', 'furnace', 'iron_ore', 'iron_ingot'
        ]
        self.latest_processed_img = None 
        
        self.writer = None
        self.current_episode_num = 0

    def set_episode(self, ep_num):
        self.current_episode_num = ep_num

    def process_frame(self, obs):
        curr_inv = obs['inventory']
        if self.last_inv is not None:
            for item in self.notify_items:
                diff = curr_inv[item] - self.last_inv[item]
                if diff > 0:
                    name = item.replace('_', ' ').title()
                    msg = f"+{diff} {name}"
                    self.messages.append([msg, 30]) 
        self.last_inv = curr_inv.copy()

        img_rgb = obs['pov'].copy()
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # Overlay Messages
        self.messages = [m for m in self.messages if m[1] > 0]
        y_pos = 15
        for i, (msg, frames) in enumerate(self.messages):
            self.messages[i][1] -= 1
            cv2.putText(img_bgr, msg, (4, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 2)
            cv2.putText(img_bgr, msg, (4, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1) 
            y_pos += 15 
            
        self.latest_processed_img = img_bgr
        
        if self.writer is not None:
            self.writer.write(self.latest_processed_img)
            
        return img_rgb
    
    def reset(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            
        filepath = os.path.join(self.directory, f"eval_ep_{self.current_episode_num}.mp4")
        self.writer = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (64, 64))
        print(f"Recording Eval: {filepath}")

        obs = self.env.reset()
        self.last_inv = obs['inventory'].copy()
        self.messages = []
        obs['pov'] = self.process_frame(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs['pov'] = self.process_frame(obs)
        
        if done and self.writer is not None:
            self.writer.release()
            self.writer = None
            
        return obs, reward, done, info

class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env, repeat=4):
        super().__init__(env)
        self.repeat = repeat
    
    def step(self, action):
        total_reward = 0.0
        done = False
        combined_info = {}
        
        is_precision_action = (action['craft'] != 'none' or 
                               action['place'] != 'none' or 
                               action['nearbyCraft'] != 'none' or
                               action['nearbySmelt'] != 'none')
        
        is_attacking = action['attack'] == 1

        if is_precision_action:
            current_repeat = 1
        elif is_attacking:
            current_repeat = 8  
        else:
            current_repeat = self.repeat 
        
        for i in range(current_repeat):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            combined_info.update(info)
            if done: break
            
        return obs, total_reward, done, combined_info

# --- EVALUATION LOOP ---

def evaluate():
    print(f"Loading Model: {MODEL_PATH}")
    agent = CloneAgent().to(DEVICE)
    
    # Load weights safely for CPU/GPU compatibility
    if os.path.exists(MODEL_PATH):
        agent.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        agent.eval() # Set to Eval mode (turns off dropout if you had it)
    else:
        print(f"Error: Model {MODEL_PATH} not found!")
        return

    print("Initializing Environment...")
    env = gym.make(ENV_NAME)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_STEPS * 4) # Adjust for action repeat
    env = ActionRepeatWrapper(env, repeat=4)
    env = VideoOverlayWrapper(env, directory='eval_videos')
    
    print(f"Starting Evaluation for {NUM_EPISODES} episodes...")
    
    for episode in range(1, NUM_EPISODES + 1):
        env.set_episode(episode)
        obs = env.reset()
        
        done = False
        total_reward = 0
        step_count = 0
        
        # State variables for logic (copied from training)
        last_craft_attempt = -100
        pickaxe_spam_count = 0
        table_recovery_timer = 0
        
        # Track items for final report
        production = {k:0 for k in env.notify_items}
        last_inv_report = obs['inventory'].copy()

        while not done:
            # Prepare Image
            img = torch.from_numpy(obs['pov'].copy()).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            img = img.to(DEVICE)
            
            # Get Action
            with torch.no_grad():
                action_struct = agent.get_action(img)
            
            # --- SAME LOGIC BLOCK AS TRAINING ---
            # We must use the EXACT same hard-coded logic as training, 
            # or the agent will perform differently.
            
            craft_idx = action_struct['craft'].item()
            cmd_craft, cmd_nearby, cmd_place, cmd_smelt = 'none', 'none', 'none', 'none'
            inv = obs['inventory']
            
            # Calculate Production stats
            for k in production.keys():
                delta = inv[k] - last_inv_report[k]
                if delta > 0: production[k] += delta
            last_inv_report = inv.copy()
            
            tables_made = production['crafting_table']

            # 1. PLANKS
            if craft_idx == 1: 
                if inv['log'] > 0: cmd_craft = 'planks'
                else: cmd_craft = 'none'
            
            # 2. STICKS
            elif craft_idx == 2: 
                if inv['stick'] < 4 and inv['planks'] >= 5: 
                    cmd_craft = 'stick'
                else: 
                    cmd_craft = 'none'
            
            # 3. TABLES 
            elif craft_idx == 3: 
                if tables_made > 0 or inv['crafting_table'] > 0:
                    cmd_craft = 'none'
                    ready_wood_pick = (inv['planks'] >= 3 and inv['stick'] >= 2 and inv['wooden_pickaxe'] == 0)
                    ready_stone_pick = (inv['cobblestone'] >= 3 and inv['stick'] >= 2 and inv['stone_pickaxe'] == 0)
                    
                    if inv['crafting_table'] > 0 and (ready_wood_pick or ready_stone_pick) and table_recovery_timer == 0:
                        cmd_place = 'crafting_table'
                    elif cmd_place == 'crafting_table':
                        cmd_craft = 'none'
                else:
                    cmd_craft = 'crafting_table'
            
            # 4. WOODEN PICKAXE
            elif craft_idx == 4: 
                 if inv['wooden_pickaxe'] > 0: cmd_nearby = 'none'
                 else: cmd_nearby = 'wooden_pickaxe'
            
            # 5. STONE PICKAXE
            elif craft_idx == 5: 
                if inv['stone_pickaxe'] > 0: cmd_nearby = 'none'
                else: cmd_nearby = 'stone_pickaxe'

            # --- AUTO-CRAFT LOGIC ---
            can_craft_wood = (inv['stick'] >= 2 and inv['planks'] >= 3 and inv['wooden_pickaxe'] == 0)
            can_craft_stone = (inv['stick'] >= 2 and inv['cobblestone'] >= 3 and inv['stone_pickaxe'] == 0)
            
            target_item = 'none'
            if can_craft_stone: target_item = 'stone_pickaxe'
            elif can_craft_wood: target_item = 'wooden_pickaxe'
            
            minerl_action_override = False
            anchor_action = {'forward': 0, 'jump': 0, 'left': 0, 'right': 0, 'back': 0, 'sneak': 0, 'sprint': 0}

            if target_item != 'none' and pickaxe_spam_count < 200:
                if step_count - last_craft_attempt > 20:
                    cmd_nearby = target_item
                    last_craft_attempt = step_count
                    pickaxe_spam_count += 1
                else:
                    cmd_nearby = 'none' 
                minerl_action_override = True
            elif target_item != 'none' and pickaxe_spam_count >= 200:
                cmd_nearby = 'none' 
            elif target_item == 'none':
                if pickaxe_spam_count > 0:
                    if inv['wooden_pickaxe'] > 0 or inv['stone_pickaxe'] > 0:
                        table_recovery_timer = 25 
                pickaxe_spam_count = 0

            # 6. FURNACE & 7. IRON
            if craft_idx == 6:
                if inv['furnace'] > 0: cmd_place = 'furnace' 
                else: cmd_nearby = 'furnace' 
            elif craft_idx == 7:
                if inv['iron_ingot'] > 0: cmd_nearby = 'iron_pickaxe'
                elif inv['iron_ore'] > 0: cmd_smelt = 'iron_ingot'
                else: cmd_nearby = 'iron_pickaxe'
            
            camera_pitch = action_struct['camera'][0].cpu().numpy()[0] * 5.0
            camera_yaw = action_struct['camera'][0].cpu().numpy()[1] * 5.0
            
            equip_cmd = 'none'
            if inv['iron_pickaxe'] > 0: equip_cmd = 'iron_pickaxe'
            elif inv['stone_pickaxe'] > 0: equip_cmd = 'stone_pickaxe'
            elif inv['wooden_pickaxe'] > 0: equip_cmd = 'wooden_pickaxe'
            
            holding_pickaxe = (equip_cmd != 'none')
            busy_with_table = (target_item != 'none' or table_recovery_timer > 0)
            needs_stuff = (inv['cobblestone'] < 3 or (inv['stone_pickaxe'] > 0 and inv['iron_ore'] == 0))
            force_drill = holding_pickaxe and needs_stuff and not busy_with_table

            minerl_action = {
                'camera': [camera_pitch, camera_yaw], 
                'attack': action_struct['attack'].item(),
                'forward': action_struct['forward'].item(),
                'jump': action_struct['jump'].item(),
                'back': 0, 'left': 0, 'right': 0, 'sneak': 0, 'sprint': 0,
                'craft': cmd_craft, 'nearbyCraft': cmd_nearby, 'nearbySmelt': cmd_smelt,
                'place': cmd_place, 'equip': equip_cmd
            }

            if minerl_action_override: minerl_action.update(anchor_action)
            
            if table_recovery_timer > 0:
                minerl_action.update(anchor_action) 
                minerl_action['camera'] = [15.0, camera_yaw] 
                minerl_action['attack'] = 1 
                minerl_action['jump'] = 0
                table_recovery_timer -= 1
            elif force_drill:
                minerl_action['camera'] = [15.0, camera_yaw]
                minerl_action['attack'] = 1
                minerl_action['jump'] = 0

            if cmd_place == 'crafting_table':
                minerl_action['forward'] = 0
                minerl_action['jump'] = 0

            # Step Env
            obs, reward, done, info = env.step(minerl_action)
            total_reward += reward
            step_count += 1
            
            if step_count >= MAX_STEPS:
                done = True

        print(f"--- Episode {episode} Finished ---")
        print(f"Total Reward: {total_reward:.2f}")
        print("Inventory Gathered:")
        print(f"  Logs: {production['log']}")
        print(f"  Wood Pick: {'YES' if production['wooden_pickaxe'] > 0 else 'No'}")
        print(f"  Cobblestone: {production['cobblestone']}")
        print(f"  Stone Pick: {'YES' if production['stone_pickaxe'] > 0 else 'No'}")
        print(f"  Iron Ore: {production['iron_ore']}")
        print(f"  Iron Ingot: {production['iron_ingot']}")
        print("-" * 30)

    env.close()

if __name__ == "__main__":
    evaluate()