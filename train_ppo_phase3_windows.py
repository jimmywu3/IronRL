import minerl
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
import os
import csv
import cv2 
import glob

# --- Hyperparameters ---
ENV_NAME = "MineRLObtainIronPickaxe-v0" 
LR = 0.00002
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01 
VALUE_LOSS_COEF = 0.5
MAX_GRAD_NORM = 0.5
BATCH_SIZE = 64
ROLLOUT_STEPS = 4000 // 4 
TOTAL_UPDATES = 3000        
DEVICE = torch.device("cpu") 

# --- WRAPPERS ---

class VideoOverlayWrapper(gym.Wrapper):
    def __init__(self, env, directory='videos_phase3'):
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
        
        # Video Recording State
        self.writer = None
        self.current_episode_num = 0
        self.is_recording = False

    def set_episode(self, ep_num):
        self.current_episode_num = ep_num
        # Logic: Record every 5th episode
        self.is_recording = (ep_num % 5 == 0 or ep_num == 1)

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

        # Convert to BGR for OpenCV (MineRL is RGB)
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
            
        self.latest_processed_img = img_bgr # Stored as BGR for video writer
        
        # Record Frame if active
        if self.is_recording and self.writer is not None:
            self.writer.write(self.latest_processed_img)
            
        return img_rgb # Return RGB to agent
    
    def reset(self):
        # Close previous writer if exists
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            
        # Start new writer if this episode is flagged for recording
        if self.is_recording:
            filepath = os.path.join(self.directory, f"episode_{self.current_episode_num}.mp4")
            # MineRL is 64x64
            self.writer = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (64, 64))
            print(f"Recording started: {filepath}")

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
    
    def render(self, mode='human', **kwargs):
        # We handle our own rendering via process_frame logic, 
        # but keep this for compatibility if needed.
        pass

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

class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

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

    def get_action_and_value(self, x, action=None):
        cam_mean, atk_logits, fwd_logits, jmp_logits, crf_logits, value = self.forward(x)
        cam_std = self.camera_log_std.exp().expand_as(cam_mean)
        dist_cam = Normal(cam_mean, cam_std)
        dist_atk = Categorical(logits=atk_logits)
        dist_fwd = Categorical(logits=fwd_logits)
        dist_jmp = Categorical(logits=jmp_logits)
        dist_crf = Categorical(logits=crf_logits)
        
        if action is None:
            cam = dist_cam.sample()
            atk = dist_atk.sample()
            fwd = dist_fwd.sample()
            jmp = dist_jmp.sample()
            crf = dist_crf.sample()
        else:
            cam = action['camera']
            atk = action['attack']
            fwd = action['forward']
            jmp = action['jump']
            crf = action['craft']
            
        log_prob = (dist_cam.log_prob(cam).sum(1) + dist_atk.log_prob(atk) + 
                    dist_fwd.log_prob(fwd) + dist_jmp.log_prob(jmp) + 
                    dist_crf.log_prob(crf))
        entropy = (dist_cam.entropy().sum(1) + dist_atk.entropy() + 
                   dist_fwd.entropy() + dist_jmp.entropy() + 
                   dist_crf.entropy())
        return {'camera': cam, 'attack': atk, 'forward': fwd, 'jump': jmp, 'craft': crf}, log_prob, entropy, value

class CaveRewardWrapper(gym.Wrapper):
    """
    🏆 IRON AGE REWARDS: Incentivize the Upgrade!
    """
    def __init__(self, env, classifier):
        super().__init__(env)
        self.classifier = classifier
        self.last_cobble = 0
        self.has_wood_pick = False
        self.has_stone_pick = False
    
    def reset(self):
        obs = self.env.reset()
        self.last_cobble = 0
        self.has_wood_pick = False
        self.has_stone_pick = False
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        curr_inv = obs['inventory']
        
        # 1. WOOD PICKAXE BONUS (+5.0)
        if not self.has_wood_pick and curr_inv['wooden_pickaxe'] > 0:
            reward += 5.0
            self.has_wood_pick = True
            print("PICKAXE CRAFTED! +5.0 Reward")

        # 2. STONE PICKAXE BONUS (+10.0)
        if not self.has_stone_pick and curr_inv['stone_pickaxe'] > 0:
            reward += 10.0
            self.has_stone_pick = True
            print("STONE PICKAXE UPGRADE! +10.0 Reward")

        # 3. STONE BOUNTY (+1.0)
        cobble_diff = curr_inv['cobblestone'] - self.last_cobble
        if cobble_diff > 0:
            reward += 1.0 
            print(f"STONE MINED! +{reward}")
            
        self.last_cobble = curr_inv['cobblestone']

        # 4. VISUAL REWARDS
        img = torch.from_numpy(obs['pov'].copy()).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        curr_brightness = img.mean().item()
        
        if curr_brightness > 0.45: 
            reward -= 0.01

        has_pickaxe = (curr_inv['wooden_pickaxe'] > 0 or 
                       curr_inv['stone_pickaxe'] > 0 or
                       curr_inv['iron_pickaxe'] > 0)

        if has_pickaxe and curr_brightness < 0.40:
            with torch.no_grad():
                output = self.classifier(img)
                cave_prob = torch.softmax(output, dim=1)[0][1].item()
            if cave_prob > 0.70:
                reward += 0.05
                info['cave_triggered'] = True
            else:
                info['cave_triggered'] = False
        else:
            info['cave_triggered'] = False
            
        reward = np.clip(reward, -1.0, 15.0) 
        return obs, reward, done, info

class StatTracker(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.track_items = [
            'log', 'planks', 'stick', 'crafting_table', 'wooden_pickaxe', 
            'cobblestone', 'stone_pickaxe', 'furnace', 'iron_ore', 'iron_ingot'
        ]
        self.reset_stats()
        
    def reset_stats(self):
        self.step_count = 0
        self.total_raw_reward = 0.0
        self.production = {item: 0 for item in self.track_items}
        self.last_inventory = None
        
    def reset(self):
        obs = self.env.reset()
        self.reset_stats()
        self.last_inventory = obs['inventory'].copy()
        return obs
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.step_count += 1
        self.total_raw_reward += reward
        
        curr_inv = obs['inventory']
        if self.last_inventory is not None:
            for item in self.track_items:
                delta = curr_inv[item] - self.last_inventory[item]
                if delta > 0:
                    self.production[item] += delta
        
        self.last_inventory = curr_inv.copy()
        info['production'] = self.production
        info['raw_reward'] = self.total_raw_reward
        info['got_wood'] = curr_inv['log'] > 0
        info['got_iron'] = curr_inv['iron_ingot'] > 0
        
        return obs, reward, done, info

def main():
    print("Initializing Agent...")
    agent = CloneAgent().to(DEVICE)
    
    if os.path.exists("ppo_agent_latest.pth"):
        print("Resuming from 'ppo_agent_latest.pth'...")
        agent.load_state_dict(torch.load("ppo_agent_latest.pth", map_location=torch.device('cpu')))
    
    classifier = SimpleClassifier().to(DEVICE)
    if os.path.exists("cave_classifier.pth"):
        print("Loading Cave Classifier...")
        classifier.load_state_dict(torch.load("cave_classifier.pth", map_location=torch.device('cpu')))
        classifier.eval()

    env = gym.make(ENV_NAME)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=12000)
    
    env = StatTracker(env)
    env = ActionRepeatWrapper(env, repeat=4) 
    env = CaveRewardWrapper(env, classifier)
    
    #  CUSTOM VIDEO WRAPPER (Phase 3 Directory)
    video_wrapper = VideoOverlayWrapper(env, directory='videos_phase3')
    env = video_wrapper 
    
    optimizer = optim.Adam(agent.parameters(), lr=LR)
    
    log_filename = "episode_log.csv"
    # Ensure log file existence to read last episode count
    if not os.path.exists(log_filename):
        with open(log_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Episode", "Update", "Score", 
                "Logs_Found", "Tables_Made", "Sticks_Made", 
                "Wood_Picks", "Stone_Mined", "Stone_Picks", 
                "Furnaces", "Iron_Ore", "Iron_Ingots"
            ])
            global_episode_count = 0
    else:
        # Read last episode
        try:
            with open(log_filename, "r") as f:
                lines = f.readlines()
                if len(lines) > 1:
                    last_line = lines[-1].strip()
                    if last_line:
                        global_episode_count = int(last_line.split(',')[0])
                    else:
                        global_episode_count = 0
                else:
                    global_episode_count = 0
        except:
            global_episode_count = 0
            
    print(f"Resuming Episode Counter from {global_episode_count}")
    
    log_file = open(log_filename, "a", newline="")
    writer = csv.writer(log_file)
    
    # Update wrapper with initial count
    video_wrapper.set_episode(global_episode_count + 1)
    
    obs = env.reset()
    info = {}
    
    print("Starting Training Loop (Iron Age - Phase 3 Video)...")
    
    for update in range(1, TOTAL_UPDATES + 1):
        states, actions_hist, rewards, log_probs, values, dones = [], [], [], [], [], []

        last_craft_attempt = -100
        pickaxe_spam_count = 0  
        table_recovery_timer = 0 
        
        for step in range(ROLLOUT_STEPS):
            img = torch.from_numpy(obs['pov'].copy()).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            img = img.to(DEVICE)
            with torch.no_grad():
                action_struct, log_prob, _, value = agent.get_action_and_value(img)
            
            # --- LOGIC BLOCK ---
            craft_idx = action_struct['craft'].item()
            cmd_craft, cmd_nearby, cmd_place, cmd_smelt = 'none', 'none', 'none', 'none'
            inv = obs['inventory']
            
            tables_made = info.get('production', {}).get('crafting_table', 0)

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
            
            # 3. TABLES (Strict Cap - RECYCLER MODE)
            elif craft_idx == 3: 
                if tables_made > 0 or inv['crafting_table'] > 0:
                    cmd_craft = 'none'
                    
                    # SMART PLACEMENT
                    ready_wood_pick = (inv['planks'] >= 3 and inv['stick'] >= 2 and inv['wooden_pickaxe'] == 0)
                    # Priority Upgrade: Stone Pick needs 3 cobble + 2 sticks
                    ready_stone_pick = (inv['cobblestone'] >= 3 and inv['stick'] >= 2 and inv['stone_pickaxe'] == 0)
                    
                    if inv['crafting_table'] > 0 and (ready_wood_pick or ready_stone_pick) and table_recovery_timer == 0:
                        cmd_place = 'crafting_table'
                        if step % 50 == 0: print("PLACING TABLE: Ready to build Tool!")
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
            if can_craft_stone: 
                target_item = 'stone_pickaxe' # PRIORITY!
            elif can_craft_wood: 
                target_item = 'wooden_pickaxe'
            
            # ANCHORED CRAFTING
            minerl_action_override = False
            anchor_action = {'forward': 0, 'jump': 0, 'left': 0, 'right': 0, 'back': 0, 'sneak': 0, 'sprint': 0}

            if target_item != 'none' and pickaxe_spam_count < 200:
                if step - last_craft_attempt > 20:
                    cmd_nearby = target_item
                    last_craft_attempt = step
                    pickaxe_spam_count += 1
                    if step % 50 == 0: 
                        print(f"AUTO-CRAFT: Attempting {target_item} ({pickaxe_spam_count}/200)")
                else:
                    cmd_nearby = 'none' 
                minerl_action_override = True
            
            elif target_item != 'none' and pickaxe_spam_count >= 200:
                cmd_nearby = 'none' 
                if step % 100 == 0: print("GIVING UP on Auto-Craft.")
            
            elif target_item == 'none':
                # CHECK FOR SUCCESS & TRIGGER RECOVERY
                if pickaxe_spam_count > 0:
                    if inv['wooden_pickaxe'] > 0 or inv['stone_pickaxe'] > 0:
                        print("CRAFT SUCCESS! Smashing table to recover it...")
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
            
            # --- CAMERA & EQUIP ---
            camera_pitch = action_struct['camera'][0].cpu().numpy()[0] * 5.0
            camera_yaw = action_struct['camera'][0].cpu().numpy()[1] * 5.0
            
            equip_cmd = 'none'
            if inv['iron_pickaxe'] > 0: equip_cmd = 'iron_pickaxe'
            elif inv['stone_pickaxe'] > 0: equip_cmd = 'stone_pickaxe'
            elif inv['wooden_pickaxe'] > 0: equip_cmd = 'wooden_pickaxe'
            
            # --- FORCE DRILL ---
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
            
            # TABLE RECOVERY OVERRIDE
            if table_recovery_timer > 0:
                minerl_action.update(anchor_action) 
                minerl_action['camera'] = [15.0, camera_yaw] 
                minerl_action['attack'] = 1 
                minerl_action['jump'] = 0
                action_struct['attack'] = torch.tensor([1]).to(DEVICE) 
                table_recovery_timer -= 1
                if table_recovery_timer == 0:
                    print("RECOVERY COMPLETE (Hopefully)")

            elif force_drill:
                minerl_action['camera'] = [15.0, camera_yaw]
                minerl_action['attack'] = 1
                minerl_action['jump'] = 0
                action_struct['attack'] = torch.tensor([1]).to(DEVICE)
                if step % 50 == 0: print("FORCE DRILL ACTIVE")

            if cmd_place == 'crafting_table':
                minerl_action['forward'] = 0
                minerl_action['jump'] = 0

            next_obs, reward, done, info = env.step(minerl_action)
            
            curr_brightness = img.mean().item()
            if curr_brightness > 0.40: reward -= 0.01
            reward = np.clip(reward, -1.0, 15.0)
            
            states.append(img.cpu())
            actions_hist.append(action_struct)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value.item())
            dones.append(done)
            
            if done:
                global_episode_count += 1
                ep_score = info.get('raw_reward', 0.0)
                stats = info.get('production', {})
                
                summ_str = f" Ep {global_episode_count} | Score: {ep_score:.1f} | "
                if stats.get('wooden_pickaxe', 0) > 0: summ_str += " WOOD | "
                if stats.get('stone_pickaxe', 0) > 0: summ_str += " STONE PICK | "
                if stats.get('iron_ore', 0) > 0: summ_str += " IRON FOUND!"
                print(summ_str)
                
                writer.writerow([
                    global_episode_count, update, round(ep_score, 1),
                    stats.get('log', 0), stats.get('crafting_table', 0), stats.get('stick', 0),
                    stats.get('wooden_pickaxe', 0), stats.get('cobblestone', 0), stats.get('stone_pickaxe', 0),
                    stats.get('furnace', 0), stats.get('iron_ore', 0), stats.get('iron_ingot', 0)
                ])
                log_file.flush()
                # Update Video Wrapper count for NEXT episode
                video_wrapper.set_episode(global_episode_count + 1)
                
                obs = env.reset()
                info = {}
            else:
                obs = next_obs

        with torch.no_grad():
            img = torch.from_numpy(obs['pov'].copy()).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            _, _, _, next_value = agent.get_action_and_value(img.to(DEVICE))
            next_value = next_value.item()

        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        lastgaelam = 0
        
        for t in reversed(range(ROLLOUT_STEPS)):
            if t == ROLLOUT_STEPS - 1:
                nextnonterminal = 1.0 - 0 
                nextvals = next_value
            else:
                nextnonterminal = 1.0 - dones[t+1]
                nextvals = values[t+1]
            delta = rewards[t] + GAMMA * nextvals * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
            returns[t] = advantages[t] + values[t]
            
        b_states = torch.cat(states).to(DEVICE) 
        if len(b_states.shape) == 5: b_states = b_states.squeeze(1)
        b_logprobs = torch.tensor(log_probs).to(DEVICE)
        b_returns = torch.tensor(returns).float().to(DEVICE)
        b_advantages = torch.tensor(advantages).float().to(DEVICE)
        b_values = torch.tensor(values).float().to(DEVICE)
        
        b_actions = {
            'camera': torch.cat([a['camera'] for a in actions_hist]).to(DEVICE),
            'attack': torch.cat([a['attack'] for a in actions_hist]).to(DEVICE),
            'forward': torch.cat([a['forward'] for a in actions_hist]).to(DEVICE),
            'jump': torch.cat([a['jump'] for a in actions_hist]).to(DEVICE),
            'craft': torch.cat([a['craft'] for a in actions_hist]).to(DEVICE)
        }
        b_inds = np.arange(ROLLOUT_STEPS)
        
        for epoch in range(3): 
            np.random.shuffle(b_inds)
            for start in range(0, ROLLOUT_STEPS, BATCH_SIZE):
                end = start + BATCH_SIZE
                mb_inds = b_inds[start:end]
                
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_states[mb_inds], action={k: v[mb_inds] for k,v in b_actions.items()}
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                pg_loss = torch.max(-mb_advantages * ratio, -mb_advantages * torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS)).mean()
                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()
                loss = pg_loss - ENTROPY_COEF * entropy.mean() + VALUE_LOSS_COEF * v_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        torch.save(agent.state_dict(), "ppo_agent_latest.pth")
        if update % 10 == 0:
            torch.save(agent.state_dict(), f"ppo_agent_step_{update}.pth")
            print(f"Checkpoint saved (Update {update})")

if __name__ == "__main__":
    main()