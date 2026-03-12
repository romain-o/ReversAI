import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
import time
import queue
from env import ReversiEnv
from mcts import MCTS
import torch.optim as optim
from collections import deque
import random
import torch.nn.functional as F
import pickle

# --- 1. Dummy Neural Network (Replace with your ResNet) ---
class ResBlock(nn.Module):
    """
    A standard Residual Block. 
    The skip connection (out += identity) allows the network to learn 
    deep strategies without the vanishing gradient problem.
    """
    def __init__(self, channels):
        super().__init__()
        # kernel_size=3 with padding=1 keeps the 8x8 spatial dimensions intact
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # The crucial Skip Connection
        out += identity
        out = F.relu(out)
        
        return out

class DualHeadResNet(nn.Module):
    """
    The AlphaZero-style architecture optimized for an 8x8 board.
    """
    def __init__(self, num_blocks=5, channels=128):
        super().__init__()
        
        # --- 1. Initial Convolutional Block ---
        # Takes the 3-channel input (Black, White, Turn) and expands it to 128 channels
        self.conv_input = nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(channels)

        # --- 2. The Residual Tower ---
        # Stacks 'num_blocks' identical ResBlocks. 
        # 5 is a great starting point for fast training. 10+ is for superhuman performance.
        self.res_blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_blocks)])

        # --- 3. The Policy Head (Predicts the Move) ---
        # 1x1 conv crushes the 128 channels down to 2, saving massive compute
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        # 2 channels * 8x8 board = 128 inputs to the linear layer
        self.policy_fc = nn.Linear(2 * 8 * 8, 65) 

        # --- 4. The Value Head (Predicts the Winner) ---
        # 1x1 conv crushes the 128 channels down to 1
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        # 1 channel * 8x8 board = 64 inputs. Maps to a hidden layer of 256.
        self.value_fc1 = nn.Linear(1 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # 1. Input Block
        x = self.conv_input(x)
        x = self.bn_input(x)
        x = F.relu(x)

        # 2. Residual Tower
        for block in self.res_blocks:
            x = block(x)

        # 3. Policy Head processing
        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = F.relu(p)
        p = p.view(p.size(0), -1) # Flatten from (batch, 2, 8, 8) to (batch, 128)
        policy = F.softmax(self.policy_fc(p), dim=1) # Output probability distribution

        # 4. Value Head processing
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = F.relu(v)
        v = v.view(v.size(0), -1) # Flatten from (batch, 1, 8, 8) to (batch, 64)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v)) # Output strictly between -1.0 and 1.0

        return policy, value
    
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        
    def save_game(self, game_history):
        """game_history is a list of (state, mcts_policy, player_who_moved, result)"""
        for state, policy, player, result in game_history:
            # If the player who moved won, value is 1. If they lost, -1.
            value = 1.0 if player == result else -1.0
            if result == 0: value = 0.0 # Draw
            self.buffer.append((state, policy, value))
            
    def sample_batch(self, batch_size=128):
        batch = random.sample(self.buffer, batch_size)
        states, policies, values = zip(*batch)
        return (np.array(states), np.array(policies), np.array(values, dtype=np.float32))
    
    def save_buffer(self, filepath):
        """Serializes the deque to a file."""
        with open(filepath, 'wb') as f:
            # Convert deque to a list for safe pickling
            pickle.dump(list(self.buffer), f)
            
    def load_buffer(self, filepath):
        """Loads a saved buffer from a file."""
        try:
            with open(filepath, 'rb') as f:
                loaded_list = pickle.load(f)
                self.buffer = deque(loaded_list, maxlen=self.buffer.maxlen)
            print(f"Successfully loaded {len(self.buffer)} positions into the Replay Buffer.")
        except FileNotFoundError:
            print("No existing buffer found. Starting fresh.")


# --- 2. The GPU Batching Engine ---
def gpu_batch_evaluator(input_queue, output_pipes, weight_sync_queue, batch_size=16):
    """
    Dedicated process that sits on the RTX 4080.
    It waits for board states from the CPU workers, batches them, 
    and sends the predictions back.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualHeadResNet().to(device)
    model.eval() # Set to evaluation mode for MCTS
    
    print(f"GPU Evaluator online using: {device}")

    with torch.no_grad():
        while True:
            try:
                new_state_dict = weight_sync_queue.get_nowait()
                model.load_state_dict(new_state_dict)
                print("GPU Evaluator: Brain updated with new weights!")
            except queue.Empty:
                pass
            batch_states = []
            worker_ids = []
            
            # 1. Gather requests until the batch is full or the queue is empty
            while len(batch_states) < batch_size:
                try:
                    # Non-blocking get. If empty, break and process what we have.
                    worker_id, state = input_queue.get_nowait()
                    worker_ids.append(worker_id)
                    batch_states.append(state)
                except queue.Empty:
                    if len(batch_states) > 0:
                        break # Process partial batch if no more requests are waiting
                    else:
                        time.sleep(0.001) # Prevent CPU spinning if completely idle

            if not batch_states:
                continue
                
            # 2. Convert batch to a single PyTorch Tensor and send to GPU
            batch_tensor = torch.tensor(np.array(batch_states), dtype=torch.float32).to(device)
            
            # 3. Massive parallel inference
            policies, values = model(batch_tensor)
            
            # Move results back to CPU memory
            policies = policies.cpu().numpy()
            values = values.cpu().numpy()

            # 4. Route the results back to the specific CPU workers
            for i, w_id in enumerate(worker_ids):
                output_pipes[w_id].send((policies[i], values[i].item()))


class RemoteEvaluator:
    """
    Acts as a bridge between the MCTS search tree and the GPU process.
    To MCTS, this looks like a normal neural network.
    """
    def __init__(self, worker_id, input_queue, pipe_conn):
        self.worker_id = worker_id
        self.input_queue = input_queue
        self.pipe_conn = pipe_conn

    def predict(self, state):
        # 1. Send the state to the GPU batcher
        self.input_queue.put((self.worker_id, state))
        
        # 2. Block and wait for the GPU to send the batch results back
        policy, value = self.pipe_conn.recv()
        
        # MCTS expects a dictionary of {action: probability}
        # In Reversi, we have 65 actions. We map the policy array to a dict.
        action_probs = {action: prob for action, prob in enumerate(policy)}
        
        return action_probs, value
    
def get_symmetries(board, policy):
    """
    Takes a 3x8x8 board and a 65-length policy.
    Returns 8 mathematically equivalent (board, policy) pairs.
    """
    symmetries = []
    
    # Reshape the 64 board squares of the policy into an 8x8 grid
    policy_board = policy[:64].reshape(8, 8)
    pass_prob = policy[64]
    
    for i in range(4): # 4 Rotations (0, 90, 180, 270 degrees)
        for flip in [False, True]: # Normal and Mirrored
            
            b = np.rot90(board, i, axes=(1, 2))
            p = np.rot90(policy_board, i, axes=(0, 1))
            
            if flip:
                b = np.flip(b, axis=2)
                p = np.flip(p, axis=1)
                
            # Flatten the policy back out and re-attach the Pass probability
            p_flat = np.append(p.flatten(), pass_prob)
            
            symmetries.append((b.copy(), p_flat.copy()))
            
    return symmetries

def self_play_worker(worker_id, input_queue, pipe_conn, experience_queue, num_games=10000):
    print(f"Worker {worker_id} started.")
    env = ReversiEnv()
    mcts = MCTS(num_simulations=200) # Lowered to 100 for faster generation
    evaluator = RemoteEvaluator(worker_id, input_queue, pipe_conn)
    
    for game in range(num_games):
        obs, info = env.reset()
        terminated = False
        game_history = []
        
        while not terminated:
            # 1. Get the action AND the policy from MCTS
            best_action, mcts_policy = mcts.search(env, evaluator)
            if env.pass_count == 0 and len(game_history) < 8:
                # Add Dirichlet noise here for even better exploration!
                best_action = np.random.choice(65, p=mcts_policy)
            # 2. Record the state, policy, and whose turn it is
            current_player = 1 if env.is_black_turn else -1
            
            for sym_obs, sym_policy in get_symmetries(obs.copy(), mcts_policy):
                game_history.append((sym_obs, sym_policy, current_player))
            
            # 3. Step the environment
            obs, reward, terminated, truncated, info = env.step(best_action)
            
        # Game Over! Determine the absolute winner
        # reward is relative to the last action. Let's find the true winner.
        final_black_bb = env.current_player_bb if env.is_black_turn else env.opp_bb
        final_white_bb = env.opp_bb if env.is_black_turn else env.current_player_bb
        
        p1_score = final_black_bb.bit_count()
        p2_score = final_white_bb.bit_count()
        
        if p1_score > p2_score: true_winner = 1
        elif p2_score > p1_score: true_winner = -1
        else: true_winner = 0

        # Package the history with the true winner and send it to the training loop
        history_with_result = [(s, p, curr_p, true_winner) for s, p, curr_p in game_history]
        experience_queue.put(history_with_result)
        
    print(f"Worker {worker_id} finished all games.")
    
def train_network(model, optimizer, replay_buffer, batch_size=128, device="cuda"):
    """Pulls a batch of MCTS data and updates the ResNet weights."""
    if len(replay_buffer.buffer) < batch_size:
        return 0.0, 0.0 # Not enough data to train yet
        
    model.train() # Set to training mode
    states, target_policies, target_values = replay_buffer.sample_batch(batch_size)
    
    # Move to RTX 4080
    states = torch.tensor(states, dtype=torch.float32).to(device)
    target_policies = torch.tensor(target_policies, dtype=torch.float32).to(device)
    target_values = torch.tensor(target_values).unsqueeze(1).to(device)
    
    optimizer.zero_grad()
    
    # 1. Forward Pass
    predicted_policies, predicted_values = model(states)
    
    # 2. Calculate Loss
    value_loss = torch.nn.functional.mse_loss(predicted_values, target_values)
    # Policy loss: Cross Entropy (using log of predicted policy)
    policy_loss = -torch.sum(target_policies * torch.log(predicted_policies + 1e-8)) / batch_size
    
    total_loss = value_loss + policy_loss
    
    # 3. Backpropagate and Update Weights
    total_loss.backward()
    optimizer.step()
    
    return value_loss.item(), policy_loss.item()


# --- 4. Main Execution ---
if __name__ == "__main__":
    # Required for PyTorch multiprocessing
    mp.set_start_method('spawn')
    
    # Configuration to saturate hardware
    NUM_WORKERS = 14 # Leaves 2 threads for the GPU orchestrator and OS
    BATCH_SIZE = 16 

    # Communication channels
    input_queue = mp.Queue()
    parent_pipes = []
    child_pipes = []
    
    for _ in range(NUM_WORKERS):
        parent_conn, child_conn = mp.Pipe()
        parent_pipes.append(parent_conn)
        child_pipes.append(child_conn)

    weight_sync_queue = mp.Queue()
    # Launch GPU Evaluator Process
    gpu_process = mp.Process(
        target=gpu_batch_evaluator, 
        args=(input_queue, parent_pipes, weight_sync_queue, BATCH_SIZE)
    )
    gpu_process.start()


    experience_queue = mp.Queue()
    
    # Launch CPU Worker Processes
    workers = []
    for i in range(NUM_WORKERS):
        p = mp.Process(
            target=self_play_worker, 
            args=(i, input_queue, child_pipes[i], experience_queue)
        )
        p.start()
        workers.append(p)

  
    
    # ... launch GPU evaluator and CPU workers (passing experience_queue to args) ...

    # --- THE MAIN TRAINING LOOP ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # We need a master copy of the model on the main process to train
    
    master_model = DualHeadResNet().to(device)
    optimizer = optim.Adam(master_model.parameters(), lr=0.0005, weight_decay=1e-4) 
    checkpoint_path = r"checkpoints\reversi_model_game_46400.pth"
    #checkpoint_bundle = torch.load(checkpoint_path, map_location=device, weights_only=False)

    #master_model.load_state_dict(checkpoint_bundle['model_state_dict'])
    #optimizer.load_state_dict(checkpoint_bundle['optimizer_state_dict'])
    #games_played = checkpoint_bundle['games_played']
    games_played = 46400
    replay_buffer = ReplayBuffer(capacity=100000)
    
    # Try to load the matching buffer so it doesn't start empty
    buffer_path = checkpoint_path.replace("reversi_bundle_", "reversi_buffer_").replace(".pth", ".pkl").replace("checkpoints", "buffer_checkpoints")
    replay_buffer.load_buffer(buffer_path)
    
    #new_lr = 0.0005 
    #for param_group in optimizer.param_groups:
    #    param_group['lr'] = new_lr
        
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

    print(f"Starting training loop. Games already played: {games_played}")
    
    try:
        while True:
            try:
                # Pull finished games from workers
                game_history = experience_queue.get(timeout=10) # Wait 10 seconds for a game
                replay_buffer.save_game(game_history)
                games_played += 1
                
                # Train the network after every 10 games
                if games_played % 100 == 0 and len(replay_buffer.buffer) > 10000:
                    v_loss, p_loss = train_network(master_model, optimizer, replay_buffer, batch_size=512, device=device)
                    
                    scheduler.step()
                    print(f"Game {games_played} | Value Loss: {v_loss:.4f} | Policy Loss: {p_loss:.4f}")
                    cpu_state_dict = {k: v.cpu() for k, v in master_model.state_dict().items()}
                    weight_sync_queue.put(cpu_state_dict)
                    # Periodically save the model weights so you don't lose progress!
                    if games_played % 500 == 0:
                        checkpoint_bundle = {
                            'games_played': games_played,
                            'model_state_dict': master_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()
                        }
                        torch.save(checkpoint_bundle, f"checkpoints/reversi_bundle_game_{games_played}.pth")
                        replay_buffer.save_buffer(f"buffer_checkpoints/reversi_buffer_game_{games_played}.pkl")
                        print("--> Model and Replay Buffer Checkpoints Saved!")
                        
                    # Note: In a fully distributed system, you would periodically push these 
                    # new weights back to the `gpu_batch_evaluator` process so it uses the updated brain.
                    
            except queue.Empty:
                # Check if all workers are dead
                if not any(p.is_alive() for p in workers):
                    break

        # Wait for all self-play games to finish
        for p in workers:
            p.join()

        # Clean up
        gpu_process.terminate()
        print("Self-play data generation complete.")
        
    except KeyboardInterrupt:
        # --- THE EMERGENCY SAVE TRIGGER ---
        print(f"\n[!] Ctrl+C detected! Halting training at Game {games_played}...")
        
        print("Saving emergency checkpoints...")
        torch.save(master_model.state_dict(), f"reversi_model_EMERGENCY_game_{games_played}.pth")
        replay_buffer.save_buffer(f"reversi_buffer_EMERGENCY_game_{games_played}.pkl")
        print("--> Model and Replay Buffer safely saved to disk.")
        
        print("Terminating background worker processes (this may take a second)...")
        # Terminate CPU workers
        for p in workers:
            p.terminate()
            p.join()
            
        # Terminate GPU evaluator
        gpu_process.terminate()
        gpu_process.join()
        
        print("All processes cleanly shut down. Exiting.")