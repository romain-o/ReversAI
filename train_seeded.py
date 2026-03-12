import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
import time
import queue
import random
import pickle
import torch.optim as optim

from env import ReversiEnv
from mcts import MCTS
# Import the architecture and buffer from your existing train file
from train import DualHeadResNet, ReplayBuffer, RemoteEvaluator, get_symmetries, gpu_batch_evaluator, train_network

import json

# --- 1. Load The Seed Book ---
try:
    with open("ffo_openings.json", "r", encoding='utf-8') as f:
        full_book = json.load(f)
        # Extract just the lists of integers to feed into the environment
        OPENING_BOOK = [entry["sequence"] for entry in full_book]
    print(f"Loaded {len(OPENING_BOOK)} openings from JSON.")
except FileNotFoundError:
    print("[!] 'ffo_openings.json' not found. Please run build_opening_book.py first.")
    exit()

# --- 2. The Seeded Worker ---
def seeded_self_play_worker(worker_id, input_queue, pipe_conn, experience_queue, num_games=10000):
    print(f"Worker {worker_id} started (Seeded Mode).")
    env = ReversiEnv()
    mcts = MCTS(num_simulations=100)
    evaluator = RemoteEvaluator(worker_id, input_queue, pipe_conn)
    
    for game in range(num_games):
        obs, info = env.reset()
        terminated = False
        game_history = []
        
        # --- NEW: Inject the Seed Opening ---
        # Pick a random opening from our book
        opening_sequence = random.choice(OPENING_BOOK)
        
        for move in opening_sequence:
            # Execute the move silently (without MCTS or recording history)
            obs, reward, terminated, truncated, info = env.step(move)
            
        # The environment is now perfectly set up at Turn 4. 
        # Now, we hand control over to MCTS for the rest of the game.
        # ------------------------------------
        
        while not terminated:
            best_action, mcts_policy = mcts.search(env, evaluator)
            
            if env.pass_count == 0 and len(game_history) < 15:
                # Keep Dirichlet noise active for exploration
                best_action = np.random.choice(65, p=mcts_policy)
                
            current_player = 1 if env.is_black_turn else -1
            
            for sym_obs, sym_policy in get_symmetries(obs.copy(), mcts_policy):
                game_history.append((sym_obs, sym_policy, current_player))
            
            obs, reward, terminated, truncated, info = env.step(best_action)
            
        # Game Over! Determine the absolute winner
        final_black_bb = env.current_player_bb if env.is_black_turn else env.opp_bb
        final_white_bb = env.opp_bb if env.is_black_turn else env.current_player_bb
        
        p1_score = final_black_bb.bit_count()
        p2_score = final_white_bb.bit_count()
        
        if p1_score > p2_score: true_winner = 1
        elif p2_score > p1_score: true_winner = -1
        else: true_winner = 0

        history_with_result = [(s, p, curr_p, true_winner) for s, p, curr_p in game_history]
        experience_queue.put(history_with_result)
        
    print(f"Worker {worker_id} finished all seeded games.")

# --- 3. Main Execution ---
if __name__ == "__main__":
    mp.set_start_method('spawn')
    
    NUM_WORKERS = 14 
    BATCH_SIZE = 16 

    input_queue = mp.Queue()
    parent_pipes = []
    child_pipes = []
    
    for _ in range(NUM_WORKERS):
        parent_conn, child_conn = mp.Pipe()
        parent_pipes.append(parent_conn)
        child_pipes.append(child_conn)

    weight_sync_queue = mp.Queue()
    
    gpu_process = mp.Process(
        target=gpu_batch_evaluator, 
        args=(input_queue, parent_pipes, weight_sync_queue, BATCH_SIZE)
    )
    gpu_process.start()

    experience_queue = mp.Queue()
    workers = []
    
    for i in range(NUM_WORKERS):
        p = mp.Process(
            target=seeded_self_play_worker, # Use the new seeded worker
            args=(i, input_queue, child_pipes[i], experience_queue)
        )
        p.start()
        workers.append(p)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    master_model = DualHeadResNet().to(device)
    

    # --- CRITICAL: LOWER LEARNING RATE ---
    # Dropped from 0.001 to 0.0001 to protect endgame weights
    optimizer = optim.Adam(master_model.parameters(), lr=0.0001, weight_decay=1e-4) 
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
    
    checkpoint_path = r"checkpoints/reversi_bundle_game_1000.pth"
    checkpoint_bundle = torch.load(checkpoint_path, map_location=device, weights_only=False)

    master_model.load_state_dict(checkpoint_bundle['model_state_dict'])
    optimizer.load_state_dict(checkpoint_bundle['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint_bundle['scheduler_state_dict'])
    games_played = checkpoint_bundle['games_played']
    replay_buffer = ReplayBuffer(capacity=500000)
    
    # Try to load the matching buffer so it doesn't start empty
    buffer_path = checkpoint_path.replace("reversi_bundle_", "reversi_buffer_").replace(".pth", ".pkl").replace("checkpoints", "buffer_checkpoints")
    replay_buffer.load_buffer(buffer_path)

    
    try:
        while True:
            try:
                game_history = experience_queue.get(timeout=10) 
                replay_buffer.save_game(game_history)
                games_played += 1
                
                if games_played % 100 == 0 and len(replay_buffer.buffer) > 10000:
                    v_loss, p_loss = train_network(master_model, optimizer, replay_buffer, batch_size=512, device=device)
                    scheduler.step()
                    
                    print(f"Seeded Game {games_played} | Value Loss: {v_loss:.4f} | Policy Loss: {p_loss:.4f}")
                    
                    cpu_state_dict = {k: v.cpu() for k, v in master_model.state_dict().items()}
                    weight_sync_queue.put(cpu_state_dict)
                    
                    if games_played % 500 == 0:
                        # Bundle everything into one dictionary
                        checkpoint_bundle = {
                            'games_played': games_played,
                            'model_state_dict': master_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()
                        }
                        torch.save(checkpoint_bundle, f"checkpoints/reversi_bundle_game_{games_played}.pth")
                        replay_buffer.save_buffer(f"buffer_checkpoints/reversi_buffer_game_{games_played}.pkl")
                        print("--> Seeded Checkpoints Saved!")
                        
            except queue.Empty:
                if not any(p.is_alive() for p in workers):
                    break

        for p in workers: p.join()
        gpu_process.terminate()
        
    except KeyboardInterrupt:
        print(f"\n[!] Ctrl+C detected! Halting seeded training at Game {games_played}...")
        checkpoint_bundle = {
                            'games_played': games_played,
                            'model_state_dict': master_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()
                        }
        torch.save(checkpoint_bundle, f"checkpoints/reversi_bundle_EMERGENCY_game_{games_played}.pth")
        replay_buffer.save_buffer(f"buffer_checkpoints/reversi_buffer_EMERGENCY_game_{games_played}.pkl")
        for p in workers: p.terminate(); p.join()
        gpu_process.terminate(); gpu_process.join()
        print("Clean shutdown.")