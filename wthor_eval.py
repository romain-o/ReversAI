import struct
import torch
import numpy as np
import os
import glob
import re

from env import ReversiEnv
from train import DualHeadResNet

# --- 1. WTHOR Binary Parser ---

def parse_wthor_file(filepath):
    """
    Parses a standard .wtb file and returns a list of games.
    Each game is a list of sequential actions (integers 0-63).
    """
    games = []
    
    with open(filepath, "rb") as f:
        # The first 16 bytes are the main file header
        header = f.read(16)
        if len(header) < 16:
            raise ValueError("File is too small to be a valid WTHOR file.")
            
        # Number of games is stored in bytes 4-7 as a 32-bit little-endian integer
        num_games = struct.unpack('<I', header[4:8])[0]
        print(f"Detected {num_games} games in WTHOR file.")
        
        for _ in range(num_games):
            game_data = f.read(68) # Each game record is exactly 68 bytes
            if len(game_data) < 68:
                break
                
            moves_bytes = game_data[8:68] # The 60 sequential moves
            moves = []
            
            for b in moves_bytes:
                if b == 0: # 0 means the game ended early or a player wiped out
                    break 
                    
                # WTHOR coordinates: (Col * 10) + Row. (e.g., C4 = 34)
                # We convert this to 0-indexed column and row
                col = (b // 10) - 1
                row = (b % 10) - 1
                action = row * 8 + col
                moves.append(action)
                
            games.append(moves)
            
    return games

# --- 2. Static Evaluator ---

def evaluate_checkpoint(model, device, games):
    """
    Plays through the WTHOR games and tracks how often the AI's 
    top policy prediction matches the human expert's actual move.
    """
    correct_predictions = 0
    total_predictions = 0
    
    # Track accuracy by phase of the game
    phase_stats = {
        "Opening (1-20)":   {"correct": 0, "total": 0},
        "Midgame (21-40)":  {"correct": 0, "total": 0},
        "Endgame (41-60)":  {"correct": 0, "total": 0}
    }

    env = ReversiEnv()
    
    for game_idx, human_moves in enumerate(games):
        obs, info = env.reset()
        move_number = 0
        
        for human_action in human_moves:
            action_mask = info["action_mask"]
            
            # WTHOR dataset implicitly skips passes. 
            # If our environment says a pass is forced, we must execute it first
            # to keep our board synchronized with the WTHOR dataset.
            while np.array_equal(np.where(action_mask == 1)[0], [64]):
                obs, reward, terminated, truncated, info = env.step(64)
                action_mask = info["action_mask"]
                if terminated: break

            if env.pass_count >= 2:
                break # Game over

            # 1. Ask the AI what it would do in this exact position
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                policy, _ = model(obs_tensor)
                policy = policy.squeeze(0).cpu().numpy()
                
                # Mask out illegal moves and find the highest probability
                masked_policy = policy * action_mask
                ai_predicted_action = int(np.argmax(masked_policy))

            # 2. Check if the AI agreed with the human expert
            move_number += 1
            is_correct = (ai_predicted_action == human_action)
            
            correct_predictions += is_correct
            total_predictions += 1
            
            # Record phase stats
            if move_number <= 20: phase = "Opening (1-20)"
            elif move_number <= 40: phase = "Midgame (21-40)"
            else: phase = "Endgame (41-60)"
            
            phase_stats[phase]["total"] += 1
            phase_stats[phase]["correct"] += is_correct

            # 3. Force the environment to play the HUMAN's move 
            # (even if the AI disagreed) to keep following the real game
            obs, reward, terminated, truncated, info = env.step(human_action)

        if (game_idx + 1) % 100 == 0:
            print(f"Processed {game_idx + 1}/{len(games)} games...")

    # --- Print Final Results ---
    overall_accuracy = (correct_predictions / total_predictions) * 100
    print("\n" + "="*40)
    print(f"WTHOR STATIC EVALUATION RESULTS")
    print("="*40)
    print(f"Total Positions Evaluated: {total_predictions}")
    print(f"Overall AI Accuracy:       {overall_accuracy:.2f}%\n")
    
    for phase, stats in phase_stats.items():
        if stats["total"] > 0:
            acc = (stats["correct"] / stats["total"]) * 100
            print(f"{phase} Accuracy: {acc:.2f}%")
    print("="*40)


if __name__ == "__main__":
    # 1. Load the WTHOR file
    WTHOR_FILE = r"WTHOR/WTH_2025.wtb" 
    
    try:
        wthor_games = parse_wthor_file(WTHOR_FILE)
    except FileNotFoundError:
        print(f"Could not find '{WTHOR_FILE}'. Please check the path.")
        exit()

    # 2. Load your highest checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualHeadResNet().to(device)
    
    # Automatically find the latest checkpoint in the checkpoints folder
    files = glob.glob(r"checkpoints\reversi_bundle_game_*.pth")
    if not files:
        print("No checkpoints found in 'checkpoints/' directory.")
        exit()
        
    latest_checkpoint_bundle_path = max(files, key=os.path.getctime)
    print(f"\nLoading Model: {latest_checkpoint_bundle_path}")
    
    checkpoint_bundle = torch.load(latest_checkpoint_bundle_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint_bundle['model_state_dict'])
    model.eval()
    
    # 3. Run the Gauntlet
    print("\nStarting evaluation. This will test the raw policy network (no MCTS)...")
    evaluate_checkpoint(model, device, wthor_games)