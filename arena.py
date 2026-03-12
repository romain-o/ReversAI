import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
import time
import queue
import json
import os
import random

# Import your architecture and environment from your existing files
from train import DualHeadResNet, get_symmetries
from env import ReversiEnv
from mcts import MCTS

# =====================================================================
# --- 1. ARENA CONFIGURATION (EDIT THESE) ---
# =====================================================================
MODEL_A_PATH = r"checkpoints/reversi_model_game_31700.pth"  # The Old Champion
MODEL_B_PATH = r"checkpoints/reversi_bundle_game_65000.pth" # The New Challenger

NUM_MATCHES = 100       # Total games to play (Should be an even number)
MCTS_SIMULATIONS = 200  # Give them enough time to think deeply
NUM_WORKERS = 14        # Maximize your CPU threads
BATCH_SIZE = 16         # GPU Batch size

# Try to load FFO Openings for game diversity
try:
    with open("ffo_openings.json", "r", encoding='utf-8') as f:
        OPENING_BOOK = [entry["sequence"] for entry in json.load(f)]
    print(f"[Arena] Loaded {len(OPENING_BOOK)} FFO openings for match diversity.")
except FileNotFoundError:
    OPENING_BOOK = []
    print("[Arena] FFO Openings not found. Games will start from empty board with slight temperature noise.")

# =====================================================================
# --- 2. THE DUAL-BRAIN GPU BATCHER ---
# =====================================================================
def load_smart_model(path, device):
    """Universal loader that handles both legacy weights and new bundles."""
    model = DualHeadResNet().to(device)
    model.eval()
    try:
        data = torch.load(path, map_location=device, weights_only=False)
        if 'model_state_dict' in data:
            model.load_state_dict(data['model_state_dict']) # New Bundle
        else:
            model.load_state_dict(data) # Legacy
    except Exception as e:
        print(f"[!] FATAL: Could not load model at {path}. Error: {e}")
        exit()
    return model

def dual_gpu_evaluator(input_queue, output_pipes, path_a, path_b, batch_size=16):
    """Holds BOTH models on the GPU and routes the MCTS queries to the correct one."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GPU] Booting Dual-Evaluator on {device}...")
    
    model_a = load_smart_model(path_a, device)
    model_b = load_smart_model(path_b, device)
    print(f"[GPU] Both Champions successfully loaded into VRAM.")

    with torch.no_grad():
        while True:
            batch_a_states, batch_a_ids = [], []
            batch_b_states, batch_b_ids = [], []
            
            # Gather a batch
            total_gathered = 0
            while total_gathered < batch_size:
                try:
                    worker_id, state, model_idx = input_queue.get_nowait()
                    if model_idx == 0:
                        batch_a_states.append(state); batch_a_ids.append(worker_id)
                    else:
                        batch_b_states.append(state); batch_b_ids.append(worker_id)
                    total_gathered += 1
                except queue.Empty:
                    if total_gathered > 0:
                        break
                    else:
                        time.sleep(0.001)

            if total_gathered == 0:
                continue

            # Process Model A
            if batch_a_states:
                t_a = torch.tensor(np.array(batch_a_states), dtype=torch.float32).to(device)
                pol_a, val_a = model_a(t_a)
                pol_a, val_a = pol_a.cpu().numpy(), val_a.cpu().numpy()
                for i, w_id in enumerate(batch_a_ids):
                    output_pipes[w_id].send((pol_a[i], val_a[i].item()))

            # Process Model B
            if batch_b_states:
                t_b = torch.tensor(np.array(batch_b_states), dtype=torch.float32).to(device)
                pol_b, val_b = model_b(t_b)
                pol_b, val_b = pol_b.cpu().numpy(), val_b.cpu().numpy()
                for i, w_id in enumerate(batch_b_ids):
                    output_pipes[w_id].send((pol_b[i], val_b[i].item()))

class ArenaEvaluator:
    def __init__(self, worker_id, input_queue, pipe_conn, model_idx):
        self.worker_id = worker_id
        self.input_queue = input_queue
        self.pipe_conn = pipe_conn
        self.model_idx = model_idx # 0 for A, 1 for B

    def predict(self, state):
        self.input_queue.put((self.worker_id, state, self.model_idx))
        policy, value = self.pipe_conn.recv()
        return {a: p for a, p in enumerate(policy)}, value

# =====================================================================
# --- 3. THE ARENA WORKER ---
# =====================================================================
def arena_worker(worker_id, input_queue, pipe_conn, result_queue, num_games, model_a_is_black):
    """Plays a full game. model_a_is_black determines who goes first."""
    env = ReversiEnv()
    
    # Each model gets its own MCTS tree so they don't share thoughts
    eval_a = ArenaEvaluator(worker_id, input_queue, pipe_conn, model_idx=0)
    eval_b = ArenaEvaluator(worker_id, input_queue, pipe_conn, model_idx=1)
    
    for _ in range(num_games):
        obs, _ = env.reset()
        mcts_a = MCTS(num_simulations=MCTS_SIMULATIONS)
        mcts_b = MCTS(num_simulations=MCTS_SIMULATIONS)
        
        terminated = False # <-- Initialize to False here
        
        # 1. Inject FFO Opening (if available) to guarantee unique games
        if OPENING_BOOK:
            opening = random.choice(OPENING_BOOK)
            for move in opening:
                obs, _, terminated, _, _ = env.step(move)
                if terminated: break
        
        # (Deleted the env.is_game_over() line that was causing the crash)
        
        # 2. Main Game Loop
        while not terminated:
            # Determine whose turn it is
            is_model_a_turn = (env.is_black_turn == model_a_is_black)
            
            if is_model_a_turn:
                best_action, policy = mcts_a.search(env, eval_a)
            else:
                best_action, policy = mcts_b.search(env, eval_b)
            
            # Pure competitive play: Always pick the absolute best move (Argmax)
            # No Dirichlet noise, No temperature exploration. Pure strength.
            best_action = np.argmax(policy) 
            
            obs, _, terminated, _, _ = env.step(best_action)
            
        # 3. Game Over - Calculate the Winner
        black_score = env.current_player_bb.bit_count() if env.is_black_turn else env.opp_bb.bit_count()
        white_score = env.opp_bb.bit_count() if env.is_black_turn else env.current_player_bb.bit_count()
        
        if black_score > white_score:
            winner = "Model A" if model_a_is_black else "Model B"
        elif white_score > black_score:
            winner = "Model B" if model_a_is_black else "Model A"
        else:
            winner = "Draw"
            
        # Send result back to the main process
        result_queue.put({
            "winner": winner, 
            "model_a_was_black": model_a_is_black,
            "black_score": black_score,
            "white_score": white_score
        })

# =====================================================================
# --- 4. MAIN EXECUTION & DASHBOARD ---
# =====================================================================
if __name__ == "__main__":
    mp.set_start_method('spawn')
    print("="*60)
    print(" ⚔️  THE ALPHA-ARENA BENCHMARK  ⚔️ ")
    print("="*60)
    print(f"Model A (Champion): {MODEL_A_PATH.split('/')[-1]}")
    print(f"Model B (Challenger): {MODEL_B_PATH.split('/')[-1]}")
    print(f"Total Games: {NUM_MATCHES} ({NUM_MATCHES//2} as Black, {NUM_MATCHES//2} as White)")
    print("="*60)

    input_queue = mp.Queue()
    result_queue = mp.Queue()
    parent_pipes, child_pipes = [], []
    
    for _ in range(NUM_WORKERS):
        p, c = mp.Pipe()
        parent_pipes.append(p); child_pipes.append(c)

    # Boot GPU
    gpu_process = mp.Process(target=dual_gpu_evaluator, args=(input_queue, parent_pipes, MODEL_A_PATH, MODEL_B_PATH, BATCH_SIZE))
    gpu_process.start()

    # Distribute games evenly among workers
    games_per_worker = NUM_MATCHES // NUM_WORKERS
    remainder = NUM_MATCHES % NUM_WORKERS
    
    workers = []
    games_assigned = 0
    for i in range(NUM_WORKERS):
        games = games_per_worker + (1 if i < remainder else 0)
        
        # Half the time Model A plays Black, Half the time it plays White
        model_a_is_black = (games_assigned < NUM_MATCHES // 2) 
        
        p = mp.Process(target=arena_worker, args=(i, input_queue, child_pipes[i], result_queue, games, model_a_is_black))
        p.start()
        workers.append(p)
        games_assigned += games

    # Live Dashboard Loop
    results = {"Model A": 0, "Model B": 0, "Draw": 0}
    games_finished = 0
    
    try:
        while games_finished < NUM_MATCHES:
            res = result_queue.get()
            results[res["winner"]] += 1
            games_finished += 1
            
            # Terminal UI
            print(f"[{games_finished}/{NUM_MATCHES}] {res['winner']} won! "
                  f"(A was {'Black' if res['model_a_was_black'] else 'White'}) "
                  f"| Score: {max(res['black_score'], res['white_score'])} - {min(res['black_score'], res['white_score'])}")

        # Wait for clean shutdown
        for p in workers: p.join()
        gpu_process.terminate()

        # Final Report
        print("\n" + "="*60)
        print(" 🏆 TOURNAMENT FINAL RESULTS 🏆 ")
        print("="*60)
        print(f"Model A (Champion) Wins: {results['Model A']}")
        print(f"Model B (Challenger) Wins: {results['Model B']}")
        print(f"Draws: {results['Draw']}")
        print("="*60)
        
        win_rate = (results['Model B'] + (results['Draw']*0.5)) / NUM_MATCHES * 100
        print(f"Model B Winrate: {win_rate:.1f}%")
        
        if win_rate >= 55.0:
            print("\n✅ VERDICT: Model B is significantly stronger. Replace the champion!")
        elif win_rate <= 45.0:
            print("\n❌ VERDICT: Model B suffered catastrophic forgetting. Model A remains champion.")
        else:
            print("\n⚠️ VERDICT: Models are roughly equal. More training required.")

    except KeyboardInterrupt:
        print("\n[!] Tournament Interrupted.")
        for p in workers: p.terminate()
        gpu_process.terminate()