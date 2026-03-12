import numpy as np
import random
import torch
import time
from env import ReversiEnv
from train import DualHeadResNet
from mcts import MCTS

# =====================================================================
# --- 1. THE CLASSIC ALGORITHMS (GYM API COMPLIANT) ---
# =====================================================================

class RandomAgent:
    def get_action(self, env, action_mask):
        valid_moves = np.where(action_mask == 1)[0]
        return random.choice(valid_moves)

class GreedyAgent:
    def get_action(self, env, action_mask):
        valid_moves = np.where(action_mask == 1)[0]
        if len(valid_moves) == 1 and valid_moves[0] == 64:
            return 64 # Pass

        max_flips = -1
        best_moves = []
        
        for move in valid_moves:
            env_copy = ReversiEnv()
            env_copy.current_player_bb, env_copy.opp_bb = env.current_player_bb, env.opp_bb
            env_copy.is_black_turn, env_copy.pass_count = env.is_black_turn, env.pass_count
            
            score_before = env.current_player_bb.bit_count()
            env_copy.step(move) 
            score_after = env_copy.opp_bb.bit_count() 
            
            flips = score_after - score_before
            
            # THE STOCHASTIC TIE-BREAKER
            if flips > max_flips:
                max_flips = flips
                best_moves = [move]       # New absolute best, reset the list
            elif flips == max_flips:
                best_moves.append(move)   # Equally good, add to the pool
                
        return random.choice(best_moves)

class MinimaxAgent:
    def __init__(self, depth=4):
        self.depth = depth
        self.weights = np.array([
            100, -20,  10,   5,   5,  10, -20, 100,
            -20, -50,  -2,  -2,  -2,  -2, -50, -20,
             10,  -2,  -1,  -1,  -1,  -1,  -2,  10,
              5,  -2,  -1,  -1,  -1,  -1,  -2,   5,
              5,  -2,  -1,  -1,  -1,  -1,  -2,   5,
             10,  -2,  -1,  -1,  -1,  -1,  -2,  10,
            -20, -50,  -2,  -2,  -2,  -2, -50, -20,
            100, -20,  10,   5,   5,  10, -20, 100
        ])

    def evaluate(self, env, maximizing_player_is_black):
        black_bb = env.current_player_bb if env.is_black_turn else env.opp_bb
        white_bb = env.opp_bb if env.is_black_turn else env.current_player_bb
        
        black_score = sum(self.weights[i] for i in range(64) if black_bb & (1 << i))
        white_score = sum(self.weights[i] for i in range(64) if white_bb & (1 << i))
            
        return (black_score - white_score) if maximizing_player_is_black else (white_score - black_score)

    def minimax(self, env, depth, alpha, beta, maximizing_player, root_player_is_black, action_mask, terminated):
        if depth == 0 or terminated:
            return self.evaluate(env, root_player_is_black)

        valid_moves = np.where(action_mask == 1)[0]
        
        if maximizing_player:
            max_eval = -float('inf')
            for move in valid_moves:
                child_env = ReversiEnv()
                child_env.current_player_bb, child_env.opp_bb = env.current_player_bb, env.opp_bb
                child_env.is_black_turn, child_env.pass_count = env.is_black_turn, env.pass_count
                _, _, child_term, _, child_info = child_env.step(move)
                
                still_maximizing = (child_env.is_black_turn == env.is_black_turn)
                eval_score = self.minimax(child_env, depth - 1, alpha, beta, still_maximizing, root_player_is_black, child_info["action_mask"], child_term)
                
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha: break
            return max_eval
        else:
            min_eval = float('inf')
            for move in valid_moves:
                child_env = ReversiEnv()
                child_env.current_player_bb, child_env.opp_bb = env.current_player_bb, env.opp_bb
                child_env.is_black_turn, child_env.pass_count = env.is_black_turn, env.pass_count
                _, _, child_term, _, child_info = child_env.step(move)
                
                still_minimizing = (child_env.is_black_turn == env.is_black_turn)
                eval_score = self.minimax(child_env, depth - 1, alpha, beta, not still_minimizing, root_player_is_black, child_info["action_mask"], child_term)
                
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha: break
            return min_eval

    def get_action(self, env, action_mask):
        valid_moves = np.where(action_mask == 1)[0]
        if len(valid_moves) == 1 and valid_moves[0] == 64: return 64
        
        max_eval = -float('inf')
        best_moves = []
        root_player_is_black = env.is_black_turn
        
        for move in valid_moves:
            child_env = ReversiEnv()
            child_env.current_player_bb, child_env.opp_bb = env.current_player_bb, env.opp_bb
            child_env.is_black_turn, child_env.pass_count = env.is_black_turn, env.pass_count
            
            _, _, child_term, _, child_info = child_env.step(move)
            still_maximizing = (child_env.is_black_turn == root_player_is_black)
            eval_score = self.minimax(child_env, self.depth - 1, -float('inf'), float('inf'), still_maximizing, root_player_is_black, child_info["action_mask"], child_term)
            
            # THE STOCHASTIC TIE-BREAKER
            if eval_score > max_eval:
                max_eval = eval_score
                best_moves = [move]
            elif eval_score == max_eval:
                best_moves.append(move)
                
        return random.choice(best_moves)

# =====================================================================
# --- 2. ALPHAZERO SYNC EVALUATOR ---
# =====================================================================
class SyncEvaluator:
    """Evaluates the board immediately on the GPU without multiprocessing overhead."""
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict(self, state):
        with torch.no_grad():
            t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            p, v = self.model(t)
            return {a: prob for a, prob in enumerate(p.squeeze(0).cpu().numpy())}, v.item()

# =====================================================================
# --- 3. THE MATCHUP ARENA ---
# =====================================================================
if __name__ == "__main__":
    # --- CONFIGURATION ---
    GAMES_TO_PLAY = 20
    CHECKPOINT = r"checkpoints/reversi_bundle_game_65000.pth" # Update this to your latest
    
    # CHOOSE YOUR OPPONENT HERE: RandomAgent(), GreedyAgent(), or MinimaxAgent(depth=4)
    baseline_bot = MinimaxAgent(depth=4)
    baseline_name = baseline_bot.__class__.__name__
    # ---------------------

    print(f"\nLoading Neural Network: {CHECKPOINT}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualHeadResNet().to(device)
    bundle = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    
    # Smart Loader to handle both legacy and bundle checkpoints
    if 'model_state_dict' in bundle:
        model.load_state_dict(bundle['model_state_dict'])
    else:
        model.load_state_dict(bundle)
        
    model.eval()
    evaluator = SyncEvaluator(model, device)
    
    wins, losses, draws = 0, 0, 0
    
    print(f"\n--- TOURNAMENT: AlphaZero vs {baseline_name} ({GAMES_TO_PLAY} Games) ---")
    
    for game in range(GAMES_TO_PLAY):
        env = ReversiEnv()
        obs, info = env.reset()
        terminated = False
        mcts = MCTS(num_simulations=200) 
        
        # AlphaZero is Black for even games, White for odd games
        az_is_black = (game % 2 == 0)
        
        while not terminated:
            action_mask = info["action_mask"]
            
            if env.is_black_turn == az_is_black:
                # AlphaZero's Turn
                _, policy = mcts.search(env, evaluator, add_noise=False)
                action = int(np.argmax(policy))
            else:
                # Baseline Bot's Turn
                action = baseline_bot.get_action(env, action_mask)
                
            obs, _, terminated, _, info = env.step(action)
            
        # Calculate Winner
        b_score = env.current_player_bb.bit_count() if env.is_black_turn else env.opp_bb.bit_count()
        w_score = env.opp_bb.bit_count() if env.is_black_turn else env.current_player_bb.bit_count()
        
        if b_score > w_score:
            az_won = az_is_black
            draw = False
        elif w_score > b_score:
            az_won = not az_is_black
            draw = False
        else:
            draw = True
            
        if draw:
            draws += 1
            res = "DRAW"
        elif az_won:
            wins += 1
            res = "WIN"
        else:
            losses += 1
            res = "LOSS"
            
        color = "Black" if az_is_black else "White"
        print(f"Game {game+1:02d} (AZ as {color:5s}): {res} | Score: AZ {b_score if az_is_black else w_score} - {w_score if az_is_black else b_score} {baseline_name}")

    print("\n" + "="*40)
    print(f" FINAL RESULTS vs {baseline_name}")
    print("="*40)
    print(f" AlphaZero Wins:   {wins}")
    print(f" AlphaZero Losses: {losses}")
    print(f" Draws:            {draws}")
    print(f" Win Rate:         {((wins + (draws*0.5)) / GAMES_TO_PLAY) * 100:.1f}%")
    print("="*40)