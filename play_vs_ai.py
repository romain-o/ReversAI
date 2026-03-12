import pygame
import sys
import numpy as np
import time
from env import ReversiEnv  
import torch
from train import DualHeadResNet 
from mcts import MCTS 

# --- UI Configuration & Modern Palette ---
SQUARE_SIZE = 80
BOARD_SIZE = SQUARE_SIZE * 8
PANEL_WIDTH = 250  # Space for the new HUD
WINDOW_SIZE = (BOARD_SIZE + PANEL_WIDTH, BOARD_SIZE)

# Premium Colors
BOARD_GREEN = (39, 119, 73)      # Rich, calm green
LINE_COLOR = (25, 80, 48)        # Subtle dark green for the grid
PANEL_BG = (33, 37, 43)          # Slate gray/blue for the modern dashboard
TEXT_LIGHT = (220, 224, 232)     # Off-white for text
TEXT_MUTED = (130, 137, 151)     # Gray for secondary text
PIECE_BLACK = (30, 32, 34)       # Charcoal instead of pure black
PIECE_WHITE = (240, 242, 245)    # Eggshell instead of pure white
SHADOW_COLOR = (20, 60, 35, 120) # Translucent dark green for piece depth
HIGHLIGHT = (255, 255, 255, 60)  # Translucent white for valid moves
OVERLAY = (0, 0, 0, 180)         # Dark overlay for the end screen
YELLOW = (255, 215, 0)          # Gold for victory text

class LocalEvaluator:
    """A fast, synchronous bridge to evaluate boards without multiprocessing."""
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict(self, state):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            policy, value = self.model(state_tensor)
            policy = policy.squeeze(0).cpu().numpy()
            return {a: p for a, p in enumerate(policy)}, value.item()

def draw_board(screen, env, action_mask, font, large_font, is_human_turn, is_game_over, is_ai_thinking=False):
    # 1. Fill Backgrounds
    screen.fill(BOARD_GREEN, (0, 0, BOARD_SIZE, BOARD_SIZE))
    screen.fill(PANEL_BG, (BOARD_SIZE, 0, PANEL_WIDTH, BOARD_SIZE))
    
    # Calculate Scores dynamically
    black_bb = env.current_player_bb if env.is_black_turn else env.opp_bb
    white_bb = env.opp_bb if env.is_black_turn else env.current_player_bb
    black_score = black_bb.bit_count()
    white_score = white_bb.bit_count()

    # --- DRAW BOARD ---
    # Draw Grid Lines
    for i in range(1, 8):
        pygame.draw.line(screen, LINE_COLOR, (0, i * SQUARE_SIZE), (BOARD_SIZE, i * SQUARE_SIZE), 2)
        pygame.draw.line(screen, LINE_COLOR, (i * SQUARE_SIZE, 0), (i * SQUARE_SIZE, BOARD_SIZE), 2)

    # We need a surface with alpha channel for shadows and highlights
    transparent_surface = pygame.Surface((BOARD_SIZE, BOARD_SIZE), pygame.SRCALPHA)

    # Draw Pieces and Highlights
    for i in range(64):
        r, c = divmod(i, 8)
        center = (c * SQUARE_SIZE + SQUARE_SIZE // 2, r * SQUARE_SIZE + SQUARE_SIZE // 2)
        shadow_offset = (center[0] + 3, center[1] + 4)
        radius = SQUARE_SIZE // 2 - 8
        
        # Draw Pieces with Drop Shadows
        if black_bb & (1 << i):
            pygame.draw.circle(transparent_surface, SHADOW_COLOR, shadow_offset, radius)
            pygame.draw.circle(screen, PIECE_BLACK, center, radius)
        elif white_bb & (1 << i):
            pygame.draw.circle(transparent_surface, SHADOW_COLOR, shadow_offset, radius)
            pygame.draw.circle(screen, PIECE_WHITE, center, radius)
            
        # Draw Valid Move Indicators
        if action_mask[i] == 1 and is_human_turn and not is_game_over:
            pygame.draw.circle(transparent_surface, HIGHLIGHT, center, radius // 2)
            # Add a subtle ring
            pygame.draw.circle(transparent_surface, (255, 255, 255, 100), center, radius // 2, 2)

    # Blit the transparent elements onto the main screen
    screen.blit(transparent_surface, (0, 0))

    # --- DRAW HUD / DASHBOARD ---
    panel_center_x = BOARD_SIZE + (PANEL_WIDTH // 2)
    
    # Title
    title_text = large_font.render("REVERSAI", True, TEXT_LIGHT)
    screen.blit(title_text, title_text.get_rect(center=(panel_center_x, 40)))

    # Black Score Box
    pygame.draw.rect(screen, PIECE_BLACK, (BOARD_SIZE + 25, 100, 200, 60), border_radius=10)
    b_score_text = font.render(f"Black: {black_score}", True, PIECE_WHITE)
    screen.blit(b_score_text, b_score_text.get_rect(center=(panel_center_x, 130)))

    # White Score Box
    pygame.draw.rect(screen, PIECE_WHITE, (BOARD_SIZE + 25, 180, 200, 60), border_radius=10)
    w_score_text = font.render(f"White: {white_score}", True, PIECE_BLACK)
    screen.blit(w_score_text, w_score_text.get_rect(center=(panel_center_x, 210)))

    # Status / Turn Indicator
    if is_game_over:
        status = "Game Over"
        status_color = YELLOW
    elif is_ai_thinking:
        status = "AI is thinking..."
        status_color = TEXT_MUTED
    else:
        status = "Your Turn" if is_human_turn else "Waiting..."
        status_color = TEXT_LIGHT

    turn_text = font.render(status, True, status_color)
    screen.blit(turn_text, turn_text.get_rect(center=(panel_center_x, 320)))
    
    # Whose turn color visualizer
    if not is_game_over:
        turn_color = PIECE_BLACK if env.is_black_turn else PIECE_WHITE
        pygame.draw.circle(screen, turn_color, (panel_center_x, 380), 20)
        pygame.draw.circle(screen, TEXT_MUTED, (panel_center_x, 380), 21, 2) # Outline

    pygame.display.flip()

def main():
    pygame.init()
    
    # Setup Modern Fonts (Fallback to Pygame defaults if system fonts are missing)
    try:
        main_font = pygame.font.SysFont("Segoe UI, Helvetica, Arial", 28, bold=True)
        title_font = pygame.font.SysFont("Segoe UI, Helvetica, Arial", 42, bold=True)
        giant_font = pygame.font.SysFont("Segoe UI, Helvetica, Arial", 72, bold=True)
    except:
        main_font = pygame.font.Font(None, 36)
        title_font = pygame.font.Font(None, 48)
        giant_font = pygame.font.Font(None, 80)

    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("ReversAI - Grandmaster Edition")

    env = ReversiEnv()
    obs, info = env.reset()
    
    # --- Load the Trained Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ai_model = DualHeadResNet().to(device)
    
    checkpoint_path = r"checkpoints\reversi_bundle_game_65000.pth" 
    try:
        checkpoint_bundle = torch.load(checkpoint_path, map_location=device, weights_only=False)
        ai_model.load_state_dict(checkpoint_bundle['model_state_dict'])
    except Exception as e:
        print(f"Warning: Could not load model. Error: {e}")
        
    ai_model.eval() 
    print("AI Model loaded and ready!")
    
    mcts = MCTS(num_simulations=1000)
    evaluator = LocalEvaluator(ai_model, device)

    human_is_black = True
    game_over = False
    clock = pygame.time.Clock()

    while True:
        action_mask = info["action_mask"]
        is_human_turn = (env.is_black_turn == human_is_black)

        # Handle Automatic Passing
        if np.array_equal(np.where(action_mask == 1)[0], [64]) and not game_over:
            print("No valid moves. Auto-passing.")
            time.sleep(0.5) 
            obs, reward, terminated, truncated, info = env.step(64)
            game_over = terminated
            continue

        # Draw the Board
        draw_board(screen, env, action_mask, main_font, title_font, is_human_turn, game_over)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Human Turn (Mouse Click)
            if event.type == pygame.MOUSEBUTTONDOWN and is_human_turn and not game_over:
                x, y = event.pos
                # Only register clicks inside the board area
                if x < BOARD_SIZE:
                    col = x // SQUARE_SIZE
                    row = y // SQUARE_SIZE
                    action = row * 8 + col
                    
                    if action_mask[action] == 1:
                        obs, reward, terminated, truncated, info = env.step(action)
                        game_over = terminated

        # Handle AI Turn
        if not is_human_turn and not game_over:
            pygame.event.pump() 
            
            # Visually update the UI to show the AI is thinking BEFORE we block the thread
            draw_board(screen, env, action_mask, main_font, title_font, is_human_turn, game_over, is_ai_thinking=True)
            
            _, mcts_policy = mcts.search(env, evaluator, add_noise=False)
            action = int(np.argmax(mcts_policy))
            
            obs, reward, terminated, truncated, info = env.step(action)
            game_over = terminated

        # Handle Game Over Overlay
        if game_over:
            # Create a cinematic dark translucent overlay over the whole screen
            overlay = pygame.Surface(WINDOW_SIZE, pygame.SRCALPHA)
            overlay.fill(OVERLAY)
            screen.blit(overlay, (0, 0))
            
            black_bb = env.current_player_bb if env.is_black_turn else env.opp_bb
            white_bb = env.opp_bb if env.is_black_turn else env.current_player_bb
            
            black_score = black_bb.bit_count()
            white_score = white_bb.bit_count()
            
            if black_score > white_score:
                text = "Black Wins!"
            elif white_score > black_score:
                text = "White Wins!"
            else:
                text = "It's a Tie!"
                
            # Draw Game Over Text
            text_surf = giant_font.render(text, True, HIGHLIGHT[:3])
            text_rect = text_surf.get_rect(center=(WINDOW_SIZE[0]//2, WINDOW_SIZE[1]//2 - 40))
            
            score_surf = title_font.render(f"{black_score} - {white_score}", True, TEXT_LIGHT)
            score_rect = score_surf.get_rect(center=(WINDOW_SIZE[0]//2, WINDOW_SIZE[1]//2 + 30))
            
            screen.blit(text_surf, text_rect)
            screen.blit(score_surf, score_rect)
            pygame.display.flip()
            
            # Wait for user to close window
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

        clock.tick(30) 

if __name__ == "__main__":
    main()