import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Bitboard Masks to prevent rays from wrapping around the board
NOT_A_FILE = 0xFEFEFEFEFEFEFEFE
NOT_H_FILE = 0x7F7F7F7F7F7F7F7F

class ReversiEnv(gym.Env):
    """
    High-performance Reversi Environment using Bitboards.
    Actions: 0-63 correspond to board squares. 64 is the PASS action.
    """
    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(self):
        super().__init__()
        # 64 squares + 1 Pass action
        self.action_space = spaces.Discrete(65)
        
        # Observation: [Current Player Pieces, Opponent Pieces, Turn Indicator]
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, 8, 8), dtype=np.int8)
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initial Reversi Setup
        # Black pieces at D5 (35) and E4 (28)
        self.black_bb = (1 << 28) | (1 << 35)
        # White pieces at D4 (27) and E5 (36)
        self.white_bb = (1 << 27) | (1 << 36)
        
        self.is_black_turn = True
        self.current_player_bb = self.black_bb
        self.opp_bb = self.white_bb
        
        self.pass_count = 0
        
        return self._get_obs(), self._get_info()

    def is_game_over(self):
        """Checks if the game has ended."""
        if self.pass_count >= 2:
            return True
        if (self.current_player_bb | self.opp_bb) == 0xFFFFFFFFFFFFFFFF:
            return True
        return False

    def _get_obs(self):
        """Converts the bitboards into a 3x8x8 NumPy array for the Neural Network."""
        obs = np.zeros((3, 8, 8), dtype=np.int8)
        for i in range(64):
            r, c = divmod(i, 8)
            if self.current_player_bb & (1 << i):
                obs[0, r, c] = 1
            if self.opp_bb & (1 << i):
                obs[1, r, c] = 1
                
        # Fill the 3rd channel with 1s if Black's turn, 0s if White's turn
        obs[2, :, :] = 1 if self.is_black_turn else 0
        return obs

    def _get_info(self):
        """Provides the action mask for the current state."""
        legal_bb = self._get_valid_moves(self.current_player_bb, self.opp_bb)
        mask = np.zeros(65, dtype=np.int8)
        
        has_moves = False
        for i in range(64):
            if legal_bb & (1 << i):
                mask[i] = 1
                has_moves = True
                
        if not has_moves:
            mask[64] = 1 # Allow PASS only if no valid moves exist
            
        return {"action_mask": mask}

    def _get_valid_moves(self, player_bb, opp_bb):
        """Finds all legal moves using bitwise ray-casting."""
        empty = ~(player_bb | opp_bb) & 0xFFFFFFFFFFFFFFFF
        legal_moves = 0

        # Directions: (Shift Amount, Wrap Mask)
        directions = [
            (1, NOT_A_FILE),          # Right
            (-1, NOT_H_FILE),         # Left
            (8, 0xFFFFFFFFFFFFFFFF),  # Down
            (-8, 0xFFFFFFFFFFFFFFFF), # Up
            (9, NOT_A_FILE),          # Down-Right
            (7, NOT_H_FILE),          # Down-Left
            (-7, NOT_A_FILE),         # Up-Right
            (-9, NOT_H_FILE)          # Up-Left
        ]

        for shift, mask in directions:
            if shift > 0:
                candidates = (player_bb << shift) & mask & opp_bb
            else:
                candidates = (player_bb >> (-shift)) & mask & opp_bb

            while candidates != 0:
                if shift > 0:
                    shifted = (candidates << shift) & mask
                else:
                    shifted = (candidates >> (-shift)) & mask
                
                legal_moves |= (shifted & empty)
                candidates = shifted & opp_bb

        return legal_moves

    def _apply_move(self, action, player_bb, opp_bb):
        """Executes a move and returns the updated bitboards."""
        move_bb = 1 << action
        flipped = 0

        directions = [
            (1, NOT_A_FILE), (-1, NOT_H_FILE), (8, 0xFFFFFFFFFFFFFFFF),
            (-8, 0xFFFFFFFFFFFFFFFF), (9, NOT_A_FILE), (7, NOT_H_FILE),
            (-7, NOT_A_FILE), (-9, NOT_H_FILE)
        ]

        for shift, mask in directions:
            ray_flipped = 0
            if shift > 0:
                curr = (move_bb << shift) & mask
            else:
                curr = (move_bb >> (-shift)) & mask

            while (curr & opp_bb) != 0:
                ray_flipped |= curr
                if shift > 0:
                    curr = (curr << shift) & mask
                else:
                    curr = (curr >> (-shift)) & mask

            if (curr & player_bb) != 0:
                flipped |= ray_flipped

        new_player_bb = player_bb | move_bb | flipped
        new_opp_bb = opp_bb & ~flipped
        return new_player_bb, new_opp_bb

    def step(self, action):
        action = int(action)
        info = self._get_info()
        
        # 1. Illegal Move Penalty
        if info["action_mask"][action] == 0:
            return self._get_obs(), -1.0, True, False, {"error": "Illegal move"}

        # 2. Execute Move
        if action == 64:
            self.pass_count += 1
        else:
            self.pass_count = 0
            self.current_player_bb, self.opp_bb = self._apply_move(
                action, self.current_player_bb, self.opp_bb
            )

        # 3. Swap Turns
        self.current_player_bb, self.opp_bb = self.opp_bb, self.current_player_bb
        self.is_black_turn = not self.is_black_turn

        # 4. Check Termination & Reward
        terminated = False
        reward = 0.0

        if self.pass_count >= 2 or (self.current_player_bb | self.opp_bb) == 0xFFFFFFFFFFFFFFFF:
            terminated = True
            
            # Use bit_count() (Available in Python 3.10+)
            next_player_score = self.current_player_bb.bit_count()
            just_moved_score = self.opp_bb.bit_count()

            if just_moved_score > next_player_score:
                reward = 1.0  # The action taken resulted in a win
            elif just_moved_score < next_player_score:
                reward = -1.0 # The action taken resulted in a loss

        return self._get_obs(), reward, terminated, False, self._get_info()

    def render(self):
        """Prints the board to the terminal."""
        board = np.full(64, '.')
        for i in range(64):
            if self.current_player_bb & (1 << i):
                board[i] = 'B' if self.is_black_turn else 'W'
            elif self.opp_bb & (1 << i):
                board[i] = 'W' if self.is_black_turn else 'B'

        print("\n  0 1 2 3 4 5 6 7")
        for r in range(8):
            row_str = f"{r} " + " ".join(board[r*8:(r+1)*8])
            print(row_str)
        print()
        
    def get_state(self):
        """Ultra-fast snapshot of the current board."""
        return (self.current_player_bb, self.opp_bb, self.is_black_turn, self.pass_count)

    def set_state(self, state):
        """Restores the board instantly."""
        self.current_player_bb, self.opp_bb, self.is_black_turn, self.pass_count = state