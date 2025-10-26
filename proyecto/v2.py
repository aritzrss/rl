#!/usr/bin/env python3
"""
hideandseek_env_reset.py

Clean reset: Gymnasium-compatible Hide & Seek environment with:
- Deterministic 30x30 DFS maze
- Player-controlled Hider (W/A/S/D)
- Automatic Seeker (BFS chase)
- 3-minute timer rendered on right panel (mm:ss)
- Suitable for headless training (render_mode=None) or play (render_mode="human")
"""

import random
import time
from collections import deque
import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
import os
import optuna
import multiprocessing

# ---------------- CONFIG ----------------
GRID_SIZE = 30
CELL_SIZE = 20
MAZE_PIXELS = GRID_SIZE * CELL_SIZE
SIDE_MARGIN = 80
WINDOW_WIDTH = MAZE_PIXELS + SIDE_MARGIN
WINDOW_HEIGHT = MAZE_PIXELS

PLAYER_COLOR = (220, 40, 40)   # red
SEEKER_COLOR = (40, 110, 240)  # blue
WALL_COLOR = (40, 40, 40)
FLOOR_COLOR = (230, 230, 230)
PANEL_COLOR = (245, 245, 245)
TEXT_COLOR = (10, 10, 10)

FPS = 60
PLAYER_MOVE_COOLDOWN = 0.12     # seconds between player moves while holding
SEEKER_MOVE_INTERVAL = 0.12    # seeker moves every 0.12s (fast)
GAME_DURATION = 180            # seconds (3 minutes)
MAZE_DOOR_DENSITY = 0.12
MAZE_SEED = 42
VISIBILITY_RADIUS = 6
MAX_STEPS = 400
# ----------------------------------------


def generate_static_maze(grid_size=GRID_SIZE, door_density=MAZE_DOOR_DENSITY, seed=MAZE_SEED):
    """Deterministic DFS maze with optional small 'door' openings between corridors."""
    maze = [[1 for _ in range(grid_size)] for _ in range(grid_size)]

    def carve(r, c):
        dirs = [(2, 0), (-2, 0), (0, 2), (0, -2)]
        random.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 1 <= nr < grid_size - 1 and 1 <= nc < grid_size - 1 and maze[nr][nc] == 1:
                maze[r + dr // 2][c + dc // 2] = 0
                maze[nr][nc] = 0
                carve(nr, nc)

    random.seed(seed)
    maze[1][1] = 0
    carve(1, 1)

    # Add small number of doors to create more turns/connections
    for r in range(1, grid_size - 1):
        for c in range(1, grid_size - 1):
            if maze[r][c] == 1:
                vert = (maze[r - 1][c] == 0 and maze[r + 1][c] == 0)
                horiz = (maze[r][c - 1] == 0 and maze[r][c + 1] == 0)
                if (vert or horiz) and random.random() < door_density:
                    maze[r][c] = 0

    return np.array(maze, dtype=np.int8)


def bfs_next_step(start, goal, walls_set, grid_size):
    """Return next cell (r,c) along a BFS shortest path from start -> goal, or start if unreachable."""
    if start == goal:
        return start
    q = deque([start])
    parents = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        r, c = cur
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size and (nr, nc) not in parents and (nr, nc) not in walls_set:
                parents[(nr, nc)] = cur
                q.append((nr, nc))
    if goal not in parents:
        return start
    # backtrack to first step
    node = goal
    while parents[node] != start and parents[node] is not None:
        node = parents[node]
    return node

class HideAndSeekEnv(gym.Env):
    """Gymnasium environment for Hide & Seek."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        self.grid_size = GRID_SIZE
        self.cell_size = CELL_SIZE
        self.window_size = MAZE_PIXELS
        self.side_margin = SIDE_MARGIN

        self.maze = generate_static_maze(self.grid_size)
        self.walls = {(r, c) for r in range(self.grid_size) for c in range(self.grid_size) if self.maze[r, c] == 1}

        # actions: 0=up,1=down,2=left,3=right
        self._actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low=np.array([-1, -1, 0, 0, 0, 0], dtype=np.int32),
            high=np.array([1, 1, 1, 1, 1, 1], dtype=np.int32),
            dtype=np.int32,
        )

        self.visibility_radius = VISIBILITY_RADIUS
        self.max_steps = MAX_STEPS

        # rendering attributes
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
        self.big_font = None
        self.play_start_time = None

        # state
        self.seeker_pos = None
        self.hider_pos = None
        self.steps = 0

        # ensure deterministic initial state
        self.reset()

    # ---------------- Gym API ----------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        free_cells = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) if (r, c) not in self.walls]
        # deterministic placements: first free cell and last free cell
        self.seeker_pos = free_cells[0]
        self.hider_pos = free_cells[-1]
        self.steps = 0
        return self._get_obs(), {}

    def shortest_path_distance(self, start, goal):
        """Return BFS path distance accounting for walls."""
        if start == goal:
            return 0
        visited = set([start])
        queue = deque([(start, 0)])
        while queue:
            (x, y), dist = queue.popleft()
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) == goal:
                    return dist + 1
                if (0 <= nx < self.grid_size and 0 <= ny < self.grid_size
                        and (nx, ny) not in self.walls and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append(((nx, ny), dist + 1))
        return self.grid_size**2  # unreachable

    def step(self, seeker_action, hider_action=None, auto_hider=True):
        """
        seeker_action: discrete action for the seeker (0..3)
        hider_action: optional manual action for hider (0..3).
        auto_hider: if False and hider_action is None, hider will NOT move this step (for manual play).
        Returns: obs, reward, terminated, truncated, info
        """
        self.steps += 1

        # --- Seeker movement ---
        old_pos = self.seeker_pos
        new_pos = self._move(self.seeker_pos, seeker_action)

        hit_wall = (new_pos == old_pos)
        self.seeker_pos = new_pos

        # --- Hider movement ---
        if hider_action is not None:
            self.hider_pos = self._move(self.hider_pos, hider_action)
        elif auto_hider:
            self.hider_pos = self._smart_hider_move()
        # else: hider stays still (manual play)

        # --- Reward shaping ---
        d = self.shortest_path_distance(self.seeker_pos, self.hider_pos)
        if hasattr(self, "_prev_distance"):
            # reward for reducing distance
            if d < self._prev_distance:
                reward = +0.05
            elif d > self._prev_distance:
                reward = -0.05
            else:
                reward = -0.01
        else:
            reward = -0.01

        self._prev_distance = d
        truncated = False
        terminated = False

        # Big rewards for catching hider / failure
        if d < 2:
            reward = 1.0
            terminated = True
        elif self.steps >= self.max_steps:
            reward = -1.0
            terminated = True
        elif hit_wall:
            reward = -0.2  # penalty for bumping wall

        # Optionally encourage approaching the hider
        elif d < self._manhattan(old_pos, self.hider_pos):
            reward += 0.02  # positive if moved closer

        return self._get_obs(), float(reward), terminated, truncated, {}

    def render(self):
        """Render maze, seeker vision overlay, seeker and hider, and the timer on right panel."""
        if self.render_mode != "human":
            return

        if self.screen is None:
            pygame.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption("Hide & Seek (Gym Env)")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
            self.big_font = pygame.font.Font(None, 64)

        # draw panel background
        self.screen.fill(PANEL_COLOR)

        # draw maze floor area
        pygame.draw.rect(self.screen, FLOOR_COLOR, (0, 0, MAZE_PIXELS, MAZE_PIXELS))

        # walls
        for (r, c) in self.walls:
            pygame.draw.rect(self.screen, WALL_COLOR, (c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size))

        # draw hider and seeker (rectangles with small padding)
        hr, hc = self.hider_pos
        sr, sc = self.seeker_pos
        pygame.draw.rect(self.screen, PLAYER_COLOR, (hc * self.cell_size + 2, hr * self.cell_size + 2, self.cell_size - 4, self.cell_size - 4))
        pygame.draw.rect(self.screen, SEEKER_COLOR, (sc * self.cell_size + 2, sr * self.cell_size + 2, self.cell_size - 4, self.cell_size - 4))

        pygame.draw.rect(self.screen, (30, 30, 30), (MAZE_PIXELS, 0, self.side_margin, MAZE_PIXELS))
        # draw timer in right panel (mm:ss)
        elapsed = int(self.steps * PLAYER_MOVE_COOLDOWN if hasattr(self, "player_timer_mode") else 0)
        # better: use real time since env doesn't track play start; allow play loop to draw timer instead
        # We'll expose a small helper: if self.play_start_time exists, use it, otherwise compute from steps.
        if hasattr(self, "play_start_time"):
            remaining = max(0, GAME_DURATION - int(time.time() - self.play_start_time))
        else:
            remaining = GAME_DURATION  # default if no play_start_time set

        mins, secs = divmod(remaining, 60)
        timer_text = f"{mins:02}:{secs:02}"
        text_surf = self.font.render(timer_text, True, TEXT_COLOR)
        text_rect = text_surf.get_rect(center=(MAZE_PIXELS + self.side_margin // 2, 40))
        self.screen.blit(text_surf, text_rect)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None

    # -------------- Internal helpers --------------
    def _get_obs(self):
        """
        Compact observation:
        - sign(dx), sign(dy): rough direction of hider
        - four wall flags around seeker (up, down, left, right)
        """
        dx = self.hider_pos[0] - self.seeker_pos[0]
        dy = self.hider_pos[1] - self.seeker_pos[1]

        sx = np.sign(dx)
        sy = np.sign(dy)

        x, y = self.seeker_pos
        wall_flags = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size or (nx, ny) in self.walls:
                wall_flags.append(1)
            else:
                wall_flags.append(0)

        return np.array([sx, sy, *wall_flags], dtype=np.int32)
    
    def _move(self, pos, action):
        dr, dc = self._actions[action]
        nr, nc = pos[0] + dr, pos[1] + dc
        if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size and (nr, nc) not in self.walls:
            return (nr, nc)
        return pos

    def _wall_flags(self, pos):
        r, c = pos
        flags = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= self.grid_size or nc < 0 or nc >= self.grid_size or (nr, nc) in self.walls:
                flags.append(1)
            else:
                flags.append(0)
        return tuple(flags)

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def has_line_of_sight(self, a, b):
        """Bresenham-like check (grid) constrained by visibility radius."""
        (x0, y0), (x1, y1) = a, b
        if self._manhattan(a, b) > self.visibility_radius:
            return False
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1
        err = dx - dy
        while True:
            if (x, y) != (x0, y0) and (x, y) != (x1, y1) and (x, y) in self.walls:
                return False
            if (x, y) == (x1, y1):
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return True

    def _smart_hider_move(self):
        """
        Evasive hider:
        - If visible, move away from seeker.
        - Otherwise wander randomly with small chance to pause.
        """
        possible_moves = []
        for a in self._actions.values():
            new = (self.hider_pos[0] + a[0], self.hider_pos[1] + a[1])
            if 0 <= new[0] < self.grid_size and 0 <= new[1] < self.grid_size and new not in self.walls:
                possible_moves.append(new)

        if not possible_moves:
            return self.hider_pos

        # chance to stay still
        if random.random() < 0.15:
            return self.hider_pos

        # if visible, move away from seeker
        if self.has_line_of_sight(self.seeker_pos, self.hider_pos):
            best = max(possible_moves, key=lambda p: self._manhattan(p, self.seeker_pos))
            return best

        # otherwise, move randomly
        return random.choice(possible_moves)

class SARSAAgent:
    def __init__(self, action_size, alpha=0.1, gamma=0.9,
                 epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.05):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.Q = {}
        self.action_size = action_size

    def choose_action(self, s, env=None):
        """
        Choose an action using ε-greedy policy.
        Compatible with calls like choose_action(s) or choose_action(s, env).
        """
        # Infer number of actions safely
        n_actions = env.action_space.n if env else 4

        if s not in self.Q:
            self.Q[s] = np.zeros(n_actions)

        if np.random.rand() < self.epsilon:
            return np.random.randint(n_actions)
        return int(np.argmax(self.Q[s]))

    def update(self, s, a, r, ns, na):
        """SARSA update rule."""
        s, ns = tuple(s), tuple(ns)
        if ns not in self.Q:
            self.Q[ns] = np.zeros(self.action_size)
        self.Q[s][a] += self.alpha * (
            r + self.gamma * self.Q[ns][na] - self.Q[s][a]
        )

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

import os
import numpy as np

def train_sarsa(episodes=50000, save_path="q_table_sarsa.npy",
                alpha=0.1, gamma=0.95, epsilon=1.0,
                epsilon_decay=0.9995, epsilon_min=0.05):
    """
    Train SARSA agent to control the seeker.
    Automatically loads Optuna best parameters if available (unless manually overridden).
    The hider moves automatically each step (auto_hider=True).
    """

    # ✅ Load Optuna best parameters if they exist
    best_params = {}
    if os.path.exists("best_sarsa_params.npy"):
        try:
            best_params = np.load("best_sarsa_params.npy", allow_pickle=True).item()
            print(f"📦 Found best Optuna parameters: {best_params}")
        except Exception as e:
            print(f"⚠️ Could not load best_sarsa_params.npy: {e}")

    # ✅ Merge with manually provided ones (manual args override Optuna)
    alpha = best_params.get("alpha", alpha)
    gamma = best_params.get("gamma", gamma)
    epsilon_decay = best_params.get("epsilon_decay", epsilon_decay)
    epsilon_min = best_params.get("epsilon_min", epsilon_min)

    print(f"\n🚀 Training SARSA agent for {episodes} episodes...")
    print(f"   Using parameters: α={alpha:.4f}, γ={gamma:.4f}, "
          f"ε_decay={epsilon_decay:.6f}, ε_min={epsilon_min:.3f}")

    env = HideAndSeekEnv(render_mode=None)
    agent = SARSAAgent(env.grid_size, alpha=alpha, gamma=gamma,
                       epsilon=epsilon, epsilon_decay=epsilon_decay,
                       epsilon_min=epsilon_min)

    total_rewards = []

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        s = tuple(obs)
        a = agent.choose_action(s, env)
        done = False
        total_r = 0

        while not done:
            next_obs, r, done, _, _ = env.step(a, hider_action=None, auto_hider=True)
            ns = tuple(next_obs)
            na = agent.choose_action(ns, env)

            agent.update(s, a, r, ns, na)
            agent.decay_epsilon()

            s, a = ns, na
            total_r += r

        total_rewards.append(total_r)

        # 🟢 Logging
        if ep % 1000 == 0:
            avg_r = np.mean(total_rewards[-1000:])
            print(f"Ep {ep}/{episodes} | Avg Reward={avg_r:.3f} | ε={agent.epsilon:.3f}")

        # 💾 Save checkpoint every 10k episodes
        if ep % 10000 == 0:
            np.save(save_path, agent.Q, allow_pickle=True)
            print(f"💾 Saved checkpoint at {save_path} ({len(agent.Q)} states)")

    # ✅ Final save
    np.save(save_path, agent.Q, allow_pickle=True)
    print(f"\n✅ Training complete! Model saved to '{save_path}'.")
    env.close()


def objective(trial):
    """Single Optuna trial — trains a SARSA agent briefly and returns avg reward."""
    alpha = trial.suggest_float("alpha", 0.01, 0.3, log=True)
    gamma = trial.suggest_float("gamma", 0.7, 0.99)
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.995, 0.99999)
    epsilon_min = trial.suggest_float("epsilon_min", 0.01, 0.1)

    env = HideAndSeekEnv(render_mode=None)
    agent = SARSAAgent(env.grid_size, alpha=alpha, gamma=gamma,
                       epsilon=1.0, epsilon_decay=epsilon_decay,
                       epsilon_min=epsilon_min)

    total_rewards = []
    episodes = 1000  # small for speed
    for ep in range(episodes):
        obs, _ = env.reset()
        s = tuple(obs)
        a = agent.choose_action(s, env)
        done = False
        ep_reward = 0

        while not done:
            ns, r, done, _, _ = env.step(a, auto_hider=True)
            ns = tuple(ns)
            na = agent.choose_action(ns, env)
            agent.update(s, a, r, ns, na)
            s, a = ns, na
            ep_reward += r

        agent.decay_epsilon()
        total_rewards.append(ep_reward)

    env.close()
    avg_reward = np.mean(total_rewards[-200:])
    return avg_reward

def run_optuna_search(n_trials=15):
    """Run Optuna search for best SARSA hyperparameters, then train final model."""
    print("\n🧠 Running Optuna hyperparameter search for SARSA agent...")
    n_jobs = max(1, multiprocessing.cpu_count() - 1)
    print(f"⚙️ Using {n_jobs} parallel workers for optimization...")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    best_params = study.best_params
    best_score = study.best_value
    print(f"\n✅ Optuna complete! Best avg reward: {best_score:.3f}")
    print(f"🏆 Best params: {best_params}")

    np.save("best_sarsa_params.npy", best_params, allow_pickle=True)



# ----------------- Play wrapper (standalone) -----------------
def play_interactive():
    """
    Manual play: you control the hider with W/A/S/D.
    The seeker moves automatically using BFS, but the hider only moves when you press keys.
    """

    # load trained model before loop
    agent = train_sarsa(HideAndSeekEnv(render_mode=None))  # or load from file

    pygame.init()
    pygame.font.init()

    env = HideAndSeekEnv(render_mode="human")
    obs, _ = env.reset()

    env.play_start_time = time.time()
    env.render()

    last_hider_move = 0.0
    last_seeker_move = 0.0

    running = True
    game_result = None

    print("Play: W/A/S/D to move, ESC to quit. Survive 3 minutes to win.")

    while running:
        now = time.time()

        # events (safe — pygame initialized and display exists)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # read keyboard (do NOT use event KEYDOWN alone; allow held keys)
        keys = pygame.key.get_pressed()
        # decide hider action only when cooldown passed AND key pressed
        hider_action = None
        if now - last_hider_move >= PLAYER_MOVE_COOLDOWN:
            if keys[pygame.K_w]:
                hider_action = 0  # Up
            elif keys[pygame.K_s]:
                hider_action = 1  # Down
            elif keys[pygame.K_a]:
                hider_action = 2  # Left
            elif keys[pygame.K_d]:
                hider_action = 3  # Right

        # inside loop
        state = env._get_obs()
        seeker_action = agent.choose_action(state)

        # Move seeker at its interval, WITHOUT running the hider AI (auto_hider=False)
        if now - last_seeker_move >= SEEKER_MOVE_INTERVAL:
            # pass auto_hider=False so the hider DOES NOT move automatically on seeker ticks
            obs, reward, terminated, truncated, info = env.step(seeker_action, hider_action=None, auto_hider=False)
            last_seeker_move = now
        # Apply hider movement only when player pressed a key (separate call).
        # We call env.step with a dummy seeker_action (0) but auto_hider=False so only hider moves.
        if hider_action is not None and now - last_hider_move >= PLAYER_MOVE_COOLDOWN:
            # use the current seeker_action as dummy (or 0); auto_hider=False prevents AI movement.
            obs, reward, terminated, truncated, info = env.step(seeker_action, hider_action=hider_action, auto_hider=False)
            last_hider_move = now

        # render after moves
        env.render()

        # draw timer on right panel using env.play_start_time
        if hasattr(env, "play_start_time"):
            remaining = max(0, GAME_DURATION - int(time.time() - env.play_start_time))
        else:
            remaining = GAME_DURATION
        mins, secs = divmod(remaining, 60)
        timer_text = f"{mins:02}:{secs:02}"
        if env.screen and env.font:
            # clear previous timer area (optional: redraw panel in env.render instead)
            env.screen.blit(env.font.render(timer_text, True, TEXT_COLOR), (MAZE_PIXELS + 10, 10))
            pygame.display.flip()

        # termination checks
        if env.seeker_pos == env.hider_pos:
            game_result = "lose"
            running = False
        elif remaining <= 0:
            game_result = "win"
            running = False

        env.clock.tick(FPS)

    # show end overlay
    if env.screen and env.big_font:
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        env.screen.blit(overlay, (0, 0))
        msg = "YOU WIN!" if game_result == "win" else "GAME OVER"
        color = (20, 160, 20) if game_result == "win" else (200, 40, 40)
        surf = env.big_font.render(msg, True, color)
        rect = surf.get_rect(center=(WINDOW_WIDTH // 2 - SIDE_MARGIN // 3, WINDOW_HEIGHT // 2))
        env.screen.blit(surf, rect)
        pygame.display.flip()
        time.sleep(2)

    env.close()
    print("Exiting play.")

def play_against_bot(model_path="q_table_sarsa.npy"):
    """Play as the hider (manual) against a trained SARSA seeker."""

    if not pygame.get_init():
        pygame.init()
    if not pygame.font.get_init():
        pygame.font.init()

    # Load trained SARSA model
    try:
        Q = np.load(model_path, allow_pickle=True).item()
        print(f"✅ Loaded trained SARSA model ({len(Q)} states) from '{model_path}'.")
    except Exception as e:
        print(f"⚠️ Could not load model: {e}")
        return

    # Initialize environment and seeker agent
    env = HideAndSeekEnv(render_mode="human")
    agent = SARSAAgent(env.grid_size)
    agent.Q = Q
    clock = pygame.time.Clock()
    env.play_start_time = time.time()

    print("🎮 PLAY MODE (Hider = You, Seeker = Bot). Use W/A/S/D to move. ESC to quit.")

    # Timer setup
    start_time = time.time()
    GAME_DURATION = 180  # 3 minutes
    font = pygame.font.Font(None, 36)

    # Game loop
    done = False
    running = True

    while running and not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # --- Player (Hider) Input ---
        keys = pygame.key.get_pressed()
        hider_action = None
        if keys[pygame.K_w]:
            hider_action = 0  # up
        elif keys[pygame.K_s]:
            hider_action = 1  # down
        elif keys[pygame.K_a]:
            hider_action = 2  # left
        elif keys[pygame.K_d]:
            hider_action = 3  # right

        # --- Seeker (SARSA Bot) Chooses Action ---
        state = tuple(env._get_obs())
        seeker_action = agent.choose_action(state, env)

        # --- Step Environment ---
        obs, reward, terminated, truncated, _ = env.step(
            seeker_action,
            hider_action=hider_action,
            auto_hider=False  # disable auto-hider movement
        )

        # --- Render Everything ---
        env.render()

        # --- Timer (top-right panel) ---
        elapsed = int(time.time() - start_time)
        remaining = max(0, GAME_DURATION - elapsed)

        # --- Win / Loss conditions ---
        if remaining <= 0:
            print("⏰ Time’s up! You win!")
            time.sleep(2)
            done = True
        elif env.seeker_pos == env.hider_pos:
            print("😱 The seeker caught you!")
            time.sleep(2)
            done = True

        clock.tick(15)

    pygame.quit()
    env.close()
    print("👋 Exiting play mode.")



def main_menu():
    while True:
        print("\n====== HIDE & SEEK RL MENU ======")
        print("1️⃣ Train Optuna (Auto-tuned)")
        print("2️⃣ Train and Save SARSA Model")
        print("3️⃣  Play Against Trained Bot (You = Hider)")
        print("4️⃣  Play Interactive Sandbox (Debug Mode)")
        print("5️⃣ Exit")

        choice = input("Select option: ").strip()

        # -------------------------
        # 1️⃣ TRAIN SARSA MODEL
        # -------------------------
        if choice == "1":
            run_optuna_search(n_trials=15)

        # -------------------------
        # 1️⃣ TRAIN SARSA MODEL
        # -------------------------
        if choice == "2":
            episode = 100000
            print(f"🚀 Starting training for {episode} episodes...")
            train_sarsa(episodes=episode)

        # -------------------------
        # 2️⃣ PLAY AGAINST BOT
        # -------------------------
        elif choice == "3":
            print("\n🎮 Starting game against trained SARSA bot...")

            model_path = "q_table_sarsa.npy"
            if not os.path.exists(model_path):
                print("⚠️ No trained model found! Please train the SARSA agent first.")
                continue

            print("✅ Loaded trained model. Launching game window...")
            play_against_bot(model_path)

        # -------------------------
        # 3️⃣ PLAY INTERACTIVE TEST MODE
        # -------------------------
        elif choice == "4":
            print("\n🧩 Launching sandbox interactive play (debug mode)...")
            play_interactive()

        # -------------------------
        # 4️⃣ EXIT
        # -------------------------
        elif choice == "5":
            print("👋 Exiting program. Goodbye!")
            break

        else:
            print("❌ Invalid option. Please choose 1–5.")

    # Ensure pygame quits safely when exiting menu
    pygame.quit()


# ----------------- Quick test runner -----------------
if __name__ == "__main__":
    # Run the interactive play loop by default for manual testing.
    main_menu()