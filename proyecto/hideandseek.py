# hideandseek_main.py
import numpy as np
import random
import os
import pygame
import gymnasium as gym
from gymnasium import spaces
import optuna
import time
import multiprocessing
import pandas as pd

# ============================================================
# DFS MAZE WITH EXTRA DOORS
# ============================================================
def static_labyrinth(door_density=0.15, seed=42):
    """
    Generate a deterministic 30x30 DFS-based maze with extra openings ("doors").
    door_density controls how open the maze is.
    """
    grid_size = 30
    maze = np.ones((grid_size, grid_size), dtype=int)  # 1 = wall, 0 = open

    def carve(x, y):
        dirs = [(2, 0), (-2, 0), (0, 2), (0, -2)]
        random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 1 <= nx < grid_size - 1 and 1 <= ny < grid_size - 1 and maze[nx, ny] == 1:
                maze[nx - dx // 2, ny - dy // 2] = 0
                maze[nx, ny] = 0
                carve(nx, ny)

    random.seed(seed)
    maze[1, 1] = 0
    carve(1, 1)

    # Add extra openings ("doors")
    for r in range(1, grid_size - 1):
        for c in range(1, grid_size - 1):
            if maze[r, c] == 1:
                if (maze[r - 1, c] == 0 and maze[r + 1, c] == 0) or (maze[r, c - 1] == 0 and maze[r, c + 1] == 0):
                    if random.random() < door_density:
                        maze[r, c] = 0

    walls = {(r, c) for r in range(grid_size) for c in range(grid_size) if maze[r, c] == 1}
    return walls, grid_size


# ============================================================
# ENVIRONMENT
# ============================================================
class HideAndSeekEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 12}

    def __init__(self, render_mode=None, door_density=0.15):
        super().__init__()
        self.door_density = door_density
        self.walls, self.grid_size = static_labyrinth(door_density=self.door_density)
        self.max_steps = 400
        self.visibility_radius = 6
        self.actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(
            low=np.array([0, -self.grid_size, -self.grid_size, 0, 0, 0, 0], dtype=np.int32),
            high=np.array([1, self.grid_size, self.grid_size, 1, 1, 1, 1], dtype=np.int32),
            dtype=np.int32,
        )
        self.cell_size = 20
        self.window_size = self.grid_size * self.cell_size

        self.render_mode = render_mode
        self.cell_size = 20
        self.screen = None
        self.clock = None
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        free_cells = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) if (r, c) not in self.walls]
        self.seeker_pos = next((cell for cell in free_cells), (1, 1))
        self.hider_pos = next((cell for cell in reversed(free_cells) if cell != self.seeker_pos),
                              (self.grid_size - 2, self.grid_size - 2))
        if self.has_line_of_sight(self.seeker_pos, self.hider_pos):
            for cell in reversed(free_cells):
                if cell != self.seeker_pos and not self.has_line_of_sight(self.seeker_pos, cell):
                    self.hider_pos = cell
                    break
        self.steps = 0
        return np.array(self._get_state(), dtype=np.int32), {}

    def _move(self, pos, action):
        dx, dy = self.actions[action]
        new_pos = (pos[0] + dx, pos[1] + dy)
        if 0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size and new_pos not in self.walls:
            return new_pos
        return pos

    def _wall_flags(self, pos):
        x, y = pos
        flags = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size or (nx, ny) in self.walls:
                flags.append(1)
            else:
                flags.append(0)
        return tuple(flags)

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def has_line_of_sight(self, a, b):
        (x0, y0), (x1, y1) = a, b
        if abs(x1 - x0) + abs(y1 - y0) > self.visibility_radius:
            return False
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx, sy = (1 if x1 > x0 else -1), (1 if y1 > y0 else -1)
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

    def _get_state(self):
        visible = 1 if self.has_line_of_sight(self.seeker_pos, self.hider_pos) else 0
        if visible:
            dx = self.hider_pos[0] - self.seeker_pos[0]
            dy = self.hider_pos[1] - self.seeker_pos[1]
        else:
            dx = dy = self.grid_size
        return (visible, dx, dy, *self._wall_flags(self.seeker_pos))

    def _smart_hider_move(self):
        moves = []
        for a, (dx, dy) in self.actions.items():
            new = (self.hider_pos[0] + dx, self.hider_pos[1] + dy)
            if 0 <= new[0] < self.grid_size and 0 <= new[1] < self.grid_size and new not in self.walls:
                moves.append(new)
        if not moves:
            return self.hider_pos
        seeker_visible = self.has_line_of_sight(self.seeker_pos, self.hider_pos)
        hidden = [p for p in moves if not self.has_line_of_sight(self.seeker_pos, p)]
        if seeker_visible and hidden:
            return hidden[0]
        return max(moves, key=lambda p: self._manhattan(p, self.seeker_pos))

    def step(self, seeker_action, hider_action=None):
        """
        Execute one environment step.
        - seeker_action: action from seeker (AI)
        - hider_action: optional manual action for player control
        """
        self.steps += 1
        old_s, old_h = self.seeker_pos, self.hider_pos

        # ✅ Move hider
        if hider_action is not None:
            # Manual control
            self._move_hider(hider_action)
        else:
            # AI movement
            self.hider_pos = self._smart_hider_move()

        # ✅ Move seeker
        self.seeker_pos = self._move(self.seeker_pos, seeker_action)

        # --- Reward logic ---
        reward, done = -0.01, False
        old_d = self._manhattan(old_s, self.hider_pos)
        new_d = self._manhattan(self.seeker_pos, self.hider_pos)
        if new_d < old_d:
            reward += 0.02
        elif new_d > old_d:
            reward -= 0.02

        # --- Terminal conditions ---
        if self.seeker_pos == self.hider_pos or (self.seeker_pos == old_h and self.hider_pos == old_s):
            reward, done = 1.0, True
        if self.steps >= self.max_steps:
            reward, done = -1.0, True

        return np.array(self._get_state(), dtype=np.int32), reward, done, False, {}


    def render(self):
        if self.render_mode != "human": return
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.grid_size * self.cell_size, self.grid_size * self.cell_size)
            )
            pygame.display.set_caption("Hide and Seek RL")
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.grid_size * self.cell_size, self.grid_size * self.cell_size))
        canvas.fill((0, 0, 0))

        for (r, c) in self.walls:
            pygame.draw.rect(canvas, (255, 255, 255),
                             (c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size))

        # draw vision overlay
        vis_surf = pygame.Surface((self.grid_size * self.cell_size, self.grid_size * self.cell_size), pygame.SRCALPHA)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.has_line_of_sight(self.seeker_pos, (r, c)):
                    pygame.draw.rect(vis_surf, (255, 255, 0, 60),
                                     (c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size))
        canvas.blit(vis_surf, (0, 0))

        pygame.draw.rect(canvas, (255, 0, 0),
                         (self.hider_pos[1] * self.cell_size + 2, self.hider_pos[0] * self.cell_size + 2,
                          self.cell_size - 4, self.cell_size - 4))
        pygame.draw.rect(canvas, (0, 0, 255),
                         (self.seeker_pos[1] * self.cell_size + 2, self.seeker_pos[0] * self.cell_size + 2,
                          self.cell_size - 4, self.cell_size - 4))
        self.screen.blit(canvas, (0, 0))
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def _move_hider(self, action):
        """Move the hider manually (player control)."""
        r, c = self.hider_pos
        nr, nc = r, c

        if action == 0:  # Up
            nr -= 1
        elif action == 1:  # Down
            nr += 1
        elif action == 2:  # Left
            nc -= 1
        elif action == 3:  # Right
            nc += 1

        if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
            if (nr, nc) not in self.walls:
                self.hider_pos = (nr, nc)

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None


# ============================================================
# SARSA AGENT
# ============================================================
class SARSAAgent:
    def __init__(self, grid_size, alpha=0.1, gamma=0.9,
                 epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.05):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.grid_size = grid_size
        # ✅ always start with an empty dict
        self.Q = {}

    def choose_action(self, s, env):
        # ensure the state exists
        if s not in self.Q:
            self.Q[s] = np.zeros(env.action_space.n)
        if np.random.rand() < self.epsilon:
            return env.action_space.sample()
        return int(np.argmax(self.Q[s]))

    def update(self, s, a, r, ns, na):
        # ensure next state exists
        if ns not in self.Q:
            self.Q[ns] = np.zeros_like(self.Q[s])
        self.Q[s][a] += self.alpha * (
            r + self.gamma * self.Q[ns][na] - self.Q[s][a]
        )

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)



# ============================================================
# TRAINING + OPTUNA + PLAY
# ============================================================
def load_best_params():
    if os.path.exists("best_params.npy"):
        try:
            best = np.load("best_params.npy", allow_pickle=True)
            if isinstance(best, np.ndarray) and best.dtype == object:
                best = best.item()
            return dict(best)
        except Exception:
            return {}
    return {}

def save_q_table_safely(agent, path):
    """
    Safely saves the agent's Q-table as a consistent dictionary format.
    Handles dicts, lists, numpy arrays, and edge cases gracefully.
    """
    try:
        Q = agent.Q

        # 🧠 Guarantee a dict structure
        if isinstance(Q, dict):
            pass  # Already good
        elif hasattr(Q, "items"):  # maybe defaultdict
            Q = dict(Q)
        elif hasattr(Q, "__iter__"):
            try:
                # Try converting iterable Q (list/array)
                Q = {tuple(k): v for k, v in enumerate(Q)}
                print(5)
            except Exception:
                # Handles NumPy scalar-like "iterables"
                Q = {("init",): Q}
                print("⚠️ Q-table looked iterable but wasn’t — wrapped as scalar.")
        else:
            # Fallback for pure scalar
            Q = {("init",): Q}

        # Save in standardized way
        np.save(path, np.array(Q, dtype=object), allow_pickle=True)
        print(f"💾 Q-table saved successfully to '{path}' ({len(Q)} states).")

    except Exception as e:
        print(f"⚠️ Failed to save Q-table: {e}")



def train_model(
    episodes=60000,
    save_path="q_table_normal.npy",
    **override
):
    """
    Trains the SARSA agent on the HideAndSeekEnv.
    - Supports Optuna and manual overrides.
    - Auto-saves checkpoints and final model safely.
    - Always saves Q-table as consistent dict format.
    """
    print(f"\n🚀 Starting training for {episodes} episodes...")
    env = HideAndSeekEnv(render_mode=None)
    best_params = {}

    # Load best Optuna params if present
    if os.path.exists("best_params.npy") and not override:
        try:
            best_params = np.load("best_params.npy", allow_pickle=True).item()
            print(f"📦 Loaded best parameters: {best_params}")
        except Exception:
            print("⚠️ Could not load best_params.npy, using defaults.")

    # Merge parameters
    params = {**best_params, **override}
    agent = SARSAAgent(env.grid_size, **params)

    # Try to load previous Q-table
    if os.path.exists(save_path):
        try:
            prev = np.load(save_path, allow_pickle=True).item()
            if isinstance(prev, dict):
                agent.Q = prev
                print(f"🔁 Loaded previous Q-table from '{save_path}' ({len(prev)} states).")
        except Exception as e:
            print(f"⚠️ Could not load previous Q-table: {e}")

    total_rewards = []
    start_time = time.time()

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        s = tuple(obs)
        a = agent.choose_action(s, env)
        done = False
        ep_reward = 0

        while not done:
            ns, r, done, _, _ = env.step(a)
            ns = tuple(ns)
            na = agent.choose_action(ns, env)
            agent.update(s, a, r, ns, na)
            s, a = ns, na
            ep_reward += r

        agent.decay_epsilon()
        total_rewards.append(ep_reward)

        # 🟢 Progress logging
        if ep % 5000 == 0:
            avgR = np.mean(total_rewards[-5000:])
            elapsed = (time.time() - start_time) / 60
            print(f"Ep {ep}/{episodes} | AvgR={avgR:.3f} | ε={agent.epsilon:.3f} | Time={elapsed:.1f}m")

        # 💾 Auto-save every 10 000 episodes
        if ep % 10000 == 0:
            save_q_table_safely(agent, save_path)

    # ✅ Final save at the end of training
    save_q_table_safely(agent, save_path)

    env.close()
    print(f"✅ Training complete! Final Q-table saved to '{save_path}'.")
    return env, agent


def objective(trial):
    """
    Run a short training to evaluate hyperparameters.
    Optuna will maximize the average episode reward.
    """
    params = {
        "alpha": trial.suggest_float("alpha", 0.01, 0.3, log=True),
        "gamma": trial.suggest_float("gamma", 0.8, 0.999),
        "epsilon_decay": trial.suggest_float("epsilon_decay", 0.995, 0.99999),
        "epsilon_min": trial.suggest_float("epsilon_min", 0.01, 0.1),
    }

    # Short evaluation training
    env = HideAndSeekEnv()
    agent = SARSAAgent(env.grid_size, **params)

    total_rewards = []
    n_eval_episodes = 1000  # short evaluation for speed

    for ep in range(n_eval_episodes):
        obs, _ = env.reset()
        s = tuple(obs)
        a = agent.choose_action(s, env)
        done = False
        ep_reward = 0

        while not done:
            ns, r, done, _, _ = env.step(a)
            ns = tuple(ns)
            na = agent.choose_action(ns, env)
            agent.update(s, a, r, ns, na)
            s, a = ns, na
            ep_reward += r

        agent.decay_epsilon()
        total_rewards.append(ep_reward)

    env.close()

    avg_reward = np.mean(total_rewards[-200:])  # average of last 200 episodes
    print(f"Trial {trial.number}: α={params['alpha']:.4f}, γ={params['gamma']:.3f}, "
          f"ε_decay={params['epsilon_decay']:.5f} → avgR={avg_reward:.3f}")
    return avg_reward


def run_optuna_search(n_trials=10):
    print(f"\n🚀 Starting Optuna search for best hyperparameters ({n_trials} trials, parallelized)...")

    def objective_wrapper(trial):
        return objective(trial)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    n_jobs = max(1, multiprocessing.cpu_count() - 1)
    print(f"🧠 Using {n_jobs} parallel workers...\n")

    start_time = time.time()

    # Run all trials in parallel (callbacks disabled in parallel mode)
    study.optimize(objective_wrapper, n_trials=n_trials, n_jobs=n_jobs)

    total_time = (time.time() - start_time) / 60
    best = study.best_params
    best_reward = study.best_value

    # ✅ Save all trial results to CSV
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df.to_csv("optuna_trials.csv", index=False)
    print(f"📊 Saved all {len(df)} trials to 'optuna_trials.csv'.")

    # ✅ Save best parameters
    np.save("best_params.npy", best, allow_pickle=True)
    print("\n✅ Optuna Search Complete!")
    print(f"🏆 Best average reward: {best_reward:.3f}")
    print(f"📦 Best parameters: {best}")
    print(f"🕒 Total optimization time: {total_time:.1f} minutes")

    # ✅ Automatically train the final Hard model
    print("\n💾 Training final Hard model using best parameters...")
    train_model(episodes=120000, save_path="q_table_hard.npy", **best)


def play_as_hider(model_path="q_table_hard.npy"):

    # ✅ Initialize pygame systems
    pygame.init()
    pygame.font.init()

    # ✅ Load trained model
    try:
        Q = np.load(model_path, allow_pickle=True).item()
        print(f"✅ Loaded Q-table from '{model_path}' ({len(Q)} states).")
    except Exception as e:
        print(f"⚠️ Could not load Q-table: {e}")
        Q = {}

    # ✅ Initialize environment and seeker agent
    env = HideAndSeekEnv(render_mode="human")
    agent = SARSAAgent(grid_size=env.grid_size)
    agent.Q = Q  # use loaded table
    clock = pygame.time.Clock()

    # ✅ Timer setup (3 minutes = 180 seconds)
    start_time = time.time()
    max_time = 180

    # ✅ Font for on-screen timer
    font = pygame.font.Font(None, 36)

    print("🎮 PLAY MODE (real-time). Use W/A/S/D to move. ESC to quit.")

    running = True
    done = False
    while running and not done:
        # --- handle input ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

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

        env.render()  
        
        # seeker (bot) logic
        state = env._get_state()
        a = agent.choose_action(state, env)
        _, _, done, _, _ = env.step(a, hider_action)


        # Draw timer (top-right)
        elapsed = int(time.time() - start_time)
        remaining = max(0, max_time - elapsed)
        mins, secs = divmod(remaining, 60)
        timer_text = f"{mins:02}:{secs:02}"
        text_surface = font.render(timer_text, True, (255, 255, 255))
        env.screen.blit(
            text_surface,
            (env.window_size - text_surface.get_width() - 10, 10),
        )
        pygame.display.flip()

        # --- win/loss conditions ---
        if remaining <= 0:
            print("⏰ Time’s up! The hider wins!")
            time.sleep(2)
            done = True
        elif env.hider_pos == env.seeker_pos:
            print("😱 You’ve been found! Game over.")
            time.sleep(2)
            done = True

        clock.tick(10)  # ~10 fps

    pygame.quit()
    print("👋 Exiting play mode. Returning to menu.")



# ============================================================
# MENU
# ============================================================
def main_menu():
    while True:
        print("\n====== HIDE & SEEK RL MENU ======")
        print("1️⃣  Calculate Hyperparameters (Optuna Search + Auto Hard Training)")
        print("2️⃣  Train Model (Manual)")
        print("3️⃣  Play Game")
        print("4️⃣  Exit")
        choice = input("Select option: ").strip()

        if choice == "1":
            print("\n⚙️ Running Optuna Hyperparameter Search...")
            print("   This will also train and save the optimized Hard model automatically.")
            run_optuna_search(n_trials=10)

        elif choice == "2":
            print("\n=== Select Difficulty for Training ===")
            print("1) Easy   – less training, more exploration (weak bot)")
            print("2) Normal – medium training, balanced bot")
            sub = input("Choose difficulty: ").strip()

            if sub == "1":
                print("🧩 Training Easy bot...")
                train_model(
                    episodes=30000,
                    save_path="q_table_easy.npy",
                    epsilon_decay=0.9995,
                    epsilon_min=0.1
                )
            elif sub == "2":
                print("🧩 Training Normal bot...")
                train_model(
                    episodes=60000,
                    save_path="q_table_normal.npy",
                    epsilon_decay=0.9998,
                    epsilon_min=0.05
                )
            else:
                print("❌ Invalid difficulty.")

        elif choice == "3":
            print("\n=== Select Difficulty for Play ===")
            print("1) Easy   – weaker bot")
            print("2) Normal – balanced bot")
            print("3) Hard   – optimized bot (from Optuna)")
            sub = input("Choose difficulty: ").strip()

            if sub == "1":
                print("🎮 Playing against Easy bot...")
                play_as_hider(model_path="q_table_easy.npy")
            elif sub == "2":
                print("🎮 Playing against Normal bot...")
                play_as_hider(model_path="q_table_normal.npy")
            elif sub == "3":
                print("🎮 Playing against Hard (Optuna-optimized) bot...")
                play_as_hider(model_path="q_table_hard.npy")
            else:
                print("❌ Invalid difficulty.")

        elif choice == "4":
            print("👋 Exiting...")
            break
        else:
            print("❌ Invalid option.")


if __name__ == "__main__":
    main_menu()
