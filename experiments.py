import random
import time
import numpy as np
import seaborn as sns
import optuna
import pandas as pd
import pickle
import optuna.samplers as samplers
from optuna.trial import TrialState
import matplotlib.pyplot as plt
import gymnasium as gym
from globals import *
from helpers import load_model, plot_rewards, save_agent_model, get_objective_value, plot_scientific_comparison
from training import train_agent
from DeepQLearning import DQNAgent


class Experiments:
    def __init__(self, env: gym.Env, seed: int=SEED):
        self.env = env
        self.seed = seed

        # Get number of actions from gym action space
        self.n_actions = env.action_space.n
        # Get the number of state observations
        state, _ = env.reset()
        self.n_observations = len(state)

        random.seed(seed)
        torch.manual_seed(seed)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def run_comparative_study(self, only_wm: bool=False):
        """Runs baseline DQN and Dyna-DQN sequentially, then compiles comparison graphs."""
        base_hyperparams = {
            "learning_rate": LR, "discount_factor": GAMMA, "initial_epsilon": EPS_START,
            "epsilon_decay": EPS_DECAY, "final_epsilon": EPS_END, "tau": TAU,
            "batch_size": BATCH_SIZE, "memory_capacity": MEMORY_CAPACITY, "seed": self.seed, "use_dyna": False
        }
        if not only_wm:
            print("=== EXECUTING CONDITION 1: PURE MODEL-FREE DQN ===")
            self.env.reset(seed=self.seed)
            base_agent = DQNAgent(env=self.env, hyperparameters=base_hyperparams)
            run_id, base_rewards, base_steps, _ = train_agent(self.env, base_agent, NUM_EPISODES)
            plot_rewards(base_rewards, f'{RESULTS_DIR}/{run_id}/base_rewards_over_episodes.png')

        print("\n=== EXECUTING CONDITION 2: MODEL-BASED DYNA-DQN ===")
        self.env.reset(seed=self.seed)
        dyna_hyperparams = base_hyperparams.copy()
        dyna_hyperparams.update({"use_dyna": True, "planning_steps": PLANNING_STEPS, "wm_lr": LR})
        dyna_agent = DQNAgent(env=self.env, hyperparameters=dyna_hyperparams)
        run_id, dyna_rewards, dyna_steps, dyna_wm_errors = train_agent(self.env, dyna_agent, NUM_EPISODES)
        plot_rewards(dyna_rewards, f'{RESULTS_DIR}/{run_id}/dyna_rewards_over_episodes_seed{self.seed}.png')

        if only_wm:
            return
        else:
            # Plot structural findings
            plot_scientific_comparison(run_id, base_rewards, base_steps, dyna_rewards, dyna_steps, dyna_wm_errors)


    def run_hyperparameter_optimization(self):

        def objective(trial: optuna.Trial):
            # Hyperparameters to optimize
            hyperparameters = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
                "discount_factor": trial.suggest_float("discount_factor", 0.9, 0.999, log=True),
                "initial_epsilon": EPS_START,
                "epsilon_decay": trial.suggest_int("epsilon_decay", 100, 10000, log=True),
                "final_epsilon": EPS_END,
                "tau": trial.suggest_float("tau", 0.001, 0.1, log=True),
                "batch_size": trial.suggest_int("batch_size", 32, 512, log=True),
                "memory_capacity": trial.suggest_int("memory_capacity", 5000, 50000, log=True),
            }

            agent = DQNAgent(env=self.env, hyperparameters=hyperparameters)

            # Train the agent and return the rewards and real steps per episode
            run_id, rewards, real_steps, _ = train_agent(self.env, agent, NUM_EPISODES, trial=trial)
            completed = [t for t in trial.study.get_trials(deepcopy=False) if t.state == optuna.trial.TrialState.COMPLETE]

            # Return a value for optuna to maximise or minimise based on the reward type
            obj_val = get_objective_value(rewards, real_steps, OBJECTIVE)

            # Save the best model so far
            if OBJECTIVE == 'real_steps':
                if not completed or obj_val < min(t.value for t in completed):
                    save_agent_model(agent, f"{run_id}/{agent.name}_optuna_trial{trial.number}.pt", episode=NUM_EPISODES, episode_rewards=rewards)
            else:
                if not completed or obj_val > max(t.value for t in completed):
                    save_agent_model(agent, f"{run_id}/{agent.name}_optuna_trial{trial.number}.pt", episode=NUM_EPISODES, episode_rewards=rewards)

            return obj_val

        # minimize or maximize the objective value based on the reward type
        direction = "minimize" if OBJECTIVE == 'real_steps' else "maximize"
        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=self.seed), pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=NUM_TRIALS, timeout=TIMEOUT)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("\tNumber of finished trials: ", len(study.trials))
        print("\tNumber of pruned trials: ", len(pruned_trials))
        print("\tNumber of complete trials: ", len(complete_trials))

        print("Best hyperparameters:")
        print(study.best_params)
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig2 = optuna.visualization.plot_param_importances(study)
        fig3 = optuna.visualization.plot_parallel_coordinate(study)
        fig4 = optuna.visualization.plot_contour(study)
        fig1.show()
        fig2.show()
        fig3.show()
        fig4.show()


    def train_deep_q_learning(self, plot: bool=True):

        hyperparameters = {"learning_rate": LR,
                           "discount_factor": GAMMA,
                           "initial_epsilon": EPS_START,
                           "epsilon_decay": EPS_DECAY,
                           "final_epsilon": EPS_END,
                           "tau": TAU,
                           "batch_size": BATCH_SIZE,
                           "memory_capacity": MEMORY_CAPACITY
                           }

        agent = DQNAgent(env=self.env, hyperparameters=hyperparameters)

        run_id, rewards, real_steps, _ = train_agent(self.env, agent, NUM_EPISODES)
        if plot: plot_rewards(rewards, f'{RESULTS_DIR}/{run_id}_{agent.name}_rewards_over_episodes.png')

    def eval_deep_q_learning(self, model_path: str, num_episodes: int=5):
        hyperparameters = {"learning_rate": LR,
                           "discount_factor": GAMMA,
                           "initial_epsilon": 0.0, # Set epsilon to 0 to always choose the action with the highest Q-value
                           "epsilon_decay": 0.0,
                           "final_epsilon": 0.0,
                           "tau": TAU,
                           "batch_size": BATCH_SIZE,
                           "memory_capacity": MEMORY_CAPACITY,
                           "seed": self.seed,
                           }

        if "Dyna" in model_path:
            hyperparameters.update({"use_dyna": True, "planning_steps": PLANNING_STEPS, "wm_lr": LR})

        agent = DQNAgent(env=self.env, hyperparameters=hyperparameters)

        load_model(agent, model_path)
        agent.policy_net.eval()

        with torch.no_grad():
            for i_episode in range(num_episodes):

                state, info = self.env.reset()
                done = False
                total_reward = 0

                while not done:
                    state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    action = agent.sample_action(state)
                    observation, reward, terminated, truncated, _ = self.env.step(action.item())
                    total_reward += reward
                    done = terminated or truncated
                    state = observation

                    time.sleep(1 / 60)  # ~60 FPS cap for visible playback

                print(f'Episode {i_episode+1}: Total Reward: {total_reward}')

        self.env.close()

    def evaluate_world_model(self, model_path: str, n_samples=10):
        hyperparameters = {"learning_rate": LR,
                           "discount_factor": GAMMA,
                           "initial_epsilon": 0.0,
                           "epsilon_decay": 0.0,
                           "final_epsilon": 0.0,
                           "tau": TAU,
                           "batch_size": BATCH_SIZE,
                           "memory_capacity": MEMORY_CAPACITY,
                           "seed": self.seed,
                           "use_dyna": True,
                           "planning_steps": PLANNING_STEPS,
                           "wm_lr": LR
                           }

        agent = DQNAgent(env=self.env, hyperparameters=hyperparameters)

        load_model(agent, model_path)
        agent.policy_net.eval()

        state, _ = self.env.reset()
        error_over_time = []
        for _ in range(n_samples):
            state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            action = agent.sample_action(state_t)
            action = action.item()
            action_t = torch.tensor([[action]], device=DEVICE)
            next_state, reward, terminated, truncated, _ = self.env.step(action)

            with torch.no_grad():
                pred_next, pred_r = agent.world_model(state_t, action_t)

            actual_next = torch.tensor(next_state, dtype=torch.float32, device=DEVICE)
            error = (pred_next.squeeze() - actual_next).abs().mean().item()
            error_over_time.append(error)
            print(f'Action: {action}, Pred reward: {pred_r.item():.3f}, Actual reward: {reward:.3f}, State MAE: {error:.4f}')

            if terminated or truncated:
                state, _ = self.env.reset()
            else:
                state = next_state

        plt.figure(1)
        data = torch.tensor(error_over_time, dtype=torch.float)
        plt.clf()
        plt.title('World Model Transition Validation Error')
        plt.xlabel('Timestep')
        plt.ylabel('Mean Absolute Error (s\')')
        plt.plot(data.numpy())
        plt.savefig(f'{RESULTS_DIR}/world_model_evaluation_error.png')

    def evaluate_world_model2(self, model_path: str, n_samples=1000):
        """
        Evaluates the world model's prediction accuracy across individual state dimensions.
        Generates an advanced dual distribution plot (Violin + Bar with Error Caps) and
        an Error Correlation Heatmap to diagnose structural physics prediction vulnerabilities.
        """
        hyperparameters = {
            "learning_rate": LR,
            "discount_factor": GAMMA,
            "initial_epsilon": 0.0,
            "epsilon_decay": 0.0,
            "final_epsilon": 0.0,
            "tau": TAU,
            "batch_size": BATCH_SIZE,
            "memory_capacity": MEMORY_CAPACITY,
            "seed": self.seed,
            "use_dyna": True,
            "planning_steps": PLANNING_STEPS,
            "wm_lr": LR
        }

        agent = DQNAgent(env=self.env, hyperparameters=hyperparameters)
        load_model(agent, model_path)
        agent.policy_net.eval()
        if hasattr(agent, 'world_model'):
            agent.world_model.eval()

        # Labels for LunarLander observation space variables
        dimension_labels = [
            "X Position",
            "Y Position",
            "X Velocity",
            "Y Velocity",
            "Angle",
            "Angular Velocity",
            "Left Leg Contact",
            "Right Leg Contact"
        ]

        all_actuals = []
        all_predictions = []

        state, _ = self.env.reset()

        print(f"\n[INFO] Gathering {n_samples} transitions from environment execution...")
        for _ in range(n_samples):
            state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            action = agent.sample_action(state_t)
            action = action.item()
            action_t = torch.tensor([[action]], device=DEVICE)

            next_state, reward, terminated, truncated, _ = self.env.step(action)

            with torch.no_grad():
                pred_next, _ = agent.world_model(state_t, action_t)

            all_actuals.append(next_state)
            all_predictions.append(pred_next.squeeze().cpu().numpy())

            if terminated or truncated:
                state, _ = self.env.reset()
            else:
                state = next_state

        # Convert matrices (Shape: n_samples, 8)
        actuals_matrix = np.array(all_actuals)
        predictions_matrix = np.array(all_predictions)

        # Calculate Absolute Errors across all data points
        errors_by_dim = np.abs(predictions_matrix - actuals_matrix)
        mean_errors_per_dim = np.mean(errors_by_dim, axis=0)
        std_errors = np.std(errors_by_dim, axis=0)

        # 1. Output Summary Statistics to Console
        print("\n" + "=" * 55)
        print(f" WORLD MODEL ERROR PROFILE BY DIMENSION (N={n_samples})")
        print("=" * 55)
        for i, label in enumerate(dimension_labels):
            print(f"{label:<22} : Mean = {mean_errors_per_dim[i]:.5f} | Std = {std_errors[i]:.5f}")
        print("=" * 55)

        # 2. Build Long-Form DataFrame for Seaborn (Optimized vectorized construction)
        df_list = []
        for i, label in enumerate(dimension_labels):
            temp_df = pd.DataFrame({
                'Dimension': label,
                'MAE': errors_by_dim[:, i]
            })
            df_list.append(temp_df)
        df_long = pd.concat(df_list, ignore_index=True)

        # 3. Plot 1: Dual Distribution Matrix (Violin + Bar chart with Error Bars)
        sns.set_theme(style="darkgrid", context="talk")
        fig1, axes = plt.subplots(1, 2, figsize=(20, 8))

        # Left Subplot: Violin Distribution Profile
        # Note: 'cut=0' restricts the violin calculation strictly to the range of observed data
        sns.violinplot(
            data=df_long, x='Dimension', y='MAE',
            ax=axes[0], palette='Blues',
            inner='box', cut=0, density_norm='width', hue='Dimension'
        )
        axes[0].set_title('World Model Error Distribution Dynamics', fontweight='bold', fontsize=16, pad=15)
        axes[0].set_ylabel('Absolute Error Magnitude', fontsize=14)
        axes[0].set_xlabel('')
        axes[0].tick_params(axis='x', rotation=30, labelsize=12)

        # Right Subplot: Bar Representation with Standard Deviation Caps
        axes[1].bar(
            dimension_labels, mean_errors_per_dim,
            yerr=std_errors, capsize=6,
            color='royalblue', edgecolor='black', alpha=0.85
        )
        # Overlay mean scalar tags safely above the maximum value boundary
        for i, (m, s) in enumerate(zip(mean_errors_per_dim, std_errors)):
            axes[1].text(
                i, m + s + (max(mean_errors_per_dim + std_errors) * 0.01),
                f'{m:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold', color='black'
            )
        axes[1].set_title('Mean Feature Deviation Metrics (Mean ± Std)', fontweight='bold', fontsize=16, pad=15)
        axes[1].set_ylabel('Mean Absolute Error (MAE)', fontsize=14)
        axes[1].tick_params(axis='x', rotation=30, labelsize=12)

        plt.tight_layout()
        dist_path = f'{RESULTS_DIR}/world_model_dimensional_distribution.png'
        plt.savefig(dist_path, dpi=300)
        print(f"[SUCCESS] Error distribution chart exported to: {dist_path}")

        # 4. Plot 2: Error Correlation Matrix Heatmap
        fig2, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = np.corrcoef(errors_by_dim.T)

        # Plotting heatmap with a divergent palette centered at zero
        sns.heatmap(
            corr_matrix,
            xticklabels=dimension_labels,
            yticklabels=dimension_labels,
            annot=True, fmt='.2f',
            cmap='coolwarm', center=0, vmin=-1.0, vmax=1.0,
            ax=ax, square=True, cbar_kws={'label': 'Pearson Correlation Coefficient'}
        )
        ax.set_title('World Model Error Cascades: Inter-Dimension Correlations', fontweight='bold', fontsize=15, pad=15)
        plt.xticks(rotation=30, ha='right', fontsize=11)
        plt.yticks(rotation=0, fontsize=11)
        plt.tight_layout()

        corr_path = f'{RESULTS_DIR}/world_model_error_correlation.png'
        plt.savefig(corr_path, dpi=300)
        print(f"[SUCCESS] Inter-dimension correlation map exported to: {corr_path}\n")

        plt.show()

    def diagnose_premature_termination(self, dqn_results_path, dyna_results_path):
        # Load your saved pkl results
        with open(dqn_results_path, 'rb') as f:
            dqn_rewards, dqn_steps = pickle.load(f)
        with open(dyna_results_path, 'rb') as f:
            dyna_rewards, dyna_steps = pickle.load(f)

        print("=" * 60)
        print("        PREMATURE TERMINATION DIAGNOSTIC CRITERIA")
        print("=" * 60)
        print(f"DQN Baseline -> Avg Steps/Episode: {np.mean(dqn_steps):.1f} | Avg Reward: {np.mean(dqn_rewards):.1f}")
        print(f"Dyna Agent   -> Avg Steps/Episode: {np.mean(dyna_steps):.1f} | Avg Reward: {np.mean(dyna_rewards):.1f}")
        print("-" * 60)

        # Check if steps are low AND rewards are highly negative
        if np.mean(dyna_steps) < np.mean(dqn_steps) * 0.5 and np.mean(dyna_rewards) < -100:
            print("DIAGNOSIS: [CONFIRMED] Pathological Premature Termination.")
            print("REASON: The agent is aggressively crashing due to a poisoned Q-network.")
        else:
            print("DIAGNOSIS: Inconclusive or True Sample Efficiency.")
        print("=" * 60)

