import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
import pandas as pd
import seaborn as sns
from DeepQLearning import DQNAgent
from globals import MODEL_DIR, RESULTS_DIR

def plot_rewards(episode_rewards, filename: str, title: str="Rewards per Episode"):
    plt.figure(1)
    data = torch.tensor(episode_rewards, dtype=torch.float)
    plt.clf()
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.plot(data.numpy())

    # Take 100 episode averages and plot them too
    if len(data) >= 100:
        means = data.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.savefig(filename)

def save_agent_model(agent: DQNAgent, filename:str, episode: int=0, episode_rewards=None):
    """
    Save a model to a file
    :param agent: the model to load
    :param filename: the filename to load from. Must end with `.pt` and be in the models directory and be of the form `run_id/model_name.pt`
    :param episode: the episode number
    :param episode_rewards: the rewards for the episodes, used for plotting the rewards curve when visualizing the trained agent
    """
    if not filename.endswith('.pt'):
        raise ValueError('Filename must end with .pt')

    # Create the run_id directory if it doesn't exist
    dirname = filename.split('/')[0]
    pathname = os.path.join(MODEL_DIR, dirname)
    os.makedirs(pathname, exist_ok=True)

    final_file = os.path.join(MODEL_DIR, filename)

    checkpoint = {
        'policy_net': agent.policy_net.state_dict(),
        'optimizer': agent.optimizer.state_dict(),
        'episode': episode,
        'rewards': episode_rewards
    }
    if agent.use_dyna:
        checkpoint['world_model'] = agent.world_model.state_dict()
        checkpoint['wm_optimizer'] = agent.wm_optimizer.state_dict()
    torch.save(checkpoint, final_file)

def load_model(model: DQNAgent, filename: str):
    """
    Load model from a file
    :param model: model to load
    :param filename: filename to load from, must be in the models directory and be of the form `run_id/model_name.pt`
    :return:
    """
    if not filename.endswith('.pt'):
        raise ValueError('Filename must end with .pt')

    checkpoint = torch.load(filename, weights_only=False)
    model.policy_net.load_state_dict(checkpoint['policy_net'])
    model.optimizer.load_state_dict(checkpoint['optimizer'])
    # model.episode = checkpoint['episode']
    # model.rewards = checkpoint['rewards']
    if checkpoint.get('world_model'):
        model.world_model.load_state_dict(checkpoint['world_model'])
        model.wm_optimizer.load_state_dict(checkpoint['wm_optimizer'])

def save_results(rewards: list, real_steps_per_episode: list, filename: str, wm_errors=None):
    """
    Save the results of the training run to a pickle file
    :param rewards: rewards for each episode
    :param real_steps_per_episode: time step taken in the environment for each episode
    :param filename: the filename to save to. Must end with `.pkl` and be in the results directory and be of the form `run_id/model_name.pkl`
    :param wm_errors: world model transition validation errors for each episode
    :return:
    """
    if not filename.endswith('.pkl'):
        raise ValueError('Filename must end with .pkl')
    # Create the run_id directory if it doesn't exist
    dirname = filename.split('/')[0]
    pathname = os.path.join(RESULTS_DIR, dirname)
    if not os.path.exists(pathname):
        os.makedirs(pathname)

    final_file = os.path.join(RESULTS_DIR, filename)
    with open(final_file, 'wb') as file:
        to_save = {'rewards': rewards, 'real_steps_per_episode': real_steps_per_episode}
        if wm_errors: to_save['wm_errors'] = wm_errors
        pickle.dump(to_save, file)


def plot_scientific_comparison(run_id: int, base_rewards, base_steps, dyna_rewards, dyna_steps, dyna_wm_errors):
    # Calculate running cumulative interaction points
    base_cum_steps = np.cumsum(base_steps)
    dyna_cum_steps = np.cumsum(dyna_steps)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Graph A: Sample Efficiency Axis
    ax1.plot(base_cum_steps, base_rewards, label='Pure DQN (Model-Free)', alpha=0.4, color='blue')
    ax1.plot(dyna_cum_steps, dyna_rewards, label='Dyna-DQN (Model-Based)', alpha=0.4, color='orange')

    # Overlay moving averages for readability
    if len(base_rewards) >= 10:
        ax1.plot(base_cum_steps[9:], np.convolve(base_rewards, np.ones(10) / 10, mode='valid'), color='blue',
                 linewidth=2, label='DQN 10-Ep MA')
        ax1.plot(dyna_cum_steps[9:], np.convolve(dyna_rewards, np.ones(10) / 10, mode='valid'), color='orange',
                 linewidth=2, label='Dyna 10-Ep MA')

    ax1.set_title("Sample Efficiency: Returns vs. Real Environment Steps")
    ax1.set_xlabel("Cumulative Real Environment Interaction Steps")
    ax1.set_ylabel("Total Episode Reward")
    ax1.legend()
    ax1.grid(True, linestyle='--')

    # Graph B: World Model Error Convergence tracking
    ax2.plot(range(1, len(dyna_wm_errors) + 1), dyna_wm_errors, color='crimson', linewidth=2,
             label='Mean Prediction MSE')
    ax2.set_title("World Model Transition Validation Error Over Training")
    ax2.set_xlabel("Training Episodes")
    ax2.set_ylabel("Mean Squared Error (s', r)")
    ax2.legend()
    ax2.grid(True, linestyle='--')

    output_path = f'{RESULTS_DIR}/{run_id}/dyna_vs_dqn_scientific_report.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\n[INFO] Empirical Analysis Plot compiled successfully saved to: {output_path}")
    plt.show()


def get_objective_value(rewards: list, real_steps: list, val_type: str):
    """
    Get the objective value based on the given arguments
    :param rewards: list of rewards for each episode
    :param real_steps: list of real steps taken in the environment for each episode
    :param val_type: the types of objective values to calculate. 'avg_100' for average reward over the last 100 episodes,
    'avg_all' for average reward over all episodes, and 'real_steps' for total real steps taken in the environment
    :return:
    """
    if rewards is None:
        raise ValueError("rewards must be a list of rewards")
    if real_steps is None:
        raise ValueError("real_steps must be a list of real steps")
    if len(rewards) != len(real_steps):
        raise ValueError("rewards and real_steps must have the same length")

    objective_values = 0.0
    if val_type == 'avg_100':
        # average reward over the last 100 episodes
        objective_values = sum(rewards[-100:]) / min(100, len(rewards))
    elif val_type == 'avg_all':
        # average reward over all episodes
        objective_values = sum(rewards) / len(rewards)
    elif val_type == 'real_steps':
        # total real steps taken in the environment for all episodes
        objective_values = sum(real_steps)
    else:
        raise ValueError(f"Invalid objective value: {val_type}")

    return objective_values

def plot_rewards_by_agent(run_ids: list[str], agents_to_plot: list[str] | None = None, title: str | None = None):
    """
    Simple plot: load rewards lists from results/<run_id>/*.pkl,
    extract agent name from filename (expected '<AgentName>_... .pkl'),
    and plot episodes vs reward with hue by agent (mean +/- sd).
    - run_ids: list of run folder names (strings). If None, scan RESULTS_DIR for subfolders.
    - agents_to_plot: optional list of agent name substrings to include (e.g. ['DQNAgent']).
    """
    rewards_series = []  # list of (rewards_list, agent_name, run_id)

    for run_id in run_ids:
        directory = os.path.join(RESULTS_DIR, str(run_id))
        if not os.path.isdir(directory):
            print(f'Warning: directory not found: {directory}')
            continue
        for fname in os.listdir(directory):
            if not fname.endswith('.pkl'):
                continue
            filepath = os.path.join(directory, fname)
            try:
                with open(filepath, 'rb') as fh:
                    data = pickle.load(fh)
            except Exception as e:
                print(f'Warning: failed to load {filepath}: {e}')
                continue

            # Try common structures: dict with 'rewards' or tuple/list (rewards, steps)
            rewards = None
            # if isinstance(data, dict) and 'rewards' in data:
            #     rewards = data['rewards']
            # elif isinstance(data, (list, tuple)) and len(data) >= 1:
            #     rewards = data[0]
            # if rewards is None:
            #     print(f'Warning: no rewards found in {filepath}')
            #     continue

            if isinstance(data, dict) and 'real_steps_per_episode' in data:
                rewards = data['real_steps_per_episode']
            elif isinstance(data, (list, tuple)) and len(data) >= 1:
                rewards = data[0]
            if rewards is None:
                print(f'Warning: no real_steps_per_episode found in {filepath}')
                continue


            # Derive the agent name from the filename. We expect filenames like '<AgentName>_<stuff>.pkl'
            base = os.path.splitext(fname)[0]  # remove extension
            agent_name = base.split('_')[0] if '_' in base else base

            if agents_to_plot and agent_name not in agents_to_plot:
                continue
            rewards_series.append({'run_id': str(run_id), 'agent': agent_name, 'rewards': list(rewards)})

    if not rewards_series:
        raise RuntimeError('No rewards series found for the given run_ids / agents.')

    # Build DataFrame: each row = one run's reward series (episodes are columns)
    df_rows = [rs['rewards'] for rs in rewards_series]
    df = pd.DataFrame(df_rows)
    # attach agent column (one value per row)
    df['agent'] = [rs['agent'] for rs in rewards_series]

    # Melt into long form: columns = ['agent', 'variable' (episode index), 'reward']
    df_long = df.melt(id_vars='agent', var_name='episode', value_name='reward')
    df_long.rename(columns={'variable': 'episode', 'value': 'reward'}, inplace=True)

    # Plot: show mean +/- sd per agent across runs (seaborn will aggregate by default)
    if title is None:
        title = 'Real Steps over Episodes by Agent'

    sns.set(style="darkgrid", context="talk", palette="rainbow")
    plt.figure(figsize=(10, 6))
    # Use estimator='mean' and show standard deviation as shaded band (ci='sd')
    sns.lineplot(x='episode', y='reward', hue='agent', data=df_long, estimator='mean', errorbar='sd')
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Real Steps')
    plt.legend(title='Agent', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_rewards_by_experiment(
        exp_legend_map: dict[str, str],
        agents_to_plot: list[str] | None = None,
        title: str | None = None,
        window_size: int | None = None
):
    """
    Plots real steps vs episodes, aggregating across multiple run/seed folders
    found inside specified experiment root directories.

    Parameters:
    - exp_legend_map: Dict where keys are paths to experiment root directories
                      (e.g., 'results/exp1/') and values are the legend suffixes to append.
    - agents_to_plot: Optional list of agent base names to filter on.
    - title: Optional title for the plot.
    - window_size: Optional integer. If provided, applies a rolling/moving average
                   to smooth each run's data over the specified number of episodes.
    """
    rewards_series = []

    # 1. Iterate through each experiment root directory provided
    for exp_dir, legend_suffix in exp_legend_map.items():
        if not os.path.isdir(exp_dir):
            print(f'Warning: Experiment directory not found: {exp_dir}')
            continue

        # 2. Look inside the experiment directory to find all run_ids (seed folders)
        for run_id in os.listdir(exp_dir):
            run_dir = os.path.join(exp_dir, run_id)
            if not os.path.isdir(run_dir):
                continue

            # 3. Look inside the run_id directory for the .pkl files
            for fname in os.listdir(run_dir):
                if not fname.endswith('.pkl'):
                    continue
                filepath = os.path.join(run_dir, fname)

                try:
                    with open(filepath, 'rb') as fh:
                        data = pickle.load(fh)
                except Exception as e:
                    print(f'Warning: failed to load {filepath}: {e}')
                    continue

                rewards = None
                if isinstance(data, dict) and 'rewards' in data:
                    rewards = data['rewards']
                elif isinstance(data, (list, tuple)) and len(data) >= 1:
                    rewards = data[0]

                if rewards is None:
                    print(f'Warning: no real_steps_per_episode found in {filepath}')
                    continue

                # Derive the agent name from the filename
                base = os.path.splitext(fname)[0]
                agent_base_name = base.split('_')[0] if '_' in base else base

                if agents_to_plot and agent_base_name not in agents_to_plot:
                    continue

                display_name = f"{agent_base_name}{legend_suffix}"

                # Convert to Python list of primitive floats/ints
                rewards_list = [float(x) for x in rewards]

                # 4. Optional: Apply moving average per individual run series
                if window_size and window_size > 1:
                    # Convert to pandas Series to quickly use the rolling feature
                    series = pd.Series(rewards_list)
                    # min_periods=1 ensures data isn't dropped at the very start of the graph
                    rewards_list = series.rolling(window=window_size, min_periods=1).mean().tolist()

                rewards_series.append({'agent': display_name, 'rewards': rewards_list})

    if not rewards_series:
        raise RuntimeError('No rewards series found for the given experiment paths.')

    # Build DataFrame
    df_rows = [rs['rewards'] for rs in rewards_series]
    df = pd.DataFrame(df_rows)
    df['agent'] = [rs['agent'] for rs in rewards_series]

    # Melt into long form for Seaborn
    df_long = df.melt(id_vars='agent', var_name='episode', value_name='reward')
    df_long['episode'] = df_long['episode'].astype(int)

    # Plot Setup
    if title is None:
        suffix = f" ({window_size}-Ep SMA)" if window_size else ""
        title = f'Real Steps over Episodes by Agent{suffix}'

    sns.set_theme(style="darkgrid", context="talk", palette="rainbow")
    plt.figure(figsize=(10, 6))

    # Seaborn plots the mean and standard deviation of the smoothed metrics
    sns.lineplot(x='episode', y='reward', hue='agent', data=df_long, estimator='mean', errorbar='sd')

    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.legend(title='Agent Configuration', loc='best')
    plt.tight_layout()
    plt.show()


def plot_single_run_scientific_report(run_dir: str, window_size: int = 10):
    """
    Loads data from a single run directory, plots its specific sample efficiency
    and world model error curves, and saves the figure directly inside that run directory.

    Parameters:
    - run_dir: Path to the specific run folder (e.g., 'results/exp1/run_0/')
    - window_size: Window size for the moving average overlay on rewards.
    """
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"The specified run directory does not exist: {run_dir}")

    # Find the .pkl file inside this specific directory
    pkl_files = [f for f in os.listdir(run_dir) if f.endswith('.pkl')]
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl data file found inside: {run_dir}")

    # Take the first pkl file found
    filepath = os.path.join(run_dir, pkl_files[0])
    print(f"[INFO] Loading single-run empirical data from: {filepath}")

    with open(filepath, 'rb') as fh:
        data = pickle.load(fh)

    # Extract core series metrics dynamically
    if isinstance(data, dict):
        rewards = data.get('real_rewards_per_episode') or data.get('rewards')
        steps = data.get('real_steps_per_episode') or data.get('steps')
        wm_errors = data.get('dyna_wm_errors') or data.get('wm_errors', [])
    else:
        rewards = data[0] if len(data) > 0 else None
        steps = data[1] if len(data) > 1 else None
        wm_errors = []

    if rewards is None or steps is None:
        raise ValueError(f"Core metrics (rewards/steps) missing or corrupted in {filepath}")

    # Convert to standard numpy arrays for calculation
    rewards = np.array(rewards, dtype=float)
    steps = np.array(steps, dtype=float)
    cum_steps = np.cumsum(steps)

    # ---------------- Plot Construction ----------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

    # Graph A: Sample Efficiency Axis (Raw + Moving Average Overlay)
    # Plot the raw, noisy background line
    ax1.plot(cum_steps, rewards, label='Raw Episode Returns', color='dodgerblue', alpha=0.3, linewidth=1.5)

    # Calculate and overlay the moving average cleanly
    if len(rewards) >= window_size:
        # Using np.convolve for a mathematically clean simple moving average matching the step lengths
        ma_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
        # Adjust the cumulative step indices to align with the trailing edge of the window
        ax1.plot(cum_steps[window_size - 1:], ma_rewards, color='darkblue', linewidth=2.5,
                 label=f'{window_size}-Episode Moving Avg')

    ax1.set_title("Sample Efficiency: Returns vs. Real Steps", fontsize=13, weight='bold')
    ax1.set_xlabel("Cumulative Real Environment Interaction Steps", fontsize=11)
    ax1.set_ylabel("Total Episode Reward", fontsize=11)
    ax1.legend(loc='best')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Graph B: World Model Error Tracking
    if len(wm_errors) > 0:
        episodes_axis = np.arange(1, len(wm_errors) + 1)
        ax2.plot(episodes_axis, wm_errors, color='crimson', linewidth=2, label='Prediction MSE')
        ax2.set_title("World Model Validation Error Over Training", fontsize=13, weight='bold')
        ax2.set_xlabel("Training Episodes", fontsize=11)
        ax2.set_ylabel("Mean Squared Error (s', r)", fontsize=11)
        ax2.legend(loc='best')
        ax2.grid(True, linestyle='--', alpha=0.7)
    else:
        ax2.text(0.5, 0.5, "No Model Error Tracking Data\nAvailable for this Run",
                 ha='center', va='center', fontsize=12, color='gray')
        ax2.set_title("World Model Metrics", fontsize=13, weight='bold')

    # Extract clean naming attributes for the main title heading
    run_name = os.path.basename(os.path.normpath(run_dir))
    plt.suptitle(f"Empirical Diagnostic Report: Single Execution ({run_name})", y=1.02, fontsize=15, weight='bold')
    plt.tight_layout()

    # Save the output figure directly back into the folder we read from
    output_path = os.path.join(run_dir, 'single_run_scientific_report.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Diagnostic plot compiled and saved to: {output_path}\n")

    plt.show()


def plot_experiment_diagnostic_suite(exp_dir: str, window_size: int = 10, exp_name: str | None = None):
    """
    Generates a unified diagnostic report for an experiment folder containing 12 runs:
    - Figure 1: A 2x3 grid of subplots. Each subplot maps a specific seed index and
      overlays BOTH the Pure DQN and Dyna-DQN curves against cumulative real steps.
    - Figure 2: A standalone trendline chart tracking World Model Loss across the Dyna seeds.
    """
    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    # Gather all run directory names inside the experiment folder
    all_run_ids = sorted([d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))])

    # Sort files into a structured dictionary mapping: seed_index -> { agent_name: run_data }
    # Since they are sorted alphabetically, the first 6 of an agent type correspond to seeds 0-5
    seeds_map = {i: {} for i in range(6)}
    dqn_count, dyna_count = 0, 0

    print("[INFO] Parsing data logs and matching seed pairs...")

    for run_id in all_run_ids:
        run_path = os.path.join(exp_dir, run_id)
        pkl_files = [f for f in os.listdir(run_path) if f.endswith('.pkl')]
        if not pkl_files:
            continue

        filepath = os.path.join(run_path, pkl_files[0])

        # Extract agent name from filename prefix
        base_filename = os.path.splitext(pkl_files[0])[0]
        agent_name = base_filename.split('_')[0] if '_' in base_filename else base_filename

        with open(filepath, 'rb') as fh:
            data = pickle.load(fh)

        # Extract core metrics
        if isinstance(data, dict):
            rewards = data.get('real_rewards_per_episode') or data.get('rewards')
            steps = data.get('real_steps_per_episode') or data.get('steps')
            wm_errors = data.get('dyna_wm_errors') or data.get('wm_errors', [])
        else:
            rewards = data[0] if len(data) > 0 else None
            steps = data[1] if len(data) > 1 else None
            wm_errors = []

        if rewards is None or steps is None:
            continue

        run_payload = {
            'run_id': run_id,
            'cum_steps': np.cumsum(steps),
            'rewards': np.array(rewards, dtype=float),
            'wm_errors': np.array(wm_errors, dtype=float) if len(wm_errors) > 0 else []
        }

        # Dynamically assign to the next available seed index slot for this model type
        if "dyna" in agent_name.lower():
            if dyna_count < 6:
                seeds_map[dyna_count]['Dyna-DQN'] = run_payload
                dyna_count += 1
        else:
            if dqn_count < 6:
                seeds_map[dqn_count]['Pure DQN'] = run_payload
                dqn_count += 1

    # Setup plotting aesthetics
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    if not exp_name:
        exp_name = os.path.basename(os.path.normpath(exp_dir))

    # ---------------- FIGURE 1: 2x3 Shared Sample Efficiency Overlay Grid ----------------
    print("[PROCESSING] Generating unified 2x3 Sample Efficiency comparison matrix...")
    fig_eff, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes_flat = axes.flatten()

    # Define unique color configurations for visual clarity
    colors = {
        'Pure DQN': {'raw': 'dodgerblue', 'ma': 'darkblue'},
        'Dyna-DQN': {'raw': 'coral', 'ma': 'crimson'}
    }

    seeds_used = [42, 1, 16, 24, 74, 123]

    for seed_idx, models in seeds_map.items():
        ax = axes_flat[seed_idx]
        ax.set_title(f"Seed = {seeds_used[seed_idx]}", fontsize=13, weight='bold')

        # Check if we have data for either model in this seed slot
        if not models:
            ax.text(0.5, 0.5, "No seed logs found", ha='center', va='center')
            continue

        for model_label, run_data in models.items():
            c = colors[model_label]

            # Plot the raw return path (faded background)
            ax.plot(run_data['cum_steps'], run_data['rewards'], color=c['raw'], alpha=0.25,
                    label=f'{model_label} (Raw)')

            # Compute and plot moving average trend line
            if len(run_data['rewards']) >= window_size:
                ma_rewards = np.convolve(run_data['rewards'], np.ones(window_size) / window_size, mode='valid')
                ax.plot(run_data['cum_steps'][window_size - 1:], ma_rewards, color=c['ma'], linewidth=2.5,
                        label=f'{model_label} ({window_size}-Ep MA)')
            else:
                ax.plot(run_data['cum_steps'], run_data['rewards'], color=c['ma'], linewidth=2, label=model_label)

        ax.set_xlabel("Cumulative Real Environment Steps", fontsize=10)
        ax.set_ylabel("Total Episode Reward", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='lower right', fontsize=9)

    fig_eff.suptitle(f"Direct Sample Efficiency Comparison Matrix: {exp_name}", y=0.98, fontsize=16, weight='bold')
    fig_eff.tight_layout()

    eff_output_path = os.path.join(exp_dir, 'sample_efficiency_overlay_matrix.png')
    fig_eff.savefig(eff_output_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Saved overlay matrix to: {eff_output_path}")
    plt.close(fig_eff)

    # ---------------- FIGURE 2: Combined World Model Loss ----------------
    print("[PROCESSING] Compiling aggregated World Model Error (Mean +/- SD)...")

    loss_records = []

    for seed_idx, models in seeds_map.items():
        if 'Dyna-DQN' in models and len(models['Dyna-DQN']['wm_errors']) > 0:
            losses = models['Dyna-DQN']['wm_errors']

            # Apply moving average to this seed's loss before aggregating
            if window_size and len(losses) >= window_size:
                smoothed_losses = np.convolve(losses, np.ones(window_size) / window_size, mode='valid')
                start_ep = window_size
            else:
                smoothed_losses = losses
                start_ep = 1

            # Record each data point into a list for pandas
            for i, val in enumerate(smoothed_losses):
                loss_records.append({
                    'Episode': start_ep + i,
                    'Loss': val,
                    'Seed': seed_idx
                })

    if loss_records:
        df_loss = pd.DataFrame(loss_records)

        fig_loss, ax_loss = plt.subplots(figsize=(10, 6))

        # Seaborn automatically calculates the mean line and the +/- standard deviation band
        sns.lineplot(
            data=df_loss,
            x='Episode',
            y='Loss',
            ax=ax_loss,
            color='crimson',
            linewidth=2.5,
            estimator='mean',
            errorbar='sd',
            label=f'Dyna World Model ({window_size}-Ep SMA)'
        )

        ax_loss.set_title(f"Aggregated World Model MSE Convergence: {exp_name}", fontsize=14, weight='bold')
        ax_loss.set_xlabel("Training Episodes", fontsize=11)
        ax_loss.set_ylabel("Mean Squared Error (s', r)", fontsize=11)
        ax_loss.grid(True, linestyle='--', alpha=0.7)
        ax_loss.legend(loc='upper right', fontsize=11)

        loss_output_path = os.path.join(exp_dir, 'world_model_loss_aggregated.png')
        fig_loss.tight_layout()
        fig_loss.savefig(loss_output_path, dpi=300, bbox_inches='tight')
        print(f"[SUCCESS] Saved aggregated loss figure to: {loss_output_path}\n")
        plt.close(fig_loss)
    else:
        print("[INFO] No World Model metrics found. Skipping Loss plot.")

    print("[FINISHED] Evaluation matrices generated cleanly.")


def plot_loss_comparison(exp_legend_map: dict[str, str], window_size: int = 10, title: str | None = None):
    """
    Plots an aggregated (Mean +/- SD) World Model Loss comparison across multiple experiments.

    Parameters:
    - exp_legend_map: Dict where keys are paths to experiment root folders
                      (e.g., {"results/exp1": "Dyna (5 steps)", "results/exp2": "Dyna (15 steps)"})
    - window_size: Window size for the moving average applied to each seed's trace.
    - title: Optional custom title for the comparison plot.
    """
    loss_records = []

    # 1. Loop through each experiment folder provided
    for exp_dir, experiment_label in exp_legend_map.items():
        if not os.path.isdir(exp_dir):
            print(f'Warning: Experiment directory not found: {exp_dir}')
            continue

        all_run_ids = sorted([d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))])

        # 2. Extract loss data from every seed/run folder inside this experiment
        for seed_idx, run_id in enumerate(all_run_ids):
            run_path = os.path.join(exp_dir, run_id)
            pkl_files = [f for f in os.listdir(run_path) if f.endswith('.pkl')]
            if not pkl_files:
                continue

            filepath = os.path.join(run_path, pkl_files[0])
            with open(filepath, 'rb') as fh:
                try:
                    data = pickle.load(fh)
                except Exception as e:
                    print(f"Warning: Failed to load {filepath}: {e}")
                    continue

            # Check for standard world model loss dictionary keys
            if isinstance(data, dict):
                wm_errors = data.get('dyna_wm_errors') or data.get('wm_errors', [])
            else:
                wm_errors = []

            if len(wm_errors) == 0:
                continue

            losses = np.array(wm_errors, dtype=float)

            # 3. Apply the simple moving average to the individual seed trace first
            if window_size and len(losses) >= window_size:
                smoothed_losses = np.convolve(losses, np.ones(window_size) / window_size, mode='valid')
                start_ep = window_size
            else:
                smoothed_losses = losses
                start_ep = 1

            # 4. Restructure data points into the long-form database format
            for i, val in enumerate(smoothed_losses):
                loss_records.append({
                    'Experiment': experiment_label,
                    'Episode': start_ep + i,
                    'Loss (MSE)': val,
                    'Seed': seed_idx
                })

    if not loss_records:
        raise RuntimeError("No World Model loss metrics found in any of the provided experiment paths.")

    # Convert to standard long-form DataFrame
    df_loss = pd.DataFrame(loss_records)

    # 5. Build and format the comparison graphic
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(11, 6))

    # Seaborn automatically tracks the 'Experiment' variable to group lines by hue,
    # while parsing the variance across different 'Seeds' to plot the shaded SD bounds.
    sns.lineplot(
        data=df_loss,
        x='Episode',
        y='Loss (MSE)',
        hue='Experiment',
        ax=ax,
        linewidth=2.5,
        estimator='mean',
        errorbar='sd',
        palette='Set1'  # Clean, high-contrast palette for multi-line comparison
    )

    if title is None:
        title = f"World Model Transition Convergence Comparison ({window_size}-Ep SMA)"

    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xlabel("Training Episodes", fontsize=11)
    ax.set_ylabel("Mean Squared Error ($s', r$)", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(title="Configurations", loc='best', fontsize=10)
    plt.tight_layout()

    # Save the chart in your current working directory execution environment
    output_path = "results/cross_experiment_loss_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Cross-experiment loss landscape figure saved to: {output_path}")

    plt.show()