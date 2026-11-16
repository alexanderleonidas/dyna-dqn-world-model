import gymnasium as gym
import time
import optuna
from itertools import count
from globals import *
from helpers import save_agent_model, save_results, get_objective_value
from DeepQLearning import DQNAgent

def train_agent(env:gym.Env, agent: DQNAgent, num_episodes: int=1000, trial=None, reward_type: str=OBJECTIVE):

    run_id = int(time.time())
    print(f'Starting run {run_id}')
    print('Episode\t\tReward\t\tReal Steps\t\tWM Error')
    print('-' * 55)
    episode_rewards = []
    real_steps_per_episode = []
    wm_errors_per_episode = []
    best_reward = -float('inf')
    for i_episode in range(num_episodes):
        # Initialize the environment and get its state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        total_reward = 0
        episode_steps = 0
        ep_wm_loss = 0.0
        for t in count():
            # Select the action based on the current state
            action = agent.sample_action(state)
            agent.decay_epsilon() # decay epsilon after sampling the action
            observation, reward, terminated, truncated, _ = env.step(action.item()) # take a step in the environment
            episode_steps += 1
            total_reward += reward
            reward = torch.tensor([reward], dtype=torch.float32, device=DEVICE)
            done = terminated or truncated
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            # Store the transition in memory
            agent.store_transition(state, action, next_state, reward)

            # Move to the next state and update the agent
            state = next_state
            agent.update()

            # Track world model loss step-by-step
            if getattr(agent, 'use_dyna', False):
                ep_wm_loss += agent.latest_wm_loss

            if done:
                episode_rewards.append(total_reward)
                real_steps_per_episode.append(episode_steps)
                # Normalize world model error by steps taken
                avg_wm_error = (ep_wm_loss / episode_steps) if episode_steps > 0 else 0.0
                if getattr(agent, 'use_dyna', False): # only save wm errors if using dyna
                    wm_errors_per_episode.append(avg_wm_error)

                if i_episode % 10 == 0 or i_episode == num_episodes - 1:
                    print(f'{i_episode+1}\t\t\t{total_reward:.2f}\t\t\t{episode_steps}\t\t\t{avg_wm_error:.4f}')

                # Save the model at checkpoint intervals and at the end of the run
                if total_reward > best_reward:
                    best_reward = total_reward
                    # save the best model
                    save_agent_model(agent, f'{run_id}/{agent.name}_episode_{i_episode}_seed{agent.seed}.pt', episode=i_episode, episode_rewards=episode_rewards)

                # Report intermediate objective value to Optuna and check if trial should be pruned
                if trial:
                    obj_val = get_objective_value(episode_rewards, real_steps_per_episode, reward_type)
                    trial.report(obj_val, i_episode)

                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

                break

    print('Training Complete')
    # Save the final model and results
    save_agent_model(agent, f'{run_id}/{agent.name}_final_model_seed{agent.seed}.pt', episode=num_episodes-1, episode_rewards=episode_rewards)
    if not wm_errors_per_episode:
        save_results(episode_rewards, real_steps_per_episode, filename=f'{run_id}/{agent.name}_results_seed{agent.seed}.pkl')
    else:
        save_results(episode_rewards, real_steps_per_episode, filename=f'{run_id}/{agent.name}_results_seed{agent.seed}.pkl', wm_errors=wm_errors_per_episode)

    return run_id, episode_rewards, real_steps_per_episode, wm_errors_per_episode