import gymnasium as gym
from experiments import Experiments
from helpers import plot_rewards_by_agent, plot_rewards_by_experiment, plot_experiment_diagnostic_suite, plot_loss_comparison


# Train the agent over different seeds
# for seed in [1, 42, 16, 24, 74, 123]:
# # for seed in [42]:
#     env = gym.make('LunarLander-v3')
    # exp = Experiments(env, seed=seed)
#     # exp.train_deep_q_learning()
#     # exp.run_hyperparameter_optimization()
#     exp.run_comparative_study(only_wm=True)

# Render the environment and evaluate the trained agent
render_env = gym.make('LunarLander-v3', render_mode="human", continuous=False)
# We can also see how our model performs under different environment conditions, such as with wind and different gravity settings
# render_env = gym.make("LunarLander-v3", render_mode="human", continuous=False, gravity=-10.0, enable_wind=False, wind_power=15.0, turbulence_power=1.5)

viz_exp = Experiments(render_env, seed=1)
# viz_exp.eval_deep_q_learning('models/1779487511/DQNAgent_episode_378.pt', num_episodes=5)
# viz_exp.eval_deep_q_learning('models/1779484561/DynaDQNAgent_episode_49.pt', num_episodes=5)
# viz_exp.eval_deep_q_learning('models/1779556833/DynaDQNAgent_final_model_seed74.pt', num_episodes=3)
# viz_exp.evaluate_world_model('models/1779484561/DynaDQNAgent_episode_49.pt', n_samples=1000)
# viz_exp.evaluate_world_model2("models/exp2/1779556833/DynaDQNAgent_final_model_seed74.pt", n_samples=100000)
# viz_exp.evaluate_world_model2("models/exp4/1779581337/DynaDQNAgent_episode_253_seed123.pt", n_samples=100000)


# viz_exp.eval_deep_q_learning('models/exp1/1779506882/DQNAgent_episode_410_seed123.pt', num_episodes=5)
viz_exp.eval_deep_q_learning('models/exp4/1779580408/DynaDQNAgent_episode_404_seed74.pt', num_episodes=8)




# These experiments were run with different seeds, but the same hyperparameters
# The World Model-based DQN agent used random actions for planning steps
rewards_over_seeds_experiments_ids_1 = ['1779487511', '1779488310', '1779494856', '1779495948', '1779497828', '1779498922',
                                      '1779500667', '1779501988', '1779503895', '1779505045', '1779506882', '1779508117']

# Experiment run with planning steps based on the current policy, rather than random actions
rewards_over_seeds_experiments_ids_2 = ['1779548681', '1779550636', '1779552361', '1779554708', '1779556833', '1779558580']

# Experiment run with planning steps based on the current policy, with 5 planning steps
rewards_over_seeds_experiments_ids_3 = ['1779566510', '1779567774', '1779569134', '1779570673', '1779572116', '1779573521']

# Experiments run with planning steps based on the current policy, with 2 planning steps
rewards_over_seeds_experiments_ids_4 = ['exp4/1779576334', 'exp4/1779577221', 'exp4/1779578356', 'exp4/1779579062', 'exp4/1779580408', 'exp4/1779581337']


# plot_rewards_by_agent(rewards_over_seeds_experiments_ids_4)

# my_runs = {
#     "results/exp1": " (random steps=10)",
#     "results/exp2":  " (policy steps=10)",
#     "results/exp3": " (steps=5)",
#     "results/exp4":  " (steps=2)",
# }
#
# plot_rewards_by_experiment(exp_legend_map=my_runs, agents_to_plot=['DynaDQNAgent'], title="Impact of planning steps on rewards", window_size=10)

# plot_loss_comparison(
#     exp_legend_map=my_runs,
#     window_size=15,
#     title="World Model Error Convergence Across Structural Setups"
# )

# plot_experiment_diagnostic_suite('results/exp4', 10, exp_name="Dyna-DQN with 2 planning steps")

