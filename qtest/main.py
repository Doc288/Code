import argparse
import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import yaml


def plot(rewards_episode):
    # Plot func
    plt.figure()

    plt.plot(rewards_episode, label="Penalty per Episode")  # changed label
    plt.xlabel('Episode')
    plt.ylabel('Penalty (lower is better)')
    plt.title('Penalty per Episode')

    plt.xlim(0, len(rewards_episode))
    plt.ylim(min(rewards_episode) - 10, 10)

    plt.grid(True)
    plt.legend()

    plt.savefig('cartpole.png')

    plt.show()

def qtable_init(is_training, env):
    # Qtable init

    if is_training:
        q_table = np.zeros((len(cart_space) + 1, len(cart_vel) + 1, len(pole_space) + 1, len(pole_vel) + 1, env.action_space.n))
        # +1 -> linespace -2.4,2.4,10 put in 10 boundaries at equal dist which makes 11 bins
    else:
        with open('cartpole.pkl', 'rb') as f:
            q_table = pickle.load(f)

    return q_table

def choose_action(is_training, q_table, state, epsilon, env):
    # Digitize
    state_p = np.digitize(state[0], cart_space)
    state_v = np.digitize(state[1], cart_vel)
    state_a = np.digitize(state[2], pole_space)
    state_av = np.digitize(state[3], pole_vel)

    if is_training and random.random() < epsilon:
        action = env.action_space.sample()  # Random action, .sample selects with equal probability out of available actions
    else:
        action = np.argmax(q_table[state_p, state_v, state_a, state_av, :])  # Best known action from Qtable

    return action, state_p, state_v, state_a, state_av

def update_q_table(q_table, state_p, state_v, state_a, state_av, action, reward, new_state, learning_rate, discount_factor):
    # Qtable update
    # Digitize new state
    new_state_p = np.digitize(new_state[0], cart_space)
    new_state_v = np.digitize(new_state[1], cart_vel)
    new_state_a = np.digitize(new_state[2], pole_space)
    new_state_av = np.digitize(new_state[3], pole_vel)

    # Q formula
    q_table[state_p, state_v, state_a, state_av, action] += learning_rate * (
        reward + discount_factor * np.max(q_table[new_state_p, new_state_v, new_state_a, new_state_av, :]) - q_table[state_p, state_v, state_a, state_av, action]
    )
    return q_table

def run():
    # Main fnc
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/conf.yaml')
    parser.add_argument('--mode', type=str, choices=['train', 'render'], required=True)
    args = parser.parse_args()

    # Select mode
    if args.mode == 'train':
        is_training = True
        render = False
    elif args.mode == 'render':
        is_training = False
        render = True

    # Open cfg
    with open(args.config) as f:
        conf = yaml.safe_load(f)

    # Def all global variables
    spaces = {}
    for name in ['cart_space', 'cart_vel', 'pole_space', 'pole_vel']:
        params = conf['discretization'][name]
        spaces[name] = np.linspace(params['start'], params['end'], params['bins'])

    global cart_space, cart_vel, pole_space, pole_vel
    cart_space = spaces['cart_space']
    cart_vel = spaces['cart_vel']
    pole_space = spaces['pole_space']
    pole_vel = spaces['pole_vel']

    learning_rate = conf['training']['learning_rate']
    discount_factor = conf['training']['discount_factor']
    epsilon = conf['training']['epsilon_start']
    epsilon_rate = conf['training']['epsilon_rate']
    epsilon_min = conf['training']['epsilon_min']
    max_rewards = conf['training']['max_rewards']
    rewards_episode = conf['logging']['rewards_episode']

    env = gym.make('CartPole-v1', render_mode='human' if render else None)

    i = 0

    # Qtable init
    q_table = qtable_init(is_training, env)

    while True:
        # Env reset
        state = env.reset()[0]
        rewards = 0
        terminated = False

        while not terminated and rewards < max_rewards:
            action, state_p, state_v, state_a, state_av = choose_action(is_training, q_table, state, epsilon, env)
            new_state, _, terminated, _, _ = env.step(action)

            # .step returns new step, reward, terminated, truncated, info
            # New steps formula -> \PycharmProjects\qtest\venv\Lib\site-packages\gymnasium\envs\classic_control

            # No reward for step, penalty at the end
            if is_training:
                q_table = update_q_table(q_table, state_p, state_v, state_a, state_av, action, 0.0, new_state, learning_rate, discount_factor)

            # State update and step counting
            state = new_state
            rewards += 1

        # Final penalty after episode
        penalty = -1.0 + (rewards / max_rewards)  # Closer to -1 - short, closer to 0 longer

        if is_training:
            # Apply terminal penalty to last taken action
            q_table[state_p, state_v, state_a, state_av, action] += learning_rate * (
                penalty - q_table[state_p, state_v, state_a, state_av, action]
            )

        # Penalty monitor
        rewards_episode.append(penalty)
        avg_rewards = np.mean(rewards_episode[-100:])

        if is_training and i % 100 == 0:
            print(f'Episode: {i}  Epsilon: {epsilon:0.2f}  Avg_penalty: {avg_rewards:0.3f}')

        # Rand rate manipulation
        if avg_rewards >= 0.0:
            break
        if i > 500:
            rate = epsilon_rate if i <= 5000 else 1.5 * epsilon_rate
            epsilon = max(epsilon - rate, epsilon_min)

        i += 1

    env.close()
    if is_training:
        plot(rewards_episode)

    # Qtable save
    if is_training:
        with open('cartpole.pkl', 'wb') as f:
            pickle.dump(q_table, f)


if __name__ == '__main__':
    # Just run
    # To tun python .\main.py --mode render/train
    run()
