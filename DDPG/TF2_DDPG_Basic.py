import gym
import random
import imageio
import datetime
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate
from tensorflow.keras.optimizers import Adam

from Prioritized_Replay import Memory

# Original paper: https://arxiv.org/pdf/1509.02971.pdf
# DDPG with PER paper: https://cardwing.github.io/files/RL_course_report.pdf

tf.keras.backend.set_floatx('float64')


def actor(state_shape, num_discrete_actions, num_continuous_actions, action_bound, action_shift, units=(400, 300)):
    state = Input(shape=state_shape)
    x = Dense(units[0], activation='relu')(state)
    for index in range(1, len(units)):
        x = Dense(units[index], activation='relu')(x)
    
    # 離散動作輸出
    discrete_output = Dense(num_discrete_actions, activation='softmax', name='discrete_out')(x)

    # 連續動作輸出
    unscaled_continuous_output = Dense(num_continuous_actions, activation='tanh', name='continuous_out')(x)
    scaled_continuous_output = Lambda(lambda op: op * action_bound)(unscaled_continuous_output)
    if action_shift != 0:
       scaled_continuous_output = Lambda(lambda op: op + action_shift)(scaled_continuous_output)  # for action range not centered at zero


    model = Model(inputs=state, outputs=[discrete_output, scaled_continuous_output])

    return model


def critic(state_shape, num_discrete_actions, num_continuous_actions, units=(48, 24)):
    state_input = Input(shape=state_shape)
    discrete_action_input = Input(shape=(num_discrete_actions,))
    continuous_action_input = Input(shape=(num_continuous_actions,))

    concat = Concatenate(axis=-1)([state_input, discrete_action_input, continuous_action_input])
    x = Dense(units[0], activation='relu')(concat)
    for index in range(1, len(units)):
        x = Dense(units[index], activation='relu')(x)
    output = Dense(1)(x)
    model = Model(inputs=[state_input, discrete_action_input, continuous_action_input], outputs=output)
    return model


def update_target_weights(model, target_model, tau=0.005):
    weights = model.get_weights()
    target_weights = target_model.get_weights()
    for i in range(len(target_weights)):  # set tau% of target model to be new weights
        target_weights[i] = weights[i] * tau + target_weights[i] * (1 - tau)
    target_model.set_weights(target_weights)


# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class OrnsteinUhlenbeckNoise:
    def __init__(self, mu, sigma=0.2, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class NormalNoise:
    def __init__(self, mu, sigma=0.15):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(scale=self.sigma, size=self.mu.shape)

    def reset(self):
        pass


class DDPG:
    def __init__(
            self,
            env,
            discrete=False,
            use_priority=False,
            lr_actor=1e-5,
            lr_critic=1e-3,
            actor_units=(24, 16),
            critic_units=(24, 16),
            noise='norm',
            sigma=0.15,
            tau=0.125,
            gamma=0.85,
            batch_size=64,
            memory_cap=100000
    ):
        self.env = env
        
        if isinstance(env.action_space, gym.spaces.Discrete):
            num_discrete_actions = env.action_space.n
            num_continuous_actions = 0
        elif isinstance(env.action_space, gym.spaces.Box):
            num_discrete_actions = 0  # 如果环境只有连续动作，离散动作数设置为0
            num_continuous_actions = env.action_space.shape[0]
        else:
            raise NotImplementedError("Action space type not supported")
        
        self.state_shape = env.observation_space.shape  # shape of observations
        self.action_dim = env.action_space.n if discrete else env.action_space.shape[0]  # number of actions
        self.discrete = discrete
        self.action_bound = (env.action_space.high - env.action_space.low) / 2 if not discrete else 1.
        self.action_shift = (env.action_space.high + env.action_space.low) / 2 if not discrete else 0.
        self.use_priority = use_priority
        self.memory = Memory(capacity=memory_cap) if use_priority else deque(maxlen=memory_cap)
        if noise == 'ou':
            self.noise = OrnsteinUhlenbeckNoise(mu=np.zeros(self.action_dim), sigma=sigma)
        else:
            self.noise = NormalNoise(mu=np.zeros(self.action_dim), sigma=sigma)
        
        num_discrete_actions = 1
        num_continuous_actions = env.action_space.shape[0] 

        # Define and initialize Actor network
        self.actor = actor(self.state_shape,num_discrete_actions, num_continuous_actions, self.action_bound, self.action_shift, actor_units)
        self.actor_target = actor(self.state_shape, num_discrete_actions, num_continuous_actions, self.action_bound, self.action_shift, actor_units)
        self.actor_optimizer = Adam(learning_rate=lr_actor)
        update_target_weights(self.actor, self.actor_target, tau=1.)

        # Define and initialize Critic network
        self.critic = critic(self.state_shape, num_discrete_actions, num_continuous_actions, critic_units)
        self.critic_target = critic(self.state_shape,num_discrete_actions, num_continuous_actions, critic_units)
        self.critic_optimizer = Adam(learning_rate=lr_critic)
        update_target_weights(self.critic, self.critic_target, tau=1.)

        # Set hyperparameters
        self.gamma = gamma  # discount factor
        self.tau = tau  # target model update
        self.batch_size = batch_size

        # Tensorboard
        self.summaries = {}
    
    def act(self, state, add_noise=True):
        state = np.expand_dims(state, axis=0).astype(np.float32)
        discrete_action_probs, continuous_action_values = self.actor.predict(state)
        discrete_action = np.argmax(discrete_action_probs)  # 选择概率最高的离散动作
        continuous_action = continuous_action_values[0]  # 取出连续动作值
        if add_noise:
            continuous_action += self.noise() * self.action_bound
        continuous_action = np.clip(continuous_action, -self.action_bound + self.action_shift, self.action_bound + self.action_shift)

        return discrete_action, continuous_action

#    def act(self, state, add_noise=True):
#        state = np.expand_dims(state, axis=0).astype(np.float32)
#        a = self.actor.predict(state)
#        a += self.noise() * add_noise * self.action_bound
#        a = tf.clip_by_value(a, -self.action_bound + self.action_shift, self.action_bound + self.action_shift)
#
#        self.summaries['q_val'] = self.critic.predict([state, a])[0][0]
#
#        return a 

    def save_model(self, a_fn, c_fn):
        self.actor.save(a_fn)
        self.critic.save(c_fn)

    def load_actor(self, a_fn):
        self.actor.load_weights(a_fn)
        self.actor_target.load_weights(a_fn)
        print(self.actor.summary())

    def load_critic(self, c_fn):
        self.critic.load_weights(c_fn)
        self.critic_target.load_weights(c_fn)
        print(self.critic.summary())

    def remember(self, state, action, reward, next_state, done):
        if self.use_priority:
            action = np.squeeze(action)
            transition = np.hstack([state, action, reward, next_state, done])
            self.memory.store(transition)
        else:
            state = np.expand_dims(state, axis=0)
            next_state = np.expand_dims(next_state, axis=0)
            self.memory.append([state, action, reward, next_state, done])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        if self.use_priority:
            tree_idx, samples, ISWeights = self.memory.sample(self.batch_size)
            split_shape = np.cumsum([self.state_shape[0], self.action_dim, 1, self.state_shape[0]])
            states, actions, rewards, next_states, dones = np.hsplit(samples, split_shape)
        else:
            ISWeights = 1.0
            samples = random.sample(self.memory, self.batch_size)
            s = np.array(samples).T
            states, actions, rewards, next_states, dones = [np.vstack(s[i, :]).astype(np.float) for i in range(5)]

        next_actions = self.actor_target.predict(next_states)
        q_future = self.critic_target.predict([next_states, next_actions])
        target_qs = rewards + q_future * self.gamma * (1. - dones)

        # train critic
        with tf.GradientTape() as tape:
            q_values = self.critic([states, actions])
            td_error = q_values - target_qs
            critic_loss = tf.reduce_mean(ISWeights * tf.math.square(td_error))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)  # compute critic gradient
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        # update priority
        if self.use_priority:
            abs_errors = tf.reduce_sum(tf.abs(td_error), axis=1)
            self.memory.batch_update(tree_idx, abs_errors)

        # train actor
        with tf.GradientTape() as tape:
            actions = self.actor(states)
            actor_loss = -tf.reduce_mean(self.critic([states, actions]))

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)  # compute actor gradient
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

        # tensorboard info
        self.summaries['critic_loss'] = critic_loss
        self.summaries['actor_loss'] = actor_loss

    def train(self, max_episodes=50, max_epochs=8000, max_steps=500, save_freq=50):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/DDPG_basic_' + current_time
        summary_writer = tf.summary.create_file_writer(train_log_dir)

        done, episode, steps, epoch, total_reward = False, 0, 0, 0, 0
        cur_state = self.env.reset()
        while episode < max_episodes or epoch < max_epochs:
            if done:
                episode += 1
                print("episode {}: {} total reward, {} steps, {} epochs".format(
                    episode, total_reward, steps, epoch))

                with summary_writer.as_default():
                    tf.summary.scalar('Main/episode_reward', total_reward, step=episode)
                    tf.summary.scalar('Main/episode_steps', steps, step=episode)

                summary_writer.flush()
                self.noise.reset()

                if steps >= max_steps:
                    print("episode {}, reached max steps".format(episode))
                    self.save_model("ddpg_actor_episode{}.h5".format(episode),
                                    "ddpg_critic_episode{}.h5".format(episode))

                done, cur_state, steps, total_reward = False, self.env.reset(), 0, 0
                if episode % save_freq == 0:
                    self.save_model("ddpg_actor_episode{}.h5".format(episode),
                                    "ddpg_critic_episode{}.h5".format(episode))

            a = self.act(cur_state)  # model determine action given state
            action = np.argmax(a) if self.discrete else a[0]  # post process for discrete action space
            next_state, reward, done, _ = self.env.step(action)  # perform action on env

            self.remember(cur_state, a, reward, next_state, done)  # add to memory
            self.replay()  # train models through memory replay

            update_target_weights(self.actor, self.actor_target, tau=self.tau)  # iterates target model
            update_target_weights(self.critic, self.critic_target, tau=self.tau)

            cur_state = next_state
            total_reward += reward
            steps += 1
            epoch += 1

            # Tensorboard update
            with summary_writer.as_default():
                if len(self.memory) > self.batch_size:
                    tf.summary.scalar('Loss/actor_loss', self.summaries['actor_loss'], step=epoch)
                    tf.summary.scalar('Loss/critic_loss', self.summaries['critic_loss'], step=epoch)
                tf.summary.scalar('Main/step_reward', reward, step=epoch)
                tf.summary.scalar('Stats/q_val', self.summaries['q_val'], step=epoch)

            summary_writer.flush()

        self.save_model("ddpg_actor_final_episode{}.h5".format(episode),
                        "ddpg_critic_final_episode{}.h5".format(episode))

    def test(self, render=True, fps=30, filename='test_render.mp4'):
        cur_state, done, rewards = self.env.reset(), False, 0
        video = imageio.get_writer(filename, fps=fps)
        while not done:
            a = self.act(cur_state, add_noise=False)
            action = np.argmax(a) if self.discrete else a[0]  # post process for discrete action space
            next_state, reward, done, _ = self.env.step(action)
            cur_state = next_state
            rewards += reward
            if render:
                video.append_data(self.env.render(mode='rgb_array'))
        video.close()
        return rewards


if __name__ == "__main__":
    gym_env = gym.make("CartPole-v1")
    try:
        # Ensure action bound is symmetric
        assert (gym_env.action_space.high == -gym_env.action_space.low)
        is_discrete = False
        print('Continuous Action Space')
    except AttributeError:
        is_discrete = True
        print('Discrete Action Space')

    ddpg = DDPG(gym_env, discrete=is_discrete)
    # ddpg.load_critic("ddpg_critic_episode124.h5")
    # ddpg.load_actor("ddpg_actor_episode124.h5")
    ddpg.train(max_episodes=1000)
    # rewards = ddpg.test()
    # print("Total rewards: ", rewards)
