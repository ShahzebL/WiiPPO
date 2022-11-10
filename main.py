import numpy as np

from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image, ImageGrab
import matplotlib

from memory_profiler import profile
from stable_baselines3 import PPO, A2C, DQN, SAC
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
import gym
from gym import spaces
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import cv2
import dolphin_memory_engine as dme

import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
# from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from pynput.keyboard import Key, Controller
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from gym.wrappers import FrameStack
import math

# instantiating the decorator


DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 500  # Terminal states (end of episodes)
MODEL_NAME = '2x256'
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20
LOG_DIR = "logs/{}-{}".format(MODEL_NAME, int(time.time()))
LR = 0.00025
BBOX = (0, 60, 1280, 1020)


# Environment settings
EPISODES = 20_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99
MIN_EPSILON = 0.001
SPEED_ADDRESS = 2162464600#2163835920
CHECKPOINT_ADDRESS = 2162427364#2162427428
OBSERVATION_SPACE_VALUES = (96, 128, 3)  # 4
ACTION_SPACE_SIZE = 9

#  Stats settings
AGGREGATE_STATS_EVERY = 20  # episodes
SHOW_PREVIEW = True

dme.hook()
keyboard = Controller()

tf.config.set_visible_devices([], 'GPU')


class Funky:
    def action(self, choice):
        '''
        Total moves
        1) Go straight
        2) Soft drift left
        3) Hard drift left
        4) Soft drift right
        5) Hard drift right
        6) Go straight and wheelie
        7) Turn left no wheelie
        8) Turn right no wheelie
        9) Hop

        Keys to hit-
        1) w
        2) a 
        3) d
        4) up arrow
        5) m
        6) /
        '''
        if choice == 0:
            keyboard.press("w")
            keyboard.release("a")
            keyboard.release("d")
            keyboard.release(Key.up)
            keyboard.release("m")
            keyboard.release("/")
        elif choice == 1:
            keyboard.press("w")
            keyboard.press("a")
            keyboard.press("/")
            keyboard.press("m")
            keyboard.release("d")
            keyboard.release(Key.up)
        elif choice == 2:
            keyboard.press("w")
            keyboard.press("a")
            keyboard.press("/")
            keyboard.release("m")
            keyboard.release("d")
            keyboard.release(Key.up)
        elif choice == 3:
            keyboard.press("w")
            keyboard.press("d")
            keyboard.press("/")
            keyboard.press("m")
            keyboard.release("a")
            keyboard.release(Key.up)
        elif choice == 4:
            keyboard.press("w")
            keyboard.press("d")
            keyboard.press("/")
            keyboard.release("m")
            keyboard.release("a")
            keyboard.release(Key.up)
        elif choice == 5:
            keyboard.press("w")
            keyboard.release("a")
            keyboard.release("/")
            keyboard.release("m")
            keyboard.release("d")
            keyboard.press(Key.up)
        elif choice == 6:
            keyboard.press("w")
            keyboard.press("a")
            keyboard.release("/")
            keyboard.release("m")
            keyboard.release("d")
            keyboard.release(Key.up)
        elif choice == 7:
            keyboard.press("w")
            keyboard.release("a")
            keyboard.release("/")
            keyboard.release("m")
            keyboard.press("d")
            keyboard.release(Key.up)
        elif choice == 8:
            keyboard.press("w")
            keyboard.release("a")
            keyboard.press("/")
            keyboard.release("m")
            keyboard.release("d")
            keyboard.release(Key.up)


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = 3
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3456, 512),
            nn.ReLU(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)

class MKWiiEnv(gym.Env):
    RETURN_IMAGES = True
    MOVE_PENALTY = 1

    def __init__(self):
        super(MKWiiEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=OBSERVATION_SPACE_VALUES, dtype=np.uint8) 
        self.prev_checkpoint = 1
        self.check = 1.05
        self.check_inc = 0.05
        self.check_timer = time.time()
        self.count_leftsoftdrift = 0
        self.count_leftharddrift = 0
        self.count_rightsoftdrift = 0
        self.count_rightharddrift = 0
        self.count_straightdrift = 0
        self.is_right = False
        self.reset_state = 0
        self.max_reward = 0
        self.min_reward = 10


    def reset(self):
        self.player = Funky()
        self.check_inc = 0.05
        self.check_timer = time.time()
        self.count_leftsoftdrift = 0
        self.count_leftharddrift = 0
        self.count_rightsoftdrift = 0
        self.count_rightharddrift = 0
        self.count_straightdrift = 0
        self.is_right = False
        self.episode_step = 0

        keyboard.release("w")
        keyboard.release("a")
        keyboard.release("d")
        keyboard.release("m")
        keyboard.release("/")
        keyboard.release(Key.up)
        
        if self.reset_state % 4 == 0:
            keyboard.press(Key.f1)
            self.prev_checkpoint = 1.027
            self.check = 1.077
        elif self.reset_state % 4 == 1:
            keyboard.press(Key.f2)
            self.prev_checkpoint = 1.036
            self.check = 1.086
        elif self.reset_state % 4 == 2:
            keyboard.press(Key.f3)
            self.prev_checkpoint = 1.499
            self.check = 1.549
        else:
            keyboard.press(Key.f4)
            self.prev_checkpoint = 1.654
            self.check = 1.704
        observation = self.get_image()
        keyboard.release(Key.f1)
        keyboard.release(Key.f2)
        keyboard.release(Key.f3)
        keyboard.release(Key.f4)
        time.sleep(0.5)
        self.reset_state += 1
        # self.stack = [np.array(observation)] * 4
        # return np.array(self.stack)
        return observation

    def step(self, action):
        self.episode_step += 1
        self.player.action(action)
        speed = dme.read_float(SPEED_ADDRESS)
        checkpoint = dme.read_float(CHECKPOINT_ADDRESS)
        new_observation = np.array(self.get_image())
        reward = 0
        # reward = speed / 2200
        # reward = np.sign(speed - 80) * math.sqrt(abs(speed - 80)) / 25
        # print('------------------')
        # print(reward)
        if speed > 100:
            reward += 0.2
        if speed > 90:
            reward += 0.05
        if speed < 80:
            reward -= 0.1

        # print(reward)
        reward += (checkpoint - self.prev_checkpoint) * 100
        # print(reward)
        # print(checkpoint_reward)
        self.prev_checkpoint = checkpoint

        if checkpoint > self.check:
            reward += 1 / (time.time() - self.check_timer)
            self.check += self.check_inc
            self.check_timer = time.time()

        # print(reward)


        #takes >~1 second to charge a hard drift mt 
        #takes 2.5 seconds to charge up a hard drift mt
        if self.count_rightharddrift == 0 and self.count_rightsoftdrift == 0 \
            and self.count_leftsoftdrift == 0 and self.count_leftharddrift == 0 \
            and self.count_straightdrift == 0:
            self.is_right = action in (3, 4)

        if action in (1, 2, 3, 4, 8) and (self.count_rightharddrift \
            + self.count_leftsoftdrift + self.count_leftharddrift \
            + self.count_rightsoftdrift + self.count_straightdrift) >= 3 \
            and ( \
            self.is_right and 2.5 * (self.count_rightsoftdrift + self.count_rightharddrift) \
            + (self.count_leftsoftdrift + self.count_leftharddrift + self.count_straightdrift) <= 30 \
            or \
            (not self.is_right) and 2.5 * (self.count_leftharddrift + self.count_leftsoftdrift) \
            + (self.count_rightharddrift + self.count_rightsoftdrift + self.count_straightdrift) <= 30 \
            ):
            if self.is_right:
                reward += (2.5 * (self.count_rightsoftdrift + self.count_rightharddrift) \
                + (self.count_leftsoftdrift + self.count_leftharddrift + self.count_straightdrift)) / 1500.0
            else:
                reward += (2.5 * (self.count_leftharddrift + self.count_leftsoftdrift) \
                + (self.count_rightharddrift + self.count_rightsoftdrift + self.count_straightdrift)) / 1500.0


        if action == 1:
            self.count_leftsoftdrift += 1
        elif action == 2:
            self.count_leftharddrift += 1
        elif action == 3:
            self.count_rightsoftdrift += 1
        elif action == 4:
            self.count_rightharddrift += 1
        elif action == 8:
            self.count_straightdrift += 1
        else:
            if (self.is_right and 2.5 * (self.count_rightsoftdrift + self.count_rightharddrift) \
                + (self.count_leftsoftdrift + self.count_leftharddrift + self.count_straightdrift) >= 3 \
                and 2.5 * (self.count_rightsoftdrift + self.count_rightharddrift) \
                + (self.count_leftsoftdrift + self.count_leftharddrift + self.count_straightdrift) <= 31) \
                or ((not self.is_right) and 2.5 * (self.count_leftharddrift + self.count_leftsoftdrift) \
                +  (self.count_rightharddrift + self.count_rightsoftdrift + self.count_straightdrift) >= 3 \
                and 2.5 * (self.count_leftsoftdrift + self.count_leftharddrift) \
                + (self.count_rightsoftdrift + self.count_rightharddrift + self.count_straightdrift) <= 31):
                reward = -0.3
            self.count_rightharddrift = 0
            self.count_rightsoftdrift = 0
            self.count_leftsoftdrift = 0
            self.count_leftharddrift = 0
            self.count_straightdrift = 0

        # print(reward)

        done = False

        if speed < 60 or checkpoint >= 3.99:
            done = True
        if speed < 60:
            reward = -1
        if checkpoint >= 3.99:
            reward = 10
        
        #reward is between -0.3 and 1.7


        #reward 2.2
        # print(reward)

        # print('------------------')
        # self.stack.pop(0)
        # self.stack.append(new_observation)
        # return np.array(self.stack), reward, done, {}
        return new_observation, reward, done, {}

    def render(self):
        img = self.get_image()
        img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
        cv2.imshow("image", np.array(img))  # show it!
        cv2.waitKey(1)

    # FOR CNN #
    def get_image(self):
        img = ImageGrab.grab(
            bbox=BBOX, 
            include_layered_windows=False, 
            all_screens=False, 
            xdisplay=None
        )
        img = np.array(img)
        scale_percent = 10 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
          
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        return resized
    def close(self):
        keyboard.release("w")
        keyboard.release("a")
        keyboard.release("d")
        keyboard.release("m")
        keyboard.release("/")
        keyboard.release(Key.up)
        keyboard.release(Key.f1)


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir)
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}.zip")
                  self.model.save(os.path.join(self.save_path, 'best_model'))

        return True

def fun():
    env = MKWiiEnv()
    env = Monitor(env, 'best_model')
    return env

env = MKWiiEnv()
env = Monitor(env, 'best_model')
# env = gym.wrappers.FrameStack(env, 4)
# env = gym.make('CartPole-v1')
# env = FrameStack(env, num_stack = 4)
# env = DummyVecEnv([fun])
# print(env.reset().shape)
# env = VecFrameStack(env, n_stack = 4, channels_order='last')
# print(env.observation_space)
model_name = "mkwiiv5(2.5 hours)"
# env = VecFrameStack(env, n_stack=4)
# model = PPO('CnnPolicy', env, verbose=1, tensorboard_log = LOG_DIR)
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=f'best_model')

# check_env(env, warn = True)
# model = PPO.load("mkwiiv4(3 hours)", env)
# model = PPO.load("best_model/best_model")
model = RecurrentPPO('CnnLstmPolicy', 
    env = env, 
    n_steps = 1024, 
    batch_size = 128,
    n_epochs = 4,
    learning_rate = 2.5e-4,
    gae_lambda = 0.98,
    tensorboard_log = LOG_DIR,
    verbose = 1,

    # policy_kwargs = policy_kwargs
    )
# model = PPO(
#     policy = 'CnnPolicy',
#     env = env,
#     n_steps = 1024,
#     batch_size = 64,
#     policy_kwargs = policy_kwargs,
#     n_epochs = 4,
#     gamma = 0.99,
#     learning_rate = 2.5e-4,
#     gae_lambda = 0.98,
#     ent_coef = 0.01,
#     verbose=1, tensorboard_log = LOG_DIR)
# for i in range(1, 20):
#     model.learn(15_000, progress_bar = True)
#     model_name = f"mkwiiv1{15_000 * i}"
#     model.save(model_name)
model.learn(100_000, progress_bar=True, callback=callback)
model.save(model_name)

# obs = env.reset()
# dones = False
# while not dones:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()

# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=4, deterministic=True)
# print(mean_reward)
# print(std_reward)
# Enjoy trained agent
# obs = env.reset()
# for i in range(10):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, dones, info = env.step(action)
#     env.render()

# for i in range(50):
#     dones = False
#     obs = env.reset()
#     while not dones:
#         action, _states = model.predict(obs)
#         obs, rewards, dones, info = env.step(action)
        # env.render()

'''
model = PPO2('MlpPolicy', env, verbose=1)

# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# Memory fraction, used mostly when training multiple agents
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
# backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        self.model = model
        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter

        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()


# Agent class
class DQNAgent:
    def __init__(self, model = False):

        # Main model
        if model:
            self.model = model
        else:
            self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir=LOG_DIR)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), input_shape=env.OBSERVATION_SPACE_VALUES)) #256 # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3))) #256
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(128)) # 64
        model.add(Activation('relu'))
        model.add(Dense(256)) # 64
        model.add(Activation('relu'))


        model.add(Dense(env.ACTION_SPACE_SIZE, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        model.summary()
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode

    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X)/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

cont_model = tf.keras.models.load_model('models/temp_model.model')
agent = DQNAgent(cont_model)

# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:
        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)

        new_state, reward, done = env.step(action)

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
        agent.model.save('models/temp_model.model')
        K.clear_session()
        saved_model = tf.keras.models.load_model('models/temp_model.model')
        agent = DQNAgent(saved_model)

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
'''