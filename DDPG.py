import os
import gymnasium as gym
import numpy as np
import csv
import gymnasium_robotics
from stable_baselines3 import DDPG
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

class TrainAndEvalCallback(BaseCallback):
    def __init__(self, check_freq: int, eval_freq: int, csv_path: str, eval_env, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.eval_freq = eval_freq
        self.csv_path = csv_path
        self.eval_env = eval_env
        self.file_exists = os.path.exists(self.csv_path)
        self.episodes = 0
        self.successes = []  

    def _on_training_start(self) -> None:
        if not self.file_exists:
            with open(self.csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["timestep", "mean_episode_reward", "actor_loss", "critic_loss", "success_rate"])

    def _on_step(self) -> bool:
        """Saves training and evaluation metrics periodically."""
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])  

        for info in infos:
            if "is_success" in info:
                self.successes.append(info["is_success"])  

        if any(dones):
            self.episodes += 1

            if self.episodes % self.check_freq == 0 and self.model.num_timesteps >= self.model.learning_starts:

                actor_loss = self.model.logger.name_to_value.get("train/actor_loss", None)
                critic_loss = self.model.logger.name_to_value.get("train/critic_loss", None)

                mean_success_rate = np.mean(self.successes) if self.successes else 0.0
                self.successes = []

                if actor_loss is not None and critic_loss is not None:
                    with open(self.csv_path, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([
                            self.num_timesteps,
                            self.model.logger.name_to_value.get("rollout/ep_rew_mean", 0.0),
                            actor_loss,
                            critic_loss,
                            mean_success_rate
                        ])

            if self.episodes % self.eval_freq == 0:
                eval_rewards, eval_success = self.evaluate() 
                print(f"\n🔹 Evaluation: Success Rate = {eval_success:.2f}, Reward = {eval_rewards:.2f}")

        return True

    def evaluate(self):
        """Runs evaluation episodes and returns average reward and success rate."""
        eval_rewards = []
        eval_success = []
        
        for _ in range(10):  
            obs = self.eval_env.reset()
            done = np.array([False])
            truncated = np.array([False])
            total_reward = 0.0
            episode_success = 0
            
            while not (done.any() or truncated.any()): 
                action, _ = self.model.predict(obs, deterministic=True)
                obs, rewards, done, infos = self.eval_env.step(action)
                total_reward += rewards[0]
                truncated = np.array([infos[0].get("TimeLimit.truncated", False)])
                if done.any() or truncated.any():
                    episode_success = infos[0].get('is_success', 0)
            
            eval_rewards.append(total_reward)
            eval_success.append(episode_success)
            
        return np.mean(eval_rewards), np.mean(eval_success)

def make_env(env_id, rank):
    def _init():
        env = gym.make(env_id, render_mode=None)
        env = Monitor(env, filename=f"./logs/monitor_{rank}.csv", info_keywords=("is_success",))
        return env
    return _init

def main():
    env_id = "FetchPush-v4"
    num_envs = 8


    vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_envs)])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=200.0,
        gamma=0.98,
    )


    eval_env = DummyVecEnv([lambda: Monitor(gym.make(env_id, render_mode=None), filename="./logs/eval_monitor.csv", info_keywords=("is_success",))])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=200.0, gamma=0.98)
    eval_env.obs_rms = vec_env.obs_rms 
    eval_env.training = False  
    eval_env.norm_reward = False  

    policy_kwargs = dict(net_arch=[256, 256])
    n_actions = vec_env.action_space.shape[-1]

    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions),
        theta=0.15,
        dt=1e-2
    )

    model = DDPG(
        policy="MultiInputPolicy",
        env=vec_env,
        learning_rate=1e-3,
        gamma=0.98,
        tau=0.05,
        batch_size=256,
        buffer_size=int(1e6),
        action_noise=action_noise,
        policy_kwargs=policy_kwargs,
        learning_starts=1000,
        train_freq=5,
        gradient_steps=10,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            "env": vec_env, 
            "n_sampled_goal": 4,
            "goal_selection_strategy": GoalSelectionStrategy.FUTURE
        },
        verbose=1,
    )

    callback = TrainAndEvalCallback(
        check_freq=1,
        eval_freq=10,
        csv_path="training_stats.csv",
        eval_env=eval_env
    )

    model.learn(
        total_timesteps=1_000_000,
        callback=callback
    )

    model.save("ddpg_her_fetch_reach")
    vec_env.save("vec_normalize.pkl")

    vec_env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
