#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl
import utils
import hydra

from logger import Logger
from replay_buffer import ReplayBuffer
from omegaconf import OmegaConf

class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')
        self.cfg = cfg
        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency)

        utils.set_seed_everywhere(cfg.seed)

        self.device = torch.device(cfg.device)
        self.log_success = False
        self.step = 0
        
        # make env
        if 'metaworld' in cfg.env:
            self.env = utils.make_metaworld_env(cfg)
            self.log_success = True
        else:
            self.env = utils.make_env(cfg)

        cfg.agent.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.action_dim = self.env.action_space.shape[0]
        cfg.agent.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]

        # eval the cfg 
        agent_cfg = OmegaConf.to_container(cfg.agent, resolve=True)
        agent_cfg = OmegaConf.create(agent_cfg)

        agent_cfg.critic_cfg = OmegaConf.create(agent_cfg.critic_cfg)
        agent_cfg.actor_cfg = OmegaConf.create(agent_cfg.actor_cfg)
        self.agent = hydra.utils.instantiate(agent_cfg, _recursive_=False)
        
        # no relabel
        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device)
        meta_file = os.path.join(self.work_dir, 'metadata.pkl')
        pkl.dump({'cfg': self.cfg}, open(meta_file, "wb"))

        # Replay buffer collection control
        self.collect_replay_buffer = getattr(cfg, 'collect_replay_buffer', True)
        self.target_success_rate = getattr(cfg, 'target_success_rate', 50.0)
        self.target_episode_return = getattr(cfg, 'target_episode_return', 1000.0)  # For non-MetaWorld envs
        self.replay_buffer_saved = False

    def save_replay_buffer(self):
        """Save the replay buffer to disk"""
        if self.replay_buffer_saved:
            return
            
        save_path = os.path.join(self.work_dir, f'replay_buffer_step_{self.step}.pkl')
        
        # Create a dictionary with replay buffer data and metadata
        buffer_data = {
            'observations': self.replay_buffer.obses[:self.replay_buffer.idx],
            'actions': self.replay_buffer.actions[:self.replay_buffer.idx],
            'rewards': self.replay_buffer.rewards[:self.replay_buffer.idx],
            'next_observations': self.replay_buffer.next_obses[:self.replay_buffer.idx],
            'dones': self.replay_buffer.not_dones[:self.replay_buffer.idx],
            'step': self.step,
            'size': self.replay_buffer.idx,
            'env': self.cfg.env,
        }
        
        with open(save_path, 'wb') as f:
            pkl.dump(buffer_data, f)
        
        self.replay_buffer_saved = True
        print(f"Replay buffer saved to: {save_path}")
        print(f"Buffer size: {self.replay_buffer.idx}")
        print("Training will end after saving replay buffer.")
        
        # Also save a summary file
        summary_path = os.path.join(self.work_dir, 'replay_buffer_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Replay Buffer Collection Summary\n")
            f.write(f"Environment: {self.cfg.env}\n")
            f.write(f"Collection stopped at step: {self.step}\n")
            f.write(f"Buffer size: {self.replay_buffer.idx}\n")

    def evaluate(self):
        average_episode_reward = 0
        if self.log_success:
            success_rate = 0
            
        for episode in range(self.cfg.num_eval_episodes):
            obs, info = self.env.reset()
            self.agent.reset()
            done = False
            episode_reward = 0
            
            if self.log_success:
                episode_success = 0

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, truncated, done, extra = self.env.step(action)
                episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra['success'])

            average_episode_reward += episode_reward
            if self.log_success:
                success_rate += episode_success
            
        average_episode_reward /= self.cfg.num_eval_episodes
        if self.log_success:
            success_rate /= self.cfg.num_eval_episodes
            success_rate *= 100.0

        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        
        if self.log_success:
            self.logger.log('eval/success_rate', success_rate,
                        self.step)
        self.logger.dump(self.step)
        
        # Check if we should save replay buffer based on current performance
        if self.collect_replay_buffer and not self.replay_buffer_saved and self.step % 50000 == 0:
            should_save = False
            
            if self.log_success:
                # For MetaWorld environments - check success rate
                if success_rate >= self.target_success_rate:
                    print(f"Target success rate reached: {success_rate:.2f}% >= {self.target_success_rate}%")
                    should_save = True
            else:
                # For other environments - check episode return
                if average_episode_reward >= self.target_episode_return:
                    print(f"Target episode return reached: {average_episode_reward:.2f} >= {self.target_episode_return}")
                    should_save = True
            
            if should_save:
                self.save_replay_buffer()
                self.collect_replay_buffer = False  # Stop further collection
        
    def run(self):
        episode, episode_reward, done = 0, 0, True
        if self.log_success:
            episode_success = 0
        start_time = time.time()
        fixed_start_time = time.time()

        while self.step < self.cfg.num_train_steps:
            # Check if replay buffer was saved and end training
            if self.replay_buffer_saved:
                print("Replay buffer saved. Ending training early.")
                break
                
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    self.logger.log('train/total_duration',
                                    time.time() - fixed_start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)

                if self.log_success:
                    self.logger.log('train/episode_success', episode_success,
                        self.step)
                            
                obs, info = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                if self.log_success:
                    episode_success = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update             
            if self.step == (self.cfg.num_seed_steps + self.cfg.num_unsup_steps) and self.cfg.num_unsup_steps > 0:
                # reset Q due to unsuperivsed exploration
                self.agent.reset_critic()
                # update agent
                self.agent.update_after_reset(
                    self.replay_buffer, self.logger, self.step, 
                    gradient_update=self.cfg.reset_update, 
                    policy_update=True)
            elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)
            # unsupervised exploration
            elif self.step > self.cfg.num_seed_steps:
                self.agent.update_state_ent(self.replay_buffer, self.logger, self.step, 
                                            gradient_update=1, K=self.cfg.topK)
            
            
            next_obs, reward, truncated, done, extra = self.env.step(action)    
              
            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward
            
            if self.log_success:
                episode_success = max(episode_success, extra['success'])
                
            self.replay_buffer.add(
                obs, action, 
                reward, next_obs, done,
                done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1

        self.agent.save(self.work_dir, self.step)
        
        # Save replay buffer at the end if collection is still active and training completed normally
        if self.collect_replay_buffer and not self.replay_buffer_saved:
            print("Training completed without reaching target. Saving replay buffer...")
            self.save_replay_buffer()
        
@hydra.main(config_path='config', config_name='train', version_base=None)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()

if __name__ == '__main__':
    main()
