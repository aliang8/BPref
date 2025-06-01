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
import tqdm

from logger import Logger
from replay_buffer import ReplayBuffer
from reward_model import RewardModel
from collections import deque
from omegaconf import OmegaConf

import utils
import hydra

class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.log_success = False
        
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

        agent_cfg = OmegaConf.to_container(cfg.agent, resolve=True)
        agent_cfg = OmegaConf.create(agent_cfg)
        agent_cfg.critic_cfg = OmegaConf.create(agent_cfg.critic_cfg)
        agent_cfg.actor_cfg = OmegaConf.create(agent_cfg.actor_cfg)
        self.agent = hydra.utils.instantiate(agent_cfg, _recursive_=False)

        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device)
        
        # for logging
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0

        # instantiating the reward model
        self.reward_model = RewardModel(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            ensemble_size=cfg.ensemble_size,
            size_segment=cfg.segment,
            activation=cfg.activation, 
            lr=cfg.reward_lr,
            mb_size=cfg.reward_batch, 
            large_batch=cfg.large_batch, 
            label_margin=cfg.label_margin, 
            teacher_beta=cfg.teacher_beta, 
            teacher_gamma=cfg.teacher_gamma, 
            teacher_eps_mistake=cfg.teacher_eps_mistake, 
            teacher_eps_skip=cfg.teacher_eps_skip, 
            teacher_eps_equal=cfg.teacher_eps_equal)
        
        # Replay buffer collection control
        self.collect_replay_buffer = getattr(cfg, 'collect_replay_buffer', True)
        self.target_success_rate = getattr(cfg, 'target_success_rate', 50.0)
        self.target_return_percentile = getattr(cfg, 'target_return_percentile', 0.5)  # Middle of convergence value
        self.eval_frequency_for_collection = getattr(cfg, 'eval_frequency_for_collection', 50000)  # Every 50k steps
        self.replay_buffer_saved = False
        self.performance_history = deque(maxlen=10)  # Track recent performance
        self.convergence_window = deque(maxlen=20)  # Larger window to estimate convergence
        
    def evaluate(self):
        average_episode_reward = 0
        average_true_episode_reward = 0
        success_rate = 0
        
        for episode in range(self.cfg.num_eval_episodes):
            obs, info = self.env.reset()
            self.agent.reset()
            done = False
            episode_reward = 0
            true_episode_reward = 0
            if self.log_success:
                episode_success = 0

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, truncated, done, extra = self.env.step(action)
                
                episode_reward += reward
                true_episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra['success'])
                
            average_episode_reward += episode_reward
            average_true_episode_reward += true_episode_reward
            if self.log_success:
                success_rate += episode_success
            
        average_episode_reward /= self.cfg.num_eval_episodes
        average_true_episode_reward /= self.cfg.num_eval_episodes
        if self.log_success:
            success_rate /= self.cfg.num_eval_episodes
            success_rate *= 100.0
        
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.log('eval/true_episode_reward', average_true_episode_reward,
                        self.step)
        if self.log_success:
            self.logger.log('eval/success_rate', success_rate,
                    self.step)
            self.logger.log('train/true_episode_success', success_rate,
                        self.step)
        self.logger.dump(self.step)
        
        return success_rate if self.log_success else average_true_episode_reward

    def check_replay_buffer_collection_stop(self, current_performance):
        """Check if we should stop collecting replay buffer data"""
        if not self.collect_replay_buffer or self.replay_buffer_saved:
            return False
            
        # Add current performance to history
        self.performance_history.append(current_performance)
        self.convergence_window.append(current_performance)
        
        # Need at least a few evaluations to make a decision
        if len(self.performance_history) < 3:
            return False
            
        if self.log_success:
            # For environments with success rate (like metaworld)
            avg_success_rate = np.mean(list(self.performance_history))
            print(f"Current average success rate: {avg_success_rate:.2f}%")
            
            # Stop when success rate is near 50%
            if 45.0 <= avg_success_rate <= 55.0:
                print(f"Stopping replay buffer collection at success rate: {avg_success_rate:.2f}%")
                return True
        else:
            # For DMControl environments - use episode returns
            if len(self.convergence_window) >= 10:
                # Estimate if we're at middle of convergence
                recent_returns = list(self.convergence_window)
                min_return = min(recent_returns)
                max_return = max(recent_returns)
                current_avg = np.mean(list(self.performance_history))
                
                # Check if current performance is around middle of the range
                if max_return > min_return:
                    middle_target = min_return + (max_return - min_return) * self.target_return_percentile
                    tolerance = (max_return - min_return) * 0.1  # 10% tolerance
                    
                    if abs(current_avg - middle_target) <= tolerance:
                        print(f"Stopping replay buffer collection at return: {current_avg:.2f} (target: {middle_target:.2f})")
                        return True
        
        return False

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
            'final_performance': list(self.performance_history)[-1] if self.performance_history else 0
        }
        
        with open(save_path, 'wb') as f:
            pkl.dump(buffer_data, f)
        
        self.replay_buffer_saved = True
        print(f"Replay buffer saved to: {save_path}")
        print(f"Buffer size: {self.replay_buffer.idx}")
        
        # Also save a summary file
        summary_path = os.path.join(self.work_dir, 'replay_buffer_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Replay Buffer Collection Summary\n")
            f.write(f"Environment: {self.cfg.env}\n")
            f.write(f"Collection stopped at step: {self.step}\n")
            f.write(f"Buffer size: {self.replay_buffer.idx}\n")
            if self.log_success:
                f.write(f"Final success rate: {list(self.performance_history)[-1]:.2f}%\n")
            else:
                f.write(f"Final episode return: {list(self.performance_history)[-1]:.2f}\n")
            f.write(f"Performance history: {list(self.performance_history)}\n")
        
    def learn_reward(self, first_flag=0):
                
        # get feedbacks
        labeled_queries, noisy_queries = 0, 0
        if first_flag == 1:
            # if it is first time to get feedback, need to use random sampling
            labeled_queries = self.reward_model.uniform_sampling()
        else:
            if self.cfg.feed_type == 0:
                labeled_queries = self.reward_model.uniform_sampling()
            elif self.cfg.feed_type == 1:
                labeled_queries = self.reward_model.disagreement_sampling()
            elif self.cfg.feed_type == 2:
                labeled_queries = self.reward_model.entropy_sampling()
            elif self.cfg.feed_type == 3:
                labeled_queries = self.reward_model.kcenter_sampling()
            elif self.cfg.feed_type == 4:
                labeled_queries = self.reward_model.kcenter_disagree_sampling()
            elif self.cfg.feed_type == 5:
                labeled_queries = self.reward_model.kcenter_entropy_sampling()
            else:
                raise NotImplementedError
        
        self.total_feedback += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries
        
        train_acc = 0
        if self.labeled_feedback > 0:
            # update reward
            for epoch in range(self.cfg.reward_update):
                if self.cfg.label_margin > 0 or self.cfg.teacher_eps_equal > 0:
                    train_acc = self.reward_model.train_soft_reward()
                else:
                    train_acc = self.reward_model.train_reward()
                total_acc = np.mean(train_acc)
                
                if total_acc > 0.97:
                    break;
                    
        print("Reward function is updated!! ACC: " + str(total_acc))

    def run(self):
        episode, episode_reward, done = 0, 0, True
        if self.log_success:
            episode_success = 0
        true_episode_reward = 0
        
        # store train returns of recent 10 episodes
        avg_train_true_return = deque([], maxlen=10) 
        start_time = time.time()

        # save replay buffer at the beginning
        # self.save_replay_buffer()

        interact_count = 0
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    current_performance = self.evaluate()
                    
                    # Check for replay buffer collection stopping every 50k steps
                    if (self.collect_replay_buffer and 
                        self.step % self.eval_frequency_for_collection == 0 and 
                        self.step > self.cfg.num_seed_steps):
                        
                        should_stop = self.check_replay_buffer_collection_stop(current_performance)
                        if should_stop:
                            self.save_replay_buffer()
                            self.collect_replay_buffer = False
                elif self.step > 0 and self.step % self.eval_frequency_for_collection == 0:
                    # Even if not regular eval time, check for replay buffer collection
                    if self.collect_replay_buffer and self.step > self.cfg.num_seed_steps:
                        self.logger.log('eval/episode', episode, self.step)
                        current_performance = self.evaluate()

                        should_stop = self.check_replay_buffer_collection_stop(current_performance)
                        if should_stop:
                            self.save_replay_buffer()
                            self.collect_replay_buffer = False
                
                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.logger.log('train/true_episode_reward', true_episode_reward, self.step)
                self.logger.log('train/total_feedback', self.total_feedback, self.step)
                self.logger.log('train/labeled_feedback', self.labeled_feedback, self.step)
                
                if self.log_success:
                    self.logger.log('train/episode_success', episode_success,
                        self.step)
                    self.logger.log('train/true_episode_success', episode_success,
                        self.step)
                
                obs, info = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                avg_train_true_return.append(true_episode_reward)
                true_episode_reward = 0
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
            if self.step == (self.cfg.num_seed_steps + self.cfg.num_unsup_steps):
                # self.save_replay_buffer()
                # update schedule
                if self.cfg.reward_schedule == 1:
                    frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                    if frac == 0:
                        frac = 0.01
                elif self.cfg.reward_schedule == 2:
                    frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                else:
                    frac = 1
                self.reward_model.change_batch(frac)
                
                # update margin --> not necessary / will be updated soon
                new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                self.reward_model.set_teacher_thres_skip(new_margin)
                self.reward_model.set_teacher_thres_equal(new_margin)
                
                # first learn reward
                self.learn_reward(first_flag=1)
                
                # relabel buffer
                self.replay_buffer.relabel_with_predictor(self.reward_model)
                
                # reset Q due to unsuperivsed exploration
                self.agent.reset_critic()
                
                # update agent
                self.agent.update_after_reset(
                    self.replay_buffer, self.logger, self.step, 
                    gradient_update=self.cfg.reset_update, 
                    policy_update=True)
                
                # reset interact_count
                interact_count = 0
            elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:
                # update reward function
                if self.total_feedback < self.cfg.max_feedback:
                    if interact_count == self.cfg.num_interact:
                        # update schedule
                        if self.cfg.reward_schedule == 1:
                            frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                            if frac == 0:
                                frac = 0.01
                        elif self.cfg.reward_schedule == 2:
                            frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                        else:
                            frac = 1
                        self.reward_model.change_batch(frac)
                        
                        # update margin --> not necessary / will be updated soon
                        new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                        self.reward_model.set_teacher_thres_skip(new_margin * self.cfg.teacher_eps_skip)
                        self.reward_model.set_teacher_thres_equal(new_margin * self.cfg.teacher_eps_equal)
                        
                        # corner case: new total feed > max feed
                        if self.reward_model.mb_size + self.total_feedback > self.cfg.max_feedback:
                            self.reward_model.set_batch(self.cfg.max_feedback - self.total_feedback)
                            
                        self.learn_reward()
                        self.replay_buffer.relabel_with_predictor(self.reward_model)
                        interact_count = 0
                        
                self.agent.update(self.replay_buffer, self.logger, self.step, 1)
                
            # unsupervised exploration
            elif self.step > self.cfg.num_seed_steps:
                self.agent.update_state_ent(self.replay_buffer, self.logger, self.step, 
                                            gradient_update=1, K=self.cfg.topK)
                
            next_obs, reward, truncated, done, extra = self.env.step(action)
            reward_hat = self.reward_model.r_hat(np.concatenate([obs, action], axis=-1))

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward_hat
            true_episode_reward += reward
            
            if self.log_success:
                episode_success = max(episode_success, extra['success'])
                
            # adding data to the reward training data
            self.reward_model.add_data(obs, action, reward, done)
            self.replay_buffer.add(
                obs, action, reward_hat, 
                next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
            interact_count += 1
            
        self.agent.save(self.work_dir, self.step)
        self.reward_model.save(self.work_dir, self.step)
        
        # Save replay buffer at the end if collection is still active
        if self.collect_replay_buffer and not self.replay_buffer_saved:
            print("Saving replay buffer at end of training...")
            self.save_replay_buffer()
        
@hydra.main(config_path='config', config_name='train_PEBBLE', version_base=None)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()

if __name__ == '__main__':
    main()