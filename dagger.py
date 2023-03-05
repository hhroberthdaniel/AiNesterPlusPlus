import gc
import os
import random
import time
import uuid

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.cross_attention_vit import CrossAttentionVit
from nesting_environment import NestingEnvironment, NestEnvConfig
import ray
import numpy as np
import wandb
from nesting_states_dataset import NestingStatesDataset, prepare_state

import os
import wandb
from utils import count_parameters

run = wandb.init(project="AiNesterPlusPlus")
SINGLE_THREADED = False
TEST = False

ray.init()


class DaggerConfig:
    NUM_EPOCHS_PER_ITERATION = 10 if not TEST else 1
    NUM_STEPS = 100_000 if not TEST else 100
    # NUM_STEPS = 4000 if not TEST else 100
    NUM_EVAL_STEPS = 1000 if not TEST else 200
    NUM_ROUNDS = 10 if not TEST else 2
    SCRATCH_DIR = os.path.join("./scratch", time.strftime("%Y%m%d-%H%M%S"))
    NUM_ACTORS = 1 if SINGLE_THREADED else 6

    DEVICE = torch.device("cuda")


class Policy:
    def get_action(self, state, info=None):
        return None


class StochasticPolicy:
    def __init__(self, model, ne, device="cuda", temperature=None):
        self.device = device
        self.ne = ne
        self.temperature = temperature
        self.model = model
        self.model = model.to(self.device)
        self.model.eval()

    def get_action(self, state, info=None):
        cfg = NestEnvConfig()
        poly_idxs = state["poly_idxs"]
        num_polys = poly_idxs.shape[0]

        state, poly_idxs, mask = prepare_state(state, cfg.max_num_of_polygons_after_merge)
        state, poly_idxs, mask = torch.from_numpy(state), torch.from_numpy(poly_idxs), torch.from_numpy(mask)
        state, poly_idxs, mask = state.unsqueeze(0), poly_idxs.unsqueeze(0), mask.unsqueeze(0)
        state, poly_idxs, mask = state.to(self.device), poly_idxs.to(self.device), mask.to(self.device)

        x = (state, poly_idxs, mask)
        with torch.no_grad():
            model_action = self.model(x)[0]
        model_action[..., 0] *= self.ne.cfg.raster_width
        model_action[..., 1] *= self.ne.cfg.raster_height
        model_action = model_action[:num_polys]
        model_action = model_action.cpu().numpy().tolist()
        # gc.collect()
        # torch.cuda.empty_cache()

        return model_action


class DaggerPolicy(Policy):
    def __init__(self, model, beta, ne, device="cuda"):
        self.device = torch.device(device)
        self.ne = ne
        self.beta = beta
        self.model = model.to(self.device)
        self.model.eval()
        self.last_ideal_action = None

    def get_action(self, state, ideal_actions=None):
        resulting_actions = []
        cfg = NestEnvConfig()
        state, poly_idxs, mask = prepare_state(state, cfg.max_num_of_polygons_after_merge)
        state, poly_idxs, mask = torch.from_numpy(state), torch.from_numpy(poly_idxs), torch.from_numpy(mask)
        state, poly_idxs, mask = state.unsqueeze(0), poly_idxs.unsqueeze(0), mask.unsqueeze(0)
        state, poly_idxs, mask = state.to(self.device), poly_idxs.to(self.device), mask.to(self.device)

        x = (state, poly_idxs, mask)

        with torch.no_grad():
            model_action = self.model(x)[0]


        model_action[...,0] *= self.ne.cfg.raster_width
        model_action[...,1] *= self.ne.cfg.raster_height
        model_action = model_action.cpu().numpy().tolist()



        for ideal_a, model_a in zip(ideal_actions, model_action):
            if random.random() > self.beta:
                resulting_actions.append(model_a)
            else:
                resulting_actions.append(ideal_a)

        return resulting_actions



# @ray.remote
@ray.remote(num_gpus=0.33)
class PolicyUnroller(object):
    def __init__(self, ne: NestingEnvironment, policy, out_dir=None, save=False):
        self.save_states = save
        self.out_dir = out_dir
        self.policy = policy
        self.ne = ne

    def run_steps(self, n_steps):
        steps_executed = 0
        done = True
        state = None

        episode_rewards = []
        while steps_executed < n_steps:
            if done:
                if steps_executed > 0:
                    episode_rewards.append(ep_reward)
                state = self.ne.reset()
                ep_reward = 0

            ideal_action = self.ne.get_ideal_actions()
            action = self.policy.get_action(state, self.ne.get_ideal_actions())
            obs, reward, done, info = self.ne.step(action)
            ep_reward += reward
            if self.save_states:
                self.save(obs, ideal_action, info)
            steps_executed += 1

        return episode_rewards

    def save(self, obs, ideal_action, info=None):
        obs["ideal_action"] = ideal_action
        uid = uuid.uuid1()
        out_path = os.path.join(self.out_dir, f"{str(uid)}")
        np.savez_compressed(out_path, **obs)


class DaggerTrainer:
    def __init__(self, cfg: DaggerConfig):
        self.cfg = cfg
        nec = NestEnvConfig()
        self.model = CrossAttentionVit(image_size=nec.raster_height, patch_size=10, vit_channels=2)
        self.data_dirs = []
        print("Number of parameters", count_parameters(self.model))

    def run_round(self, run_idx):
        self.collect_data(run_idx)
        self.train_model(run_idx)

    def collect_data(self, run_idx):
        print("Starting to collect data")
        start_time = time.time()
        out_dir = os.path.join(self.cfg.SCRATCH_DIR, str(run_idx))
        os.makedirs(out_dir, exist_ok=True)

        policy_unrollers = []
        for _ in range(self.cfg.NUM_ACTORS):
            nec = NestEnvConfig()
            ne = NestingEnvironment(nec)
            dp = DaggerPolicy(self.model, self.beta, ne)
            policy_unrollers.append(PolicyUnroller.remote(ne, dp, out_dir, True))

        steps_per_actor = self.cfg.NUM_STEPS // self.cfg.NUM_ACTORS
        ray.get([pu.run_steps.remote(steps_per_actor) for pu in policy_unrollers])

        print("Finished collecting data, total time :", time.time() - start_time)
        self.data_dirs.append(out_dir)


    def train_model(self, run_idx):
        torch.cuda.empty_cache()
        nest_cfg = NestEnvConfig()
        #todo add dirs from all the last rounds
        ds = NestingStatesDataset(self.data_dirs, nest_cfg.max_num_of_polygons_after_merge, nest_cfg.raster_width, nest_cfg.raster_height)
        print(f"Training on {len(ds)} samples")
        dl = DataLoader(ds, batch_size=128, num_workers=2)
        loss = nn.MSELoss(reduction='none')
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-7)
        self.model.train()
        # self.model.to(self.cfg.DEVICE)
        parallel_net = nn.DataParallel(self.model, device_ids = list(range(torch.cuda.device_count()))).cuda()
        model_save_path = './checkpoints'
        os.makedirs(model_save_path, exist_ok=True)
        for epoch in range(self.cfg.NUM_EPOCHS_PER_ITERATION):
            epoch_losses = []
            for batch in dl:
                state, poly_idxs, ideal_action, mask = batch
                state, poly_idxs, ideal_action, mask = state.to(self.cfg.DEVICE), poly_idxs.to(self.cfg.DEVICE), ideal_action.to(self.cfg.DEVICE), mask.to(self.cfg.DEVICE)
                x = (state, poly_idxs, mask)

                optimizer.zero_grad()
                out = parallel_net(x)
                error = loss(out, ideal_action)
                error = (error * mask.unsqueeze(-1)).mean()
                error.backward()
                optimizer.step()

                wandb.log({"batch loss" : error.item()})
                epoch_losses.append(error.item())

            torch.save(parallel_net.state_dict(), os.path.join(model_save_path, f"{epoch}.pth"))
            gc.collect()
            ep_rewards = self.evaluate_model()
            avg_reward = sum(ep_rewards) / len(ep_rewards)
            num_episodes = len(ep_rewards)
            avg_loss = sum(epoch_losses) / len(epoch_losses)

            print(sum(ep_rewards) / len(ep_rewards))

            wandb.log({
                "Epoch loss":avg_loss,
                "Epoch Avg Reward": avg_reward,
                "Epoch Num Episodes": num_episodes,

            })


    def evaluate_model(self):
        policy_unrollers = []
        for _ in range(self.cfg.NUM_ACTORS):
            nec = NestEnvConfig()
            ne = NestingEnvironment(nec)
            sp = StochasticPolicy(self.model, ne)
            policy_unrollers.append(PolicyUnroller.remote(ne, sp))

        steps_per_actor = self.cfg.NUM_EVAL_STEPS // self.cfg.NUM_ACTORS
        ep_rewards = ray.get([pu.run_steps.remote(steps_per_actor) for pu in policy_unrollers])
        print(ep_rewards)
        ep_rewards = [item for sublist in ep_rewards for item in sublist]
        return ep_rewards


    def run(self):
        for run_idx in range(1, self.cfg.NUM_ROUNDS + 1):
            print("Starting run", run_idx)
            self.beta = 1 - run_idx / self.cfg.NUM_ROUNDS
            start_time = time.time()
            self.run_round(run_idx)
            print("Round time ", time.time() - start_time)


if __name__ == "__main__":
    print("NUM GPUS", torch.cuda.device_count())
    dt = DaggerTrainer(DaggerConfig())
    dt.run()
