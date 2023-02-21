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

from nesting_states_dataset import NestingStatesDataset, prepare_state
# import tracemalloc
# tracemalloc.start()

SINGLE_THREADED = False
TEST = False
# os.environ['OMP_NUM_THREADS'] = '1'

ray.init(local_mode=SINGLE_THREADED)


class DaggerConfig:
    NUM_EPOCHS_PER_ITERATION = 20 if not TEST else 1
    NUM_STEPS = 10000 if not TEST else 100
    NUM_EVAL_STEPS = 300 if not TEST else 200
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
        state, poly_idxs, mask = prepare_state(state, cfg.max_num_of_polygons_after_merge)
        state, poly_idxs, mask = torch.from_numpy(state), torch.from_numpy(poly_idxs), torch.from_numpy(mask)
        state, poly_idxs, mask = state.unsqueeze(0), poly_idxs.unsqueeze(0), mask.unsqueeze(0)
        state, poly_idxs, mask = state.to(self.device), poly_idxs.to(self.device), mask.to(self.device)

        x = (state, poly_idxs, mask)
        with torch.no_grad():
            model_action = self.model(x)[0]
        model_action[..., 0] *= self.ne.cfg.raster_width
        model_action[..., 1] *= self.ne.cfg.raster_height
        model_action = model_action.cpu().numpy().tolist()

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

        # snapshot = tracemalloc.take_snapshot()
        # top_stats = snapshot.statistics('lineno')
        #
        # print("[ Top 10 ]")
        # for stat in top_stats[:10]:
        #     print(stat)

        x = (state, poly_idxs, mask)
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a  # free inside reserved
        torch.cuda.empty_cache()
        gc.collect()

        while f <= 0.3 % t:
            r = torch.cuda.memory_reserved(0)
            a = torch.cuda.memory_allocated(0)
            f = r - a  # free inside rese
            torch.cuda.empty_cache()
            gc.collect()


        print(f)
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



@ray.remote
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
        self.model = CrossAttentionVit(image_size=nec.raster_height, patch_size=40, vit_channels=2)


    def run_round(self, run_idx):
        data_dir = self.collect_data(run_idx)
        self.train_model(run_idx, data_dir)

    def collect_data(self, run_idx):
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
        return out_dir

    def train_model(self, run_idx, data_dir):
        torch.cuda.empty_cache()
        nest_cfg = NestEnvConfig()
        #todo add dirs from all the last rounds
        ds = NestingStatesDataset(data_dir, nest_cfg.max_num_of_polygons_after_merge, nest_cfg.raster_width, nest_cfg.raster_height)
        dl = DataLoader(ds, batch_size=8, num_workers=0, persistent_workers=False)
        loss = nn.MSELoss(reduction='none')
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-6)
        self.model.train()
        self.model.to(self.cfg.DEVICE)

        for epoch in range(self.cfg.NUM_EPOCHS_PER_ITERATION):
            for batch in dl:
                state, poly_idxs, ideal_action, mask = batch
                state, poly_idxs, ideal_action, mask = state.to(self.cfg.DEVICE), poly_idxs.to(self.cfg.DEVICE), ideal_action.to(self.cfg.DEVICE), mask.to(self.cfg.DEVICE)
                x = (state, poly_idxs, mask)

                optimizer.zero_grad()
                out = self.model(x)
                error = loss(out, ideal_action)
                error = (error * mask.unsqueeze(-1)).mean()
                error.backward()
                optimizer.step()
            ep_rewards = self.evaluate_model()
            print(sum(ep_rewards) / len(ep_rewards))


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
            self.beta = 1 - run_idx / self.cfg.NUM_ROUNDS
            start_time = time.time()
            self.run_round(run_idx)
            print("Round time ", time.time() - start_time)


if __name__ == "__main__":
    dt = DaggerTrainer(DaggerConfig())
    dt.run()
