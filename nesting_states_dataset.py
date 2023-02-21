import os

import numpy as np
from torch.utils.data import Dataset, DataLoader

def prepare_state(sample_data, max_num_of_polys):
    # state = np.expand_dims(sample_data["state"], 0)
    state = sample_data["state"]
    poly_idxs = sample_data["poly_idxs"]

    num_polys = poly_idxs.shape[0]

    poly_idxs = np.expand_dims(np.pad(poly_idxs, (0, max_num_of_polys - num_polys), 'constant'), axis=-1)

    mask = np.zeros(max_num_of_polys)
    mask[:num_polys] = 1
    mask = np.array(mask, dtype=np.bool)
    state, poly_idxs = np.array(state, dtype=np.float32), np.array(poly_idxs, dtype=np.float32)
    return state, poly_idxs, mask


def prepare_sample(sample_data, max_grid_width, max_grid_height, max_num_of_polys):
    state, poly_idxs, mask = prepare_state(sample_data, max_num_of_polys)
    num_polys = mask.sum()
    ideal_action = np.array(sample_data["ideal_action"], dtype=np.float32)
    ideal_action[:, 0] /= max_grid_width
    ideal_action[:, 1] /= max_grid_height
    ideal_action = np.pad(ideal_action, ((0, max_num_of_polys - num_polys), (0, 0)), 'constant',
                          constant_values=(np.array([0, 0])))

    return state, poly_idxs, mask, ideal_action


class NestingStatesDataset(Dataset):
    def __init__(self, data_dir, max_num_of_polys, max_grid_width, max_grid_height):
        self.max_grid_height = max_grid_height
        self.max_grid_width = max_grid_width
        self.max_num_of_polys = max_num_of_polys
        self.all_files = os.listdir(data_dir)
        self.all_files = [os.path.join(data_dir, f) for f in self.all_files]

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        data = np.load(self.all_files[idx], allow_pickle=True)
        state, poly_idxs, mask, ideal_action = prepare_sample(data, self.max_grid_width, self.max_grid_height, self.max_num_of_polys)
        # return
        # state = np.expand_dims(data["state"], 0)
        # poly_idxs = data["poly_idxs"]
        # ideal_action = np.array(data["ideal_action"])
        # ideal_action[:,0] /= self.max_grid_width
        # ideal_action[:,1] /= self.max_grid_height
        #
        # num_polys = poly_idxs.shape[0]
        #
        # poly_idxs = np.expand_dims(np.pad(poly_idxs, (0, self.max_num_of_polys - num_polys), 'constant'), axis=-1)
        # ideal_action = np.pad(ideal_action, ((0, self.max_num_of_polys - num_polys), (0,0)), 'constant', constant_values=(np.array([np.inf, np.inf])))
        #
        # mask = np.zeros(self.max_num_of_polys)
        # mask[:num_polys] = 1
        # mask = np.array(mask, dtype=np.bool)
        # state, poly_idxs = np.array(state, dtype=np.float32), np.array(poly_idxs, dtype=np.float32)

        return state, poly_idxs, ideal_action, mask

if __name__ == "__main__":
    ds = NestingStatesDataset(r"C:\Proiecte\AiNesterPlusPlus\scratch\1", 1000, 1000, 1000)
    dl = DataLoader(ds, batch_size=16)
    from models.cross_attention_vit import CrossAttentionVit
    m = CrossAttentionVit()
    for state, poly_idxs, ideal_action, mask in dl:
        print(state.shape)
        x = (state, poly_idxs, mask)
        y = m(x)
        print(y)
    print(len(ds))
    print(ds[0])
    a = ds[0]
    poly_rasters = a["rasters"]
    print(poly_rasters)