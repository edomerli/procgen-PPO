from torch.utils.data import Dataset
import numpy as np

class TransitionsDataset(Dataset):
    """Dataset class for storing transitions. Each transition is a dictionary with the following keys:
    - 's_t': state at time t
    - 'a_t': action at time t
    - 'A_t': advantage at time t
    - 'v_target_t': value target at time t
    """
    def __init__(self, transitions, transform=None, normalize_v_targets=False, v_mu=None, v_std=None):
        """Constructor

        Args:
            transitions (list[dict]): the list of transitions
            transform (callable, optional): The transformation to apply to each state. Defaults to None.
            normalize_v_targets (bool, optional): Wether to normalize value targets. Defaults to False.
            v_mu (float, optional): Mean of value targets for normalization. Defaults to None.
            v_std (float, optional): Standard deviation of value targets for normalization. Defaults to None.
        """
        self.transitions = transitions
        self.transform = transform
        
        self.normalize_v_targets = normalize_v_targets
        if normalize_v_targets:
            self.v_mu = v_mu
            self.v_std = v_std

    def __len__(self):
        return len(self.transitions)
    
    def __getitem__(self, idx):
        state_t = self.transitions[idx]['s_t']
        action_t = self.transitions[idx]['a_t']
        advantage_t = self.transitions[idx]['A_t']
        v_target_t = self.transitions[idx]['v_target_t']

        if self.transform:
            state_t = self.transform(state_t)

        if self.normalize_v_targets:
            v_target_t = (v_target_t - self.v_mu) / max(self.v_std, 1e-6)

        return state_t, action_t, advantage_t, v_target_t.astype(np.float32)
