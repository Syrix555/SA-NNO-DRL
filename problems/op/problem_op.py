from torch.utils.data import Dataset
import torch
import math
from op_data_utils import load_op_dataset
from problems.op.state_op import StateOP
from utils.beam_search import beam_search


class OP(object):

    NAME = 'op'  # Orienteering problem

    @staticmethod
    def get_costs(dataset, pi):
        if pi.size(-1) == 1:  # In case all tours directly return to depot, prevent further problems
            assert (pi == 0).all(), "If all length 1 tours, they should be zero"
            # Return
            return torch.zeros(pi.shape[:2], dtype=torch.float, device=pi.device), None

        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(-1)[0]
        # Make sure each node visited once at most (except for depot)
        assert ((sorted_pi[:, :, 1:] == 0) | (sorted_pi[:, :, 1:] > sorted_pi[:, :, :-1])).all(), "Duplicates"

        prize_with_depot = torch.cat(
            (
                torch.zeros_like(dataset['prize'][:, :1]),
                dataset['prize']
            ),
            1
        )[:, None, :].expand(pi.size(0), pi.size(1), -1)
        p = prize_with_depot.gather(-1, pi)

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)[:, None, :, :].expand(pi.size(0), pi.size(1), -1, 2)
        d = loc_with_depot.gather(-2, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        length = (
            (d[:, :, 1:] - d[:, :, :-1]).norm(p=2, dim=-1).sum(-1)  # Prevent error if len 1 seq
            + (d[:, :, 0] - dataset['depot'][:, None]).norm(p=2, dim=-1)  # Depot to first
            + (d[:, :, -1] - dataset['depot'][:, None]).norm(p=2, dim=-1)  # Last to depot, will be 0 if depot is last
        )
        assert (length <= dataset['max_length'][:, None] + 1e-5).all(), \
            "Max length exceeded by {}".format((length - dataset['max_length'][:, None]).max())

        # We want to maximize total prize but code minimizes so return negative
        return -p.sum(-1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return OPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateOP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = OP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


def generate_instance(size, prize_type, num_samples, uav_num=5, max_length=2.0):
    """Generate exactly num_samples angular sectors of size nodes each.

    One parent mission contains uav_num * size nodes. Its sorted angular
    sectors are training instances; any surplus sectors in the last mission
    are discarded. num_samples counts sectors, rather than parent missions.
    """
    if size <= 0 or uav_num <= 0 or num_samples < 0:
        raise ValueError("size/uav_num must be positive; num_samples must be nonnegative")
    if not math.isfinite(max_length) or max_length <= 0:
        raise ValueError("max_length must be finite and positive")
    if prize_type not in ('const', 'unif', 'dist'):
        raise ValueError("OP distribution must be const, unif or dist")

    data = []
    for _ in range((num_samples + uav_num - 1) // uav_num):
        location = torch.rand(uav_num * size, 2, dtype=torch.float32)
        depot = torch.tensor([0.5, 0.5], dtype=torch.float32)
        lab = location - depot
        relocation = location[torch.atan2(lab[:, 1], lab[:, 0]).argsort()]
        for j in range(min(uav_num, num_samples - len(data))):
            loc = relocation[j * size: (j + 1) * size]
            if prize_type == 'const':
                prize = torch.ones(size, dtype=torch.float32)
            elif prize_type == 'unif':
                prize = (1 + torch.randint(0, 100, size=(size,))).float() / 100
            else:
                distance = (depot[None, :] - loc).norm(p=2, dim=-1)
                scale = distance.max().clamp_min(torch.finfo(torch.float32).eps)
                prize = (1 + (distance / scale * 99).int()).float() / 100
            data.append({
                'loc': loc,
                'prize': prize,
                'depot': depot,
                'max_length': torch.tensor(max_length, dtype=torch.float32)
            })
    return data


class OPDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0,
                 distribution='const', uav_num=5, max_length=2.0):
        super().__init__()
        if offset < 0 or num_samples < 0:
            raise ValueError("offset and num_samples must be nonnegative")
        if filename is not None:
            self.data = [
                {key: torch.from_numpy(value.copy()) for key, value in record.items()}
                for record in load_op_dataset(filename, num_samples=num_samples, offset=offset)
            ]
        else:
            if offset != 0:
                raise ValueError("offset is only supported for file-backed OP datasets")
            self.data = generate_instance(size, distribution, num_samples, uav_num, max_length)
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
