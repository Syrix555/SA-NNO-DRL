import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import math
from torch.utils.data import DataLoader
import torch
import os
import time
import argparse
from train import augment
from make_dataset import make_dataset
from inference_results import InferenceResults, instance_result
from runtime_config import resolve_model_path, resolve_device
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# load model
from utils import load_model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def angle_sort(task, depot):

    lab = task - depot
    angle = np.arctan2(lab[:, 1], lab[:, 0])
    index = angle.argsort()
    angle = angle[index]

    return angle, index

def change_node(loc, depot, MAX_LENGTHS):
    loc = np.concatenate((loc, depot[None, :]), 0)
    # distance1 = np.linalg.norm(loc[:-1] - loc[1:], axis=-1)
    # plt.scatter(loc[:, 0], loc[:, 1])
    # plt.show()
    x_max = np.max(loc[:, 0])
    x_min = np.min(loc[:, 0])
    y_max = np.max(loc[:, 1])
    y_min = np.min(loc[:, 1])
    # scale factor
    span = max(x_max - x_min, y_max - y_min)
    if span <= 0:
        raise ValueError("Cannot normalize a degenerate task geometry")
    s = 1. / span
    new_length = s*MAX_LENGTHS
    loc[:, 0] = s*(loc[:, 0] - x_min)
    loc[:, 1] = s*(loc[:, 1] - y_min)
    new_loc = loc[:-1]
    new_depot = loc[-1]
    # distance2 = np.linalg.norm(loc[:-1] - loc[1:], axis=-1)
    # a = distance2 / distance1
    # plt.scatter(loc[:, 0], loc[:, 1])
    # plt.show()
    # plt.close()
    return new_loc, new_depot, new_length

def cal_fitness(dataset, uav_num):
    if len(dataset) != uav_num:
        raise ValueError("SA model batch must contain exactly one instance per UAV")
    dataloader = DataLoader(dataset, batch_size=uav_num)
    batch = next(iter(dataloader))
    N_aug = 4
    batch = augment(batch, N_aug, 'op')

    with torch.no_grad():
        collect_p, _, log_p = model(batch, return_pi=True)
    _, k_size, n_length = log_p.size()
    collect_p = collect_p.view(N_aug, uav_num, k_size).permute(1, 2, 0).reshape(-1, N_aug * k_size)
    log_p = log_p.view(N_aug, uav_num, k_size, n_length).permute(1, 2, 0, 3).reshape(-1, N_aug * k_size, n_length)
    fitness = (-collect_p).max(-1)[0]  # 适应度值
    pi = torch.gather(log_p, 1, (-collect_p).max(-1)[1].contiguous()
                      .view(uav_num, 1, 1)
                      .expand(uav_num, 1, log_p.size(-1))).squeeze(1)

    return fitness.cpu().numpy(), pi.cpu().numpy()


def validate_allocation(task, depot, prize, max_length, index_table, line, tolerance=1e-5,
                        expected_prize=None):
    """Validate assignment uniqueness, route budgets and reported reward."""
    task = np.asarray(task, dtype=np.float32)
    depot = np.asarray(depot, dtype=np.float32).reshape(2)
    prize = np.asarray(prize, dtype=np.float32).reshape(-1)
    max_length = float(np.asarray(max_length))
    if task.ndim != 2 or task.shape[1] != 2 or len(task) == 0:
        raise ValueError("task must have shape [N, 2], with N > 0")
    if prize.shape != (len(task),):
        raise ValueError("prize must match the task count")
    if max_length <= 0 or not np.isfinite(max_length):
        raise ValueError("max_length must be finite and positive")

    visited = []
    total_prize = 0.0
    total_length = 0.0
    for assignment, selected in zip(index_table, line):
        assignment = np.asarray(assignment, dtype=np.int64)
        selected = np.asarray(selected, dtype=np.int64)
        if len(selected) and ((selected < 0).any() or (selected >= len(assignment)).any()):
            raise ValueError("SA route contains an out-of-range local node")
        global_ids = assignment[selected] if len(selected) else np.empty(0, dtype=np.int64)
        if len(global_ids):
            if (global_ids < 0).any() or (global_ids >= len(task)).any():
                raise ValueError("SA route contains an out-of-range global node")
            visited.extend(global_ids.tolist())
            route = np.concatenate((depot[None, :], task[global_ids], depot[None, :]), axis=0)
            route_length = np.linalg.norm(route[1:] - route[:-1], axis=1).sum()
            if route_length > max_length + tolerance:
                raise ValueError(f"SA route exceeds max_length: {route_length} > {max_length}")
            total_length += float(route_length)
            total_prize += float(prize[global_ids].sum())
    if len(visited) != len(set(visited)):
        raise ValueError("SA allocation assigns a task more than once")
    if expected_prize is not None:
        expected_prize = float(expected_prize)
        if not np.isfinite(expected_prize) or not np.isclose(
                total_prize, expected_prize, rtol=1e-5, atol=tolerance):
            raise ValueError(
                f"SA reward mismatch: routes collect {total_prize}, reported {expected_prize}"
            )
    return total_prize, total_length


def sa_allocate(task, depot, prize, MAX_LENGTHS, uav_num):
    # start = time.time()
    task = np.asarray(task, dtype=np.float32)
    depot = np.asarray(depot, dtype=np.float32).reshape(2)
    prize = np.asarray(prize, dtype=np.float32).reshape(-1)
    max_length = float(np.asarray(MAX_LENGTHS))
    if uav_num <= 0 or task.ndim != 2 or task.shape[1] != 2:
        raise ValueError("uav_num must be positive and task must have shape [N, 2]")
    if task.shape[0] < uav_num:
        raise ValueError("task count must be at least the UAV count")
    if prize.shape != (task.shape[0],):
        raise ValueError("prize must match task count")
    if max_length <= 0 or not np.isfinite(max_length):
        raise ValueError("max_length must be finite and positive")
    task_num = task.shape[0]
    angle, index = angle_sort(task, depot)
    groups = np.array_split(index, uav_num)
    part_angle = np.array([np.median(angle[np.isin(index, group)]) for group in groups])

    # add depot at the end of task set
    task = np.concatenate((task, [depot]), axis=0)
    # add 0 at the end of prize set
    prize = np.append(prize, 0)

    old_index_table, ll = [], []
    for i, group in enumerate(groups):
        a = group
        ll.append(len(a))
        old_index_table.append(a)
    max_ll = max(ll)
    # the gap to the longest length
    d_max = max_ll - np.array(ll)
    new_index_table = old_index_table.copy()
    for i in range(uav_num):
        for j in range(d_max[i]):
            new_index_table[i] = np.append(new_index_table[i], task_num)

    dataset = []
    for i in range(uav_num):
        loc = task[new_index_table[i]]
        prz = prize[new_index_table[i]]
        dataset.append({
            'loc': torch.FloatTensor(loc).to(device),
            'prize': torch.FloatTensor(prz).to(device),
            'valid_nodes': torch.tensor(new_index_table[i] != task_num, dtype=torch.bool, device=device),
            'depot': torch.FloatTensor(depot).to(device),
            'max_length': torch.tensor(max_length, dtype=torch.float32).to(device)
        })
    fitness, pi = cal_fitness(dataset, uav_num)
    # print(fitness)

    fit = sum(fitness)
    cur_fit = fit
    # transform pi into node location sequence
    pi = list(pi)
    line = []
    for i in range(uav_num):
        line.append(pi[i][(pi[i] != 0) & (pi[i] <= ll[i])] - 1)

    # return fit, new_index_table, line

    # =================SA=====================
    T_max = 80  # max_temp
    L = 4  # Markov chain length
    T_min = 60  # low_temp
    alpha = 0.9  # decay rate
    T = T_max
    best_fit = fit
    best_index = old_index_table
    best_line = line
    # plot_picture(depot, task, best_index, best_line)

    while T > T_min:
        for i in range(L):
            # produce new solution
            # obtain the uncovered node
            # node_set, prz_set = [], []  # uncovered node set, uncovered profit set
            # uncovered node index set
            unable_table = []
            for j in range(uav_num):
                unable_index = np.delete(old_index_table[j], line[j], axis=0)
                # divide the uncovered nodes into two parts
                angle, angle_index = angle_sort(task[unable_index], depot)
                # print(np.int64(unable_index.shape[0] / 2))
                # part_angle = angle[angle_index[np.int64(unable_index.shape[0]/2)]]

                unable_table.append(unable_index[angle < part_angle[j]])
                unable_table.append(unable_index[~(angle < part_angle[j])])

            # reassign nodes
            # i is odd，assign uncovered nodes by clockwise
            # new_index_table, fit_table, new_line = [], [], []  # new assignment scheme
            if i % 2 == 0:
                new_index_table, ll = [], []
                for j in range(uav_num):
                    # cat the covered node set with the assigned uncovered nodes
                    # assign the uncovered node set of first UAV to the last UAV
                    a = np.concatenate((old_index_table[j][line[j]], np.concatenate(
                        (unable_table[2 * (j + 1) - 1],
                         unable_table[2 * (j + 1)] if j < uav_num - 1 else unable_table[0]), 0)), 0)
                    ll.append(len(a))
                    new_index_table.append(a)
                max_ll = max(ll)

                d_max = max_ll - np.array(ll)
                renew_index_table = new_index_table.copy()
                for j in range(uav_num):
                    for _ in range(d_max[j]):
                        renew_index_table[j] = np.append(renew_index_table[j], task_num)

                dataset = []
                for j in range(uav_num):
                    loc = task[renew_index_table[j]]
                    prz = prize[renew_index_table[j]]
                    dataset.append({
                        'loc': torch.FloatTensor(loc).to(device),
                        'prize': torch.FloatTensor(prz).to(device),
                        'valid_nodes': torch.tensor(renew_index_table[j] != task_num, dtype=torch.bool, device=device),
                        'depot': torch.FloatTensor(depot).to(device),
                        'max_length': torch.tensor(max_length, dtype=torch.float32).to(device)
                    })
                fit, pi = cal_fitness(dataset, uav_num)
                new_fit = sum(fit)

                pi = list(pi)
                new_line = []
                for j in range(uav_num):
                    new_line.append(pi[j][(pi[j] != 0) & (pi[j] <= ll[j])] - 1)

            else:
                new_index_table, ll = [], []
                for j in range(uav_num):
                    a = np.concatenate((old_index_table[j][line[j]], np.concatenate(
                        (unable_table[2 * j - 1] if j > 0 else unable_table[-1], unable_table[2 * j]), 0)), 0)
                    ll.append(len(a))
                    new_index_table.append(a)

                max_ll = max(ll)

                d_max = max_ll - np.array(ll)
                renew_index_table = new_index_table.copy()
                for j in range(uav_num):
                    for _ in range(d_max[j]):
                        renew_index_table[j] = np.append(renew_index_table[j], task_num)

                dataset = []
                for j in range(uav_num):
                    loc = task[renew_index_table[j]]
                    prz = prize[renew_index_table[j]]
                    # loc, dep, length = change_node(loc, dep, length)
                    dataset.append({
                        'loc': torch.FloatTensor(loc).to(device),
                        'prize': torch.FloatTensor(prz).to(device),
                        'valid_nodes': torch.tensor(renew_index_table[j] != task_num, dtype=torch.bool, device=device),
                        'depot': torch.FloatTensor(depot).to(device),
                        'max_length': torch.tensor(max_length, dtype=torch.float32).to(device)
                    })
                fit, pi = cal_fitness(dataset, uav_num)
                new_fit = sum(fit)

                pi = list(pi)
                new_line = []
                for j in range(uav_num):
                    new_line.append(pi[j][(pi[j] != 0) & (pi[j] <= ll[j])] - 1)
                # print(new_line)
            # new_fit = sum(fit_table)
            # print(best_fit)
            # print(old_index_table)
            if new_fit > cur_fit:
                cur_fit = new_fit
                old_index_table = new_index_table
                line = new_line
                if new_fit > best_fit:
                    best_fit = new_fit
                    best_index = new_index_table
                    best_line = new_line
            else:
                if np.random.random() < math.exp((new_fit - cur_fit) / T):
                    cur_fit = new_fit
                    old_index_table = new_index_table
                    line = new_line
        T *= alpha
        # duration = time.time() - start
    validate_allocation(
        task[:task_num], depot, prize[:task_num], max_length, best_index, best_line,
        expected_prize=best_fit,
    )
    return best_fit, best_index, best_line


def plot_fig7(results, output_path):
    """Save SA-NNO-DRL routes in the layout used by paper Fig. 7(a)."""
    if not results:
        raise ValueError("At least one result is required for plotting")

    route_colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]
    ncols = len(results)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(3.8 * ncols, 3.7),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    axes = axes[0]

    for ax, result in zip(axes, results):
        task = np.asarray(result["task"], dtype=np.float32)
        depot = np.asarray(result["depot"], dtype=np.float32)
        index = result["index"]
        line = result["line"]
        best_fit = float(result["fitness"])

        # Keep all target points blue, as in Fig. 7. Route colors identify UAVs.
        ax.scatter(
            task[:, 0],
            task[:, 1],
            s=8,
            c="#1479c9",
            marker="o",
            linewidths=0,
            zorder=2,
        )

        for uav_id, local_route in enumerate(line):
            local_route = np.asarray(local_route, dtype=np.int64)
            assignment = np.asarray(index[uav_id], dtype=np.int64)
            if local_route.size == 0:
                continue
            route_nodes = task[assignment[local_route]]
            route = np.concatenate((depot[None, :], route_nodes, depot[None, :]), axis=0)
            color = route_colors[uav_id % len(route_colors)]

            xs = route[:, 0]
            ys = route[:, 1]
            ax.plot(xs, ys, color=color, linewidth=1.0, alpha=0.9, zorder=3)
            ax.quiver(
                xs[:-1],
                ys[:-1],
                np.diff(xs),
                np.diff(ys),
                angles="xy",
                scale_units="xy",
                scale=1,
                color=color,
                width=0.0035,
                headwidth=4.5,
                headlength=5.5,
                headaxislength=4.5,
                minlength=0,
                zorder=4,
            )

        ax.scatter(
            depot[0],
            depot[1],
            marker="s",
            c="black",
            s=32,
            zorder=5,
        )
        ax.set_title(f"Total profits:{best_fit:.1f}", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.tick_params(labelsize=8)

    fig.text(0.5, 0.01, "(a)", ha="center", va="bottom", fontsize=10)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0), pad=1.0)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run SA multi-UAV inference for OP')
    parser.add_argument('--model', required=True, help='checkpoint file or directory')
    parser.add_argument('--dataset', required=True, help='OP .pkl dataset')
    parser.add_argument('--num-samples', type=int, default=30)
    parser.add_argument('--uav-num', type=int, default=5)
    parser.add_argument('--device', default=str(device))
    parser.add_argument(
        '--output-dir', default='results/sa',
        help='Root for unique run directories containing CSV, routes and summary',
    )
    parser.add_argument(
        '--weight-source', choices=('self-trained', 'external', 'unspecified'),
        default='unspecified', help='Declared provenance of the checkpoint',
    )
    parser.add_argument(
        '--plot-dir',
        default=None,
        help='Save a Fig. 7-style multi-panel route figure for the evaluated samples',
    )
    parser.add_argument(
        '--plot-name',
        default='fig7_sa',
        help='Output stem for the Fig. 7-style figure (default: fig7_sa)',
    )
    parser.add_argument(
        '--plot-format',
        choices=('png', 'pdf', 'eps'),
        default='png',
        help='Format for the Fig. 7-style figure',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Seed NumPy SA moves and PyTorch augmentation (default: no fixed seed)',
    )
    opts = parser.parse_args()
    if opts.num_samples <= 0 or opts.uav_num <= 0:
        parser.error('--num-samples and --uav-num must be positive')
    device = resolve_device(opts.device)
    if opts.seed is not None:
        np.random.seed(opts.seed)
        torch.manual_seed(opts.seed)
    model_path = resolve_model_path(opts.model)
    model, _ = load_model(str(model_path))
    model.to(device)
    model.eval()
    model.set_decode_type('greedy')

    print(opts.dataset)
    print('UAV num:', opts.uav_num)
    data = make_dataset(opts.dataset, opts.num_samples)
    if not data:
        parser.error('The dataset contains no instances')
    if any(len(batch['loc']) < opts.uav_num for batch in data):
        parser.error('Each instance must have at least --uav-num targets')
    results = InferenceResults(vars(opts), model_path, device)
    print("results:", results.directory)
    print(f"samples: {len(data)} (requested: {opts.num_samples})")
    plot_results = []
    try:
        for index, batch in enumerate(data):
            task = batch['loc']
            depot = batch['depot']
            prize = batch['prize']
            MAX_LENGTHS = batch['max_length']
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            best_fit, best_index, best_line = sa_allocate(
                task, depot, prize, MAX_LENGTHS, opts.uav_num,
            )
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            duration = time.perf_counter() - start
            results.add(instance_result(
                index, batch, best_fit, duration, best_index, best_line,
            ))
            if opts.plot_dir is not None:
                plot_results.append({
                    "task": np.asarray(task),
                    "depot": np.asarray(depot),
                    "index": best_index,
                    "line": best_line,
                    "fitness": best_fit,
                })
            print(best_fit)
    except BaseException as error:
        results.save("interrupted" if isinstance(error, KeyboardInterrupt) else "failed")
        raise
    results.save("completed")
    print("fitness:", np.mean([r["reward"] for r in results.records]))
    print(np.mean([r["duration_seconds"] for r in results.records]))
    if opts.plot_dir is not None:
        os.makedirs(opts.plot_dir, exist_ok=True)
        plot_path = os.path.join(
            opts.plot_dir,
            f"{opts.plot_name}.{opts.plot_format}",
        )
        plot_fig7(plot_results, plot_path)
        print("figure:", plot_path)



