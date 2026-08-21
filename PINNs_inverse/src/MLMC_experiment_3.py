import os
import math
import torch
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as color
from src.MLMC_base import make_config, levels_to_num_evals, compute_mean_and_var, store_vals, expectation_of_estimator, variance_of_estimator
from src.trainmodel import set_seed, CONFIG_to_folder_path, sunflower_disk_points
from src.model_defn import load_model

def eval_models(total_cost = 2048, model_seed = 0, num_seeds = 100, x_data_points = 4096):
    with torch.no_grad():
        CONFIG = make_config(seed = model_seed)
        model  = load_model(CONFIG, device = 'cpu')
        model_path = os.path.join(CONFIG_to_folder_path(CONFIG), 'model.pt')
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only = True))
        model.train()
        values = sunflower_disk_points(x_data_points)
        for i in range(num_seeds):
            set_seed(seed = i)
            output = torch.stack([model(values).squeeze() for _ in range(total_cost)])
            torch.save(output, f'plots/model_evals_{i}.pt')
            print(f'Completed seed {i+1}/{num_seeds}')

def single_level_multi_fidelity_from_evals(evals, fidelity_ladder, factor=math.pi):
    running_mean = []
    running_var = []
    for fid in fidelity_ladder:
        mean_est, var_est = compute_mean_and_var( evals[:fid], factor=factor)
        running_mean.append(mean_est)
        running_var.append(var_est)
    running_mean = torch.tensor(running_mean)
    running_var = torch.tensor(running_var)
    return running_mean[0], torch.diff(running_mean), running_var[0], torch.diff(running_var)

def estimator_from_evals(evals = None, model_seed = None, fidelity_ladder = [16, 32, 64], M = [43, 35, 25], factor = math.pi):
    if evals is None:
        if model_seed is None:
            evals =  torch.load(f'plots/model_evals_0.pt', weights_only = True)
        else:
            evals =  torch.load(f'plots/model_evals_{model_seed}.pt', weights_only = True)
    evals = evals.squeeze()
    num_levels = levels_to_num_evals(M)
    cost = sum([fidelity_ladder[i]*num_levels[i] for i in range(len(fidelity_ladder))]).item()
    if cost > evals.shape[0]:
        raise ValueError("Insufficient evaluations for this fidelity ladder")
    Y_store = [[] for _ in num_levels]
    V_store = [[] for _ in num_levels]
    offset = 0
    for i, nrep in enumerate(num_levels):
        max_fid = fidelity_ladder[i]
        for _ in range(int(nrep)):
            block = evals[offset:(offset + max_fid)]
            Y0, dY, V0, dV = single_level_multi_fidelity_from_evals(block, fidelity_ladder[:i+1], factor)
            store_vals(Y_store, V_store, Y0, dY, V0, dV)
            offset += max_fid
    mean_estimate = expectation_of_estimator(Y_store)
    mean_var = variance_of_estimator(Y_store, M)
    var_estimate = expectation_of_estimator(V_store)
    var_var = variance_of_estimator(V_store, M)
    return mean_estimate, mean_var, var_estimate, var_var, Y_store, V_store

def all_M(cost = 2048, fidelity_ladder = [16, 32, 64]):
    T0, T1, T2 = fidelity_ladder
    out = []
    for n0 in range(cost // T0 + 1):
        for n1 in range((cost - T0 * n0) // T1 + 1):
            rem = cost - T0 * n0 - T1 * n1
            if rem % T2 == 0:
                n2 = rem // T2
                M2 = n2
                M1 = n1 + M2
                M0 = n0 + M1
                if M0 > 1 and M1 > 1 and M2 > 1:
                    out.append([M0, M1, M2])
    return out

def main_experiment(cost = 2048, fidelity_ladder = [16, 32, 64], model_seed = 0, num_seeds = 100,  factor = math.pi):
    all_possible_Ms = all_M(cost, fidelity_ladder)
    output = []
    total_jobs = len(all_possible_Ms) * num_seeds
    completed = 0
    start_time = time.time()
    for count, M in enumerate(all_possible_Ms):
        temp_output = M.copy()
        for i in range(num_seeds):
            evals = torch.load(f'plots/model_evals_{i}.pt', weights_only = True)
            _, mean_var, _, var_var, _, _ = estimator_from_evals(evals = evals, fidelity_ladder = fidelity_ladder, M = M, factor = factor)
            temp_output.append(mean_var.item())
            temp_output.append(var_var.item() )

            completed += 1
            elapsed = time.time() - start_time
            rate = elapsed / completed
            eta = rate * (total_jobs - completed)
            print(
                f"\rProgress: {completed:5d}/{total_jobs} "
                f"({100*completed/total_jobs:5.1f}%) | "
                f"Elapsed: {elapsed/60:6.1f} min | "
                f"ETA: {eta/60:6.1f} min",
                end="",
                flush=True,
            )

        output.append(temp_output)
    torch.save(output, f'plots/triangle_cost_{cost}_fidelity_ladder_{fidelity_ladder}_num_seeds_{num_seeds}.pt')

def triangle_plot(M0, M1, data, file_name):
    plt.figure(figsize=(6,5))
    norm = color.TwoSlopeNorm(vmin=np.min(data), vcenter=np.median(data), vmax=np.max(data))
    plt.tripcolor(M0, M1, data, shading="gouraud", cmap="viridis", norm=norm)
    cbar = plt.colorbar()
    qticks = np.quantile(data, [0, 0.5, 1.0])
    cbar.set_ticks(qticks)
    plt.scatter(M0, M1, c="k", s=1)
    imin = np.argmin(data)
    plt.scatter(M0[imin], M1[imin], marker="x", s=5, c="red")
    m0_theo = 2048/(16*(1+np.sqrt(2))),
    m1_theo = 2048/(16*np.sqrt(2)*(1+np.sqrt(2)))
    plt.scatter(m0_theo, m1_theo, marker="x", s=5, c="orange")
    plt.xlabel(r"$M_0$")
    plt.ylabel(r"$M_1$")
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

def plot_results(file_name = None):
    if file_name is None:
        cost = 2048
        fidelity_ladder = [16, 32, 64]
        num_seeds = 100
        file_name = f'plots/triangle_cost_{cost}_fidelity_ladder_{fidelity_ladder}_num_seeds_{num_seeds}.pt'
    outputs = torch.load(file_name, weights_only = True)
    plotting_outputs = []
    for line in outputs:
        if not any(math.isinf(x) for x in line):
            plotting_outputs.append(line)
    plotting_outputs = torch.tensor(plotting_outputs)
    M0 = plotting_outputs[:,0].numpy()
    M1 = plotting_outputs[:,1].numpy()
    mean_var = plotting_outputs[:, 3::2].mean(dim=1).numpy()
    var_var  = plotting_outputs[:, 4::2].mean(dim=1).numpy()
    triangle_plot(M0, M1, mean_var, f'plots/sample_mean_estimator_variance.pdf')
    triangle_plot(M0, M1, var_var,  f'plots/sample_var_estimator_variance.pdf' )
    imin = np.argmin(mean_var)
    print('Minima:', M0[imin], M1[imin], plotting_outputs[imin,2].numpy())
    imin = np.argmin(var_var)
    print('Minima:', M0[imin], M1[imin], plotting_outputs[imin,2].numpy())

if __name__ == "__main__":
    eval_models()
    main_experiment()
    plot_results()
