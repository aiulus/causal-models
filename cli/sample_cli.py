import argparse
import os
from os.path import basename
import json
from scm.base import SCM
from scm import sampler, counterfactuals
from utils import io, plot


def main():
    config = io.config
    parser = argparse.ArgumentParser(description="Sample data from SCMs (L1, L2, L3).")

    parser.add_argument('--file_name', required=True, help="SCM filename (in PATH_SCMs)")
    parser.add_argument('--mode', choices=['l1', 'l2', 'l3'], required=True)
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--do', nargs='+', help="Interventions in format (X, value)")
    parser.add_argument('--observations_path', help="Required for counterfactuals (L3)")
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--save', action='store_true')

    args = parser.parse_args()

    #file_path = os.path.join(config['PATH_SCMs'], args.file_name)
    if os.path.isabs(args.file_name) or os.path.dirname(args.file_name):
        file_path = args.file_name
    else:
        file_path = os.path.join(config['PATH_SCMs'], args.file_name)

    #data_path = os.path.join(config['PATH_DATA'], args.file_name.replace(".json", ".csv"))

    scm_filename = basename(args.file_name)
    data_path = os.path.join(config['PATH_DATA'], scm_filename.replace(".json", ".csv"))

    scm = SCM(file_path)

    if args.mode == 'l1':
        data = sampler.sample_L1(scm, args.n_samples)

    elif args.mode == 'l2':
        if not args.do:
            raise ValueError("Please specify interventions with --do for L2 sampling.")
        data = sampler.sample_L2(scm, args.n_samples, args.do)

    elif args.mode == 'l3':
        if not args.do or not args.observations_path:
            raise ValueError("Both --do and --observations_path are required for L3 sampling.")
        with open(args.observations_path, 'r') as f:
            obs = json.load(f)
        data = counterfactuals.sample_L3(scm, L1_obs=obs, interventions=args.do, n_samples=args.n_samples)

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    if args.save:
        io.save_to_csv(data, data_path)
        print(f"Data saved to: {data_path}")

    if args.plot:
        fig = plot.plot_distributions_from_dict(data)
        if args.save:
            plot_path = os.path.join(config['PATH_PLOTS'], args.file_name.replace('.json', '.png'))
            fig.savefig(plot_path)
            print(f"Plot saved to: {plot_path}")
