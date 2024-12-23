import json
import argparse
import os

import matplotlib.pyplot as plt

def plot_loss(args):
    iterations = []
    loss = []
    
    with open(args.log_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            iterations.append(entry['iter'])
            loss.append(entry['loss'])
    
    plt.figure()
    plt.plot(iterations, loss, label='Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    if args.log_scale: plt.yscale('log')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(args.log_file), 'plot.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot loss from log file.')
    parser.add_argument('log_file', type=str, help='Path to the log JSON file.')
    parser.add_argument('--log_scale', type=bool, default=True, help='Use log scale for the loss axis.')
    args = parser.parse_args()
    plot_loss(args)

if __name__ == '__main__':
    main()