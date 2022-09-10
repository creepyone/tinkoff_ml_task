import pickle

import torch
import os
import argparse
from train import Model, RNN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and test the model")
    parser.add_argument('--model',
                        type=str,
                        help='Path to model',
                        required=True)

    parser.add_argument('--prefix',
                        type=str,
                        help="Start of the string",
                        required=False)

    parser.add_argument('--length',
                        type=int,
                        help="Length of the string to generate",
                        required=True)

    args = parser.parse_args()
    model_path = args.model
    prefix = args.prefix
    predict_len = args.length

    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
            print(model.generate(initial_str=prefix, predict_len=predict_len))
    else:
        print("No model!")
