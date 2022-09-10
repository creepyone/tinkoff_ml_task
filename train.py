import argparse
import pickle
import random
import string

import torch
from sys import stdin

from torch import nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, carry):
        out = self.embedding(x)
        out, (hidden, carry) = self.lstm(out.unsqueeze(1), (hidden, carry))
        out = self.decoder(out.reshape(out.shape[0], -1))
        return out, (hidden, carry)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        carry = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden, carry


class Model:
    def __init__(self, chunk_len=200, n_epochs=50, batch_size=1, hidden_size=256, num_layers=1, lr=0.005):
        self.chunk_len = chunk_len
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.num_layers = num_layers
        self.rnn = None
        self.text = None
        self.w2i = dict()
        self.i2w = dict()
        self.chars = set()

    def _get_tensor(self, string_sequence: str) -> torch.Tensor:
        tensor = torch.zeros(len(string_sequence)).long()
        for i in range(len(string_sequence)):
            tensor[i] = self.w2i[string_sequence[i]]
        return tensor

    def _get_batch(self):
        x = torch.zeros(self.batch_size, self.chunk_len)
        y = torch.zeros(self.batch_size, self.chunk_len)
        try:
            start_idx = random.randint(0, len(self.text) - self.chunk_len)
            end_idx = start_idx + self.chunk_len + 1
            text = self.text[start_idx:end_idx]

            for i in range(self.batch_size):
                x[i, :] = self._get_tensor(text[:-1])
                y[i, :] = self._get_tensor(text[1:])
            return x.long(), y.long()
        except ValueError:
            print("Failure: chunk length is greater than text length")
            exit()
    def train(self, corpus):
        self.text = corpus
        self.chars = set(corpus)
        n_chars = len(self.chars)

        for i, key in enumerate(self.chars):
            self.w2i[key] = i
            self.i2w[i] = key

        self.rnn = RNN(input_size=n_chars, hidden_size=self.hidden_size,
                       output_size=n_chars, num_layers=self.num_layers)

        optimizer = torch.optim.SGD(self.rnn.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        print("Training: ")

        for i in range(1, self.n_epochs + 1):

            inp, target = self._get_batch()
            hidden, carry = self.rnn.init_hidden(batch_size=self.batch_size)

            self.rnn.zero_grad()
            loss = 0

            try:
                for c in range(self.chunk_len):
                    output, (hidden, carry) = self.rnn(inp[:, c], hidden, carry)
                    loss += criterion(output, target[:, c])
            except:
                continue

            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(i, loss.item() / self.chunk_len, self.generate(predict_len=100, initial_str="А"))

    def generate(self, initial_str, predict_len=250, temperature=0.5):
        if initial_str is None:
            initial_str = chr(random.randint(1072, 1103))

        hidden, cell = self.rnn.init_hidden(batch_size=self.batch_size)
        initial_input = self._get_tensor(initial_str)
        predicted = initial_str

        for p in range(len(initial_str) - 1):
            _, (hidden, cell) = self.rnn(initial_input[p].view(1), hidden, cell)

        last_char = initial_input[-1]

        for p in range(predict_len):
            output, (hidden, cell) = self.rnn(last_char.view(1), hidden, cell)
            output_dist = output.data.view(-1).div(temperature).exp()
            top_char = torch.multinomial(output_dist, 1)[0]
            predicted_char = self.i2w[top_char.item()]
            predicted += predicted_char
            last_char = self._get_tensor(predicted_char)

        return predicted


class TextReader:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.text = str()

    def read_data(self):
        if not self.data_path:
            print("Enter text (press ctrl+d to confirm): ")
            return stdin.read()
        else:
            try:
                with open(self.data_path, 'r', encoding='utf-8') as file:
                    self.text = file.read()
                    return self.text
            except OSError:
                print("Error: File not found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument('--input-dir',
                        type=str,
                        help='Path to file with text docs')

    parser.add_argument('--model',
                        type=str,
                        help="Path to model file",
                        required=True)

    args = parser.parse_args()

    input_dir = args.input_dir
    model_dir = args.model

    text_reader = TextReader(input_dir)
    text = text_reader.read_data()
    model = Model(batch_size=1, n_epochs=5000, num_layers=2, chunk_len=500, lr=0.005)
    model.train(text)

    with open(model_dir, 'wb') as file:
        pickle.dump(model, file)
