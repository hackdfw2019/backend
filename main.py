import sys
import select
import tty
import termios
import time
import queue
from threading import Thread

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NonBlockingConsole:
    def __enter__(self):
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, type, value, traceback):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def get_data(self):
        try:
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                return sys.stdin.read(1)
        except:
            return '[CTRL-C]'
        return False

class RectifiedLogisticCurve(nn.Module):
    """
    Custom logistic curve that intersects the origin and asymptotically approaches 1.
    Used to transform time values in the range (0,inf) to the range (0,1).
    Also used on the output layer of the RNN, in which case the value is in the range
    (-inf, inf), and is thresholded (rectified) at 0.

    See https://www.desmos.com/calculator/p4tcoqfwn1
    """
    def __init__(self, growth_value, inflection_point):
        super().__init__()

        self.growth_value = growth_value
        self.inflection_point = inflection_point

    def forward(self, x):
        v_inv = 1 / self.inflection_point
        v_inv_2 = 2**v_inv

        result = (v_inv_2 * torch.pow(torch.exp(x * -self.growth_value) + 1, -v_inv) - 1) / (v_inv_2 - 1)
        return F.threshold(result, 0, 0)

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_activation):
        super().__init__()

        self.hidden_layer = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh()
        )

        self.output_layer = nn.Sequential(
            nn.Linear(input_size + hidden_size, 1),
            output_activation
        )

    def forward(self, input_, hidden):
        input_combined = torch.cat((hidden, input_))
        return self.output_layer(input_combined), self.hidden_layer(input_combined)

class CharRNNTrainer(Thread):
    CHARSET = tuple(map(chr, range(128)))  # ASCII values 0-127
    CHARMAP = {x: i for i, x in enumerate(CHARSET)}  # maps char values to indices

    HIDDEN_SIZE = 128
    LEARNING_RATE = 0.05
    TRUNCATE_LENGTH = 10

    # see https://www.desmos.com/calculator/p4tcoqfwn1
    TIME_GROWTH_VALUE = 2.7
    TIME_INFLECTION_VALUE = 0.33

    def __init__(self, char_queue):
        super().__init__()
        self.char_queue = char_queue

        self.time_curve = RectifiedLogisticCurve(self.TIME_GROWTH_VALUE, self.TIME_INFLECTION_VALUE)
        self.rnn = CharRNN(len(self.CHARSET), self.HIDDEN_SIZE, self.time_curve)
        self.criterion = nn.MSELoss()

        self.states = [torch.zeros(self.HIDDEN_SIZE)]  # use zero-initialization for hidden layer

    def char_to_one_hot(self, char):
        one_hot = np.zeros(len(self.CHARSET), dtype=np.float32)
        one_hot[self.CHARMAP[char]] = 1
        return torch.from_numpy(one_hot)

    def run(self):
        # don't train on first character, just feed through network to get initial hidden state
        char, timestamp = self.char_queue.get()
        if (char not in self.CHARSET):
            raise ValueError("Illegal char '{}' ({}) not in charset".format(char, ord(char)))
        _, hidden = self.rnn(self.char_to_one_hot(char), self.states[-1])
        self.states.append(hidden)

        while True:
            char, new_timestamp = self.char_queue.get()
            if (char not in self.CHARSET):
                raise ValueError("Illegal char '{}' ({}) not in charset".format(char, ord(char)))


            if len(self.states) < self.TRUNCATE_LENGTH:
                output, hidden = self.rnn(self.char_to_one_hot(char), self.states[-1])
                self.states.append(hidden)
            else:
                # do some sketch shit
                for curr, next_ in zip(self.states, self.states[1:]):
                    curr.data = next_.data
                output, self.states[-1] = self.rnn(self.char_to_one_hot(char), self.states[-2])

            target = self.time_curve(torch.tensor(new_timestamp - timestamp))
            loss = self.criterion(output, target)
            print(loss)

            self.rnn.zero_grad()
            loss.backward(retain_graph=True)
            for p in self.rnn.parameters():
                p.data.add_(-self.LEARNING_RATE, p.grad.data)

            timestamp = new_timestamp

def main():
    char_queue = queue.Queue()
    trainer = CharRNNTrainer(char_queue)
    trainer.start()

    with NonBlockingConsole() as nbc:
        while True:
            c = nbc.get_data()
            if c:
                if c == '\x1b':
                    break
                char_queue.put((c, time.process_time()))
                sys.stdout.write(c)
                sys.stdout.flush()

if __name__ == "__main__":
    main()
