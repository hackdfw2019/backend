from threading import Thread

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class RectifiedLogisticCurve(nn.Module):
    """
    Custom logistic curve that intersects the origin and asymptotically approaches 1.
    Used to transform time values in the range (0,inf) to the range (0,1).
    Also used on the output layer of the RNN, in which case the value is in the range
    (-inf, inf), and is thresholded (rectified) at 0.

    See https://www.desmos.com/calculator/grilveheq
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
    """
    Asynchronously trains a CharRNN, taking characters and timestamps from a Queue

    Args:
        char_queue (Queue): queue to poll from, containing tuples of the form (char: string, timestamp: float)
        charset (iterable): charset to use, also determines size of input vectors to RNN
        hidden_size (int): number of dimensions in hidden layer
        learning_rate (float): learning rate per time step
        truncate_length (int): number of time steps to backpropagate gradients backwards through time
        time_logistic_growth_constant, time_logistic_inflection_constant (float): used to normalize time
        values into the range (0, 1), see https://www.desmos.com/calculator/grilveheq
    """

    def __init__(self, char_queue, charset, hidden_size, learning_rate, truncate_length,
                 time_logistic_growth_constant, time_logistic_inflection_constant):
        super().__init__()
        self.char_queue = char_queue
        self.charset = charset
        self.charmap = {x: i for i, x in enumerate(charset)}  # maps char values to indices
        self.learning_rate = learning_rate

        self.time_curve = RectifiedLogisticCurve(time_logistic_growth_constant, time_logistic_inflection_constant)
        self.rnn = CharRNN(len(self.charset), hidden_size, self.time_curve)
        self.criterion = nn.MSELoss()

        self.states = [torch.zeros(hidden_size)]  # use zero-initialization for hidden layer

    def char_to_one_hot(self, char):
        one_hot = np.zeros(len(self.charset), dtype=np.float32)
        one_hot[self.charmap[char]] = 1
        return torch.from_numpy(one_hot)

    def run(self):
        # don't train on first character, just feed through network to get initial hidden state
        char, timestamp = self.char_queue.get()
        if (char not in self.charset):
            raise ValueError("Illegal char '{}' ({}) not in charset".format(char, ord(char)))
        _, hidden = self.rnn(self.char_to_one_hot(char), self.states[-1])
        self.states.append(hidden)

        while True:
            char, new_timestamp = self.char_queue.get()
            if (char not in self.charset):
                raise ValueError("Illegal char '{}' ({}) not in charset".format(char, ord(char)))


            if len(self.states) < self.truncate_length:
                output, hidden = self.rnn(self.char_to_one_hot(char), self.states[-1])
                self.states.append(hidden)
            else:
                # do some sketch shit
                for curr, next_ in zip(self.states, self.states[1:]):
                    curr.data = next_.data
                output, self.states[-1] = self.rnn(self.char_to_one_hot(char), self.states[-2])

            target = self.time_curve(torch.tensor(new_timestamp - timestamp))
            loss = self.criterion(output, target)

            self.rnn.zero_grad()
            loss.backward(retain_graph=True)
            for p in self.rnn.parameters():
                p.data.add_(-self.learning_rate, p.grad.data)

            timestamp = new_timestamp
