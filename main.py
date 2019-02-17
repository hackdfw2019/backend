import sys
import select
import tty
import termios
import time
import queue

from neural_network import CharRNNTrainer
from parser.annotation_to_madlib import annotations_to_madlibs

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
            return False


def main():
    char_queue = queue.Queue()
    trainer = CharRNNTrainer(char_queue,
                             charset=tuple(range(128)),
                             hidden_size=128,
                             learning_rate=0.05,
                             truncate_length=30,
                             time_logistic_growth_constant=2.7,
                             time_logistic_inflection_constant=0.33)
    trainer.start()

    with NonBlockingConsole() as nbc:
        while True:
            c = nbc.get_data()
            if c:
                char_queue.put((c, time.process_time()))
                sys.stdout.write(c)
                sys.stdout.flush()
            else:
                break

if __name__ == "__main__":
    main()
