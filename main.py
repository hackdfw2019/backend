import sys
import select
import tty
import termios
import websockets
import json
import asyncio
from threading import Thread

from neural_network import CharRNN
from parser2.factory import Factory

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

class WebsocketHandler:
    def __init__(self, model, producer_threshold, num_lines, num_candidates):
        self.model = model
        self.producer_threshold = producer_threshold
        self.num_lines = num_lines
        self.num_candidates = num_candidates

    async def handle(self, websocket, path):
        timestamp = None
        parser_factory = Producer(websocket, path, self.model, self.num_lines, self.num_candidates)
        async for message in websocket:
            decoded = json.loads(message)

            if decoded["request"] < self.producer_threshold:
                asyncio.ensure_future(parser_factory.produce())

            if not timestamp:
                new_timestamp = decoded["time"]
                self.model.train(decoded["char"], None)
            else:
                new_timestamp = decoded["time"]
                print(self.model.train(decoded["char"], (new_timestamp - timestamp) / 1000.0))
            timestamp = new_timestamp

class Producer:
    def __init__(self, websocket, path, model, num_lines, num_candidates):
        self.websocket = websocket
        self.path = path
        self.model = model
        self.num_lines = num_lines

        self.factory = Factory().get_candidates(count=num_candidates)

    async def produce(self):
        lines = []
        while True:
            for strings in self.factory:
                times = list(map(self.model.eval, strings))
                best = strings[times.index(max(times))]
                lines.append(best)
                if len(lines) == self.num_lines:
                    break
            if len(lines) == self.num_lines:
                break
            self.factory = Factory().get_candidates(count=self.num_candidates)
        await self.websocket.send("\n".join(lines))

def main():
    model = CharRNN(charset=tuple(map(chr, range(128))),
                    hidden_size=128,
                    learning_rate=0.05,
                    truncate_length=30,
                    time_logistic_growth_constant=2.7,
                    time_logistic_inflection_constant=0.33)

    handler = WebsocketHandler(model, producer_threshold=10, num_lines=10, num_candidates=50)
    start_server = websockets.serve(handler.handle, '127.0.0.1', 5678)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()

if __name__ == "__main__":
    main()
