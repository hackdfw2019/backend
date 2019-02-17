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
        parser_factory = Producer(websocket, path, self.model, self.num_candidates)
        producer_task = None
        async for message in websocket:
            decoded = json.loads(message)

            if decoded["char"] is None:
                producer_task = asyncio.create_task(parser_factory.produce(self.num_lines * 2))
                self.model.train(None, None)
            elif decoded["request"] < self.producer_threshold and producer_task.done():
                producer_task = asyncio.create_task(parser_factory.produce(self.num_lines))

            if decoded["char"]:
                if not timestamp:
                    new_timestamp = decoded["time"]
                    self.model.train(decoded["char"], None)
                else:
                    new_timestamp = decoded["time"]
                    output, target, loss = self.model.train(decoded["char"], (new_timestamp - timestamp) / 1000.0)
                    print("char: {}\t\toutput: {:5.3f}\t\ttarget: {:5.3f}\t\tMSE loss: {:5.3f}".format(decoded["char"], output, target, loss))
                timestamp = new_timestamp

class Producer:
    def __init__(self, websocket, path, model, num_candidates):
        self.websocket = websocket
        self.path = path
        self.model = model
        self.num_candidates = num_candidates

        self.factory = Factory().get_candidates(count=num_candidates)

    async def produce(self, num_lines):
        print("Producing {} new lines based on {} candidates...".format(num_lines, num_lines * self.num_candidates))
        lines = []
        while True:
            for strings in self.factory:
                times = list(map(self.model.eval, strings))
                best = strings[times.index(max(times))]
                lines.append(best)
                if len(lines) == num_lines:
                    break
            if len(lines) == num_lines:
                break
            self.factory = Factory().get_candidates(count=self.num_candidates)
        await self.websocket.send("\n".join(lines))
        await asyncio.sleep(1)

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
    print("started server")
    asyncio.get_event_loop().run_forever()

if __name__ == "__main__":
    main()
