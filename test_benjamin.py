import _thread
import time
from matplotlib import pyplot as plt
import numpy as np

class GraphCallback:
    def __init__(self):
        self.n_values = 100
        self.data = None
        self.has_changed = True
        self.figure_handler = None
        self.thread_is_closed = True
    def eval(self, new_data):
        self.data = new_data
        self.has_changed = True

my_callback = GraphCallback()

def print_data(callback_data):
    callback_data.thread_is_closed = False
    plt.figure()
    plot_handler = plt.plot(np.linspace(0, 10, callback_data.n_values), np.zeros((callback_data.n_values,)))
    axis_handler = plt.gca()
    axis_handler.set_ylim(-1, 1)
    plt.show(block=False)
    while plt.get_fignums():
        if callback_data.has_changed:
            plot_handler[0].set_ydata(callback_data.data)
            callback_data.has_changed = False
        plt.draw()
        plt.pause(.001)
    callback_data.thread_is_closed = True
    print("bye")

_thread.start_new_thread(print_data, (my_callback,))
cmp = 0
start = time.time()

while True:
    time.sleep(2)
    print(f"New data! at iteration {cmp}")
    print(f"Time : {time.time() - start}")
    my_callback.eval(np.random.rand(my_callback.n_values,))
    if cmp > 3:
        break
    else:
        cmp += 1
        print("Wait so you can have a look")
while not my_callback.thread_is_closed:
    pass
time.sleep(0.005)
print("Programs end")