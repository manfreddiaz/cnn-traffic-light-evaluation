import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

figure, axis = plt.subplots()
line, = axis.plot([], [], lw=2)
axis.grid()
samples, accuracies = [], []


def init():
    axis.set_ylim(0, 1)
    axis.set_yticks(np.arange(0, 1, 0.05))
    axis.set_ylabel('Accuracy')
    axis.set_xlim(0, 100)
    axis.set_xlabel('Iteration')
    del samples[:]
    del accuracies[:]
    line.set_data(samples, accuracies)
    return line,


def update_batch_graph(batch_data):
    # update the data
    sample, loss = batch_data
    print (loss)
    samples.append(sample)
    accuracies.append(loss)
    xmin, xmax = axis.get_xlim()

    if sample >= xmax:
        axis.set_xlim(xmin, 2 * xmax)
        axis.figure.canvas.draw()
    line.set_data(samples, accuracies)
    return line,


def plot(learning_pipeline, export):
    plt.ion()

    animated_plot = animation.FuncAnimation(figure, update_batch_graph, learning_pipeline, blit=False, interval=10,
                                            repeat=False, init_func=init)

    return animated_plot

