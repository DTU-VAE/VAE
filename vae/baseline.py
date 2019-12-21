import numpy as np
import matplotlib.pylab as plt

def calculate_priors(dataset):
    prior_container = np.zeros((dataset.midi_data[0].shape[0],1))
    dataset_length = 0

    for midi in dataset.midi_data:
        dataset_length += midi.shape[1]
        for row in range(dataset.midi_data[0].shape[0]):
            prior_container[row] += np.sum(midi[row])

    prior_container /= dataset_length
    return prior_container

def generate_baseline(dataset, length=64):
    priors = calculate_priors(dataset)
    baseline = np.zeros((dataset.midi_data[0].shape[0],length))
    for i in range(dataset.midi_data[0].shape[0]):
        row = np.random.binomial(1, priors[i], size=length)
        baseline[i, :] = row

    return baseline

def plot_midi(midi):
	midi = np.repeat(midi, 3, axis=1)

	# For print
	plt.figure(figsize=(10,5))
	plt.axis("off")
	plt.imshow(midi, cmap="binary")
	plt.show()