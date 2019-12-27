import numpy as np
import matplotlib.pyplot as plt
import pretty_midi

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

def synthetize_baseline(dataset, length=64):
	prior_container = calculate_priors(dataset)
	baseline = generate_baseline(dataset_length, length)

	program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
	midi_from_proll = vae.midi_utils.piano_roll_to_pretty_midi(baseline, fs = 16, program = program)

	# save midi to specified location
	save_path = '../results/baseline.midi'
	midi_from_proll.write(save_path)
	print('Saved baseline at {}'.format(save_path))

if __name__ == '__main__':
	root_path = '../data/maestro-v2.0.0'
	train_dataset = vae.midi_dataloader.MIDIDataset(root_path, split='train', year=2004)
	synthetize_baseline(train_dataset, length=64*4)
