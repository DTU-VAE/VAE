from os import walk, path
from pathlib import Path
import numpy as np
import pickle
import pretty_midi
import torch
from torch.utils.data import Dataset, DataLoader

root_path = path.expanduser(r'C:\Users\Kronos\Downloads\maestro-v2.0.0-midi')


## TODO
# Remove modifiers
# Filter given signature

class MIDIDataset(Dataset):
	def __init__(self, root_path, sequence_length=50, fs=16, year=-1, add_limit_tokens=True, binarize=False):
		year = str(year) if not isinstance(year, str) else year
		self.sequence_length = sequence_length
		self.add_limit_tokens = add_limit_tokens
		self.binarize = binarize

		# Load pickle if exists and return
		if add_limit_tokens:
			pickled_file = root_path + "/year_" + year + "_89.pkl"
		else:
			pickled_file = root_path + "/year_" + year + "_88.pkl"
		if Path(pickled_file).is_file():
			with open(pickled_file, 'rb') as f:
				pickle_content = pickle.load(f)
				self.midi_data = pickle_content[0]
				self.dataset_length = pickle_content[1]
				self.end_tokens = pickle_content[2]
				print("Loaded pickled dataset. Size = {}, Path: {}/training_data_year_{}.npy" .format(self.dataset_length, root_path, year))

			return

		# Create dataset
		self.midi_files = []
		for (dirpath, dirnames, filenames) in walk(root_path):
			ff = [dirpath + "/" + file for file in filenames if ".midi" in file]
			if year == -1 or year in dirpath:
				self.midi_files.extend(ff)

		# Logging
		counter = 0
		self.file_count = len(self.midi_files)

		self.dataset_length = 0
		self.end_tokens = []
		self.midi_data = []
		for idx, file in enumerate(self.midi_files):
			piano_midi = pretty_midi.PrettyMIDI(file)
			piano_roll = piano_midi.get_piano_roll(fs=fs)[21:109, :]
			if self.add_limit_tokens:
				limit_array = np.zeros((1, piano_roll.shape[1]))
				limit_array[:,0] = 1
				limit_array[:,-1] = 1
				piano_roll = np.vstack((piano_roll, limit_array))

			self.midi_data.append(piano_roll)

			self.dataset_length += piano_roll.shape[1] - (self.sequence_length - 1) # Remove uncomplete sequences from choices
			self.end_tokens.append(self.dataset_length-1)
			counter += 1
			if counter % 5 == 0:
				print("loading progress {:.2f}%" .format(counter / self.file_count * 100))

		# Pickle dataset
		with open(pickled_file, 'wb') as f:
			pickle.dump((self.midi_data, self.dataset_length, self.end_tokens), f)


	def __len__(self):
		return self.dataset_length


	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		start, list_idx = 0, 0
		for end_keys in self.end_tokens:
			if idx <= end_keys:
				break
			start = end_keys + 1
			list_idx += 1

		# Adjust for uncomplete sequences
		relative_idx = idx - start + (self.sequence_length - 1)
				
		# Get sequence
		sequence = self.midi_data[list_idx][:, relative_idx - (self.sequence_length - 1) : relative_idx + 1]

		# Binarize if set
		if self.binarize:
			sequence = np.clip(sequence, 0, 1)

		return np.transpose(sequence)
	
allMIDI = MIDIDataset(root_path, fs=16, year=2004, add_limit_tokens=True, binarize=False)

dataloader = DataLoader(allMIDI, batch_size=10, shuffle=True, num_workers=0)


for i_batch, sample_batched in enumerate(dataloader):
	print(i_batch, sample_batched.shape)
	
	if i_batch == 50000:
		break
