from os import walk, path, makedirs
from pathlib import Path
import numpy as np
import pickle
import pretty_midi
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from tqdm import tqdm # used to pretty print loading progress
from itertools import *


class MIDIDataset(Dataset):
    def __init__(self, root_path, sequence_length=50, fs=16, year=-1, binarize=True, save_pickle=False):
        year = str(year) if not isinstance(year, str) else year
        self.sequence_length = sequence_length
        self.binarize = binarize
        self.save_pickle = save_pickle

        # Load pickle if it exists and return
        pickled_file = root_path + "/pickle/year_" + year + ".pkl"
        if Path(pickled_file).is_file():
            print('Found pickle dataset at {}. Start loading...'.format(pickled_file))
            with open(pickled_file, 'rb') as f:
                pickle_content = pickle.load(f)
                self.midi_data = pickle_content[0]
                self.trackID = len(self.midi_data.keys())
                self.dataset_length = pickle_content[1]
                print("Loaded pickled dataset. Size = {}, Path: {}" .format(self.dataset_length, pickled_file))

            return

        #TEST: we need to filter the dataset (potentially) and only include 4/4
        # Create dataset
        self.midi_files = []
        for (dirpath, dirnames, filenames) in walk(root_path):
            ff = [dirpath + "/" + file for file in filenames if ".midi" in file]
            if year == -1 or year in dirpath:
                self.midi_files.extend(ff)

        self.dataset_length = 0
        self.trackID = 0
        self.midi_data = {}
        print('Start loading dataset..')
        # tqdm() only perform pretty loading print, does not interact with the data in any other way
        for idx, file in enumerate(tqdm(self.midi_files)):
            piano_midi = pretty_midi.PrettyMIDI(file)

            if len(piano_midi.time_signature_changes) != 1 or piano_midi.time_signature_changes[0].numerator != 4 or piano_midi.time_signature_changes[0].denominator != 4:
                continue # if the time signature of the music is not 4/4 we skip this music

            # get the key of the music (by estimating the dominant semitone)
            total_velocity = sum(sum(piano_midi.get_chroma()))
            semitones = [sum(semitone)/total_velocity for semitone in piano_midi.get_chroma()]
            midi_key = np.argmax(semitones)

            # Shift all notes down by midi_key semitones if major, midi_key + 3 semitones if minor
            transpose_key = midi_key if semitones[(midi_key + 4) % 12] > semitones[(midi_key + 3) % 12] else midi_key + 3

            # Shift all notes down by transpose_key semitones
            for instrument in piano_midi.instruments:
                for note in instrument.notes:
                    note.pitch -= transpose_key if note.pitch - transpose_key >= 0 else transpose_key - 12

            # this is the required sampling frequency to get 16 x 16th notes in a bar (1 bar = 4 beats)
            fs = (piano_midi.estimate_tempo() * 16.0) / (4.0 * 60.0);

            piano_roll = piano_midi.get_piano_roll(fs=fs)[21:109, :]

            self.midi_data[self.trackID] = piano_roll
            self.trackID += 1
            self.dataset_length += piano_roll.shape[1] - (self.sequence_length - 1) # Remove uncomplete sequences from choices

        print('Loaded dataset. Number of tracks = {}, Total sample size = {}'.format(self.trackID + 1, self.dataset_length))

        # Pickle dataset
        if self.save_pickle:
            if not path.exists(root_path + "/pickle"):
                makedirs(root_path + "/pickle")
            with open(pickled_file, 'wb') as f:
                pickle.dump((self.midi_data, self.dataset_length), f)
                print('Saved dataset into pickle file at {}'.format(pickled_file))

    def __len__(self):
        return self.trackID + 1


    def __getitem__(self, idx):
        return self.midi_data[idx]


def create_sequential_data_loader(dataset, batch_size=10, test_split=0.15, validation_split=0.15, sequence_length=16, shuffle=True):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    test_split = int(np.floor(test_split * dataset_size))
    validation_split = int(np.floor(validation_split * dataset_size))
    shuffle = shuffle

    train_indices = indices[(validation_split + test_split):]
    test_indices = indices[validation_split:(validation_split + test_split)]
    validation_indices = indices[:validation_split]

    train_loader = sequential_data_loader(dataset, train_indices, batch_size, sequence_length, shuffle=shuffle)
    test_loader = sequential_data_loader(dataset, test_indices, batch_size, sequence_length, shuffle=shuffle)
    validation_loader = sequential_data_loader(dataset, validation_indices, batch_size, sequence_length, shuffle=shuffle)

    return train_loader, test_loader, validation_loader


class sequential_data_loader():
    def __init__(self, dataset, indices, batch_size, sequence_length, shuffle=True):
        self.dataset = dataset.midi_data
        self.indices = indices
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.shuffle = shuffle

        self.reset_data()
        self.reset_iter(hardReset=True)

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        self.reset_iter(hardReset=True)
        return self

    def __next__(self):
        while self.idx <= len(self):
            try:
                sequence = np.array(next(self.sequence_iter))
                return sequence
            except:
                self.idx += 1
                self.reset_iter(hardReset=False)
        
        # Reset data loader and iterator
        self.reset_data()
        self.reset_iter(hardReset=True)
        raise StopIteration # epoch complete if return is None

    def reset_data(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

        self.concat_data = []
        for i in range(len(self)):
            self.concat_data.append(self.dataset[i])
            
    def reset_iter(self, hardReset):
        if hardReset:
            self.idx = 0
        self.sequence_iter =  zip(*[islice(iter([self.dataset[self.idx][:,i:i+self.sequence_length].T for i in range(self.dataset[self.idx].shape[1] - self.sequence_length + 1)]), j, None) for j in range(self.batch_size)])


if __name__ == '__main__':
    root_path = r'C:\Users\Kronos\Downloads\maestro-v2.0.0-midi'
    allMIDI = MIDIDataset(root_path, fs=16, year=2004, binarize=True, save_pickle=True)

    batch_size = 10
    train_loader, test_loader, validation_loader = create_sequential_data_loader(allMIDI)

    print('Train size: {}'.format(len(train_loader)))
    print('Test size: {}'.format(len(test_loader)))
    print('Validation size: {}'.format(len(validation_loader)))
    print('\n------------------------------------------\nShape examples\n')


    for _ep in range(2):
        for i_batch, sample_batched in enumerate(train_loader):
            i, s = i_batch, sample_batched.shape
            print('train',i_batch, sample_batched.shape)

        for i_batch, sample_batched in enumerate(test_loader):
            i, s = i_batch, sample_batched.shape
            print('train',i_batch, sample_batched.shape)

        for i_batch, sample_batched in enumerate(validation_loader):
            i, s = i_batch, sample_batched.shape
            print('train',i_batch, sample_batched.shape)
