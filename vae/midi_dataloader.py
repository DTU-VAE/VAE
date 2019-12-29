from os import walk, path, makedirs
from pathlib import Path
import numpy as np
import pickle
import pretty_midi
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, RandomSampler, SequentialSampler, BatchSampler
from tqdm import tqdm # used to pretty print loading progress
import itertools


class PianoRoll(Dataset):
    def __init__(self, piano_midi, sequence_length=16, binarize=True):
        self.sequence_length = sequence_length

        # get the key of the music (by estimating the dominant semitone)
        total_velocity = sum(sum(piano_midi.get_chroma()))
        semitones = [sum(semitone)/total_velocity for semitone in piano_midi.get_chroma()]
        midi_key = np.argmax(semitones)

        # Shift all notes down by midi_key semitones if major, midi_key + 3 semitones if minor
        #transpose_key = midi_key if semitones[(midi_key + 4) % 12] > semitones[(midi_key + 3) % 12] else midi_key + 3
        transpose_key = 0 #TODO: temporarly disabled transposition of pitches

        # Shift all notes down by transpose_key semitones
        for instrument in piano_midi.instruments:
            for note in instrument.notes:
                note.pitch -= transpose_key if note.pitch - transpose_key >= 0 else transpose_key - 12

        # this is the required sampling frequency to get 16 x 16th notes in a bar (1 bar = 4 beats)
        fs = (piano_midi.estimate_tempo() * 16.0) / (4.0 * 60.0);

        self.piano_roll = piano_midi.get_piano_roll(fs=fs)[21:109, :]
        self.piano_roll = self.piano_roll.astype(np.float32)
        # binarize if set
        if binarize:
            self.piano_roll = np.clip(self.piano_roll, 0, 1)

        self.piano_roll = np.transpose(self.piano_roll)
        
    def __len__(self):
        return len(self.piano_roll//self.sequence_length)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx * self.sequence_length + self.sequence_length >= len(self.piano_roll):
            return self.piano_roll[len(self.piano_roll) - self.sequence_length:,:]

        return self.piano_roll[idx * self.sequence_length:idx * self.sequence_length + self.sequence_length, :]


class MIDIDataset(Dataset):
    def __init__(self, root_path, split='train', sequence_length=16, year=-1, binarize=True, save_pickle=True):
        # Load pickle if it exists and return
        pickled_file = root_path + "/pickle/year_" + str(year) + "_" + split + ".pkl"
        if Path(pickled_file).is_file():
            print('Found pickle dataset at {}. Start loading...'.format(pickled_file))
            with open(pickled_file, 'rb') as f:
                self.midi_data = pickle.load(f)
                print("Loaded pickled dataset from {}" .format(pickled_file))

            return

        self.midi_data = []
        midi_csv = pd.read_csv(root_path+'/maestro-v2.0.0.csv')
        for i in tqdm(range(len(midi_csv))):
            row = midi_csv.iloc[i]
            if (year == -1 or row['year'] == year) and row['split'] == split:
                piano_midi = pretty_midi.PrettyMIDI(root_path+'/'+row['midi_filename'])
                if len(piano_midi.time_signature_changes) != 1 or piano_midi.time_signature_changes[0].numerator != 4 or piano_midi.time_signature_changes[0].denominator != 4:
                    continue # if the time signature of the music is not 4/4 we skip this music

                self.midi_data.append(PianoRoll(piano_midi, sequence_length, binarize))

        # Pickle dataset
        if save_pickle:
            pickled_file = root_path + "/pickle/year_" + str(year) + "_" + split + ".pkl"
            if not path.exists(root_path + "/pickle"):
                makedirs(root_path + "/pickle")
            with open(pickled_file, 'wb') as f:
                pickle.dump(self.midi_data, f)
                print('Saved dataset into pickle file at {}'.format(pickled_file))


    def __len__(self):
        return len(self.midi_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.midi_data[idx]


class MIDIRandomDataset(Dataset):
    def __init__(self, root_path, sequence_length=50, fs=16, year=-1, add_limit_tokens=True, binarize=True, save_pickle=False):
        year = str(year) if not isinstance(year, str) else year
        self.sequence_length = sequence_length
        self.add_limit_tokens = add_limit_tokens
        self.binarize = binarize
        self.save_pickle = save_pickle

        # Load pickle if it exists and return
        if add_limit_tokens:
            pickled_file = root_path + "/pickle/year_" + year + "_89.pkl"
        else:
            pickled_file = root_path + "/pickle/year_" + year + "_88.pkl"
        if Path(pickled_file).is_file():
            print('Found pickle dataset at {}. Start loading...'.format(pickled_file))
            with open(pickled_file, 'rb') as f:
                pickle_content = pickle.load(f)
                self.midi_data = pickle_content[0]
                self.dataset_length = pickle_content[1]
                self.end_tokens = pickle_content[2]
                print("Loaded pickled dataset. Size = {}, Path: {}" .format(self.dataset_length, pickled_file))

            return

        # Create dataset
        self.midi_files = []
        for (dirpath, dirnames, filenames) in walk(root_path):
            ff = [dirpath + "/" + file for file in filenames if ".midi" in file]
            if year == -1 or year in dirpath:
                self.midi_files.extend(ff)

        self.dataset_length = 0
        self.end_tokens = []
        self.midi_data = []
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
            transpose_key = 0

            # Shift all notes down by transpose_key semitones
            for instrument in piano_midi.instruments:
                for note in instrument.notes:
                    note.pitch -= transpose_key if note.pitch - transpose_key >= 0 else transpose_key - 12

            # this is the required sampling frequency to get 16 x 16th notes in a bar (1 bar = 4 beats)
            fs = (piano_midi.estimate_tempo() * 16.0) / (4.0 * 60.0);

            piano_roll = piano_midi.get_piano_roll(fs=fs)[21:109, :]
            if self.add_limit_tokens:
                limit_array = np.zeros((1, piano_roll.shape[1]))
                limit_array[:,0] = 1
                limit_array[:,-1] = 1
                piano_roll = np.vstack((piano_roll, limit_array))

            self.midi_data.append(piano_roll)
            self.dataset_length += piano_roll.shape[1] - (self.sequence_length - 1) # Remove uncomplete sequences from choices
            self.end_tokens.append(self.dataset_length-1)

        print('Loaded dataset. Size = {}'.format(self.dataset_length))

        # Pickle dataset
        if self.save_pickle:
            if not path.exists(root_path + "/pickle"):
                makedirs(root_path + "/pickle")
            with open(pickled_file, 'wb') as f:
                pickle.dump((self.midi_data, self.dataset_length, self.end_tokens), f)
                print('Saved dataset into pickle file at {}'.format(pickled_file))


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
        sequence = sequence.astype(np.float32)

        # Binarize if set
        if self.binarize:
            sequence = np.clip(sequence, 0, 1)

        return np.transpose(sequence)


class SINUSDataset(Dataset):
    def __init__(self, sequence_length):
        self.sequence = 16 * 80
        x = np.linspace(-np.pi, np.pi, self.sequence + 1)
        features = [int(i) for i in 44 + 43 * np.sin(x)]

        sinus_array = []
        for i in range(self.sequence):
            column = np.zeros(88)
            column[features[i]] = 1
            sinus_array.append(column)
        self.sinus_roll = np.transpose(np.asarray(sinus_array))
        self.sequence_length = sequence_length

    def __len__(self):
        return (self.sequence-self.sequence_length)*1000

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx = idx % (self.sequence-self.sequence_length)

        sequence = self.sinus_roll[:, idx:idx+self.sequence_length]
        sequence = sequence.astype(np.float32)

        return np.transpose(sequence)


def data_loader(dataset, sampler):
    for indices in sampler:
        samplers = []
        for index in indices:
            samplers.append(SequentialSampler(dataset[index]))
            
        #for samples in itertools.zip_longest(*samplers):
        for samples in zip(*samplers):
            batch = []
            for i in range(len(samples)):
                if samples[i] is not None:
                    batch.append(torch.unsqueeze(torch.from_numpy(dataset[indices[i]][samples[i]]),0))

            yield torch.cat(batch, 0)
        yield None


def split_dataset(dataset, test_split=0.15, validation_split=0.15, shuffle=True):
    """
    Splits a given dataset into train, test, and validation sets.
    The train set is given by the ratio 1-(test_split+validation split).

    Arguments:
        dataset (Dataset): the dataset to be split
        test_split (float): ratio of the total (1) data to be split into test set
        validation_split (float): ratio of the total (1) data to be split into validation set
        shuffle (bool): indicates whether the dataset indices are randomly distributed within the splits

    Returns:
        (train_sampler, test_sampler, validation_sampler) (tuple): SubsetRandomSampler instances for train, test, and validation
    """

    # Creating data indices for training and validation splits
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    test_split = int(np.floor(test_split * dataset_size))
    validation_split = int(np.floor(validation_split * dataset_size))
    if shuffle:
        np.random.shuffle(indices)
    train_indices, test_indices, validation_indices = indices[(validation_split+test_split):], indices[validation_split:(validation_split+test_split)], indices[:validation_split]

    # Creating PT data samplers and loaders:	
    train_sampler      = SubsetRandomSampler(train_indices)
    test_sampler       = SubsetRandomSampler(test_indices)
    validation_sampler = SubsetRandomSampler(validation_indices)

    return train_sampler, test_sampler, validation_sampler


if __name__ == '__main__':
    batch_size = 10
    
    root_path = '../data/maestro-v2.0.0'

    
    
    
    
    
    allMIDI = SINUSDataset(256)

    batch_size = 10
    train_sampler, test_sampler, validation_sampler = split_dataset(allMIDI)

    train_loader = DataLoader(allMIDI, batch_size=batch_size, sampler=train_sampler, drop_last=True, num_workers=0)
    test_loader = DataLoader(allMIDI, batch_size=batch_size, sampler=test_sampler, drop_last=True, num_workers=0)
    validation_loader = DataLoader(allMIDI, batch_size=batch_size, sampler=validation_sampler, drop_last=True, num_workers=0)
    dataloader_all = DataLoader(allMIDI, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    print('Dataset size: {}'.format(len(dataloader_all)))
    print('Train size: {}'.format(len(train_loader)))
    print('Test size: {}'.format(len(test_loader)))
    print('Validation size: {}'.format(len(validation_loader)))
    print('\n------------------------------------------\nShape examples\n')

    for i_batch, sample_batched in enumerate(dataloader_all):
        print('all',i_batch, sample_batched.shape)
        if i_batch == 2:
            break

    for i_batch, sample_batched in enumerate(train_loader):
        print('train',i_batch, sample_batched.shape)
        if i_batch == 2:
            break

    for i_batch, sample_batched in enumerate(test_loader):
        print('test',i_batch, sample_batched.shape)
        if i_batch == 2:
            break

    for i_batch, sample_batched in enumerate(validation_loader):
        print('valid',i_batch, sample_batched.shape)
        if i_batch == 2:
            break

    exit()










    
    train_dataset = MIDIDataset(root_path, split='train', year=2004)
    valid_dataset = MIDIDataset(root_path, split='validation', year=2004)
    test_dataset  = MIDIDataset(root_path, split='test', year=2004)

    train_sampler = BatchSampler(RandomSampler(train_dataset), batch_size=batch_size, drop_last=True)
    valid_sampler = BatchSampler(RandomSampler(valid_dataset), batch_size=batch_size, drop_last=True)
    test_sampler  = BatchSampler(RandomSampler(test_dataset),  batch_size=batch_size, drop_last=True)
    
    train_loader = data_loader(train_dataset, train_sampler)
    valid_loader = data_loader(valid_dataset, valid_sampler)
    test_loader  = data_loader(test_dataset,  test_sampler)

    for i, data in enumerate(train_loader):
        print('train', i, data.shape)
        if i == 10:
            break

    for i, data in enumerate(valid_loader):
        print('valid', i, data.shape)
        if i == 10:
            break

    for i, data in enumerate(test_loader):
        print('test', i, data.shape)
        if i == 10:
            break


#if __name__ == '__main__':
#    root_path = '../data/maestro-v2.0.0'
#    allMIDI = MIDIDataset(root_path, fs=16, year=2004, add_limit_tokens=False, binarize=True, save_pickle=True)

#    batch_size = 10
#    train_sampler, test_sampler, validation_sampler = split_dataset(allMIDI)

#    train_loader = DataLoader(allMIDI, batch_size=batch_size, sampler=train_sampler, drop_last=True, num_workers=0)
#    test_loader = DataLoader(allMIDI, batch_size=batch_size, sampler=test_sampler, drop_last=True, num_workers=0)
#    validation_loader = DataLoader(allMIDI, batch_size=batch_size, sampler=validation_sampler, drop_last=True, num_workers=0)
#    dataloader_all = DataLoader(allMIDI, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

#    print('Dataset size: {}'.format(len(dataloader_all)))
#    print('Train size: {}'.format(len(train_loader)))
#    print('Test size: {}'.format(len(test_loader)))
#    print('Validation size: {}'.format(len(validation_loader)))
#    print('\n------------------------------------------\nShape examples\n')

#    for i_batch, sample_batched in enumerate(dataloader_all):
#        print('all',i_batch, sample_batched.shape)
#        if i_batch == 2:
#            break

#    for i_batch, sample_batched in enumerate(train_loader):
#        print('train',i_batch, sample_batched.shape)
#        if i_batch == 2:
#            break

#    for i_batch, sample_batched in enumerate(test_loader):
#        print('test',i_batch, sample_batched.shape)
#        if i_batch == 2:
#            break

#    for i_batch, sample_batched in enumerate(validation_loader):
#        print('valid',i_batch, sample_batched.shape)
#        if i_batch == 2:
#            break