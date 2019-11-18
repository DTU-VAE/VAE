import pretty_midi
import numpy as np
from vae import midi_utils


# name of the midi file to load
midi = 'piano_example'

# load MIDI file into PrettyMIDI object
midi_data = pretty_midi.PrettyMIDI(f'{midi}.midi')
instrument = midi_data.instruments[0]
notes = instrument.notes

# sampling rate for midi->piano roll and backwards
fs = 16

# convert MIDI to piano roll -> 128 row (sound) times x number of columns based on `fs`
proll = midi_data.get_piano_roll(fs=fs)

# random piano roll imitation with shape (128, x)
test = np.random.randint(80, size=(128, 8000))

# convert piano roll to midi
program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
midi_from_proll = midi_utils.piano_roll_to_pretty_midi(proll, fs = fs, program = program)

# save midi
midi_from_proll.write(f'{midi}_processed.midi')

# synthesize the resulting MIDI data using sine waves
#audio_data = midi_data.synthesize()




"""
# Create a PrettyMIDI object
cello_c_chord = pretty_midi.PrettyMIDI()
# Create an Instrument instance for a cello instrument
program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
cello = pretty_midi.Instrument(program=program)
# Iterate over note names, which will be converted to note number later
for note_name in ['C5', 'E5', 'G5']:
    # Retrieve the MIDI note number for this note name
    note_number = pretty_midi.note_name_to_number(note_name)
    # Create a Note instance, starting at 0s and ending at .5s
    note = pretty_midi.Note(
        velocity=100, pitch=note_number, start=0, end=2)
    # Add it to our cello instrument
    cello.notes.append(note)
for note_name in ['A5', 'C5', 'E5']:
    # Retrieve the MIDI note number for this note name
    note_number = pretty_midi.note_name_to_number(note_name)
    # Create a Note instance, starting at 0s and ending at .5s
    note = pretty_midi.Note(
        velocity=100, pitch=note_number, start=2, end=4)
    # Add it to our cello instrument
    cello.notes.append(note)
# Add the cello instrument to the PrettyMIDI object
cello_c_chord.instruments.append(cello)
# Write out the MIDI data
cello_c_chord.write('cello-C-chord.midi')
"""