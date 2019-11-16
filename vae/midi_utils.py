import pretty_midi
import numpy as np

def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0, offset=21):
    """
    Convert a piano roll array into a pretty_midi object with a single instrument.

    Parameters
    ----------
    piano_roll : np.ndarray, shape=(notes,frames), dtype=int
        Piano roll of one instrument. If notes |= 128, offset should be provided.
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart by ``1/fs`` seconds.
    program : int
        The program number of the instrument.
    offset : int
        If notes |= 128, offset specifies how many rows are missing in the beginning of the 0-127 spectrum of midi notes.

    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing the piano roll.
    """

    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    if notes == 0 or notes > 128:
        raise ValueError('The number of notes cannot be 0 nor larger than 128.')

    # TEST: implement `piano_roll` padding row-wise if notes |= 128 <- require offset (how many missing on top and bottom)
    # also pad 1 column of zeros to beginning/end so we can acknowledge inital and final events
    if notes != 128 and offset == 0:
        offset = 128-notes
    if notes == 128:
        offset = 0
    piano_roll = np.pad(piano_roll, [(offset, 128-offset-notes), (1, 1)], 'constant')
    notes, frames = piano_roll.shape

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        # if the sound persists, keep track of on-time
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        # if the note has finished playing, create a pretty_midi.Note
        else:
            pm_note = pretty_midi.Note(velocity = prev_velocities[note],
                                       pitch    = note,
                                       start    = note_on_time[note],
                                       end      = time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0

    pm.instruments.append(instrument)
    return pm
