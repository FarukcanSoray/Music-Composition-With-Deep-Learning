import os, random
import midi
import numpy as np



def loadPieces(dirpath, batch_len=16*8):

    pieces = {}

    for fname in os.listdir(dirpath):
        if fname[-4:] not in ('.mid','.MID'):
            continue

        name = fname[:-4]

        outMatrix = midiToNoteStateMatrix(os.path.join(dirpath, fname))
        if len(outMatrix) < batch_len:
            continue
        print(outMatrix.shape)

        pieces[name] = outMatrix
        print("Loaded {}".format(name))

    return pieces

lowerBound = 24
upperBound = 102
def midiToNoteStateMatrix(dirpath):
    pattern = midi.read_midifile(dirpath)
    timeleft = np.array([track[0].tick for track in pattern], dtype=np.int32)

    posns = np.array([0 for track in pattern],dtype=np.int32)

    statematrix = np.array([],dtype=np.int32)
    span = upperBound-lowerBound
    time = 0

    state = np.array([0 for x in range(span)],dtype=np.int32)

    statematrix = np.hstack((statematrix, state))

    while True:
        if time % (pattern.resolution / 4) == (pattern.resolution / 8):
            # Crossed a note boundary. Create a new state, defaulting to holding notes
            statematrix = np.vstack((statematrix, state))
            state = [0 for x in range(span)]
            

        for i in range(len(timeleft)):
            while timeleft[i] == 0:
                track = pattern[i]
                pos = posns[i]

                evt = track[pos]
                if isinstance(evt, midi.NoteEvent):
                    if (evt.pitch < lowerBound) or (evt.pitch >= upperBound):
                        pass
                        print ("Note {} at time {} out of bounds (ignoring)".format(evt.pitch, time))
                    else:
                        if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                            state[evt.pitch-lowerBound] = 0
                        else:
                            state[evt.pitch-lowerBound] = 1
                """elif isinstance(evt, midi.TimeSignatureEvent):
                    if evt.numerator not in (2, 4):
                        # We don't want to worry about non-4 time signatures. Bail early!
                        print ("Found time signature event {}. Bailing!".format(evt))
                        return statematrix"""

                try:
                    timeleft[i] = track[pos + 1].tick
                    posns[i] += 1
                except IndexError:
                    timeleft[i] = -1

            if timeleft[i] != -1:
                timeleft[i] -= 1

        if all(t == -1 for t in timeleft):
            break

        time += 1

    return statematrix