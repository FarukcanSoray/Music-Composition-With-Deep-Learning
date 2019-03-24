import midi
pattern = midi.Pattern(resolution = 480)
track = midi.Track()
pattern.append(track)

pattern2 = midi.read_midifile("/home/farukcan/MIDI_DATASET/bach_846_format0.mid")
midi.write_midifile("example.mid", pattern2)


for i in range(50):
    track.append(pattern2[0][i])
print(pattern)

midi.write_midifile("example.mid", pattern)