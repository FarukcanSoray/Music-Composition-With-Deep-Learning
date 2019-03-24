import midi
pattern = midi.read_midifile("/home/farukcan/MIDI_DATASET/bach_846_format0.mid")
print(pattern[0][50])
