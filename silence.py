import os
import librosa
import soundfile as sf

count = 0
for filename in os.listdir("wav"):
    y, sr = librosa.load("wav/"+filename)
    yt, index = librosa.effects.trim(y)
    sf.write('wavs/'+filename, yt, sr, subtype='PCM_16')
    count += 1
    print("Converted: ", count, "(", round((count/2951) * 100, 2),"%)")