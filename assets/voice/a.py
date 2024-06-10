import librosa
import soundfile

raw_path = "test.wav"
out_path = "out.wav"
original_sr = 48000
target_sr = 16000
wav, sr = librosa.load(raw_path, sr=original_sr)
wav16k = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
wav16k = wav16k
soundfile.write(out_path, wav16k, target_sr)
