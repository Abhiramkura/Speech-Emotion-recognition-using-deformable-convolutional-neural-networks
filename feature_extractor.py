
import librosa
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def extract_features(audio_file, max_len=173, n_mfcc=40, n_mels=40):
    y, sr = librosa.load(audio_file, sr=None)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    mfcc = pad_sequences([mfcc.T], maxlen=max_len, padding='post', truncating='post', dtype='float32')[0]
    mel_db = pad_sequences([mel_db.T], maxlen=max_len, padding='post', truncating='post', dtype='float32')[0]

    combined = np.stack((mfcc, mel_db), axis=-1) 
    return np.transpose(combined, (1, 0, 2))       

