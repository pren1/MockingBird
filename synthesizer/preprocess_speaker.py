import librosa
import numpy as np

from encoder import inference as encoder
from utils import logmmse
from synthesizer import audio
from pathlib import Path
from pypinyin import Style
from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
from pypinyin.converter import DefaultConverter
from pypinyin.core import Pinyin
import pdb
import soundfile as sf

class PinyinConverter(NeutralToneWith5Mixin, DefaultConverter):
    pass

pinyin = Pinyin(PinyinConverter()).pinyin


def _process_utterance(wav: np.ndarray, text: str, out_dir: Path, basename: str, 
                      skip_existing: bool, hparams, speaker_name):
    ## FOR REFERENCE:
    # For you not to lose your head if you ever wish to change things here or implement your own
    # synthesizer.
    # - Both the audios and the mel spectrograms are saved as numpy arrays
    # - There is no processing done to the audios that will be saved to disk beyond volume  
    #   normalization (in split_on_silences)
    # - However, pre-emphasis is applied to the audios before computing the mel spectrogram. This
    #   is why we re-apply it on the audio on the side of the vocoder.
    # - Librosa pads the waveform before computing the mel spectrogram. Here, the waveform is saved
    #   without extra padding. This means that you won't have an exact relation between the length
    #   of the wav and of the mel spectrogram. See the vocoder data loader.
    
    
    # Skip existing utterances if needed
    mel_fpath = out_dir.joinpath("mels", "mel-%s.npy" % basename)
    speaker_path = out_dir.joinpath("audio", speaker_name)
    speaker_path.mkdir(exist_ok=True)
    
    wav_fpath = speaker_path.joinpath("%s" % basename)
    # print(wav_fpath)
    # if skip_existing and mel_fpath.exists() and wav_fpath.exists():
    #     print("not exist exception")
    #     return None

    # Trim silence
    if hparams.trim_silence:
        wav = encoder.preprocess_wav(wav, normalize=False, trim_silence=True)
    
    # Skip utterances that are too short
    if len(wav) < hparams.utterance_min_duration * hparams.sample_rate:
#         print(f"short exception: {text}")
        return None
    
    # Compute the mel spectrogram
    mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
    mel_frames = mel_spectrogram.shape[1]
    
    # Skip utterances that are too long
    if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
#         print(f"long exception: {text}")
        return None
    
    # # Write the spectrogram, embed and audio to disk
    # np.save(mel_fpath, mel_spectrogram.T, allow_pickle=False)
    # np.save(wav_fpath, wav, allow_pickle=False)

    sf.write(wav_fpath, wav, hparams.sample_rate)
    
    # Return a tuple describing this training example
    return wav_fpath.name, mel_fpath.name, "embed-%s.npy" % basename, len(wav), mel_frames, text
 

def _split_on_silences(wav_fpath, words, hparams):
    # Load the audio waveform
    wav, _ = librosa.load(wav_fpath, hparams.sample_rate)
    wav = librosa.effects.trim(wav, top_db= 40, frame_length=2048, hop_length=512)[0]
    if hparams.rescale:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max
    # denoise, we may not need it here.
    if len(wav) > hparams.sample_rate*(0.3+0.1):
        noise_wav = np.concatenate([wav[:int(hparams.sample_rate*0.15)],
                                    wav[-int(hparams.sample_rate*0.15):]])
        profile = logmmse.profile_noise(noise_wav, hparams.sample_rate)
        wav = logmmse.denoise(wav, profile, eta=0)

    resp = pinyin(words, style=Style.TONE3)
    res = [v[0] for v in resp if v[0].strip()]
    res = " ".join(res)

    return wav, res

def preprocess_speaker_general(speaker_dir, out_dir: Path, skip_existing: bool, hparams, dict_info, no_alignments: bool):
    metadata = []
    extensions = ["*.wav", "*.flac", "*.mp3"]
    for extension in extensions:
        wav_fpath_list = speaker_dir.glob(extension)
        # Iterate over each wav
        for wav_fpath in wav_fpath_list:
            words = dict_info.get(wav_fpath.name[:-4])
            words = dict_info.get(wav_fpath.name) if not words else words # try with wav 
            if not words:
                print(f"no wordS: {wav_fpath.name[:-4]}")
                continue
            sub_basename = wav_fpath.name
            wav, text = _split_on_silences(wav_fpath, words, hparams)
            # print(f"current text: {text}")
            metadata.append(_process_utterance(wav, text, out_dir, sub_basename, 
                                                skip_existing, hparams, speaker_name = speaker_dir.as_posix().split("/")[-1]))   
    print(f"processed: {speaker_dir}")
    return [m for m in metadata if m is not None]
