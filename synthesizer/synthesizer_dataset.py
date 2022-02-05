import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from synthesizer.utils.text import text_to_sequence
import pdb
import pickle

class SynthesizerDataset(Dataset):
    def __init__(self, metadata_fpath: Path, mel_dir: Path, embed_dir: Path, hparams):
        print("Using inputs from:\n\t%s\n\t%s\n\t%s" % (metadata_fpath, mel_dir, embed_dir))
        
        with metadata_fpath.open("r", encoding="utf-8") as metadata_file:
            metadata = [line.split("|") for line in metadata_file]
        
        mel_fnames = [x[1] for x in metadata if int(x[4])]
        mel_fpaths = [mel_dir.joinpath(fname) for fname in mel_fnames]
        embed_fnames = [x[2] for x in metadata if int(x[4])]
        embed_fpaths = [embed_dir.joinpath(fname) for fname in embed_fnames]
        self.samples_fpaths = list(zip(mel_fpaths, embed_fpaths))
        self.samples_texts = [x[5].strip() for x in metadata if int(x[4])]
        self.metadata = metadata
        self.hparams = hparams
        
        a_file = open("data.pkl", "rb")
        self.embed_dict = pickle.load(a_file)
        a_file.close()

        print(len(self.embed_dict))

        print("Found %d samples" % len(self.samples_fpaths))
    
    def __getitem__(self, index):  
        # Sometimes index may be a list of 2 (not sure why this happens)
        # If that is the case, return a single item corresponding to first element in index
        if index is list:
            index = index[0]

        mel_path, embed_path = self.samples_fpaths[index]
        mel = np.load(mel_path).T.astype(np.float32)
        
        # Load the embed
        # embed = np.load(embed_path)
        cur_name = embed_path.as_posix().split("/")[-1][6:].split(".")[0]

        # print(embed_path)
        # print(cur_name)
        if (cur_name in self.embed_dict):
          embed_res = self.embed_dict[cur_name]
          if embed_res == "嘉然":
            embed = np.asarray([0.15] * 256)
          elif embed_res == "向晚":
            embed = np.asarray([0.35] * 256)
          elif embed_res == "贝拉":
            embed = np.asarray([0.55] * 256)
          elif embed_res == "珈乐":
            embed = np.asarray([0.75] * 256)
          elif embed_res == "乃琳":
            embed = np.asarray([0.95] * 256)
          else:
            print(f"Fatal error: {embed_res}")
        else:
          embed = np.asarray([0.15] * 256)

        # Get the text and clean it
        text = text_to_sequence(self.samples_texts[index], self.hparams.tts_cleaner_names)
        
        # Convert the list returned by text_to_sequence to a numpy array
        text = np.asarray(text).astype(np.int32)

        return text, mel.astype(np.float32), embed.astype(np.float32), index

    def __len__(self):
        return len(self.samples_fpaths)


def collate_synthesizer(batch):
    # Text
    x_lens = [len(x[0]) for x in batch]
    max_x_len = max(x_lens)

    chars = [pad1d(x[0], max_x_len) for x in batch]
    chars = np.stack(chars)

    # Mel spectrogram
    spec_lens = [x[1].shape[-1] for x in batch]
    max_spec_len = max(spec_lens) + 1 
    if max_spec_len % 2 != 0:  # FIXIT: Hardcoded due to incompatibility with Windows (no lambda)
        max_spec_len += 2 - max_spec_len % 2

    # WaveRNN mel spectrograms are normalized to [0, 1] so zero padding adds silence
    # By default, SV2TTS uses symmetric mels, where -1*max_abs_value is silence.
    # if hparams.symmetric_mels:
    #     mel_pad_value = -1 * hparams.max_abs_value
    # else:
    #     mel_pad_value = 0
    mel_pad_value = -4 # FIXIT: Hardcoded due to incompatibility with Windows (no lambda)
    mel = [pad2d(x[1], max_spec_len, pad_value=mel_pad_value) for x in batch]
    mel = np.stack(mel)

    # Speaker embedding (SV2TTS)
    embeds = [x[2] for x in batch]
    embeds = np.stack(embeds)

    # Index (for vocoder preprocessing)
    indices = [x[3] for x in batch]


    # Convert all to tensor
    chars = torch.tensor(chars).long()
    mel = torch.tensor(mel)
    embeds = torch.tensor(embeds)

    return chars, mel, embeds, indices

def pad1d(x, max_len, pad_value=0):
    return np.pad(x, (0, max_len - len(x)), mode="constant", constant_values=pad_value)

def pad2d(x, max_len, pad_value=0):
    return np.pad(x, ((0, 0), (0, max_len - x.shape[-1])), mode="constant", constant_values=pad_value)
