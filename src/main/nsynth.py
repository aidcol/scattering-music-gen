# Copyright 2019 Maurice Frank
#
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the "Software"), to deal 
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
# SOFTWARE.

import json
from glob import glob
from os import path
from typing import Optional, List, Dict

from operator import itemgetter
import librosa
import torch
from torch import dtype as torch_dtype
from torch.utils.data import Dataset, DataLoader

from utils import normalize_audio


# path to nsynth data dir
ROOT = "C:\\Users\\aidan\\source\\datasets\\nsynth-full"


class NSynthDataset(Dataset):
    """
    Dataset to handle the NSynth data in json/wav format.
    """

    def __init__(self, root: str = ROOT, subset: str = 'train',
                 families: Optional[List[str]] = None,
                 sources: Optional[List[str]] = None,
                 dtype: torch_dtype = torch.float32, mono: bool = True):
        """
        :param root: The path to the dataset. Should contain the sub-folders
            for the splits as extracted from the .tar.gz.
        :param subset: The subset to use.
        :param families: Only keep those Instrument families
        :param sources: Only keep those instrument sources
        :param dtype: The data type to output for the audio signals.
        :param mono: Whether to use mono signal instead of stereo.
        """
        self.dtype = dtype
        self.subset = subset.lower()
        self.mono = mono

        if isinstance(families, str):
            families = [families]
        if isinstance(sources, str):
            sources = [sources]

        assert self.subset in ['valid', 'test', 'train']

        self.root = path.normpath(f'{root}/nsynth-{subset}')
        if not path.isdir(self.root):
            raise ValueError('The given root path is not a directory.'
                             f'\nI got {self.root}')

        if not path.isfile(f'{self.root}/examples.json'):
            raise ValueError('The given root path does not contain an'
                             'examples.json')

        print(f'Loading NSynth data from split {self.subset} at {self.root}')

        with open(f'{self.root}/examples.json', 'r') as fp:
            self.attrs = json.load(fp)

        if families:
            self.attrs = {k: a for k, a in self.attrs.items()
                          if a['instrument_family_str'] in families}
        if sources:
            self.attrs = {k: a for k, a in self.attrs.items()
                          if a['instrument_source_str'] in sources}

        print(f'\tFound {len(self)} samples.')

        files_on_disk = set(map(lambda x: path.basename(x)[:-4],
                                glob(f'{self.root}/audio/*.wav')))
        if not set(self.attrs) <= files_on_disk:
            raise FileNotFoundError

        self.names = list(self.attrs.keys())

    def __len__(self):
        return len(self.attrs)

    def __str__(self):
        return f'NSynthDataset: {len(self):>7} samples in subset {self.subset}'

    def __getitem__(self, item: int):
        name = self.names[item]
        attrs = self.attrs[name]
        path = f'{self.root}/audio/{name}.wav'
        raw, _ = librosa.load(path, mono=self.mono, sr=attrs['sample_rate'])
        # Add channel dimension.
        if raw.ndim == 1:
            raw = raw[None, ...]
        attrs['audio'] = torch.tensor(raw, dtype=self.dtype)
        return attrs
    

class AudioOnlyNSynthDataset(NSynthDataset):
    def __init__(self, *args, targets: Optional[List[str]] = None, **kwargs):
        super(AudioOnlyNSynthDataset, self).__init__(*args, **kwargs)
        target_set = set(["instrument_source", "instrument_family", 
                        "pitch", "velocity"]) 
        for target in targets:
            assert target in target_set
        self.targets = targets

    def __getitem__(self, item: int):
        attrs = super(AudioOnlyNSynthDataset, self).__getitem__(item)
        audio = normalize_audio(attrs['audio'])
        audio_target = None
        if self.targets:
            audio_target = itemgetter(*self.targets)(attrs)
        return audio, audio_target


def make_loaders(subsets: List[str], nbatch: int,
                 targets: Optional[List[str]] = None,
                 families: Optional[List[str]] = None,
                 sources: Optional[List[str]] = None) -> Dict[str, DataLoader]:
    data_loaders = dict()
    for subset in subsets:
        dset = AudioOnlyNSynthDataset(root=ROOT, 
                                      subset=subset,
                                      families=families, 
                                      sources=sources,
                                      targets=targets)
        data_loaders[subset] = DataLoader(dset, batch_size=nbatch, 
                                          num_workers=4, shuffle=True)
    return data_loaders
