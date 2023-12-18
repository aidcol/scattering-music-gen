import unittest

import torch
from torchaudio.transforms import MFCC as MFCCTorch
import numpy as np
from kymatio.torch import Scattering1D, TimeFrequencyScattering

from src.main.features import MFCC, Scat1D, JTFS
from src.main.utils import make_exp_chirp, make_hann_window


# Build AcousticFeature instances ##############################################

assert torch.cuda.is_available()
device = 'cuda'
sr = 16000
N = 4 * sr
batch_size = 2

# MFCC parameters
n_mfcc = 40
log_mels=True

# scattering transform parameters
J = 10
Q1 = 12
Q2 = 1
log2_T = J
T = 2 ** log2_T
J_fr = 5
Q_fr = 2
F = None

# AcousticFeature objects
mfcc = MFCC(sr=sr,
            batch_size=batch_size,
            device=device,
            n_mfcc=40,
            log_mels=log_mels)
st = Scat1D(shape=N,
            sr=sr,
            batch_size=batch_size,
            device=device,
            J=J,
            Q=(Q1, Q2),
            T=T)
jtfst = JTFS(shape=N, 
             sr=sr,
             batch_size=batch_size,
             device=device,
             J=J, 
             Q=(Q1, Q2), 
             T=T, 
             J_fr=J_fr, 
             Q_fr=Q_fr, 
             F=F)

# Build toy signals ############################################################

chirp = make_exp_chirp(n_samples=N,
                       sr=sr,
                       start_freq=512,
                       end_freq=1024) * make_hann_window(n_samples=N)
chirp_batch = torch.from_numpy(np.array([chirp, chirp])).to(device)

################################################################################


class TestMFCC(unittest.TestCase):
    def test_get_sr(self):
        self.assertEqual(mfcc.sr, sr)

    def test_get_batch_size(self):
        self.assertEqual(mfcc.batch_size, batch_size)

    def test_get_transform(self):
        mfcc_test = MFCCTorch(sample_rate=sr).cuda()
        self.assertEqual(repr(mfcc.transform), repr(mfcc_test))

    def test_compute_features(self):
        X = mfcc.compute_features(chirp_batch)
        self.assertEqual(X.shape[1], n_mfcc)


class TestScat1D(unittest.TestCase):
    def test_get_sr(self):
        self.assertEqual(st.sr, sr)

    def test_get_batch_size(self):
        self.assertEqual(st.batch_size, batch_size)

    def test_get_transform(self):
        self.assertEqual(repr(st.transform), 'Scattering1D()')

    def test_output_size(self):
        st_test = Scattering1D(shape=N, J=J, Q=(Q1, Q2), T=T)
        print(f"output size: {st_test.output_size(detail=True)}")
        self.assertEqual(st.transform.output_size(detail=True),
                         st_test.output_size(detail=True))

    def test_compute_features(self):
        X = st.compute_features(chirp_batch)
        self.assertEqual(X.shape[1], st.transform.output_size())


class TestJTFS(unittest.TestCase):
    def test_get_sr(self):
        self.assertEqual(jtfst.sr, sr)

    def test_get_batch_size(self):
        self.assertEqual(jtfst.batch_size, batch_size)

    def test_get_transform(self):
        self.assertEqual(repr(jtfst.transform), 'TimeFrequencyScattering()')

    def test_output_size(self):
        jtfst_test = TimeFrequencyScattering(shape=N,
                                             J=J,
                                             Q=(Q1, Q2),
                                             T=T,
                                             J_fr=J_fr,
                                             Q_fr=Q_fr,
                                             F=F,
                                             format="time")
        print(f"output size: {jtfst_test.output_size(detail=True)}")
        self.assertEqual(jtfst.transform.output_size(detail=True),
                         jtfst_test.output_size(detail=True))

    def test_compute_features(self):
        X = jtfst.compute_features(chirp_batch)
        self.assertEqual(X.shape[1], jtfst.transform.output_size())


if __name__ == '__main__':
    unittest.main()
