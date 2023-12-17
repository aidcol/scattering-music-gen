import unittest

from src.main.features import MFCC, Scat1D, JTFS

from src.main.utils import make_exp_chirp


S1d = Scat1D(shape=1000, sr=16000, batch_size=3)


class TestScat1D(unittest.TestCase):
    def test_get_sr(self):
        self.assertEqual(S1d.sr, 16000)

    def test_get_batch_size(self):
        self.assertEqual(S1d.batch_size, 64)

    def test_get_transform(self):
        self.assertEqual(repr(S1d.transform), 'Scattering1D()')


if __name__ == '__main__':
    unittest.main()
