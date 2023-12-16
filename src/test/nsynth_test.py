import unittest

from src.main.nsynth import NSynthDataset

ROOT = "C:\\Users\\aidan\\source\\datasets\\nsynth-full"

class TestNSynthDataset(unittest.TestCase):
    def setUp(self):
        self.train_ds = NSynthDataset(root=ROOT, subset="train")
        self.valid_ds = NSynthDataset(root=ROOT, subset="valid")
        self.test_ds = NSynthDataset(root=ROOT, subset="test")

    def test_len_train_ds(self):
        self.assertEqual(len(self.train_ds), 289205)

    def test_len_valid_ds(self):
        self.assertEqual(len(self.valid_ds), 12678)

    def test_len_test_ds(self):
        self.assertEqual(len(self.test_ds), 4096)


if __name__ == '__main__':
    unittest.main()
