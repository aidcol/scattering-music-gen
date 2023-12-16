import unittest

from src.main.nsynth import NSynthDataset

ROOT = "C:\\Users\\aidan\\source\\datasets\\nsynth-full"

class TestNSynthDataset(unittest.TestCase):
    def test_len_train_ds(self):
        """Verify fetching of the train split."""
        train_ds = NSynthDataset(root=ROOT, subset='train')
        self.assertEqual(len(train_ds), 289205)

    def test_len_valid_ds(self):
        """Verify fetching of the valid split."""
        valid_ds = NSynthDataset(root=ROOT, subset='valid')
        self.assertEqual(len(valid_ds), 12678)

    def test_len_test_ds(self):
        """Verify fetching of the test split."""
        test_ds = NSynthDataset(root=ROOT, subset='test')
        self.assertEqual(len(test_ds), 4096) 

    def test_choose_source(self):
        """Verify instrument source selection for loading NSynth."""
        instr_src = 'acoustic'
        ds = NSynthDataset(root=ROOT, subset='test', sources=[instr_src])
        for sample in ds:
            self.assertEqual(sample['instrument_source_str'], instr_src) 
    
    def test_choose_family(self):
        """Verify instrument family selection for loading NSynth."""
        instr_fam = 'bass'
        ds = NSynthDataset(root=ROOT, subset='test', families=[instr_fam])
        for sample in ds:
            self.assertEqual(sample['instrument_family_str'], instr_fam)


if __name__ == '__main__':
    unittest.main()
