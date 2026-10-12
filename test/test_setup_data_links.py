"""The setup helper must preserve existing data and preflight before writing."""
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/setup_data_links.sh'
SPLITS = ('UAVD4L-LoD/inTraj', 'UAVD4L-LoD/outTraj',
          'Swiss-EPFL/inPlace', 'Swiss-EPFL/outPlace')


class DataSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.repo = root / 'repo'
        self.external = root / 'external data'
        self.checkpoints = root / 'external checkpoints'
        (self.repo / 'scripts').mkdir(parents=True)
        (self.repo / 'data').mkdir()
        self.checkpoints.mkdir()
        self.script = self.repo / 'scripts/setup_data_links.sh'
        shutil.copy2(SCRIPT, self.script)
        for index, split in enumerate(SPLITS):
            name = 'Query_image_full' if index == 1 else 'Query_image'
            (self.external / split / name).mkdir(parents=True)
        self.relative = 'UAVD4L-LoD/inTraj/GT_pose.txt'
        (self.external / self.relative).write_bytes(b'fixture\n')
        digest = hashlib.sha256(b'fixture\n').hexdigest()
        (self.repo / 'data/SHA256SUMS').write_text(f'{digest}  data/{self.relative}\n')

    def run_setup(self):
        return subprocess.run(['bash', str(self.script), str(self.external),
                               str(self.checkpoints)], capture_output=True, text=True)

    def test_links_metadata_images_and_checkpoints_idempotently(self):
        for _ in range(2):
            result = self.run_setup()
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.repo / 'data' / self.relative).resolve(),
                         self.external / self.relative)
        for index, split in enumerate(SPLITS):
            name = 'Query_image_full' if index == 1 else 'Query_image'
            self.assertEqual((self.repo / 'data' / split / 'Query_image').resolve(),
                             self.external / split / name)
        self.assertEqual((self.repo / 'checkpoints').resolve(), self.checkpoints)

    def test_wrong_hash_creates_no_links(self):
        (self.external / self.relative).write_bytes(b'wrong\n')
        result = self.run_setup()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('identity mismatch', result.stderr)
        self.assertFalse(any(p.is_symlink() for p in self.repo.rglob('*')))

    def test_conflict_preserves_files_and_creates_no_links(self):
        existing = self.repo / 'data' / SPLITS[-1] / 'Query_image'
        existing.mkdir(parents=True)
        (existing / 'keep.txt').write_text('preserve this')
        result = self.run_setup()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Existing path is preserved', result.stderr)
        self.assertEqual((existing / 'keep.txt').read_text(), 'preserve this')
        self.assertFalse(any(p.is_symlink() for p in self.repo.rglob('*')))

    def test_existing_matching_metadata_is_preserved(self):
        target = self.repo / 'data' / self.relative
        target.parent.mkdir(parents=True)
        target.write_bytes(b'fixture\n')
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(target.is_symlink())
        self.assertEqual(target.read_bytes(), b'fixture\n')


if __name__ == '__main__':
    unittest.main()
