import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.prepare_expert_audit import REQUIRED, prepare, select_validation_rows, write_csv


def row(path, obs, tier, *, split="val"):
    return {
        "split": split, "path": path, "obs_id": obs, "observer_id": "observer",
        "species": "Synthetic species", "tier": tier, "who_category": "not_listed",
    }


class ExpertAuditTests(unittest.TestCase):
    def test_selection_is_seeded_and_uses_unique_validation_observations(self):
        rows = [
            row("a.jpg", "1", "tier1"), row("b.jpg", "1", "tier1"),
            row("c.jpg", "2", "tier2"), row("d.jpg", "3", "tier2"),
            row("test.jpg", "4", "tier1", split="test"),
        ]
        sample = select_validation_rows(rows, 3, 42)
        self.assertEqual(sample, select_validation_rows(reversed(rows), 3, 42))
        self.assertEqual({item["obs_id"] for item in sample}, {"1", "2", "3"})

    def test_review_package_keeps_reference_labels_private(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "species-name-in-filename.jpg"
            exif = Image.Exif()
            exif[270] = "unblinded source description"
            Image.new("RGB", (40, 30)).save(source, exif=exif)
            metadata = root / "metadata.csv"
            write_csv(metadata, sorted(REQUIRED), [row(source.name, "1", "tier1")])
            output = root / "audit"
            self.assertEqual(prepare(metadata, root, output, count=1), 1)
            with (output / "review" / "responses.csv").open(encoding="utf-8") as file:
                record = next(csv.DictReader(file))
            self.assertEqual(record["image"], "images/A0001.jpg")
            self.assertNotIn("species", record)
            with Image.open(output / "review" / record["image"]) as photo:
                self.assertFalse(photo.getexif())
            key = (output / "private" / "answer-key.csv").read_text(encoding="utf-8")
            self.assertIn(source.name, key)
            self.assertIn("Synthetic species", key)
