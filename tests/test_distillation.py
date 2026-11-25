import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from serpentguard.data import read_teacher, run_config, run_teacher
from serpentguard.models import distillation_loss, total_loss
from serpentguard.train import check_config, target_names

N_CLASSES = 6
KNOWN = [0, 1, 2]  # the classes the teacher can name
BASE = dict(run_id="kd", encoder="mobilevit_s", head="linear", losses=["ce", "kd"], front_end="F0",
            sampler="weighted", hierarchy=False)


def teacher_rows(n, seed=0):
    rows = np.zeros((n, N_CLASSES), np.float32)
    rows[:, KNOWN] = np.random.default_rng(seed).dirichlet(np.ones(len(KNOWN)), n)
    return torch.from_numpy(rows)


class DistillationLossTests(unittest.TestCase):
    def test_zero_when_the_student_matches_the_teacher(self):
        teacher = torch.softmax(torch.randn(4, N_CLASSES), dim=1)
        logits = torch.log(teacher)  # the same distribution, so equal at any temperature
        loss = distillation_loss(logits, teacher, torch.ones(4, dtype=torch.bool), 2.0)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_mass_off_the_teachers_classes_costs(self):
        teacher, valid = teacher_rows(8), torch.ones(8, dtype=torch.bool)
        on = torch.full((8, N_CLASSES), -10.0)
        on[:, KNOWN] = 2.0 * torch.log(teacher[:, KNOWN])
        off = on.clone()
        off[:, 4] = 5.0  # an unknown class takes most of the mass
        self.assertLess(distillation_loss(on, teacher, valid, 2.0).item(),
                        distillation_loss(off, teacher, valid, 2.0).item())

    def test_only_photos_with_a_teacher_count(self):
        teacher, logits = teacher_rows(6), torch.randn(6, N_CLASSES, requires_grad=True)
        valid = torch.tensor([True, False, True, False, False, False])
        part = distillation_loss(logits, teacher, valid, 2.0)
        alone = distillation_loss(logits[valid], teacher[valid], torch.ones(2, dtype=torch.bool), 2.0)
        self.assertAlmostEqual(part.item(), alone.item(), places=6)
        part.backward()
        self.assertTrue(torch.all(logits.grad[~valid] == 0))

    def test_no_teacher_photo_gives_zero_with_a_graph(self):
        logits = torch.randn(3, N_CLASSES, requires_grad=True)
        loss = distillation_loss(logits, teacher_rows(3), torch.zeros(3, dtype=torch.bool), 2.0)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()  # still part of the graph, so a batch without teacher photos trains

    def test_total_loss_adds_the_weighted_term(self):
        cfg = run_config({**BASE, "kd_weight": 0.5})
        out = {"species_logits": torch.randn(4, N_CLASSES), "genus_logits": None,
               "family_logits": None, "who_logits": torch.randn(4, 3), "mask_logits": None,
               "embedding": None}
        batch = {"class_id": torch.tensor([0, 1, 2, 0]), "who_risk_id": torch.tensor([0, 1, 2, 0]),
                 "teacher": teacher_rows(4), "teacher_valid": torch.ones(4, dtype=torch.bool)}
        total, parts = total_loss(out, batch, cfg)
        without, _ = total_loss(out, batch, {**cfg, "losses": ["ce"]})
        self.assertAlmostEqual(total.item(), without.item() + 0.5 * parts["kd"], places=5)


class TeacherConfigTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.file = self.folder / "teacher.npz"
        np.savez(self.file, paths=np.array(["a.jpg", "b.jpg"]), probs=teacher_rows(2).numpy())

    def test_kd_needs_an_existing_teacher_file(self):
        with self.assertRaises(ValueError):
            check_config(run_config(BASE))
        with self.assertRaises(ValueError):
            check_config(run_config({**BASE, "teacher_probs": str(self.folder / "missing.npz")}))
        check_config(run_config({**BASE, "teacher_probs": str(self.file)}))

    def test_temperature_must_be_positive(self):
        with self.assertRaises(ValueError):
            check_config(run_config({**BASE, "teacher_probs": str(self.file), "kd_temperature": 0}))

    def test_teacher_is_read_by_path_and_only_for_kd(self):
        cfg = run_config({**BASE, "teacher_probs": str(self.file)})
        teacher = run_teacher(cfg)
        self.assertEqual(sorted(teacher), ["a.jpg", "b.jpg"])
        self.assertEqual(teacher["a.jpg"].shape, (N_CLASSES,))
        self.assertIsNone(run_teacher({**cfg, "losses": ["ce"]}))
        self.assertIs(read_teacher(str(self.file)), teacher)
        self.assertIn("teacher", target_names(cfg, False))
        self.assertNotIn("teacher", target_names({**cfg, "losses": ["ce"]}, False))


if __name__ == "__main__":
    unittest.main()
