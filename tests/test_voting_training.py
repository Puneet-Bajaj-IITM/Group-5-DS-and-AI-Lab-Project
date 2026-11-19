import hashlib
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from uuid import uuid4
from zipfile import ZipFile

from PIL import Image

from serpentguard.voting_api import RequestError, export_training, make_app, provision_user


class TrainingExportTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.db = self.folder / "voting.sqlite3"
        self.reporter = provision_user(self.db, "citizen")
        self.reviewer = provision_user(self.db, "expert")
        self.curator = provision_user(self.db, "expert", training_curator=True)
        self.app = make_app(self.db)
        with BytesIO() as image:
            Image.new("RGB", (90, 64), "green").save(image, "JPEG")
            self.jpeg = image.getvalue()

    def request(self, method, path, user, expected, *, data=None, jpeg=None):
        body, content_type = b"", ""
        if data is not None:
            body, content_type = json.dumps(data).encode(), "application/json"
        if jpeg is not None:
            body, content_type = jpeg, "image/jpeg"
        environ = {
            "REQUEST_METHOD": method, "PATH_INFO": path, "CONTENT_TYPE": content_type,
            "CONTENT_LENGTH": str(len(body)), "wsgi.input": BytesIO(body),
            "HTTP_AUTHORIZATION": f"Bearer {user['token']}",
        }
        statuses = []
        response = self.app(environ, lambda status, _: statuses.append(status))
        result = json.loads(b"".join(response))
        self.assertEqual(int(statuses[0].split()[0]), expected, result)
        return result

    def submit(self, consent, *, upload=True):
        sighting_id = str(uuid4())
        path = f"/sightings/{sighting_id}"
        self.request("POST", "/sightings", self.reporter, 201, data={
            "id": sighting_id, "trigger": "unknown", "source": "photo",
            "snake_found": True, "captured_at": "2026-11-16T08:30:00Z",
            "training_consent": consent,
            "evidence": [{"sha256": hashlib.sha256(self.jpeg).hexdigest()}],
        })
        if upload:
            self.request("PUT", path + "/evidence/0", self.reporter, 201, jpeg=self.jpeg)
        return sighting_id, path

    def vote(self, path, label, *, previous=None):
        vote = {"id": str(uuid4()), "rank": "genus", "label": label}
        if previous is not None:
            vote["previous_vote_id"] = previous
        self.request("POST", path + "/votes", self.reviewer, 201, data=vote)
        return vote["id"]

    def training_queue(self):
        return self.request("GET", "/training/review-feed", self.curator, 200)["items"]

    def test_training_requires_consent_consensus_and_curator_grant(self):
        _, no_consent = self.submit(False)
        self.vote(no_consent, "Naja")
        self.submit(True, upload=False)
        self.assertEqual(self.training_queue(), [])
        _, pending = self.submit(True)
        self.assertEqual(self.training_queue(), [])
        self.request("GET", "/training/review-feed", self.reviewer, 403)
        for path in (no_consent, pending):
            self.request(
                "POST", path + "/approve-for-training", self.curator, 409, data={"revision": 1},
            )
        self.vote(pending, "Naja")
        self.request(
            "POST", pending + "/approve-for-training", self.reviewer, 403, data={"revision": 1},
        )
        output = self.folder / "unauthorized.zip"
        with self.assertRaises(RequestError):
            export_training(self.db, output, self.reviewer["token"])
        self.assertFalse(output.exists())

    def test_export_uses_only_the_current_approved_revision(self):
        sighting_id, path = self.submit(True)
        first_vote = self.vote(path, "Naja")
        self.assertEqual(self.training_queue()[0]["consensus"]["revision"], 1)
        approval_path = path + "/approve-for-training"
        approval = self.request("POST", approval_path, self.curator, 201, data={"revision": 1})
        self.assertEqual(self.request("POST", approval_path, self.curator, 200, data={"revision": 1}), approval)
        self.assertEqual(self.training_queue(), [])

        output = self.folder / "approved.zip"
        self.assertEqual(export_training(self.db, output, self.curator["token"]), 1)
        with ZipFile(output) as archive:
            row = json.loads(archive.read("manifest.jsonl"))
            self.assertEqual(row["sighting_id"], sighting_id)
            self.assertEqual(row["consensus"]["label"], "Naja")
            self.assertEqual(row["approval"]["curator_id"], self.curator["user_id"])
            jpeg = archive.read(row["images"][0]["path"])
            self.assertEqual(row["images"][0]["sha256"], hashlib.sha256(jpeg).hexdigest())

        self.vote(path, "Ptyas", previous=first_vote)
        self.request("POST", path + "/votes", self.reviewer, 409, data={
            "id": str(uuid4()), "rank": "genus", "label": "Naja", "previous_vote_id": first_vote,
        })
        own_vote = self.request("GET", path + "/my-vote", self.reviewer, 200)["vote"]
        self.assertEqual(own_vote["label"], "Ptyas")
        self.assertIsNone(self.request("GET", path + "/my-vote", self.curator, 200)["vote"])
        self.request("POST", approval_path, self.curator, 409, data={"revision": 1})
        self.assertEqual(self.training_queue()[0]["consensus"]["revision"], 2)
        empty = self.folder / "stale.zip"
        self.assertEqual(export_training(self.db, empty, self.curator["token"]), 0)
        with ZipFile(empty) as archive:
            self.assertEqual(archive.namelist(), ["manifest.jsonl"])
            self.assertEqual(archive.read("manifest.jsonl"), b"")

        self.request("POST", approval_path, self.curator, 201, data={"revision": 2})
        with self.assertRaises(FileExistsError):
            export_training(self.db, output, self.curator["token"])
        with ZipFile(output) as archive:
            self.assertEqual(json.loads(archive.read("manifest.jsonl"))["consensus"]["revision"], 1)
        fresh = self.folder / "reapproved.zip"
        self.assertEqual(export_training(self.db, fresh, self.curator["token"]), 1)
        with ZipFile(fresh) as archive:
            self.assertEqual(json.loads(archive.read("manifest.jsonl"))["consensus"]["label"], "Ptyas")
