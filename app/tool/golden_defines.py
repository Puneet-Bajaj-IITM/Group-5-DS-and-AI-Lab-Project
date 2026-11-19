"""Prints the --dart-define-from-file JSON for integration_test/golden_photos_test.dart."""
import json
from pathlib import Path

golden = Path(__file__).resolve().parents[1] / "test/golden/golden_photos.json"
# One line: a dart-define value cannot hold newlines.
print(json.dumps({"GOLDEN_PHOTOS": json.dumps(json.loads(golden.read_text()), separators=(",", ":"))}))
