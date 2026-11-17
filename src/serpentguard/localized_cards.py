"""Validate and select reviewed, offline first-aid card text by stable ID."""

import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit


CARD_LANGUAGES = ("en", "hi", "ta", "te")
FIRST_AID_CARD_IDS = (
    "who-110-01", "who-119-04", "who-118-04", "who-119-03", "who-120-02", "who-120-05",
)


def validate_card_pack(pack, required_ids):
    """Require a complete set of reviewed cards in every supported language."""
    if not isinstance(pack, dict) or set(pack) != {"schema_version", "cards"}:
        raise ValueError("Expected schema_version and cards")
    if type(pack["schema_version"]) is not int or pack["schema_version"] != 1:
        raise ValueError("Unsupported card pack version")
    if not isinstance(pack["cards"], dict) or set(pack["cards"]) != set(CARD_LANGUAGES):
        raise ValueError("Provide English, Hindi, Tamil and Telugu card sets")
    required = set(required_ids)
    if not required:
        raise ValueError("At least one card ID is required")
    for language, cards in pack["cards"].items():
        if not isinstance(cards, dict) or set(cards) != required:
            raise ValueError(f"{language} must cover the same required card IDs")
        for card_id, card in cards.items():
            if not isinstance(card, dict) or set(card) != {"text", "source_url", "review_id"}:
                raise ValueError(f"{language}/{card_id} requires text, source_url and review_id")
            if any(not isinstance(value, str) or not value.strip() for value in card.values()):
                raise ValueError(f"{language}/{card_id} has an empty or invalid field")
            source = urlsplit(card["source_url"])
            if source.scheme != "https" or not source.hostname:
                raise ValueError(f"{language}/{card_id} requires an HTTPS source")


def resolve_card(pack, card_id, locale):
    """Return the display text and its actual language for offline voice selection."""
    language = locale.replace("_", "-").split("-", 1)[0].lower()
    for candidate in dict.fromkeys((language, "en")):
        card = pack["cards"].get(candidate, {}).get(card_id)
        if card is not None:
            return {"text": card["text"], "language": candidate}
    raise KeyError(card_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack", type=Path)
    parser.add_argument("--card-id", action="append", help="Override the default required card IDs")
    args = parser.parse_args()
    try:
        with args.pack.open(encoding="utf-8") as file:
            pack = json.load(file)
        required = args.card_id or FIRST_AID_CARD_IDS
        validate_card_pack(pack, required)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"Validated {len(set(required))} card IDs across {', '.join(CARD_LANGUAGES)}.")


if __name__ == "__main__":
    main()
