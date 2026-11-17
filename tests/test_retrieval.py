import unittest

from serpentguard.localized_cards import CARD_LANGUAGES, resolve_card, validate_card_pack
from serpentguard.retrieval import reciprocal_rank_fusion


class RetrievalTests(unittest.TestCase):
    def test_rank_fusion_gives_credit_for_agreement(self):
        self.assertEqual(
            reciprocal_rank_fusion([["b", "shared"], ["a", "shared"]]),
            ["shared", "a", "b"],
        )

    def test_duplicate_is_not_a_second_vote_and_empty_is_valid(self):
        self.assertEqual(reciprocal_rank_fusion([["b", "b", "a"], ["a", "b"]]), ["a", "b"])
        self.assertEqual(reciprocal_rank_fusion([[], []]), [])


class CardPackTests(unittest.TestCase):
    def setUp(self):
        self.pack = {
            "schema_version": 1,
            "cards": {
                locale: {"demo": {
                    "text": f"{locale} test fixture", "source_url": "https://example.org/source",
                    "review_id": f"fixture-{locale}",
                }}
                for locale in CARD_LANGUAGES
            },
        }

    def test_speech_gets_the_actual_language_of_the_displayed_text(self):
        validate_card_pack(self.pack, ["demo"])
        self.assertEqual(resolve_card(self.pack, "demo", "ta-IN")["language"], "ta")
        self.assertEqual(resolve_card(self.pack, "demo", "te_IN")["language"], "te")
        self.assertEqual(resolve_card(self.pack, "demo", "fr-FR")["language"], "en")

    def test_pack_requires_complete_card_reviews(self):
        self.pack["cards"]["hi"]["demo"]["review_id"] = ""
        with self.assertRaises(ValueError):
            validate_card_pack(self.pack, ["demo"])
