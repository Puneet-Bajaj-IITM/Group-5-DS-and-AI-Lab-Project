"""Taxonomy-aware community voting for uncertain snake sightings."""

import csv
from collections import Counter
from math import exp, fsum, isclose, log
from pathlib import Path

ROLE_WEIGHT = {"citizen": 1, "verified_rescuer": 2, "expert": 3}
RULE_VERSION = "taxonomic-bayes-v1"
# Each pair gives genus accuracy and species accuracy conditional on the correct genus.
ROLE_ACCURACY = {
    "citizen": (0.65, 0.60),
    "verified_rescuer": (0.80, 0.75),
    "expert": (0.90, 0.85),
}
MIN_POSTERIOR = 0.85
MIN_SUPPORT = 3

CATALOG_PATH = Path(__file__).resolve().parents[2] / "data" / "classes.csv"
with CATALOG_PATH.open(encoding="utf-8", newline="") as catalog_file:
    SPECIES_TO_GENUS = {
        row["species"]: row["genus"] for row in csv.DictReader(catalog_file)
    }


def label_likelihood(reported, truth, accuracy, option_count):
    """Categorical response model with uniform errors."""
    if reported == truth:
        return accuracy
    return (1 - accuracy) / (option_count - 1)


def consensus_label(probabilities, support):
    """Return a named label only if it clears both evidence gates."""
    candidate = max(probabilities, key=probabilities.get)
    if candidate is None or support[candidate] < MIN_SUPPORT:
        return None

    confidence = probabilities[candidate]
    tied = any(
        label != candidate and isclose(probability, confidence, rel_tol=1e-12)
        for label, probability in probabilities.items()
    )
    if confidence >= MIN_POSTERIOR and not tied:
        return candidate
    return None


class SightingBallot:
    """Collect and aggregate active votes for one sighting."""

    def __init__(self, reviewer_roles, species_to_genus=SPECIES_TO_GENUS):
        self._reviewer_roles = dict(reviewer_roles)
        self._species_to_genus = dict(species_to_genus)
        if not self._species_to_genus:
            raise ValueError("The species catalog is empty.")
        if any(role not in ROLE_WEIGHT for role in self._reviewer_roles.values()):
            raise ValueError("Every reviewer must have a recognized role.")

        # None represents an unlisted alternative at each rank.
        self._species_options = {None: [None]}
        for species, genus in self._species_to_genus.items():
            self._species_options.setdefault(genus, [None]).append(species)
        self._votes = {}

    def cast_vote(self, reviewer_id, rank, label):
        """Validate and save the reviewer's current choice."""
        if reviewer_id not in self._reviewer_roles:
            raise ValueError(f"Unknown reviewer: {reviewer_id!r}")
        if rank == "species" and label in self._species_to_genus:
            genus, species = self._species_to_genus[label], label
        elif rank == "genus" and label is not None and label in self._species_options:
            genus, species = label, None
        else:
            raise ValueError(f"Invalid {rank!r} vote for {label!r}")

        self._votes[reviewer_id] = (genus, species)

    def result(self):
        weighted_genus, weighted_species = Counter(), Counter()
        vote_evidence = []
        for reviewer_id, (genus, species) in self._votes.items():
            role = self._reviewer_roles[reviewer_id]
            weight = ROLE_WEIGHT[role]
            weighted_genus[genus] += weight
            if species is not None:
                weighted_species[species] += weight
            vote_evidence.append((genus, species, *ROLE_ACCURACY[role]))

        genus_count = len(self._species_options)
        log_scores = {}
        for true_genus, species_options in self._species_options.items():
            log_prior = -log(genus_count) - log(len(species_options))
            for true_species in species_options:
                log_score = log_prior
                for genus, species, genus_accuracy, species_accuracy in vote_evidence:
                    log_score += log(label_likelihood(
                        genus, true_genus, genus_accuracy, genus_count
                    ))
                    if species is not None:
                        option_count = len(self._species_options[genus])
                        likelihood = 1 / option_count
                        if genus == true_genus:
                            likelihood = label_likelihood(
                                species, true_species, species_accuracy, option_count
                            )
                        log_score += log(likelihood)
                log_scores[true_genus, true_species] = log_score

        # Shift log scores before exponentiating to preserve numerical stability.
        offset = max(log_scores.values())
        masses = {taxon: exp(score - offset) for taxon, score in log_scores.items()}
        total_mass = fsum(masses.values())
        species_posterior = {taxon: mass / total_mass for taxon, mass in masses.items()}
        genus_posterior = {
            genus: fsum(species_posterior[genus, species] for species in species_options)
            for genus, species_options in self._species_options.items()
        }
        named_species_posterior = {
            species: species_posterior[genus, species]
            for species, genus in self._species_to_genus.items()
        }
        # Leave unlisted species as individual alternatives when checking the winner.
        weighted_paths = Counter({
            (self._species_to_genus[species], species): weight
            for species, weight in weighted_species.items()
        })
        species_path = consensus_label(species_posterior, weighted_paths)
        genus_label = consensus_label(genus_posterior, weighted_genus)

        status, rank, label, confidence = "pending", None, None, None
        if species_path is not None:
            status, rank, label = "species_consensus", "species", species_path[1]
            confidence = species_posterior[species_path]
        elif genus_label is not None:
            status, rank, label = "genus_consensus", "genus", genus_label
            confidence = genus_posterior[genus_label]

        return {
            "status": status,
            "rank": rank,
            "label": label,
            "posterior_confidence": confidence,
            "reviewer_count": len(self._votes),
            "support_weight": {"genus": dict(weighted_genus), "species": dict(weighted_species)},
            "genus_posterior": genus_posterior,
            "species_posterior": named_species_posterior,
            "unlisted_species_posterior": {
                genus: species_posterior[genus, None]
                for genus in self._species_options if genus is not None
            },
        }
