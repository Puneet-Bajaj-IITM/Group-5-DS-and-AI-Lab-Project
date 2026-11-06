"""Splits the runs share: the calibration set, the unknown-pool halves and K1's held-out species. Notebook 10 draws
the training subset itself with these settings."""
import numpy as np

SUBSET_FRACTION, SUBSET_FLOOR, SEED = 0.15, 20, 42
NOT_LISTED = "not_listed"
HELDOUT_SPECIES = 10
HELDOUT_MIN_OBSERVATIONS = 15  # 10 to enrol from, at least 5 left to score
HELDOUT_MIN_PHOTOGRAPHERS, HELDOUT_MIN_VAL = 3, 2
DANGEROUS_FAMILIES = ("Viperidae", "Elapidae")  # never held out, nor are WHO-listed species
DANGEROUS_GENERA_EXTRA = ("Rhabdophis",)        # keelbacks, medically important outside those families


def calibration_set(train, subset):
    return train[~train["path"].isin(subset["path"]) & ~train["observer_id"].isin(subset["observer_id"])]


def unknown_split(unknown, seed=SEED):
    """(half_a, half_b) of the unknown pool by species: every WHO-listed species and a seeded half of the rest in A."""
    listed = set(unknown.loc[unknown["who_category"].astype(str) != NOT_LISTED, "species"])
    others = sorted(set(unknown["species"]) - listed)
    shuffled = np.random.default_rng(seed).permutation(others)
    in_a = unknown["species"].isin(listed | set(shuffled[:len(others) // 2]))
    return unknown[in_a], unknown[~in_a]


def heldout_species(classes, metadata, n=HELDOUT_SPECIES, seed=SEED):
    """K1's held-out species, sorted: n seeded tier 2 classes that aren't dangerous and have enough training
    observations, photographers and validation photos."""
    train, val = metadata[metadata["split"] == "train"], metadata[metadata["split"] == "val"]
    observations = train.groupby("species")["obs_id"].nunique()
    photographers = train.groupby("species")["observer_id"].nunique()
    val_photos = val.groupby("species").size()
    dangerous = ((classes["who_category"].astype(str) != NOT_LISTED) | classes["family"].isin(DANGEROUS_FAMILIES)
                 | classes["genus"].isin(DANGEROUS_GENERA_EXTRA))
    candidates = classes.loc[(classes["tier"] == "tier2") & ~dangerous, "species"]
    eligible = sorted(name for name in candidates if observations.get(name, 0) >= HELDOUT_MIN_OBSERVATIONS
                      and photographers.get(name, 0) >= HELDOUT_MIN_PHOTOGRAPHERS
                      and val_photos.get(name, 0) >= HELDOUT_MIN_VAL)
    return sorted(np.random.default_rng(seed).choice(eligible, n, replace=False).tolist())
