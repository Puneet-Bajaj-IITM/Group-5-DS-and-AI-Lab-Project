"""The danger table, data/m3/medical_importance.csv: whether each class in classes.csv counts as dangerous, and on what
basis, citing pages of the WHO South-East Asia snakebite guideline (2nd edition, 2016) but never its wording.
Rebuild it with python -m serpentguard.medical_importance."""
from pathlib import Path

import pandas as pd

from serpentguard.subset import DANGEROUS_FAMILIES, DANGEROUS_GENERA_EXTRA, NOT_LISTED

REPO_ROOT = Path(__file__).resolve().parents[2]
CLASSES_CSV = REPO_ROOT / "data" / "classes.csv"
MEDICAL_CSV = REPO_ROOT / "data" / "m3" / "medical_importance.csv"

GUIDELINE = "WHO SEARO 2016"
PRINTED_PAGE_OFFSET = 3       # PDF page labels restart at 1 on PDF page 4; the printed footers agree
CATEGORY_DEFINITIONS_PAGE, INDIA_LISTS_PAGE = 84, 87
VENOMOUS_FAMILIES_PAGE = 30
GUIDELINE_NAMED_PAGE, KEELBACK_SPECIES_PAGE = 65, 30  # medically important colubrids; keelbacks among the dangerous

WHO_CATEGORY_1, WHO_CATEGORY_2 = "who_category_1", "who_category_2"
VENOMOUS_FAMILY, GUIDELINE_NAMED, NO_BASIS = "venomous_family", "who_guideline_named", "none"
BASES = (WHO_CATEGORY_1, WHO_CATEGORY_2, VENOMOUS_FAMILY, GUIDELINE_NAMED, NO_BASIS)
CATEGORY_BASIS = {"1": WHO_CATEGORY_1, "2": WHO_CATEGORY_2}
COLUMNS = ["species", "family", "genus", "who_category", "dangerous", "basis", "source", "note"]

GENUS_NAMED_NOTE = "genus named through R. subminiatus and R. tigrinus; this species not named"
LISTED_AS = {  # the guideline's older names for our classes
    "Ophiophagus kaalinga": "listed as Ophiophagus hannah, which O. kaalinga was later split from",
    "Macrovipera lebetinus": "listed as Macrovipera lebetina",
    "Craspedocephalus malabaricus": "listed as Trimeresurus (Craspedocephalus) malabaricus",
    "Craspedocephalus gramineus": "listed as Trimeresurus (Craspedocephalus) gramineus",
}
NAMED_AS = {"Rhabdophis helleri": "named as Rhabdophis subminiatus, which R. helleri was split from"}


def pages(*pdf_pages):
    """'PDF p. N (printed p. M)', or the 'pp.' form for several pages."""
    prefix = "p." if len(pdf_pages) == 1 else "pp."
    printed = (page - PRINTED_PAGE_OFFSET for page in pdf_pages)
    return f"PDF {prefix} {', '.join(map(str, pdf_pages))} (printed {prefix} {', '.join(map(str, printed))})"


CATEGORY_SOURCE = (f"{GUIDELINE}, {pages(INDIA_LISTS_PAGE)}, India lists; categories defined "
                   f"{pages(CATEGORY_DEFINITIONS_PAGE)}")
VENOMOUS_SOURCE = f"{GUIDELINE}, {pages(VENOMOUS_FAMILIES_PAGE)}, venomous families; family from classes.csv"
NAMED_SOURCE = f"{GUIDELINE}, {pages(GUIDELINE_NAMED_PAGE, KEELBACK_SPECIES_PAGE)}"


def build_medical_importance(classes):
    """The danger table in class_id order: WHO category, then viper or elapid, then keelback."""
    rows = []
    for record in classes.sort_values("class_id").itertuples(index=False):
        who = str(record.who_category)
        if who != NOT_LISTED:
            basis, source = CATEGORY_BASIS[who], CATEGORY_SOURCE
            note = "; ".join(filter(None, (f"India Category {who}", LISTED_AS.get(record.species))))
        elif record.family in DANGEROUS_FAMILIES:
            basis, source, note = VENOMOUS_FAMILY, VENOMOUS_SOURCE, f"{record.family}, not in the WHO category lists"
        elif record.genus in DANGEROUS_GENERA_EXTRA:
            basis, source, note = GUIDELINE_NAMED, NAMED_SOURCE, NAMED_AS.get(record.species, GENUS_NAMED_NOTE)
        else:
            basis, source, note = NO_BASIS, "classes.csv", ""
        rows.append({"species": record.species, "family": record.family, "genus": record.genus, "who_category": who,
                     "dangerous": basis != NO_BASIS, "basis": basis, "source": source, "note": note})
    return pd.DataFrame(rows, columns=COLUMNS)


if __name__ == "__main__":
    build_medical_importance(pd.read_csv(CLASSES_CSV)).to_csv(MEDICAL_CSV, index=False)
