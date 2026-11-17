"""The non-snake set: openly licensed photos with no snake in them, to train and judge the no-snake gate.

Open Images V7 photos (CC BY 2.0) and iNaturalist photos of snake lookalikes (CC0 or CC BY). Every photo
keeps its licence, source url and owner, and a split keeps each owner's photos together.
"""
import re
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw
from scipy.fft import dctn

SEED = 42
SPLITS, SHARES = ("train", "val", "test"), (0.6, 0.2, 0.2)
MANIFEST_COLUMNS = ["file", "source", "owner", "licence", "url", "category", "split"]

OI_SNAKE = "/m/078jl"
# Open Images has species-level snake classes the 600-class hierarchy doesn't list under Snake. Matching
# their names also catches a few non-snakes (cars named Cobra, feather boas); dropping those costs nothing.
SNAKE_WORDS = re.compile(
    r"\b(?:snakes?|serpents?|python|cobra|viper|adder|boas?|rattlesnakes?|mambas?|krait|anaconda|constrictor"
    r"|colubrid\w*|elapid\w*|taipan|copperhead|cottonmouth|moccasin|racer|sidewinder|massasauga|kingsnake"
    r"|bullsnake|groundsnake|garter|hognose|hog-nosed)\b", re.I)

INAT_API = "https://api.inaturalist.org/v1"
INAT_PAGE = 200
INAT_PAUSE_SECONDS = 1.0     # iNaturalist asks API clients for about one request a second
INAT_PHOTO_SIZE = "medium"   # 500 px on the long side, the size of the snake photos
SERPENTES_ID = 85553         # iNaturalist's suborder Snakes

MAX_SIDE = 500               # every photo is saved at the snake photos' size, so size can't give a snake away
JPEG_QUALITY = 90
RETRY_STATUS = (429, 500, 502, 503, 504)
RETRIES, FIRST_WAIT_SECONDS, TIMEOUT_SECONDS = 6, 2.0, 60
HEADERS = {"User-Agent": "serpentguard-course-project"}

PHASH_SIZE, PHASH_KEEP = 32, 8   # the same hash as notebooks 03 and 09
NEAR_DUP_BITS = 8
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def snake_mids(classes, hierarchy):
    """Snake, every class under it in the hierarchy, and every class whose name names a snake."""
    names = classes.set_index("LabelName")["DisplayName"]
    if names.get(OI_SNAKE) != "Snake":
        raise ValueError(f"{OI_SNAKE} is not Snake in the class descriptions")
    found = {OI_SNAKE} | set(classes.loc[classes["DisplayName"].str.contains(SNAKE_WORDS), "LabelName"])

    def walk(node, under_snake):
        under_snake = under_snake or node["LabelName"] == OI_SNAKE
        if under_snake:
            found.add(node["LabelName"])
        for child in node.get("Subcategory", []):
            walk(child, under_snake)

    walk(hierarchy, False)
    return found


def open_images_candidates(image_labels, wanted, snake_mids, seed=SEED):
    """Per wanted label, up to that many images verified to show it; an image carrying a snake label is out.

    image_labels has ImageID, LabelName and Confidence (1 present, 0 verified absent; machine labels and
    boxes may be added for snake labels). Labels are filled in the order given, each image only once.
    """
    if set(wanted) & set(snake_mids):
        raise ValueError("a wanted label is a snake label")
    present = image_labels[image_labels["Confidence"] > 0]
    snakes = set(present.loc[present["LabelName"].isin(snake_mids), "ImageID"])
    verified = present[(present["Confidence"] == 1) & ~present["ImageID"].isin(snakes)]
    by_label = verified.groupby("LabelName")["ImageID"].unique()
    rng = np.random.default_rng(seed)
    taken, rows = set(), []
    for label, n in wanted.items():
        pool = sorted(set(by_label.get(label, [])) - taken)
        picked = [pool[i] for i in rng.permutation(len(pool))[:n]]
        taken.update(picked)
        rows += [(image_id, label) for image_id in picked]
    return pd.DataFrame(rows, columns=["ImageID", "LabelName"])


def get(url, params=None):
    """One GET, retried with doubling waits on HTTP 429, 5xx and dropped connections."""
    wait = FIRST_WAIT_SECONDS
    for _ in range(RETRIES):
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT_SECONDS)
        except requests.RequestException:
            response = None
        if response is not None and response.status_code not in RETRY_STATUS:
            response.raise_for_status()
            return response
        retry_after = response.headers.get("Retry-After", "") if response is not None else ""
        time.sleep(float(retry_after) if retry_after.isdigit() else wait)
        wait *= 2
    raise RuntimeError(f"gave up after {RETRIES} tries: {url}")


def get_json(url, params=None):
    """An iNaturalist API call, paced to one a second."""
    body = get(url, params).json()
    time.sleep(INAT_PAUSE_SECONDS)
    return body


def inat_taxon(taxon_id):
    """The taxon record, fetched by id: id, name, rank and common name."""
    record = get_json(f"{INAT_API}/taxa/{taxon_id}")["results"][0]
    return {"id": record["id"], "name": record["name"], "rank": record["rank"],
            "common_name": record.get("preferred_common_name", "")}


def inat_observations(taxon_id, licences):
    """Every research-grade observation of the taxon with a photo under one of the licences, by id."""
    observations, last = [], 0
    while True:
        page = get_json(f"{INAT_API}/observations",
                        {"taxon_id": taxon_id, "quality_grade": "research", "photo_license": ",".join(licences),
                         "photos": "true", "order_by": "id", "order": "asc", "per_page": INAT_PAGE,
                         "id_above": last})["results"]
        if not page:
            return observations
        observations += page
        last = page[-1]["id"]


def photo_row(observation, taxon_id, licences):
    """The observation's first photo under an allowed licence, or None for a snake, another taxon or no photo."""
    taxon = observation.get("taxon") or {}
    lineage = set(taxon.get("ancestor_ids", [])) | {taxon.get("id")}
    if observation.get("quality_grade") != "research" or SERPENTES_ID in lineage or taxon_id not in lineage:
        return None
    for photo in observation.get("photos", []):
        licence = (photo.get("license_code") or "").lower()
        if licence in licences and not photo.get("hidden"):
            return {"source": "inaturalist", "source_id": str(photo["id"]), "obs_id": str(observation["id"]),
                    "owner": observation["user"]["login"], "licence": licence,
                    "url": f"https://www.inaturalist.org/observations/{observation['id']}",
                    "image_url": photo["url"].replace("/square.", f"/{INAT_PHOTO_SIZE}."),
                    "attribution": photo.get("attribution", "")}
    return None


def take_turns(frame, n, seed=SEED):
    """n rows in which owners take turns, so no one photographer fills the quota."""
    if frame.empty:
        return frame
    shuffled = frame.sample(frac=1, random_state=seed)
    shuffled = shuffled.assign(turn=shuffled.groupby("owner").cumcount())
    return shuffled.sort_values("turn", kind="stable").head(n).drop(columns="turn")


def inat_lookalikes(taxon_ids, per_taxon, licences=("cc0", "cc-by"), seed=SEED):
    """Up to per_taxon research-grade photos of each taxon, one per observation, queried by taxon id.

    taxon_ids maps a scientific name to its iNaturalist id; the record fetched by that id must carry the name.
    """
    picked = []
    for name, taxon_id in taxon_ids.items():
        record = inat_taxon(taxon_id)
        if record["name"] != name:
            raise ValueError(f"taxon {taxon_id} is {record['name']}, not {name}")
        rows = [row for obs in inat_observations(taxon_id, licences)
                if (row := photo_row(obs, taxon_id, licences)) is not None]
        chosen = take_turns(pd.DataFrame(rows), per_taxon, seed)
        picked.append(chosen.assign(taxon=name, taxon_id=taxon_id, rank=record["rank"]))
    return pd.concat(picked, ignore_index=True)


def split_negatives(frame, shares=SHARES, seed=SEED):
    """Adds split (train, val, test): each owner's photos go to one split, filling every category's shares."""
    if not np.isclose(sum(shares), 1):
        raise ValueError("shares must add up to 1")
    category = frame["category"] if "category" in frame else pd.Series("all", index=frame.index)
    counts = pd.crosstab(frame["owner"], category)
    order = counts.sample(frac=1, random_state=seed)
    order = order.iloc[np.argsort(-order.sum(axis=1).to_numpy(), kind="stable")]  # big owners first
    totals = counts.sum().to_numpy(dtype=float)
    targets, filled = np.outer(shares, totals), np.zeros((len(shares), len(totals)))
    split_of = {}
    for owner, photos in zip(order.index, order.to_numpy(dtype=float)):
        gap = ((targets - filled) / totals * photos).sum(axis=1)  # how far each split is behind, for this owner
        best = int(np.argmax(gap))
        filled[best] += photos
        split_of[owner] = SPLITS[best]
    return frame.assign(split=frame["owner"].map(split_of))


def save_photo(url, dest, max_side=MAX_SIDE):
    """Downloads one photo and saves it as a JPEG no longer than max_side; returns its width and height."""
    with Image.open(BytesIO(get(url).content)) as image:
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side))
        dest.parent.mkdir(parents=True, exist_ok=True)
        image.save(dest, "JPEG", quality=JPEG_QUALITY)
        return image.size


def download_photos(frame, root, workers=4):
    """Saves every row's image_url at root/file; adds width, height and error (empty when saved)."""
    root = Path(root)

    def fetch(row):
        try:
            return (*save_photo(row.image_url, root / row.file), "")
        except Exception as error:  # a bad photo is recorded and dropped, not fatal
            return 0, 0, f"{type(error).__name__}: {error}"[:200]

    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(fetch, frame.itertuples()))
    width, height, error = zip(*results) if results else ((), (), ())
    return frame.assign(width=width, height=height, error=error)


def phash(image):
    """The 64-bit DCT hash of notebooks 03 and 09, as 16 hex digits."""
    gray = np.asarray(image.convert("L").resize((PHASH_SIZE, PHASH_SIZE), Image.LANCZOS), dtype=np.float32)
    low = dctn(gray, norm="ortho")[:PHASH_KEEP, :PHASH_KEEP].flatten()
    bits = low > np.median(low[1:])
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def hash_bits(hashes):
    return np.array([int(h, 16) for h in hashes], dtype=np.uint64)


def distances(bank, one):
    return POPCOUNT[(bank ^ one).view(np.uint8).reshape(-1, 8)].sum(axis=1)


def near_duplicate_rows(hashes, bits=NEAR_DUP_BITS):
    """Positions of photos within `bits` of an earlier photo, keeping the first of each group."""
    bank, repeats, kept = hash_bits(hashes), [], []
    for i, one in enumerate(bank):
        if kept and distances(bank[kept], one).min() <= bits:
            repeats.append(i)
        else:
            kept.append(i)
    return repeats


def cross_split_pairs(frame, bits=NEAR_DUP_BITS):
    """Pairs of positions (i < j) within `bits` of each other but in different splits."""
    bank, splits, pairs = hash_bits(frame["phash"]), frame["split"].to_numpy(), []
    for i in range(len(bank) - 1):
        close = np.flatnonzero(distances(bank[i + 1:], bank[i]) <= bits) + i + 1
        pairs += [(i, int(j)) for j in close if splits[j] != splits[i]]
    return pairs


def contact_sheet(frame, root, cols=10, thumb=160, caption=24):
    """A grid of the rows' photos, each captioned with its category and owner."""
    sheet = Image.new("RGB", (cols * thumb, -(-len(frame) // cols) * (thumb + caption)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(frame.itertuples()):
        with Image.open(Path(root) / row.file) as photo:
            tile = photo.convert("RGB")
            tile.thumbnail((thumb, thumb))
        left, top = (i % cols) * thumb, (i // cols) * (thumb + caption)
        sheet.paste(tile, (left + (thumb - tile.width) // 2, top))
        draw.text((left + 2, top + thumb + 1), f"{i} {row.category}"[:26], fill="black")
        draw.text((left + 2, top + thumb + 12), str(row.owner)[:26], fill="gray")
    return sheet
