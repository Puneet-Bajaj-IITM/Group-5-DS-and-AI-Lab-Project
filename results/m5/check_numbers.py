"""Checks the Milestone 5 report's headline numbers against the result files in this folder.

Reads only files that are in git, needs no GPU and no third-party package, and stops at the first number that
disagrees with its source. A deliberately wrong expectation runs first and must be caught, so a check that
can never fail is noticed.

Usage: python results/m5/check_numbers.py
"""
import csv
import json
from pathlib import Path

M5 = Path(__file__).resolve().parent
SCORES = M5 / "test/test_scores.json"
REPORT_NUMBERS = M5 / "report/data/report_numbers.json"
PER_SPECIES = M5 / "test/per_species_heldout.csv"
ROBUSTNESS_TABLE = M5 / "robustness/results/robustness_table.csv"
GRADCAM = M5 / "robustness/results/gradcam.json"
G5_FROZEN = M5 / "test/g5/results/g5_summary.csv"

MEASURABLE = 5
F1_BINS = ((5, 19), (20, 99), (100, None))
STRIDE8 = "trained, stride-8 stage (stages.2)"
NEW_CLIPS = "new_test_scores"
G5_SPEEDS = ("0.1", "0.25", "0.4", "0.8")


class Mismatch(AssertionError):
    pass


def expect(name, actual, wanted, places):
    if round(actual, places) != round(wanted, places):
        raise Mismatch(f"{name}: the file says {actual}, the report says {wanted}")
    print(f"ok  {name} = {round(actual, places)}")


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def instrument_check(goals):
    try:
        expect("control: a wrong safety bound", goals["safety"]["lower_bound"], 0.95, 3)
    except Mismatch:
        print("ok  control: the wrong value was caught")
        return
    raise SystemExit("the check passed a value it should have caught")


def goals_on_test(scores):
    goals = scores["subsets"]["all"]["goals"]
    instrument_check(goals)
    expect("G1 scenery gap", goals["G1"]["value"], 0.057, 3)
    expect("G2 species at F1 0.5", goals["G2"]["value"], 89, 0)
    expect("G2 measurable species", goals["G2"]["of"], 165, 0)
    for measure, (diff, low, high) in {"top1": (-0.006, -0.020, 0.008),
                                       "macro_f1": (0.006, -0.017, 0.028)}.items():
        g3 = goals["G3"][measure]
        expect(f"G3 {measure} minus B1", g3["diff"], diff, 3)
        expect(f"G3 {measure} interval low", g3["low"], low, 3)
        expect(f"G3 {measure} interval high", g3["high"], high, 3)
    expect("G3 slice photos", goals["G3"]["photos"], 12203, 0)
    expect("safety lower bound", goals["safety"]["lower_bound"], 0.926, 3)
    expect("safety recall", goals["safety"]["recall"], 0.944, 3)
    expect("WHO-listed photos", goals["safety"]["who_listed"], 2633, 0)
    expect("G4 non-snakes called a snake", goals["G4"]["false_snake"], 0.05, 4)
    expect("G4 false snakes, count", goals["G4"]["false_snake_count"], 30, 0)
    expect("G4 snakes kept", goals["G4"]["kept"], 0.9931, 4)


def safety_rows(scores):
    safety = scores["subsets"]["all"]["safety"]
    gate = safety["frozen, behind the M4 gate"]
    check = safety["frozen, behind the snake check"]
    expect("false alarms behind the gate", gate["false_alarm"], 0.216, 3)
    expect("bound behind the snake check", check["lower_bound"], 0.946, 3)
    expect("recall behind the snake check", check["who_recall"], 0.960, 3)
    expect("false alarms behind the snake check", check["false_alarm"], 0.223, 3)
    expect("test-only safety bound",
           scores["subsets"]["test only"]["goals"]["safety"]["lower_bound"], 0.910, 3)


def derived_counts(numbers):
    composition, errors = numbers["composition"], numbers["errors"]
    expect("held-out photos", composition["photos"], 18533, 0)
    expect("species present", composition["species"], 181, 0)
    expect("photographers", composition["photographers"], 2281, 0)
    expect("species wrong", errors["species_wrong"], 5298, 0)
    expect("WHO-listed not called dangerous", errors["who_listed_not_dangerous"], 147, 0)
    expect("of them dropped by the gate", errors["who_listed_gate_dropped"], 61, 0)
    expect("harmless called dangerous", errors["harmless_called_dangerous"], 3022, 0)
    expect("snakes dropped by the check", errors["snake_check_dropped"], 128, 0)


def rare_against_common():
    rows = [row for row in read_csv(PER_SPECIES) if int(row["photos"]) >= MEASURABLE]
    expect("measurable species", len(rows), 165, 0)
    expect("frozen species with F1 0", sum(float(r["frozen_f1"]) == 0 for r in rows), 33, 0)
    expect("A3 species with F1 0", sum(float(r["a3_f1"]) == 0 for r in rows), 21, 0)
    expect("species where M5 is worse than A3",
           sum(float(r["frozen_f1"]) < float(r["a3_f1"]) for r in rows), 74, 0)
    expect("species where M5 is better than A3",
           sum(float(r["frozen_f1"]) > float(r["a3_f1"]) for r in rows), 68, 0)
    wanted = {(5, 19): (0.269, 0.318), (20, 99): (0.478, 0.496), (100, None): (0.708, 0.686)}
    for low, high in F1_BINS:
        group = [r for r in rows if int(r["photos"]) >= low and (high is None or int(r["photos"]) <= high)]
        for column, value in zip(("frozen_f1", "a3_f1"), wanted[(low, high)]):
            mean = sum(float(r[column]) for r in group) / len(group)
            expect((f"mean {column}, {low} to {high} photos" if high else f"mean {column}, {low} or more photos"), mean, value, 3)


def robustness():
    table = {row["condition"]: row for row in read_csv(ROBUSTNESS_TABLE)}
    expect("clean top-1 through the harness", float(table["clean (harness rerun, vikas25s)"]["top1"]), 0.714, 3)
    expect("blur 0.5% top-1 change", float(table["blur_s1"]["top1_change"]), -0.109, 3)
    expect("25% covered WHO recall", float(table["occlude_25pct"]["who_recall"]), 0.599, 3)
    expect("noise sd 50 false alarms", float(table["noise_s3"]["false_alarm"]), 0.787, 3)
    stride8 = read_json(GRADCAM)["all"][STRIDE8]
    expect("Grad-CAM stride-8 heat minus area", stride8["mass_minus_area_mean"], 0.068, 3)
    expect("Grad-CAM stride-8 photos above area", stride8["share_with_mass_above_area"], 0.709, 3)


def video():
    rows = [row for row in read_csv(G5_FROZEN) if row["set"] == NEW_CLIPS]
    speeds = {row["latency_s"]: row for row in rows}
    for speed in G5_SPEEDS:
        expect(f"G5 most changes at {speed} s a frame", int(speeds[speed]["max_changes"]), 1, 0)
    if speeds["0.4"]["final_right"] != "12/19" or speeds["0.4"]["safe_species_ending_dangerous"] != "12/13":
        raise Mismatch("G5 at 0.4 s: final right or safe clips ending dangerous differ from the report")
    print("ok  G5 at 0.4 s: 12/19 final right, 12/13 safe clips end on dangerous")


def main():
    scores = read_json(SCORES)
    goals_on_test(scores)
    safety_rows(scores)
    derived_counts(read_json(REPORT_NUMBERS))
    rare_against_common()
    robustness()
    video()
    print("every checked number matches its source")


if __name__ == "__main__":
    main()
