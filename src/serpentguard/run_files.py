"""The files a training run writes, named here so a module can read a run without importing the trainer."""

CONFIG_FILE, VERSIONS_FILE = "config.yaml", "versions.json"
HASH_FILE, CLASS_MIX_FILE = "subset_hash.txt", "class_mix.json"
METRICS_FILE, FINAL_FILE, CHECKPOINT_FILE = "metrics.jsonl", "final_metrics.json", "best.pt"
VAL_OUTPUTS, CALIBRATION_OUTPUTS, UNKNOWN_OUTPUTS = "val_outputs.npz", "calib_outputs.npz", "unknown_outputs.npz"
HELDOUT_OUTPUTS, PROTOTYPES_FILE = "heldout_outputs.npz", "prototypes.npz"
