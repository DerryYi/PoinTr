import os
import json

ROOT = "data/Rice"
SPLITS = ["train", "val", "test"]
TAXONOMY_ID = "rice"
TAXONOMY_NAME = "rice"

def collect_ids(split):
    partial_root = os.path.join(ROOT, split, "partial", TAXONOMY_ID)
    ids = sorted(os.listdir(partial_root))
    return ids

def main():
    data = {
        "taxonomy_id": TAXONOMY_ID,
        "taxonomy_name": TAXONOMY_NAME,
        "train": collect_ids("train"),
        "val": collect_ids("val"),
        "test": collect_ids("test")
    }

    final = [data]

    with open("data/Rice/Rice.json", "w") as f:
        json.dump(final, f, indent=2)

    print("train:", len(data["train"]))
    print("val:", len(data["val"]))
    print("test:", len(data["test"]))
    print("Total:", len(data["train"]) + len(data["val"]) + len(data["test"]))
    print("Saved to data/Rice/Rice.json")

if __name__ == "__main__":
    main()
