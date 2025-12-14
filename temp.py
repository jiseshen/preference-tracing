# random_outfit_params.py
import csv
import random

RANGES = {
    "OutfitIndex": (0, 14),
    "OutfitMaterialIndex": (0, 6),
    "PatternOption": (0, 15),
    "PatternColor": (0, 3),
}

def sample_one(rng):
    return {
        "OutfitIndex": rng.randint(*RANGES["OutfitIndex"]),
        "OutfitMaterialIndex": rng.randint(*RANGES["OutfitMaterialIndex"]),
        "PatternOption": rng.randint(*RANGES["PatternOption"]),
        "PatternColor": rng.randint(*RANGES["PatternColor"]),
    }

def make_rows(skeleton_name, n, rng):
    rows = []
    for i in range(n):
        p = sample_one(rng)
        p["Skeleton"] = skeleton_name
        p["SampleID"] = f"{skeleton_name}_{i:02d}"
        rows.append(p)
    return rows

def main(out_path="random_outfits.csv", seed=42):
    rng = random.Random(seed)
    rows = []
    rows += make_rows("SkeletonA", 20, rng)
    rows += make_rows("SkeletonB", 20, rng)

    fieldnames = ["Skeleton", "SampleID", "OutfitIndex", "OutfitMaterialIndex", "PatternOption", "PatternColor"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

if __name__ == "__main__":
    main()