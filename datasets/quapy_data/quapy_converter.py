import os
import pickle
import pandas as pd
from quapy.data.base import LabelledCollection, Dataset

# -----------------------
# Config
# -----------------------
INPUT_DIR = "./"
OUTPUT_DIR = "./"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------
# Helper function
# -----------------------
def save_csv(X, y, out_path):
    df = pd.DataFrame(X)
    df.columns = [f"f{i}" for i in range(df.shape[1])]
    df["class"] = y
    df.to_csv(out_path, index=False)

# -----------------------
# Main loop
# -----------------------
for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith(".pkl"):
        continue

    pkl_path = os.path.join(INPUT_DIR, fname)
    base_name = os.path.splitext(fname)[0]

    print(f"Processing: {fname}")

    try:
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)

        # Case 1: LabelledCollection
        if isinstance(obj, LabelledCollection):
            out_csv = os.path.join(OUTPUT_DIR, f"{base_name}.csv")
            save_csv(obj.instances, obj.labels, out_csv)

        # Case 2: Dataset (train/test)
        elif isinstance(obj, Dataset):
            train_csv = os.path.join(OUTPUT_DIR, f"{base_name}_train.csv")
            test_csv  = os.path.join(OUTPUT_DIR, f"{base_name}_test.csv")

            save_csv(obj.training.instances, obj.training.labels, train_csv)
            save_csv(obj.test.instances, obj.test.labels, test_csv)

        else:
            print(f"⚠️ Skipped (unsupported type): {type(obj)}")

    except Exception as e:
        print(f"❌ Failed on {fname}: {e}")

print("Done ✔")
