from __future__ import annotations

from pathlib import Path
import csv


def read_arff_header(src: Path) -> list[str]:
    header: list[str] = []
    with src.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            low = line.lower()
            if low.startswith("@attribute"):
                parts = line.split(None, 2)
                if len(parts) >= 2:
                    name = parts[1].strip().strip('"').strip("'")
                    header.append(name)
                continue
            if low.startswith("@data"):
                break
    return header


def main() -> None:
    src = Path(__file__).with_name("diamonds.arff")
    csv_path = src.with_suffix(".csv")
    arff_header = read_arff_header(src)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        csv_header = next(reader)

    print("arff_header", arff_header)
    print("csv_header", csv_header)
    print("match", arff_header == csv_header)


if __name__ == "__main__":
    main()
