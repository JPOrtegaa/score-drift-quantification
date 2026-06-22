from __future__ import annotations

from pathlib import Path
import csv


def arff_to_csv(src: Path) -> Path:
    header: list[str] = []
    rows: list[list[str]] = []
    in_data = False

    with src.open("r", newline="", encoding="utf-8") as f:
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
                in_data = True
                break

        if in_data:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("%"):
                    continue
                row = next(
                    csv.reader(
                        [line],
                        delimiter=",",
                        quotechar="'",
                        skipinitialspace=True,
                    )
                )
                rows.append(row)

    dst = src.with_suffix(".csv")
    with dst.open("w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(header)
        writer.writerows(rows)

    return dst


def main() -> None:
    src = Path(__file__).with_name("diamonds.arff")
    dst = arff_to_csv(src)
    with dst.open("r", encoding="utf-8") as f:
        row_count = sum(1 for _ in f) - 1
    print(f"wrote {dst} rows {row_count} cols {len(_read_header(src))}")


def _read_header(src: Path) -> list[str]:
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


if __name__ == "__main__":
    main()
