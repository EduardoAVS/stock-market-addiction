import pandas as pd
from pathlib import Path

def detect_encoding(path, sample_bytes=20000):
    try:
        import chardet
        raw = Path(path).read_bytes()
        det = chardet.detect(raw[:sample_bytes])
        return det.get("encoding") or "latin1"
    except Exception:
        return "latin1"

def detect_delimiter(path, encoding, sample_lines=10):
    import csv
    text = Path(path).read_bytes().decode(encoding, errors="replace")
    sample = "\n".join(text.splitlines()[:sample_lines])
    try:
        return csv.Sniffer().sniff(sample).delimiter
    except Exception:
        for d in [",", ";", "\t", "|"]:
            if d in sample:
                return d
    return ","  # fallback

def to01(v):
    if pd.isna(v):
        return "0"
    s = str(v).strip()
    if s in {"1", "1.0", "1,0", "X", "x", "True", "true", "yes", "Y", "y"}:
        return "1"
    return "0"

def csv_to_slf(csv_file, output_file, encoding=None, sep=None):
    # detect encoding/delimiter if not provided
    if encoding is None:
        encoding = detect_encoding(csv_file)
    if sep is None:
        sep = detect_delimiter(csv_file, encoding)

    # read csv
    df = pd.read_csv(csv_file, sep=sep, encoding=encoding, dtype=str, keep_default_na=False)

    # Prepare objects (lines numbered 1..n) and attributes (column headers)
    num_objects = df.shape[0]
    objects = [str(i+1) for i in range(num_objects)]
    attributes = df.columns.astype(str).tolist()

    # Convert matrix to 0/1 strings
    matrix = df.applymap(to01)

    # Write SLF file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("[Lattice]\n")
        f.write(f"{num_objects}\n")
        f.write(f"{len(attributes)}\n\n")

        f.write("[Objects]\n")
        for o in objects:
            f.write(o + "\n")
        f.write("\n[Attributes]\n")
        for a in attributes:
            f.write(a + "\n")
        f.write("\n[relation]\n")
        for i in range(num_objects):
            row = " ".join(matrix.iloc[i, :].tolist())
            f.write(row + " \n")

if __name__ == "__main__":
    # Substitua pelos caminhos corretos no seu PC
    csv_to_slf("binarized_data.csv", "Teste.slf")
    print("SLF gerado: Teste.slf")