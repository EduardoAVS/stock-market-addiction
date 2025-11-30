import itertools
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from googletrans import Translator
except Exception:
    Translator = None  # keep optional; we can still run without network


EXCEL_FILE = "AFC.xlsx"
ORIGINAL_SHEET = "Original Data (Thai)"
CODED_SHEET = "Coded Data FINAL for regression"


def load_data(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    orig = pd.read_excel(path, sheet_name=ORIGINAL_SHEET)
    coded = pd.read_excel(path, sheet_name=CODED_SHEET)
    return orig, coded


def consistent_mapping(
    source: pd.Series, target: pd.Series, max_unique_ratio: float = 0.5
) -> Dict:
    """Return mapping if each source category maps to exactly one target value."""
    if source.nunique(dropna=False) > max_unique_ratio * len(source):
        return {}
    df = pd.DataFrame({"src": source, "tgt": target}).dropna()
    if df.empty:
        return {}
    mapping = df.groupby("src")["tgt"].nunique()
    if (mapping > 1).any():
        return {}
    return df.groupby("src")["tgt"].first().to_dict()


def detect_simple_mappings(
    original: pd.DataFrame, coded: pd.DataFrame
) -> Dict[str, Dict]:
    """Find 1:1 category/ordinal mappings and straight copies."""
    result = {}
    for coded_col in coded.columns:
        best = None
        for orig_col in original.columns:
            mapping = consistent_mapping(original[orig_col], coded[coded_col])
            if not mapping:
                continue
            score = len(mapping)  # prefer mappings that cover more categories
            if best is None or score > best[0]:
                best = (score, orig_col, mapping)
        if best:
            result[coded_col] = {
                "type": "category_map",
                "source_cols": [best[1]],
                "mapping": best[2],
            }
    return result


def detect_difference(
    original: pd.DataFrame, coded_col: pd.Series
) -> Tuple[List[str], str]:
    """Find two numeric columns whose difference matches the coded column."""
    numeric_cols = [
        c for c in original.columns if pd.api.types.is_numeric_dtype(original[c])
    ]
    best = None
    for c1, c2 in itertools.permutations(numeric_cols, 2):
        diff = original[c2] - original[c1]
        if diff.equals(coded_col):
            best = (c1, c2)
            break
    return list(best) if best else [], "difference c2 - c1"


def detect_mean_of_four(
    original: pd.DataFrame, coded_col: pd.Series
) -> Tuple[List[str], str]:
    """Look for an average of four ordinal 1-5 columns (used for gambling score)."""
    ord_cols = [
        c
        for c in original.columns
        if pd.api.types.is_numeric_dtype(original[c])
        and original[c].dropna().between(1, 5).all()
    ]
    for combo in itertools.combinations(ord_cols, 4):
        mean_val = original[list(combo)].mean(axis=1)
        if mean_val.equals(coded_col):
            return list(combo), "mean"
    return [], ""


def detect_stress_level(
    original: pd.DataFrame, coded_col: pd.Series
) -> Tuple[List[str], Dict]:
    """
    Detect mapping where coded = (mapped_ordinal(question about stress) + numeric_scale)/2.
    """
    cat_candidates = [
        c
        for c in original.columns
        if original[c].dtype == object and original[c].nunique() <= 10
    ]
    num_candidates = [
        c
        for c in original.columns
        if pd.api.types.is_numeric_dtype(original[c]) and original[c].nunique() <= 10
    ]
    for c_cat in cat_candidates:
        for c_num in num_candidates:
            needed = 2 * coded_col - original[c_num]
            df = pd.DataFrame({"cat": original[c_cat], "needed": needed}).dropna()
            if df.empty:
                continue
            checks = df.groupby("cat")["needed"].nunique()
            if (checks > 1).any():
                continue
            mapping = df.groupby("cat")["needed"].first().to_dict()
            reconstructed = original[c_cat].map(mapping)
            recon_avg = (reconstructed + original[c_num]) / 2
            if recon_avg.equals(coded_col):
                return [c_cat, c_num], mapping
    return [], {}


def detect_risk_tolerance(
    original: pd.DataFrame, coded_col: pd.Series
) -> Tuple[List[str], Dict[str, Dict]]:
    """Detect score that is a sum of two coded ordinal questions."""
    cat_candidates = [
        c
        for c in original.columns
        if original[c].dtype == object and original[c].nunique() <= 6
    ]
    for c1, c2 in itertools.permutations(cat_candidates, 2):
        s1 = original[c1]
        s2 = original[c2]
        # build linear system: map1[cat1] + map2[cat2] = coded
        cats1 = s1.dropna().unique().tolist()
        cats2 = s2.dropna().unique().tolist()
        rows = []
        y_vals = []
        for a, b, y in zip(s1, s2, coded_col):
            if pd.isna(a) or pd.isna(b) or pd.isna(y):
                continue
            row = [0] * (len(cats1) + len(cats2))
            row[cats1.index(a)] = 1
            row[len(cats1) + cats2.index(b)] = 1
            rows.append(row)
            y_vals.append(y)
        if not rows:
            continue
        coeffs, *_ = np.linalg.lstsq(np.array(rows), np.array(y_vals), rcond=None)
        map1 = {cat: round(val) for cat, val in zip(cats1, coeffs[: len(cats1)])}
        map2 = {
            cat: round(val)
            for cat, val in zip(cats2, coeffs[len(cats1) : len(cats1) + len(cats2)])
        }
        recon = s1.map(map1) + s2.map(map2)
        if recon.equals(coded_col):
            return [c1, c2], {"map1": map1, "map2": map2}
    return [], {}


def detect_addiction_score(
    original: pd.DataFrame, coded_col: pd.Series
) -> Tuple[List[str], str]:
    """
    Try to find subset of yes/no columns whose sum equals the coded addiction score.
    """
    bin_cols = [
        c
        for c in original.columns
        if original[c].dtype == object and set(original[c].dropna().unique()) == {"ใช่", "ไม่ใช่"}
    ]
    bin_df = pd.DataFrame({c: original[c].map({"ใช่": 1, "ไม่ใช่": 0}) for c in bin_cols})
    target = coded_col.values
    remaining_max = [len(bin_cols) - i for i in range(len(bin_cols) + 1)]

    @lru_cache(None)
    def search(idx: int, current_sum: Tuple[int, ...]) -> Tuple[bool, List[str]]:
        curr = list(current_sum)
        if idx == len(bin_cols):
            return (curr == target.tolist(), [])
        if any(curr[i] > target[i] for i in range(len(target))):
            return (False, [])
        if any(curr[i] + remaining_max[idx] < target[i] for i in range(len(target))):
            return (False, [])
        col = bin_cols[idx]
        add_if_yes = bin_df[col].values
        with_col = tuple(curr[i] + add_if_yes[i] for i in range(len(target)))
        ok, subset = search(idx + 1, with_col)
        if ok:
            return True, subset + [col]
        return search(idx + 1, current_sum)

    success, subset = search(0, tuple([0] * len(target)))
    return subset, "sum of yes/no (yes=1)"


def detect_financial_literacy(
    original: pd.DataFrame, coded_col: pd.Series
) -> Tuple[List[str], Dict]:
    """Identify the correct answer per FL question so the count matches coded_col."""
    question_indices = list(range(47, 56))  # FL1 - FL9 positions
    questions = [original.columns[i] for i in question_indices]
    answers = [original[q].fillna("") for q in questions]
    uniques = [ans.unique().tolist() for ans in answers]
    target = coded_col.values
    n = len(target)

    # Pre-encode answers for quick comparison
    encoded = []
    for ans, uni in zip(answers, uniques):
        mapping = {v: i for i, v in enumerate(uni)}
        encoded.append(ans.map(mapping).fillna(-1).astype(int).values)

    remaining = [len(questions) - i for i in range(len(questions) + 1)]

    @lru_cache(None)
    def backtrack(q_idx: int, current: Tuple[int, ...]) -> List[Dict[int, str]]:
        curr = list(current)
        if q_idx == len(questions):
            if curr == target.tolist():
                return [{}]
            return []
        if any(curr[i] > target[i] for i in range(n)):
            return []
        if any(curr[i] + remaining[q_idx] < target[i] for i in range(n)):
            return []
        results: List[Dict[int, str]] = []
        uni = uniques[q_idx]
        enc = encoded[q_idx]
        for idx_answer, answer_val in enumerate(uni):
            updated = tuple(curr[i] + (1 if enc[i] == idx_answer else 0) for i in range(n))
            for sol in backtrack(q_idx + 1, updated):
                sol = dict(sol)
                sol[q_idx] = answer_val
                results.append(sol)
        return results

    solutions = backtrack(0, tuple([0] * n))
    if not solutions:
        return [], {}
    solution = solutions[0]
    mapping = {questions[i]: {solution[i]: 1, "*other*": 0} for i in range(len(questions))}
    return questions, mapping


def translate_text(text: str) -> str:
    if not Translator:
        return text
    try:
        translator = Translator()
        translated = translator.translate(text, src="th", dest="en")
        return translated.text
    except Exception:
        return text


def build_mapping_and_apply(
    original: pd.DataFrame, coded: pd.DataFrame
) -> Tuple[pd.DataFrame, List[Dict]]:
    mapping_info: Dict[str, Dict] = detect_simple_mappings(original, coded)
    remaining = [c for c in coded.columns if c not in mapping_info]

    # special cases
    if "Est_Interval" in remaining:
        cols, note = detect_difference(original, coded["Est_Interval"])
        if cols:
            mapping_info["Est_Interval"] = {
                "type": note,
                "source_cols": cols,
                "mapping": {},
            }
            remaining.remove("Est_Interval")

    if "Gambling_Score" in remaining:
        cols, note = detect_mean_of_four(original, coded["Gambling_Score"])
        if cols:
            mapping_info["Gambling_Score"] = {
                "type": f"{note} of four gambling expense questions",
                "source_cols": cols,
                "mapping": {col: "ordinal 1-5" for col in cols},
            }
            remaining.remove("Gambling_Score")

    if "Stress_Level" in remaining:
        cols, mapping = detect_stress_level(original, coded["Stress_Level"])
        if cols:
            mapping_info["Stress_Level"] = {
                "type": "average of two stress questions",
                "source_cols": cols,
                "mapping": mapping,
            }
            remaining.remove("Stress_Level")

    if "Risk_Tolerance_Score" in remaining:
        cols, mapping = detect_risk_tolerance(original, coded["Risk_Tolerance_Score"])
        if cols:
            mapping_info["Risk_Tolerance_Score"] = {
                "type": "sum of two risk/acceptable loss codes",
                "source_cols": cols,
                "mapping": mapping,
            }
            remaining.remove("Risk_Tolerance_Score")

    if "Addiction_Score" in remaining:
        cols, note = detect_addiction_score(original, coded["Addiction_Score"])
        if cols:
            mapping_info["Addiction_Score"] = {
                "type": note,
                "source_cols": cols,
                "mapping": {"yes": 1, "no": 0},
            }
            remaining.remove("Addiction_Score")

    if "Financial_Literacy" in remaining:
        cols, mapping = detect_financial_literacy(original, coded["Financial_Literacy"])
        if cols:
            mapping_info["Financial_Literacy"] = {
                "type": "count of correct answers across nine questions",
                "source_cols": cols,
                "mapping": mapping,
            }
            remaining.remove("Financial_Literacy")

    # build encoded dataframe using discovered rules
    encoded_df = pd.DataFrame(index=original.index)
    for coded_col, info in mapping_info.items():
        if info["type"] == "category_map":
            src = info["source_cols"][0]
            encoded_df[coded_col] = original[src].map(info["mapping"])
        elif info["type"].startswith("difference"):
            c1, c2 = info["source_cols"]
            encoded_df[coded_col] = original[c2] - original[c1]
        elif info["type"].startswith("mean"):
            cols = info["source_cols"]
            encoded_df[coded_col] = original[cols].mean(axis=1)
        elif info["type"].startswith("average"):
            cat_col, num_col = info["source_cols"]
            mapped = original[cat_col].map(info["mapping"])
            encoded_df[coded_col] = (mapped + original[num_col]) / 2
        elif info["type"].startswith("sum of two"):
            c1, c2 = info["source_cols"]
            map1, map2 = info["mapping"]["map1"], info["mapping"]["map2"]
            encoded_df[coded_col] = original[c1].map(map1) + original[c2].map(map2)
        elif info["type"].startswith("sum of yes/no"):
            cols = info["source_cols"]
            bin_df = pd.DataFrame(
                {c: original[c].map({"ใช่": 1, "ไม่ใช่": 0}) for c in cols}
            )
            encoded_df[coded_col] = bin_df.sum(axis=1)
        elif info["type"].startswith("count of correct answers"):
            encoded_df[coded_col] = 0
            for q_col, q_map in info["mapping"].items():
                correct_answer = next(k for k in q_map if k != "*other*")
                encoded_df[coded_col] += (original[q_col] == correct_answer).astype(int)
        else:
            # fallback copy if nothing else matched
            src = info["source_cols"][0]
            encoded_df[coded_col] = original[src]

    # translate column names for readability
    translated_names = {col: translate_text(col) for col in original.columns}

    mapping_table = []
    for coded_col, info in mapping_info.items():
        orig_name = info["source_cols"]
        mapping_table.append(
            {
                "coded_column": coded_col,
                "original_column": " + ".join(orig_name),
                "original_column_en": " + ".join(
                    translated_names.get(col, col) for col in orig_name
                ),
                "coding_type": info["type"],
                "mapping": info["mapping"],
            }
        )

    return encoded_df, mapping_table


def main() -> None:
    orig, coded = load_data(EXCEL_FILE)
    encoded_df, mapping_table = build_mapping_and_apply(orig, coded)

    # Save to a new Excel with mapping sheet
    output_path = Path(EXCEL_FILE).with_name("AFC_auto_coded.xlsx")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        orig.to_excel(writer, sheet_name="Original Data (Thai)", index=False)
        coded.to_excel(writer, sheet_name="Coded Data FINAL for regression", index=False)
        encoded_df.to_excel(writer, sheet_name="Auto Encoded", index=False)
        pd.DataFrame(mapping_table).to_excel(writer, sheet_name="Mapping", index=False)

    # Print mapping summary to stdout
    print("Detected mappings:")
    for row in mapping_table:
        print(
            f"- {row['original_column']} -> {row['coded_column']} "
            f"({row['coding_type']}), mapping={row['mapping']}"
        )
    print(f"\nAuto-encoded sheet written to {output_path}")


if __name__ == "__main__":
    main()
