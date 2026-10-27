"""
Interactive human labelling tool for Cohen's kappa validation.

Usage:
    python scripts/label_human.py --experiment e1
    python scripts/label_human.py --experiment e2
    python scripts/label_human.py --experiment e3

Controls:
    1  →  hallucination (not supported by gold answers)
    0  →  supported (correct or acceptable)
    s  →  skip (genuinely ambiguous, will be excluded from kappa)
    q  →  quit and save progress (resumable)

Output: results/<experiment>/human_labels.csv
"""

import argparse
import json
import os
import random
import csv
from pathlib import Path

# ── ANSI colours ──────────────────────────────────────────────────────────────
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def load_existing(csv_path: Path) -> dict:
    """Load already-labelled sample ids so we can resume."""
    if not csv_path.exists():
        return {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return {row["sample_id"]: row["human_label"] for row in reader}

def save_label(csv_path: Path, sample_id: str, human_label: str,
               question: str, generated: str, judge_label: int):
    exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sample_id", "human_label", "judge_label",
            "question", "generated_text"])
        if not exists:
            writer.writeheader()
        writer.writerow({
            "sample_id":      sample_id,
            "human_label":    human_label,   # "0", "1", or "skip"
            "judge_label":    judge_label,
            "question":       question,
            "generated_text": generated,
        })

def display_sample(i: int, total: int, sample: dict, judge_label: int):
    clear()
    print(f"{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  Human labelling — sample {i}/{total}{RESET}")
    print(f"{BOLD}{'='*70}{RESET}\n")

    print(f"{CYAN}{BOLD}QUESTION:{RESET}")
    print(f"  {sample['question']}\n")

    print(f"{CYAN}{BOLD}GOLD ANSWERS:{RESET}")
    for ans in sample["gold_answers"][:5]:
        print(f"  • {ans}")
    if len(sample["gold_answers"]) > 5:
        print(f"  … ({len(sample['gold_answers'])} total)")
    print()

    print(f"{CYAN}{BOLD}MODEL RESPONSE:{RESET}")
    print(f"  {sample['generated_text']}\n")

    judge_colour = RED if judge_label == 1 else GREEN
    judge_text   = "HALLUCINATION" if judge_label == 1 else "SUPPORTED"
    print(f"  Judge (GPT-4o-mini): {judge_colour}{BOLD}{judge_text}{RESET}\n")

    print(f"{BOLD}{'─'*70}{RESET}")
    print(f"  {GREEN}0{RESET} = supported   "
          f"{RED}1{RESET} = hallucination   "
          f"{YELLOW}s{RESET} = skip   "
          f"{YELLOW}q{RESET} = quit & save")
    print(f"{BOLD}{'─'*70}{RESET}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, choices=["e1", "e2", "e3"])
    parser.add_argument("--n", type=int, default=60,
                        help="Number of samples to label (default 60)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo_root   = Path(__file__).parent.parent
    texts_file  = repo_root / "results" / args.experiment / "generated_texts.jsonl"
    labels_file = repo_root / "results" / args.experiment / "labels.json"
    out_csv     = repo_root / "results" / args.experiment / "human_labels.csv"

    if not texts_file.exists():
        print(f"ERROR: {texts_file} not found. Run the extraction step first.")
        return

    samples      = [json.loads(l) for l in texts_file.read_text().strip().split("\n")]
    judge_labels = json.loads(labels_file.read_text()) if labels_file.exists() else {}

    # Fixed stratified sample: n/2 hallucinations + n/2 supported, seed-fixed
    hall = [s for s in samples if judge_labels.get(s["sample_id"], -1) == 1]
    supp = [s for s in samples if judge_labels.get(s["sample_id"], -1) == 0]
    rng  = random.Random(args.seed)
    rng.shuffle(hall)
    rng.shuffle(supp)

    n_each = args.n // 2
    subset = hall[:n_each] + supp[:n_each]
    rng.shuffle(subset)

    done    = load_existing(out_csv)
    pending = [s for s in subset if s["sample_id"] not in done]

    print(f"\nExperiment {args.experiment.upper()}: "
          f"{len(done)} already labelled, {len(pending)} remaining.\n")

    if not pending:
        print("All samples labelled. Run the kappa check in notebook 01.")
        return

    labelled_this_session = 0
    for i, sample in enumerate(pending, start=len(done) + 1):
        j_label = judge_labels.get(sample["sample_id"], -1)
        display_sample(i, args.n, sample, j_label)

        while True:
            raw = input("\n  Your label: ").strip().lower()
            if raw in ("0", "1", "s", "q"):
                break
            print(f"  {RED}Invalid input. Press 0, 1, s, or q.{RESET}")

        if raw == "q":
            print(f"\n{YELLOW}Saved progress. Re-run to continue.{RESET}")
            break

        human_label = "skip" if raw == "s" else raw
        save_label(out_csv, sample["sample_id"], human_label,
                   sample["question"], sample["generated_text"], j_label)
        labelled_this_session += 1

        if raw != "s":
            match = (int(raw) == j_label)
            indicator = f"{GREEN}✓ agrees{RESET}" if match else f"{RED}✗ disagrees{RESET}"
            print(f"  → {indicator} with judge")

    total_done = len(load_existing(out_csv))
    print(f"\n{BOLD}Session done.{RESET} "
          f"Labelled this session: {labelled_this_session}. "
          f"Total: {total_done}/{args.n}.")

    if total_done >= args.n:
        print(f"\n{GREEN}{BOLD}All {args.n} samples labelled for {args.experiment}.{RESET}")
        print("Run the kappa check: open notebook 01_labeling_openai.ipynb "
              "and execute the last cell.")

if __name__ == "__main__":
    main()
