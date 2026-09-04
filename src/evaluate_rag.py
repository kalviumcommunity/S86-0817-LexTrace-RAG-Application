import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator import diagnose_and_recommend, score_answer, summarize_failures
from src.test_sets import CONCEPT_TEST_SET, LEXTRACE_TEST_SET


def run_evaluation(test_set_name: str = "lextrace"):
    print("\n=======================================================")
    print("      LexTrace RAG Answer Quality & Citation Evaluator  ")
    print("=======================================================")

    if test_set_name.lower() == "concept":
        selected_set = CONCEPT_TEST_SET
        print(f"Running evaluation on: CONCEPT TEST SET ({len(selected_set)} items)")
    else:
        selected_set = LEXTRACE_TEST_SET
        print(f"Running evaluation on: LEXTRACE TEST SET ({len(selected_set)} items)")

    print("-------------------------------------------------------")

    rows = []
    for i, example in enumerate(selected_set, 1):
        print(f"\n[{i}/{len(selected_set)}] Evaluating: '{example['question']}'")
        row = score_answer(example)
        rows.append(row)

        print(f"  • Answer             : {row['answer']}")
        print(f"  • Citations          : {row['citations']} (Expected: {row['expected_sources']})")
        print(f"  • Correctness Score  : {row['correctness']:.2f}")
        print(f"  • Grounding Score    : {row['grounding']:.2f}")
        print(f"  • Citation Accuracy  : {row['citation_accuracy']:.2f}")

        if row.get("diagnoses"):
            print("  • Diagnoses          :")
            for d in row["diagnoses"]:
                print(f"    - {d}")

    # Generate summary & failure breakdown
    summary = summarize_failures(rows)
    report = diagnose_and_recommend(summary)
    print(report)

    return summary


if __name__ == "__main__":
    test_choice = "lextrace"
    if len(sys.argv) > 1:
        test_choice = sys.argv[1]

    run_evaluation(test_choice)
