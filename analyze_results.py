
import pandas as pd
from pathlib import Path


# Project directory
BASE_DIR = Path(__file__).parent

# Input and output files
ANSWER_FILE = BASE_DIR / "dataset" / "answers.csv"
SUMMARY_FILE = BASE_DIR / "dataset" / "evaluation_summary.csv"
REPORT_FILE = BASE_DIR / "dataset" / "evaluation_report.txt"


def analyze_results():

    print("Loading chatbot evaluation results...")

    df = pd.read_csv(
        ANSWER_FILE,
        encoding="utf-8-sig"
    )

    df["answer"] = df["answer"].fillna("")

    total_questions = len(df)
    answered = (df["answer"].str.strip() != "").sum()
    unanswered = total_questions - answered

    # Calculate answer word counts
    df["word_count"] = df["answer"].apply(
        lambda answer: len(answer.split())
    )

    average_words = (
        round(df.loc[df["answer"].str.strip() != "", "word_count"].mean(), 2)
        if answered > 0 else 0
    )

    # Create summary
    summary = pd.DataFrame({
        "Metric": [
            "Total Questions",
            "Answered Questions",
            "Unanswered Questions",
            "Average Answer Word Count"
        ],
        "Value": [
            total_questions,
            answered,
            unanswered,
            average_words
        ]
    })

    # Save summary CSV
    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # Save text report
    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        file.write("CHATBOT EVALUATION REPORT\n")
        file.write("=" * 35 + "\n")
        file.write(summary.to_string(index=False))
        file.write("\n\nIndividual Answer Word Counts\n")
        file.write(df[["query_id", "word_count"]].to_string(index=False))

    print("\nEVALUATION ANALYSIS COMPLETED!")
    print("\n", summary.to_string(index=False))

    print("\nSummary saved to:", SUMMARY_FILE)
    print("Report saved to:", REPORT_FILE)


if __name__ == "__main__":
    analyze_results()
