
import pandas as pd
from pathlib import Path

from chatbot import generate_answer


# Project directory
BASE_DIR = Path(__file__).parent

# Dataset paths
QUERY_FILE = BASE_DIR / "dataset" / "queries.csv"
ANSWER_FILE = BASE_DIR / "dataset" / "answers.csv"


def evaluate_chatbot():

    print("Loading evaluation questions...")

    queries_df = pd.read_csv(
        QUERY_FILE,
        encoding="utf-8-sig"
    )

    results = []

    print("Total questions:", len(queries_df))

    for index, row in queries_df.iterrows():

        query_id = row["query_id"]
        question = row["question"]

        print("\n" + "=" * 50)
        print(f"Question {index + 1}/{len(queries_df)}")
        print("Question:", question)

        try:
            answer = generate_answer(question)

        except Exception as error:
            answer = f"Error: {error}"

        print("\nAI Answer:")
        print(answer)

        results.append({
            "query_id": query_id,
            "question": question,
            "answer": answer
        })

        # Save progress after each question
        pd.DataFrame(results).to_csv(
            ANSWER_FILE,
            index=False,
            encoding="utf-8-sig"
        )

    print("\n" + "=" * 50)
    print("CHATBOT EVALUATION COMPLETED!")
    print("Total questions processed:", len(results))
    print("Answers saved to:", ANSWER_FILE)


if __name__ == "__main__":
    evaluate_chatbot()
