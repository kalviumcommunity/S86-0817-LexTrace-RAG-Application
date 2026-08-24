from pathlib import Path
import re
import unicodedata



def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def load_clean_documents(data_folder="data/sample"):
    """Load files and return their cleaned text with source metadata."""

    documents = []

    for file_path in Path(data_folder).iterdir():

        if not file_path.is_file():
            continue

        try:
            raw_text = file_path.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            cleaned_text = clean_text(raw_text)

            if not cleaned_text:
                continue

            documents.append({
                "source": file_path.name,
                "text": cleaned_text
            })

        except Exception as e:
            print(f"SKIP {file_path.name}: {e}")

    return documents


if __name__ == "__main__":

    documents = load_clean_documents()

    for document in documents:
        print(f"\n--- {document['source']} ---")
        print("Cleaned characters:", len(document["text"]))
        print("Text:")
        print(document["text"][:200])