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


if __name__ == "__main__":

    data_folder = Path("data/sample")

    for file_path in data_folder.iterdir():

        if not file_path.is_file():
            continue

        try:
            raw_text = file_path.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            cleaned_text = clean_text(raw_text)

            print(f"\n--- {file_path.name} ---")
            print("Before:", len(raw_text), "characters")
            print("After :", len(cleaned_text), "characters")

            print("\nBEFORE:")
            print(raw_text[:200])

            print("\nAFTER:")
            print(cleaned_text[:200])

        except Exception as e:
            print(f"SKIP {file_path.name}: {e}")