import os
import re
import unicodedata
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


def clean_text(text):
    """Remove non-letter characters and normalize."""
    text = unicodedata.normalize('NFD', text)
    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def clean_corpora(root_dir, output_dir):
    """
    Cleans all .txt files in the MCO1 Corpora folder and saves cleaned versions.
    Keeps the same folder structure (Language/Book/Files).
    """
    for lang in os.listdir(root_dir):
        lang_path = os.path.join(root_dir, lang)
        if not os.path.isdir(lang_path):
            continue

        for book in os.listdir(lang_path):
            book_path = os.path.join(lang_path, book)
            if not os.path.isdir(book_path):
                continue

            for file_name in os.listdir(book_path):
                if not file_name.endswith(".txt"):
                    continue

                file_path = os.path.join(book_path, file_name)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                cleaned = clean_text(content)

                output_path = os.path.join(output_dir, lang, book)
                os.makedirs(output_path, exist_ok=True)

                with open(os.path.join(output_path, file_name), "w", encoding="utf-8") as f:
                    f.write(cleaned)

    print(f"Cleaned files saved in: {output_dir}")
    

def compile_per_language(cleaned_dir, output_dir="MCO1_Compiled"):
    """
    Compiles all cleaned .txt files per language into a single file.
    Example:
        Bikol.txt contains all cleaned books from MCO1_Cleaned/Bikol/
    """
    os.makedirs(output_dir, exist_ok=True)

    for lang in os.listdir(cleaned_dir):
        lang_path = os.path.join(cleaned_dir, lang)
        if not os.path.isdir(lang_path):
            continue

        compiled_texts = []

        for book in os.listdir(lang_path):
            book_path = os.path.join(lang_path, book)
            if not os.path.isdir(book_path):
                continue

            for file_name in os.listdir(book_path):
                if file_name.endswith(".txt"):
                    file_path = os.path.join(book_path, file_name)
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                        if text:
                            compiled_texts.append(text)

        # Join all texts for this language
        combined = "\n".join(compiled_texts)

        # Save as one file per language
        output_file = os.path.join(output_dir, f"{lang}.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(combined)

        print(f"Compiled {lang} → {output_file} ({len(compiled_texts)} files merged)")

    print(f"\nAll languages compiled to: {output_dir}")
