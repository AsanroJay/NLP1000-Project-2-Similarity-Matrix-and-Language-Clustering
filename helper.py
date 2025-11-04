import os
import re
import unicodedata
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import seaborn as sns


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


def load_ngp_file(file_path):
    """Loads an NGP file and returns a dictionary of ngram → frequency."""
    profile = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                ngram, freq = parts
                try:
                    profile[ngram] = float(freq)
                except ValueError:
                    continue
    return profile


def build_ngp_vectors(ngp_dir):
    """
    Loads all NGP files in a directory and returns:
      - langs: list of language names
      - vectors: list of frequency vectors (aligned by ngram vocabulary)
    """
    profiles = {}
    for filename in os.listdir(ngp_dir):
        if filename.endswith(".ngp"):
            lang = os.path.splitext(filename)[0]
            profiles[lang] = load_ngp_file(os.path.join(ngp_dir, filename))

    # Create shared n-gram vocabulary
    all_ngrams = sorted(set(ngram for profile in profiles.values() for ngram in profile))

    vectors = []
    for lang in profiles:
        vector = [profiles[lang].get(ngram, 0.0) for ngram in all_ngrams]
        vectors.append(vector)

    langs = list(profiles.keys())
    return langs, vectors

def compute_cosine_similarity_table(ngp_dir):
    """
    Computes and returns a DataFrame of cosine similarities between languages.
    """
    langs, vectors = build_ngp_vectors(ngp_dir)
    sim_matrix = cosine_similarity(vectors)
    df_sim = pd.DataFrame(sim_matrix, index=langs, columns=langs)
    return df_sim

def draw_heat_map(df_sim, title="Cosine Similarity Heatmap"):
    """
    Draws a heatmap for the cosine similarity DataFrame.
    Green = high similarity, Orange = medium, Red = low.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        df_sim,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",  # Red → Yellow → Green 
        linecolor="gray",
        cbar_kws={'label': 'Cosine Similarity'}
    )
    plt.title(title, fontsize=14, pad=12)
    plt.xlabel("Languages")
    plt.ylabel("Languages")
    plt.tight_layout()
    plt.show()


def plot_language_dendrogram(similarity_df, method="average", save_path=None):
    """
    Performs hierarchical clustering using cosine similarity and plots a dendrogram.

    Parameters:
        similarity_df (pd.DataFrame): Cosine similarity matrix (languages × languages)
        method (str): Linkage method ('average', 'ward', 'complete', etc.)
        save_path (str, optional): File path to save dendrogram image (e.g. 'output/dendrogram.png')
    """
    # Convert similarity to distance
    distance_matrix = 1 - similarity_df.values
    condensed_distance = squareform(distance_matrix, checks=False)

    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_distance, method=method)

    # Plot dendrogram
    plt.figure(figsize=(10, 6))
    dendrogram(linkage_matrix, labels=similarity_df.index, leaf_rotation=45, leaf_font_size=10)
    plt.title(f"Hierarchical Clustering of Languages ({method.title()} Linkage, Cosine Distance)")
    plt.xlabel("Languages")
    plt.ylabel("Distance")
    plt.tight_layout()

    # Save or display
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Dendrogram saved to {save_path}")
    else:
        plt.show()
