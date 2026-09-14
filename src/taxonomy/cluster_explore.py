"""Embed root complaints, cluster with HDBSCAN, project with UMAP for taxonomy discovery."""
import hashlib
import re
from pathlib import Path

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
ROOTS_PATH = DATA_DIR / "uber_roots.csv"
CLUSTERS_OUT = DATA_DIR / "uber_roots_clustered.csv"
PLOT_OUT = Path(__file__).resolve().parents[2] / "report" / "cluster_plot.png"
EMBED_CACHE_DIR = DATA_DIR / "embed_cache"

EMBED_MODEL = "all-MiniLM-L6-v2"

# clustering happens in a higher-dim UMAP space (better separation than raw 384D
# or the 2D space, which is for visualization only)
CLUSTER_N_COMPONENTS = 10
CLUSTER_MIN_DIST = 0.0

# final chosen params (update these after running the sweep once)
FINAL_MIN_CLUSTER_SIZE = 200
FINAL_MIN_SAMPLES = 5
FINAL_CLUSTER_SELECTION_EPSILON = 0.2

# set True to run a param sweep with DBCV instead of a single final clustering
SWEEP = False

# DBCV on the full 41k-point set is memory-heavy; skip unless RAM is available
COMPUTE_DBCV = False

# any top-level cluster with more than this many points gets re-clustered on its own
SUBCLUSTER_SIZE_THRESHOLD = 5000
SUBCLUSTER_MIN_CLUSTER_SIZE = 80
SUBCLUSTER_MIN_SAMPLES = 5
SUBCLUSTER_EPSILON = 0.0

HANDLE_RE = re.compile(r"@\w+")
URL_RE = re.compile(r"https?://\S+")


def clean_text(text: str) -> str:
    text = HANDLE_RE.sub("", text)
    text = URL_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def load_roots(path: Path = ROOTS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["clean_text"] = df["text"].astype(str).map(clean_text)
    return df


def embed_texts(texts: list, model_name: str = EMBED_MODEL) -> np.ndarray:
    EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(("\n".join(texts) + model_name).encode("utf-8")).hexdigest()
    cache_path = EMBED_CACHE_DIR / f"{key}.npy"
    if cache_path.exists():
        print(f"using cached embeddings: {cache_path.name}")
        return np.load(cache_path)

    model = SentenceTransformer(model_name)
    embeddings = np.asarray(model.encode(texts, show_progress_bar=True))
    np.save(cache_path, embeddings)
    return embeddings


def reduce_dims(
    embeddings,
    n_components: int = 2,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=n_components, min_dist=min_dist, random_state=random_state
    )
    return np.asarray(reducer.fit_transform(embeddings))


def cluster_embeddings(
    embeddings,
    min_cluster_size: int = 150,
    min_samples: int | None = 10,
    cluster_selection_method: str = "leaf",
    cluster_selection_epsilon: float = 0.0,
):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )
    labels = clusterer.fit_predict(embeddings)
    return labels, clusterer

def sweep_params(
    embeddings,
    mcs_candidates: list[int],
    ms_candidates: list[int],
    epsilon_candidates: list[float],
) -> None:
    """Fast initial sweep: use cluster count and noise percentage only.
    Skip DBCV here because it is too memory-intensive for 40k+ points.
    Select the best candidate, then calculate DBCV once for the final combination."""
    for mcs in mcs_candidates:
        for ms in ms_candidates:
            for eps in epsilon_candidates:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=mcs,
                    min_samples=ms,
                    cluster_selection_method="leaf",
                    cluster_selection_epsilon=eps,
                )
                labels = clusterer.fit_predict(embeddings)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = int((labels == -1).sum())
                noise_pct = n_noise / len(labels) * 100
                print(
                    f"mcs={mcs:<4} ms={ms:<3} eps={eps:<4} -> "
                    f"clusters={n_clusters:<3} noise={noise_pct:5.1f}%"
                )

def subcluster_oversized(
    cluster_space: np.ndarray,
    labels: np.ndarray,
    size_threshold: int = SUBCLUSTER_SIZE_THRESHOLD,
    min_cluster_size: int = SUBCLUSTER_MIN_CLUSTER_SIZE,
    min_samples: int = SUBCLUSTER_MIN_SAMPLES,
    epsilon: float = SUBCLUSTER_EPSILON,
) -> np.ndarray:
    """Re-cluster any cluster larger than the size threshold using only its own
    points. Assign new sub-clusters unique ids after the current maximum label.
    Keep any noise points found during re-clustering as -1."""
    new_labels = labels.copy()
    next_label = int(labels.max()) + 1

    counts = pd.Series(labels).value_counts()
    oversized = [c for c in counts.index if c != -1 and counts[c] > size_threshold]

    for parent in oversized:
        mask = labels == parent
        print(f"sub-clustering cluster {parent} (n={mask.sum()})...")
        sub_labels, _ = cluster_embeddings(
            cluster_space[mask],
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=epsilon,
        )
        n_sub = len(set(sub_labels)) - (1 if -1 in sub_labels else 0)
        print(f"  -> {n_sub} sub-clusters found")

        remapped = np.where(
            sub_labels == -1, -1, sub_labels + next_label
        )
        new_labels[mask] = remapped
        next_label += n_sub

    return new_labels


def plot_clusters(coords_2d, labels, out_path: Path = PLOT_OUT) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        coords_2d[:, 0], coords_2d[:, 1], c=labels, cmap="Spectral", s=5, alpha=0.7
    )
    plt.colorbar(scatter, label="cluster")
    plt.title("Uber support root complaints - UMAP + HDBSCAN clusters")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cluster_grid(
    coords_2d, labels, df, top_n: int = 15, out_path: Path | None = None
):
    if out_path is None:
        out_path = PLOT_OUT.parent / "cluster_grid.png"

    top_clusters = (
        df["cluster"].value_counts().drop(-1, errors="ignore").head(top_n).index
    )

    n_cols = 5
    n_rows = (len(top_clusters) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, c in enumerate(top_clusters):
        ax = axes[i]
        ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c="lightgray", s=3, alpha=0.3)
        mask = labels == c
        ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1], c="crimson", s=6)
        ax.set_title(f"cluster {c} (n={mask.sum()})", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(len(top_clusters), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")


def sample_noise(df, n: int = 30, seed: int = 7):
    noise = df[df["cluster"] == -1]["clean_text"]
    sample = noise.sample(min(n, len(noise)), random_state=seed)
    print(f"\n--- Noise sample ({len(noise)} total noise points) ---")
    for i, t in enumerate(sample, 1):
        print(f"{i}. {t}")


def main() -> None:
    df = load_roots()
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    print(f"loaded {len(df)} root complaints")

    embeddings = embed_texts(df["clean_text"].tolist())
    embeddings = normalize(embeddings)
    print(f"embedded: {embeddings.shape}")

    # separate reductions: high-dim + tight packing for clustering, 2D for viz
    cluster_space = reduce_dims(
        embeddings, n_components=CLUSTER_N_COMPONENTS, min_dist=CLUSTER_MIN_DIST
    )
    coords_2d = reduce_dims(embeddings, n_components=2, min_dist=CLUSTER_MIN_DIST)

    if SWEEP:
        sweep_params(
            cluster_space,
            mcs_candidates=[100, 150, 200],
            ms_candidates=[5, 10],
            epsilon_candidates=[0.0, 0.2],
        )
        return

    labels, clusterer = cluster_embeddings(
        cluster_space,
        min_cluster_size=FINAL_MIN_CLUSTER_SIZE,
        min_samples=FINAL_MIN_SAMPLES,
        cluster_selection_epsilon=FINAL_CLUSTER_SELECTION_EPSILON,
    )

    labels = subcluster_oversized(cluster_space, labels)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(
        f"clusters found: {n_clusters} (+{n_noise} noise points, "
        f"{n_noise / len(labels):.1%})"
    )

    if COMPUTE_DBCV:
        try:
            dbcv = hdbscan.validity.validity_index(cluster_space.astype(np.float64), labels)
            print(f"DBCV score: {dbcv:.4f}")
        except Exception as e:
            print(f"DBCV computation skipped: {e}")
    else:
        print("DBCV computation skipped (COMPUTE_DBCV=False)")

    print(pd.Series(labels).value_counts().head(20))

    df["cluster"] = labels

    print("\n Top clusters, sample texts ")
    top_clusters = df["cluster"].value_counts().drop(-1, errors="ignore").head(15).index
    for c in top_clusters:
        sample = df[df["cluster"] == c]["clean_text"].sample(
            min(8, (df["cluster"] == c).sum()), random_state=1
        )
        print(f"\n-- cluster {c} (n={(df['cluster']==c).sum()}) --")
        for t in sample:
            print(f"  - {t}")

    df["umap_x"] = coords_2d[:, 0]
    df["umap_y"] = coords_2d[:, 1]
    df.to_csv(CLUSTERS_OUT, index=False)
    print(f"wrote {CLUSTERS_OUT}")

    plot_clusters(coords_2d, labels)
    plot_cluster_grid(coords_2d, labels, df)
    print(f"wrote {PLOT_OUT}")

    sample_noise(df)

if __name__ == "__main__":
    main()