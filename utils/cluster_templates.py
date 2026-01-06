import argparse
from pathlib import Path
from typing import Dict, List
from collections import Counter
import numpy as np
import pandas as pd
import torch
import umap
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances
from transformers import AutoModel, AutoTokenizer

# pool token embeddings with attention mask to get emb of sentence
def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts

# encode templates
def embed_texts(
    texts: List[str],
    model_name: str,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            output = model(**encoded)
            pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
        all_embeddings.append(pooled.cpu().numpy())
    embeddings = np.vstack(all_embeddings)
    return normalize(embeddings, axis=1)

# to find optimal dbscan hyperparameters
def save_kdistance_plot(embeddings: np.ndarray, k: int, out_path: str) -> None:
    neighbors = NearestNeighbors(n_neighbors=k, metric="cosine")
    neighbors.fit(embeddings)
    distances, _ = neighbors.kneighbors(embeddings)
    kdist = np.sort(distances[:, -1])

    plt.figure(figsize=(8, 4))
    plt.plot(kdist)
    plt.title(f"k-distance plot (k={k})")
    plt.xlabel("Points sorted by distance")
    plt.ylabel("Distance to k-th neighbor")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"[info] Saved k-distance plot to {out_path}")

def report_distance_stats(embeddings: np.ndarray) -> None:
    dist = cosine_distances(embeddings)
    upper = dist[np.triu_indices_from(dist, k=1)]
    if upper.size == 0:
        print("[info] Not enough points to compute pairwise distances.")
        return
    stats = {
        "min": float(np.min(upper)),
        "p10": float(np.percentile(upper, 10)),
        "p25": float(np.percentile(upper, 25)),
        "median": float(np.median(upper)),
        "p75": float(np.percentile(upper, 75)),
        "p90": float(np.percentile(upper, 90)),
        "max": float(np.max(upper)),
    }
    print("[info] Cosine distance stats:", stats)

    
def dbscan_param_search(
    embeddings: np.ndarray,
    eps_list: List[float],
    min_samples_list: List[int],
) -> Dict:
    """
    Перебор eps/min_samples.
    Выбираем вариант с минимумом шума и без "слипания" в гигантский кластер.
    """
    n = len(embeddings)
    best = None
    best_score = None

    for ms in min_samples_list:
        for eps in eps_list:
            labels = DBSCAN(eps=eps, min_samples=ms, metric="cosine").fit_predict(embeddings)

            cnt = Counter(labels)
            noise = cnt.get(-1, 0)
            cluster_sizes = [v for k, v in cnt.items() if k != -1]

            n_clusters = len(cluster_sizes)
            biggest = max(cluster_sizes) if cluster_sizes else 0

            noise_ratio = noise / n if n else 1.0
            biggest_ratio = biggest / n if n else 1.0

            # score меньше = лучше
            # штрафуем случай когда кластеров нет вообще
            penalty = 1.0 if n_clusters == 0 else 0.0
            score = noise_ratio + 0.5 * biggest_ratio + penalty

            print(
                f"[sweep] eps={eps:.3f} ms={ms:2d} | "
                f"clusters={n_clusters:3d} | "
                f"noise={noise_ratio:.1%} | "
                f"biggest={biggest_ratio:.1%} | "
                f"score={score:.4f}"
            )

            if best_score is None or score < best_score:
                best_score = score
                best = {
                    "eps": eps,
                    "min_samples": ms,
                    "labels": labels,
                    "score": score,
                    "noise_ratio": noise_ratio,
                    "biggest_ratio": biggest_ratio,
                    "n_clusters": n_clusters,
                }

    return best


# transform cluster_ids to strings
def build_cluster_ids(labels: np.ndarray) -> List[str]:
    return ["NOISE" if label == -1 else f"CLUSTER_{label}" for label in labels]

# append cluster id column to templates
def add_cluster_ids(temp_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    temp_df = temp_df.copy()
    temp_df["ClusterId"] = build_cluster_ids(labels)
    return temp_df


def map_structured_logs(
    struct_df: pd.DataFrame,
    template_to_cluster: Dict[str, str],
) -> pd.DataFrame:
    struct_df = struct_df.copy()
    struct_df["ClusterId"] = struct_df["EventTemplate"].map(template_to_cluster)
    struct_df["ClusterId"] = struct_df["ClusterId"].fillna("UNMATCHED")
    return struct_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster Drain templates with BERT embeddings.")
    parser.add_argument("--temp_path", default="../data/mobservice2_temp_my.csv", help="Path to templates CSV.")
    parser.add_argument("--stru_path", default="../data/mobservice2_stru_my.csv", help="Path to structured logs CSV.")
    parser.add_argument("--temp_out", default="../data/mobservice2_temp_my_with_clusters.csv", help="Output path for templates CSV (default: overwrite).")
    parser.add_argument("--stru_out", default="../data/mobservice2_stru_my_with_clusters.csv", help="Output path for structured logs CSV (default: overwrite).")
    parser.add_argument("--model", default="bert-base-uncased", help="HuggingFace model name.")
    parser.add_argument("--batch_size", type=int, default=32, help="Embedding batch size.")
    parser.add_argument("--eps", type=float, default=0.02, help="DBSCAN eps (cosine distance).")
    parser.add_argument("--min_samples", type=int, default=2, help="DBSCAN min_samples.")
    parser.add_argument("--plot_2d", action="store_true", help="Save 2D UMAP plot of embeddings.")
    parser.add_argument("--plot_path", default="cluster_umap.png", help="Path to save the plot.")
    parser.add_argument("--kdist_plot", action="store_true", help="Save k-distance plot for eps selection.")
    parser.add_argument("--kdist_k", type=int, default=5, help="k for k-distance plot.")
    parser.add_argument("--kdist_path", default="kdistance.png", help="Path to save k-distance plot.")
    parser.add_argument("--dist_stats", action="store_true", help="Print cosine distance stats.")
    parser.add_argument("--auto_dbscan", action="store_true",
                        help="Auto-pick best eps/min_samples by sweeping.")
    parser.add_argument("--eps_list", default="0.02,0.025,0.03,0.04,0.05,0.06,0.08,0.10",
                        help="Comma-separated eps values for sweep.")
    parser.add_argument("--min_samples_list", default="2,3,5,10",
                        help="Comma-separated min_samples values for sweep.")

    
    args = parser.parse_args()

    temp_path = Path(args.temp_path)
    stru_path = Path(args.stru_path)
    temp_out = Path(args.temp_out) if args.temp_out else temp_path
    stru_out = Path(args.stru_out) if args.stru_out else stru_path

    temp_df = pd.read_csv(temp_path)
    if "EventTemplate" not in temp_df.columns:
        raise RuntimeError("EventTemplate column not found in templates CSV.")

    templates = temp_df["EventTemplate"].astype(str).tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Started computing embeddings')
    embeddings = embed_texts(templates, args.model, args.batch_size, device)
    print('Finished computing embeddings')

    if args.dist_stats:
        report_distance_stats(embeddings)

    if args.kdist_plot:
        save_kdistance_plot(embeddings, args.kdist_k, args.kdist_path)

    # ---- AUTO DBSCAN PARAM SEARCH ----
    if args.auto_dbscan:
        eps_list = [float(x.strip()) for x in args.eps_list.split(",") if x.strip()]
        ms_list = [int(x.strip()) for x in args.min_samples_list.split(",") if x.strip()]

        best = dbscan_param_search(embeddings, eps_list, ms_list)
        if best is None:
            raise RuntimeError("Auto DBSCAN failed to find any configuration.")

        print(
            f"[best] eps={best['eps']:.3f}, min_samples={best['min_samples']} | "
            f"clusters={best['n_clusters']} | "
            f"noise={best['noise_ratio']:.1%} | biggest={best['biggest_ratio']:.1%}"
        )

        args.eps = best["eps"]
        args.min_samples = best["min_samples"]

    
        
    print('Started clustering')
    clustering = DBSCAN(eps=args.eps, min_samples=args.min_samples, metric="cosine")
    print('Finished clustering')
    labels = clustering.fit_predict(embeddings)

    temp_df = add_cluster_ids(temp_df, labels)
    temp_df.to_csv(temp_out, index=False)

    if stru_path.exists():
        struct_df = pd.read_csv(stru_path)
        template_to_cluster = dict(zip(temp_df["EventTemplate"], temp_df["ClusterId"]))
        struct_df = map_structured_logs(struct_df, template_to_cluster)
        struct_df.to_csv(stru_out, index=False)

    reducer = umap.UMAP(n_components=2, metric="cosine", random_state=42)
    points = reducer.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        plt.scatter(points[mask, 0], points[mask, 1], s=8, alpha=0.7, label=str(label))
    plt.legend(markerscale=2, fontsize=8, loc="best")
    plt.title("UMAP of Template Embeddings")
    plt.tight_layout()
    plt.savefig(args.plot_path, dpi=200)
    print(f"[info] Saved plot to {args.plot_path}")

    print(f"[info] Wrote templates with ClusterId to {temp_out}")
    if stru_path.exists():
        print(f"[info] Wrote structured logs with ClusterId to {stru_out}")


if __name__ == "__main__":
    main()
