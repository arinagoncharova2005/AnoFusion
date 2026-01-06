from datetime import datetime, timezone
from hashlib import md5
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import pandas as pd
import sys
sys.path.append('../')
import config


INPUT_PATH = Path("../data/mobservice2_stru.csv")
STRUCT_OUT_PATH = Path("../data/mobservice2_stru_my.csv")
TEMP_OUT_PATH = Path("../data/mobservice2_temp_my.csv")
CHUNK_SIZE = 200_000
# Training window for Drain template extraction.
# If TRAIN_START_DATE/TRAIN_END_DATE are None, derive them from aligned data + TRAIN_PROPORTION.
TRAIN_START_DATE = None
TRAIN_END_DATE = None
TRAIN_PROPORTION = 0.6
ALIGN_SERVICE = "mobservice2"
# Output window for structured logs. Set to None to disable.
START_DATE = "2021-07-01"
END_DATE = "2021-07-15"

PARAM_STR = "<*>"


class LogCluster:
    def __init__(self, template_tokens: List[str]) -> None:
        self.template_tokens = template_tokens
        self.size = 1

    @property
    def template(self) -> str:
        return " ".join(self.template_tokens)

    @property
    def event_id(self) -> str:
        return md5(self.template.encode("utf-8")).hexdigest()[:8]


class DrainNode:
    def __init__(self, depth: int) -> None:
        self.depth = depth
        self.children: Dict[str, "DrainNode"] = {}
        self.clusters: List[LogCluster] = []


class DrainParser:
    def __init__(self, depth: int = 4, max_children: int = 100, sim_th: float = 0.4) -> None:
        self.depth = depth
        self.max_children = max_children
        self.sim_th = sim_th
        self.root = DrainNode(0)

    @staticmethod
    def has_numbers(token: str) -> bool:
        return any(char.isdigit() for char in token)

    def tree_search(self, tokens: List[str]) -> Optional[LogCluster]:
        # Traverse the Drain prefix tree to find a matching cluster.
        seq_len = len(tokens)
        if seq_len not in self.root.children:
            return None
        cur = self.root.children[seq_len]
        max_depth = min(self.depth, seq_len)
        for i in range(max_depth):
            token = tokens[i]
            if token in cur.children:
                cur = cur.children[token]
                continue
            if PARAM_STR in cur.children:
                cur = cur.children[PARAM_STR]
                continue
            return None
        return self.fast_match(cur.clusters, tokens)

    @staticmethod
    def seq_similarity(template_tokens: List[str], tokens: List[str]) -> Tuple[float, int]:
        assert len(template_tokens) == len(tokens)
        sim = 0
        num_params = 0
        for templ_token, token in zip(template_tokens, tokens):
            if templ_token == PARAM_STR:
                num_params += 1
                continue
            if templ_token == token:
                sim += 1
        return sim / len(template_tokens), num_params

    def fast_match(self, clusters: List[LogCluster], tokens: List[str]) -> Optional[LogCluster]:
        # Select the closest cluster by similarity and wildcard count.
        max_sim = -1.0
        max_params = -1
        best_cluster = None
        for cluster in clusters:
            sim, num_params = self.seq_similarity(cluster.template_tokens, tokens)
            if sim > max_sim or (sim == max_sim and num_params > max_params):
                max_sim = sim
                max_params = num_params
                best_cluster = cluster
        if best_cluster and max_sim >= self.sim_th:
            return best_cluster
        return None

    def add_cluster_to_tree(self, cluster: LogCluster) -> None:
        # Insert a new cluster by length and prefix tokens.
        seq_len = len(cluster.template_tokens)
        if seq_len not in self.root.children:
            self.root.children[seq_len] = DrainNode(1)
        cur = self.root.children[seq_len]
        max_depth = min(self.depth, seq_len)
        for i in range(max_depth):
            token = cluster.template_tokens[i]
            if self.has_numbers(token):
                token = PARAM_STR
            if token not in cur.children:
                if len(cur.children) >= self.max_children:
                    token = PARAM_STR
                    if token not in cur.children:
                        cur.children[token] = DrainNode(cur.depth + 1)
                else:
                    cur.children[token] = DrainNode(cur.depth + 1)
            cur = cur.children[token]
        cur.clusters.append(cluster)

    def update_template(self, cluster: LogCluster, tokens: List[str]) -> None:
        # Generalize mismatching tokens into the wildcard.
        new_tokens = []
        for templ_token, token in zip(cluster.template_tokens, tokens):
            if templ_token == token:
                new_tokens.append(templ_token)
            else:
                new_tokens.append(PARAM_STR)
        cluster.template_tokens = new_tokens
        cluster.size += 1

    def parse(self, tokens: List[str], create_new: bool = True) -> Optional[LogCluster]:
        # With create_new=False, only match existing templates.
        cluster = self.tree_search(tokens)
        if cluster is None:
            if not create_new:
                return None
            cluster = LogCluster(tokens)
            self.add_cluster_to_tree(cluster)
            return cluster
        if create_new:
            self.update_template(cluster, tokens)
        return cluster


def parse_timestamp(message: str) -> Tuple[datetime, int]:
    """
    Extract timestamp string from log message and convert to epoch seconds (UTC).

    The GAIA logs start with `YYYY-MM-DD HH:MM:SS,mmm`, so we grab the prefix
    before the first pipe.
    """
    time_part = message.split(" | ", 1)[0]
    dt_obj = datetime.strptime(time_part, "%Y-%m-%d %H:%M:%S,%f")
    epoch = int(dt_obj.replace(tzinfo=timezone.utc).timestamp())
    return dt_obj, epoch

# http(s), ipv4, numbers to <*>
def normalize_message(message: str) -> str:
    
    msg = str(message)
    msg = re.sub(r"https?://\S+", PARAM_STR, msg)
    msg = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", PARAM_STR, msg)
    msg = re.sub(r"\d+", PARAM_STR, msg)
    # объединаем повторяющиеся PARAM_STR и PARAM_STR:PARAM_STR
    msg = re.sub(r"(?:<\*>\s+)+<\*>", PARAM_STR, msg)
    msg = re.sub(r"(?:<\*>[-/:])+<\*>", PARAM_STR, msg)
    return msg

def get_message(row: pd.Series) -> Optional[str]:
    if "message" in row:
        return row.get("message")
    return row.get("Content")

def get_service(row: pd.Series) -> Optional[str]:
    if "service" in row:
        return row.get("service")
    return row.get("Service")

# tokenize message by whitespaces
def tokenize_message(message: str) -> List[str]:
    msg = normalize_message(message)
    msg = msg.replace("|", " | ").replace("->", " -> ").replace(";", " :")
    return msg.split()


def extract_parameters(template_tokens: List[str], tokens: List[str]) -> List[str]:
    # Extract parameter values that align with wildcards.
    params = []
    for templ_token, token in zip(template_tokens, tokens):
        if templ_token == PARAM_STR:
            params.append(token)
    return params


def train_drain(
    input_path: Path,
    chunk_size: int,
    start_date: Optional[str],
    end_date: Optional[str],
) -> DrainParser:
    # Train Drain on the specified date window.
    parser = DrainParser()
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date() if start_date else None
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else None
    
    total = 0
    valid_date = 0
    skipped_na = 0
    parsed = 0

    for chunk in pd.read_csv(input_path, chunksize=chunk_size):
        for _, row in chunk.iterrows():
            total += 1
            # message = row.get("message")
            message = get_message(row)
            if pd.isna(message):
                skipped_na += 1
                continue
            dt_obj, _ = parse_timestamp(str(message))
            if start_dt and dt_obj.date() < start_dt:
                skipped_na += 1
                continue
            if end_dt and dt_obj.date() > end_dt:
                continue
            valid_date += 1
            parser.parse(tokenize_message(message), create_new=True)
            parsed += 1
    print(f"Parsed {parsed} of {total} messages ({skipped_na} skipped due to NaN, {valid_date} valid date)")
    return parser


def _aligned_train_dates(service_s: str, proportion: float) -> Tuple[str, str]:
    metric = pd.read_csv(f"../update_linear_interpolation_data/metrics/{service_s}.csv")
    log = pd.read_csv(f"../update_linear_interpolation_data/logs/{service_s}.csv")
    trace = pd.read_csv(f"../update_linear_interpolation_data/trace/{service_s}.csv")

    metric = metric[(metric["timestamp"] >= config.start_time[service_s]) & (metric["timestamp"] <= config.end_time[service_s])]
    log = log[(log["timestamp"] >= config.start_time[service_s]) & (log["timestamp"] <= config.end_time[service_s])]
    trace = trace[(trace["timestamp"] >= config.start_time[service_s]) & (trace["timestamp"] <= config.end_time[service_s])]

    metric = metric.drop_duplicates(["timestamp"])
    log = log.drop_duplicates(["timestamp"])
    trace = trace.drop_duplicates(["timestamp"])

    all_timestamps = sorted(set(metric["timestamp"]) | set(log["timestamp"]) | set(trace["timestamp"]))
    if not all_timestamps:
        raise RuntimeError("No aligned timestamps available to derive training dates.")

    n = len(all_timestamps)
    cut = int(proportion * n)
    if cut <= 0:
        raise RuntimeError("TRAIN_PROPORTION yields empty training window.")

    start_ts = int(all_timestamps[0])
    end_ts = int(all_timestamps[cut - 1])
    start_date = datetime.fromtimestamp(start_ts, tz=timezone.utc).date().isoformat()
    end_date = datetime.fromtimestamp(end_ts, tz=timezone.utc).date().isoformat()
    return start_date, end_date


def process_logs(
    input_path: Path,
    struct_out: Path,
    temp_out: Path,
    chunk_size: int = CHUNK_SIZE,
    train_start_date: Optional[str] = TRAIN_START_DATE,
    train_end_date: Optional[str] = TRAIN_END_DATE,
    start_date: Optional[str] = START_DATE,
    end_date: Optional[str] = END_DATE,
) -> None:
    if train_start_date is None or train_end_date is None:
        train_start_date, train_end_date = _aligned_train_dates(ALIGN_SERVICE, TRAIN_PROPORTION)
        print(f"[info] Derived train window: {train_start_date} .. {train_end_date}")
    # Learn templates only from training logs.
    print("[info] Training Drain templates...")
    parser = train_drain(input_path, chunk_size, train_start_date, train_end_date)
    if not parser.root.children:
        raise RuntimeError("Drain did not learn any templates from training logs.")

    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date() if start_date else None
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else None

    # Counters for template occurrences
    occurrences: Dict[str, int] = {}
    occurrences["UNMATCHED"] = 0

    # Prepare output file
    if struct_out.exists():
        struct_out.unlink()

    line_id = 0
    write_header = True

    for chunk_idx, chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_size)):
        rows = []
        for _, row in chunk.iterrows():
            # message = row.get("message")
            message = get_message(row)
            if pd.isna(message):
                occurrences["UNMATCHED"] += 1
                continue
            message = str(message)
            # service = row["service"]
            service = get_service(row) or "UNKNOWN"
            dt_obj, epoch = parse_timestamp(message)
            if start_dt and dt_obj.date() < start_dt:
                continue
            if end_dt and dt_obj.date() > end_dt:
                continue
            tokens = tokenize_message(message)
            # Match test logs to trained templates without updating them.
            cluster = parser.parse(tokens, create_new=False)
            if cluster is None:
                event_id = "UNMATCHED"
                template = "UNMATCHED"
                params = []
            else:
                event_id = cluster.event_id
                template = cluster.template
                params = extract_parameters(cluster.template_tokens, tokens)
            occurrences[event_id] = occurrences.get(event_id, 0) + 1

            rows.append(
                {
                    "Unnamed: 0": line_id,
                    "LineId": line_id,
                    "Datetime": dt_obj.date().isoformat(),
                    "Service": service,
                    "Content": message,
                    "EventId": event_id,
                    "EventTemplate": template,
                    "ParameterList": params,
                    "timestamp": epoch,
                }
            )
            line_id += 1

        out_df = pd.DataFrame(rows)
        out_df.to_csv(struct_out, mode="a", index=False, header=write_header)
        write_header = False
        print(f"[chunk {chunk_idx}] processed {len(rows)} rows (total {line_id})")

    # Build template occurrences table
    temp_rows = []
    all_clusters = []
    # Flatten tree to a unique template list.
    for depth_node in parser.root.children.values():
        stack = [depth_node]
        while stack:
            node = stack.pop()
            all_clusters.extend(node.clusters)
            stack.extend(node.children.values())
    seen = set()
    deduped_clusters = []
    for cluster in all_clusters:
        if cluster.template not in seen:
            seen.add(cluster.template)
            deduped_clusters.append(cluster)
    for idx, cluster in enumerate(deduped_clusters):
        event_id = cluster.event_id
        template = cluster.template
        temp_rows.append(
            {
                "Unnamed: 0": idx,
                "EventId": event_id,
                "EventTemplate": template,
                "Occurrences": occurrences.get(event_id, 0),
            }
        )
    temp_df = pd.DataFrame(temp_rows)
    temp_df.to_csv(temp_out, index=False)
    print(f"Wrote structured logs to {struct_out} and templates to {temp_out}")


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH)
    start_ts = int(pd.Timestamp('2021-07-01').timestamp())
    end_ts = int(pd.Timestamp('2021-07-09').timestamp())

    train_df = df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)]

    print(f"Количество записей в обучающем окне: {len(train_df)}")
    process_logs(INPUT_PATH, STRUCT_OUT_PATH, TEMP_OUT_PATH)
