import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


INPUT_PATH = Path("data/business_table_mobservice2_2021-07.csv")
TEMPLATE_PATH = Path("data_old/mobservice2_temp.csv")
STRUCT_OUT_PATH = Path("data/mobservice2_stru.csv")
TEMP_OUT_PATH = Path("data/mobservice2_temp.csv")
CHUNK_SIZE = 200_000
# Limit processing to a date window (inclusive). Set to None to disable.
START_DATE = "2021-07-01"
END_DATE = "2021-07-15"


def _clean_template(template: str) -> str:
    """Normalize template strings by trimming quotes and whitespace."""
    template = str(template).strip()
    if template.startswith('"'):
        template = template[1:]
    if template.endswith('"'):
        template = template[:-1]
    return template


def load_templates(path: Path) -> List[Tuple[str, str, re.Pattern]]:
    """Load templates and compile regex patterns for matching."""
    template_df = pd.read_csv(path)
    compiled = []
    for _, row in template_df.iterrows():
        template = row.get("EventTemplate")
        event_id = row.get("EventId")
        if pd.isna(template) or pd.isna(event_id):
            continue
        cleaned = _clean_template(template)
        parts = cleaned.split("<*>")
        regex = "^"
        for idx, part in enumerate(parts):
            regex += re.escape(part)
            if idx < len(parts) - 1:
                regex += "(.*?)"
        regex += "$"
        compiled.append((event_id, cleaned, re.compile(regex)))
    return compiled


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


def match_event(message: str, compiled: List[Tuple[str, str, re.Pattern]]) -> Tuple[str, str, List[str]]:
    """Return (EventId, EventTemplate, params) for the first template that matches the message."""
    for event_id, template, pattern in compiled:
        match = pattern.match(message)
        if match:
            return event_id, template, list(match.groups())
    return "UNMATCHED", "UNMATCHED", []


def process_logs(
    input_path: Path,
    template_path: Path,
    struct_out: Path,
    temp_out: Path,
    chunk_size: int = CHUNK_SIZE,
    start_date: Optional[str] = START_DATE,
    end_date: Optional[str] = END_DATE,
) -> None:
    compiled_templates = load_templates(template_path)
    if not compiled_templates:
        raise RuntimeError(f"No templates loaded from {template_path}")

    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date() if start_date else None
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else None

    # Counters for template occurrences
    occurrences: Dict[str, int] = {event_id: 0 for event_id, _, _ in compiled_templates}
    occurrences["UNMATCHED"] = 0

    # Prepare output file
    if struct_out.exists():
        struct_out.unlink()

    line_id = 0
    write_header = True

    for chunk_idx, chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_size)):
        rows = []
        for _, row in chunk.iterrows():
            message = row.get("message")
            if pd.isna(message):
                occurrences["UNMATCHED"] += 1
                continue
            message = str(message)
            service = row["service"]
            dt_obj, epoch = parse_timestamp(message)
            if start_dt and dt_obj.date() < start_dt:
                continue
            if end_dt and dt_obj.date() > end_dt:
                continue
            event_id, template, params = match_event(message, compiled_templates)
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
    for idx, (event_id, template, _) in enumerate(compiled_templates):
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
    process_logs(INPUT_PATH, TEMPLATE_PATH, STRUCT_OUT_PATH, TEMP_OUT_PATH)
