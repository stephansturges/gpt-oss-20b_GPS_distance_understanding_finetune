# build_parquet_dataset.py
import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from gps_example_generator import generate_one_example

def to_harmony_row(messages):
    system_msg = next(m for m in messages if m.get("role") == "system")
    user_msg   = next(m for m in messages if m.get("role") == "user")
    analysis_msg = next((m for m in messages if m.get("role") == "assistant" and m.get("channel") == "analysis"), None)
    final_msg    = next(m for m in messages if m.get("role") == "assistant" and m.get("channel") == "final")

    developer = system_msg.get("content", "")
    user      = user_msg.get("content", "")
    analysis  = analysis_msg.get("content") if analysis_msg else None
    final     = final_msg.get("content", "")

    assistant_struct = {"role": "assistant", "content": final, "thinking": analysis}
    row_messages = [
        {"role": "system", "content": developer, "thinking": None},
        {"role": "user",   "content": user,      "thinking": None},
        assistant_struct,
    ]

    return {"developer": developer, "user": user, "analysis": analysis, "final": final, "messages": row_messages}

def build_table(rows):
    msg_struct = pa.struct([
        pa.field("role", pa.string()),
        pa.field("content", pa.string()),
        pa.field("thinking", pa.string(), nullable=True),
    ])
    schema = pa.schema([
        pa.field("developer", pa.string()),
        pa.field("user", pa.string()),
        pa.field("analysis", pa.string(), nullable=True),
        pa.field("final", pa.string()),
        pa.field("messages", pa.list_(msg_struct)),
    ])
    return pa.Table.from_pylist(rows, schema=schema)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--out", type=str, default="gps_samples.parquet")
    parser.add_argument("--no_preview", action="store_true")
    parser.add_argument("--distance_model", choices=["wgs84","sphere"], default="wgs84")
    args = parser.parse_args()

    rows = []
    for _ in range(args.n):
        msgs = generate_one_example(analysis_ratio=0.8, format_style="harmony", distance_model=args.distance_model)
        rows.append(to_harmony_row(msgs))

    table = build_table(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="zstd")

    if not args.no_preview:
        pd.set_option("display.max_colwidth", 120)
        df = pd.read_parquet(out_path, engine="pyarrow")
        print("\n=== Parquet written ===")
        print(f"File: {out_path.resolve()}\nRows: {len(df)}")
        print("\n--- Preview (developer / user / final) ---")
        print(df[["developer", "user", "final"]].head(min(10, len(df))).to_string(index=False))

        first_cell = df.iloc[0]["messages"]
        try:
            print("\n--- messages[0] (pretty) ---")
            print(json.dumps(first_cell, ensure_ascii=False, indent=2))
        except TypeError:
            import numpy as np
            import pyarrow as pa
            import pyarrow.parquet as pq
            if isinstance(first_cell, np.ndarray):
                first_cell = first_cell.tolist()
            elif isinstance(first_cell, pa.Scalar):
                first_cell = first_cell.as_py()
            elif isinstance(first_cell, (pa.Array, pa.ChunkedArray)):
                first_cell = first_cell.to_pylist()
            else:
                first_cell = pq.read_table(out_path, columns=["messages"]).column("messages").to_pylist()[0]
            print("\n--- messages[0] (pretty) ---")
            print(json.dumps(first_cell, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

