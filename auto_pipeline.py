import argparse, json, os, random, shutil, logging, base64
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from tqdm import tqdm

from config import MODELS_CONFIG, get_model_class, get_data_paths
from main import collect_mode, test_mode

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ---------------- LLM -----------------
CFG = MODELS_CONFIG["gemini-2.5-pro"]
LLM = get_model_class(CFG["class"])(model=CFG["model_name"], temperature=0)

def img_to_b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


def ask_gemini(img_b64: str, log_json: str) -> bool:
    """Send distance curve image + log excerpt to Gemini, expect YES/NO."""
    prompt_text = f"""
# ROLE
You are an AI Quality Assurance Analyst. Your task is to evaluate the performance of a geolocation agent on a specific task.

# GOAL
Your primary goal is to identify high-quality training examples. A high-quality example is one where the agent solves a *difficult* problem through *meaningful, intelligent interaction*. You are NOT looking for cases where the agent got lucky or solved an easy problem.

# INPUTS
You will be provided with two pieces of evidence:
1.  **`distance.png`**: An image showing the *outcome*. It plots the error distance (Y-axis, km) vs. the interaction step (X-axis).
2.  **`log_excerpt.json`**: A JSON text snippet showing the *process*. It contains the agent's internal thoughts, observations, and actions for each step. Here is the log excerpt:
    ```json
    {log_json}
    ```

# ANALYSIS & CRITERIA
You must synthesize information from BOTH the image and the log to make your decision. A sample should be ACCEPTED (reply 'YES') only if it meets ALL of the following criteria:

1.  **Verifiable High Difficulty:**
    * **(Image Check):** The distance curve in the PNG must show at least one of the first 3 steps with an error > 100 km.
    * **(Log Check):** The initial steps in the `log` should reflect this difficulty (e.g., expressing uncertainty, defining a large initial search area).

2.  **Interaction-Driven Success (Most Important):**
    * The agent's reasoning and actions described in the `log` must plausibly **explain** the significant drops in error distance shown in the `png`.
    * **Look for this:** Does the agent analyze an image, read a sign, or change its strategy in the log, immediately preceding a large drop in the error curve? This shows cause-and-effect.
    * **Reject this:** The error distance drops significantly, but the log shows the agent was still confused, guessing, or its actions don't logically lead to that improvement. This indicates a lucky guess, not skillful interaction.

3.  **Clear Resolution:**
    * **(Image Check):** The final step in the PNG must show a low error distance, indicating that the model was clearly closer to or solve the task.

# FINAL INSTRUCTION
Reply with a SINGLE WORD.
-   Reply 'YES' only if you can confirm a difficult start, a more accurate end, AND find clear evidence in the log that the agent's interaction drove the reduction in error.
-   Otherwise, reply 'NO'.
"""

    message = [
        {"type": "text", "text": prompt_text},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
    ]

    resp = LLM.invoke([{"role": "user", "content": message}])
    return resp.content.strip().upper().startswith("Y")


# --------------- helpers ---------------

def draw_distance_curve(dist: List[float], save_to: Path):
    save_to.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 3))
    plt.plot(range(1, len(dist) + 1), dist, marker="o")
    plt.xlabel("step"); plt.ylabel("km"); plt.grid(alpha=0.3)
    plt.title(save_to.parent.name[:8])
    plt.savefig(save_to, dpi=120, bbox_inches="tight")
    plt.close()


# -------------- pipeline ---------------

def pipeline(dataset: str, target: int, batch: int, steps: int, runs: int, headless: bool):
    ds_filtered_dir = Path(f"datasets/{dataset}_filtered")
    ds_filtered_dir.mkdir(parents=True, exist_ok=True)

    # accum kept samples (id -> sample dict)
    kept: Dict[str, dict] = {}
    with open(ds_filtered_dir / "golden_labels.json", "r") as f:
        golden_labels = json.load(f)
    for sample in golden_labels["samples"]:
        kept[sample["id"]] = sample
    logging.info("cumulative kept %d / %d", len(kept), target)
    golden_path = Path(get_data_paths(dataset)["golden_labels"])

    result_root = Path("results")
    combined_log_path = result_root / f"{dataset}_filtered/combined_log.json"

    while len(kept) < target:
        logging.info("Collecting %d unique samples", batch)
        # read current dataset samples to find uniqueness
        existing_ids = set()
        if golden_path.exists():
            existing_ids = {s["id"] for s in json.load(open(golden_path)) ["samples"]}
        existing_ids |= kept.keys()

        new_ids = set()
        while len(new_ids) < batch:
            collect_mode(dataset_name=dataset, samples=batch-len(new_ids), headless=headless)
            data = json.load(open(golden_path))["samples"]
            new_ids = {s["id"] for s in data if s["id"] not in existing_ids}
        # ensure new_ids order
        new_batch = [s for s in data if s["id"] in new_ids]

        # move new samples to front (for test_mode convenience)
        reordered = new_batch + [s for s in data if s["id"] not in new_ids]
        json.dump({"samples": reordered}, open(golden_path, "w"), indent=2, ensure_ascii=False)

        # test only new batch
        test_mode(models=["gemini-2.5-pro"], samples=batch, runs=runs, steps=steps,
                  dataset_name=dataset, temperature=1.0, headless=headless)

        # 读取本轮测试日志
        latest_res = sorted((result_root / dataset).iterdir())[-1]
        log_file   = latest_res / "gemini-2.5-pro" / "gemini-2.5-pro_log.json"
        log_data   = json.load(open(log_file))

        # 为每个 sample 生成 distance.png（存在原 log_picture 目录）
        for sid, runs_list in log_data.items():
            dists     = [p["distance"] for p in runs_list[0]["predictions"]]
            dest_dir  = latest_res / "gemini-2.5-pro" / "log_picture" / sid
            draw_distance_curve(dists, dest_dir / "distance.png")

        # ---------- 第二阶段：Gemini 评估 ----------
        logging.info("Gemini vision filtering …")
        kept_logs: Dict[str, list] = {}      # 本轮通过的日志
        for sid in tqdm(new_ids, desc="vision-filter", unit="sample"):
            dist_png = latest_res / "gemini-2.5-pro" / "log_picture" / sid / "distance.png"
            if not dist_png.exists():
                continue
            max_d = max(p["distance"] for p in log_data[sid][0]["predictions"])
            if max_d <= 100:
                continue
            ok = ask_gemini(img_to_b64(dist_png), json.dumps(log_data[sid], ensure_ascii=False))
            if ok:
                sample_dict    = next(s for s in reordered if s["id"] == sid)
                kept[sid]      = sample_dict
                kept_logs[sid] = log_data[sid]

        # ---------- 仅追加通过样本到 combined_log ----------
        combined = json.load(open(combined_log_path)) if combined_log_path.exists() else {}
        combined.update(kept_logs)
        json.dump(combined, open(combined_log_path, "w"), indent=2, ensure_ascii=False)

        # 实时保存已筛选数据集
        json.dump({"samples": list(kept.values())},
                  open(ds_filtered_dir / "golden_labels.json", "w"),
                  indent=2, ensure_ascii=False)

        logging.info("cumulative kept %d / %d", len(kept), target)

    # save final dataset
    json.dump({"samples": list(kept.values())}, open(ds_filtered_dir / "golden_labels.json", "w"), indent=2, ensure_ascii=False)
    json.dump({"samples": list(kept)}, open(ds_filtered_dir / "second_stage.json", "w"), indent=2, ensure_ascii=False)
    logging.info("✅ Done, kept %d samples", len(kept))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    pipeline(dataset=args.dataset, target=args.target, batch=args.batch, steps=args.steps, runs=args.runs, headless=args.headless)
