#!/usr/bin/env python
"""E1c — zero-shot router baseline (ADAPTIVE_SELECTION.md).

A frozen LLM (Gemini Flash, text-only v0) reads each question and picks which
view expert should answer: panorama / top-down / point-matching / no-think.
Routed accuracy = the chosen expert's official per-question correctness, taken
from the router_task_*.json files produced by e1_plurality_vote.py.

Choices are restricted to the experts whose results exist for that benchmark
(§2.1 coverage gaps). Decisions are cached to a JSONL so reruns resume.
"""

import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor

E0DIR = "/path/to/scratch/VisualCoT/eval_results/adaptive_selection_e0"

DESC = {
    "pano": ("panorama: stitch the two input views into a single 360-degree panorama seen "
             "from the observer's position. Best when the question needs the two views fused "
             "into one egocentric scene, e.g. relative direction or distance between objects "
             "that appear in different views."),
    "td": ("top_down: render an overhead room-overview of the scene. Best when the question "
           "needs an allocentric layout: many input views, camera motion, map-like reasoning, "
           "or positions of several objects in the room."),
    "pm": ("point_matching: mark the same physical object with the same colored dot in both "
           "input views. Best when the question hinges on cross-view correspondence, e.g. "
           "which object is visible in both views, or matching a queried object across views."),
    "no_think": ("no_think: answer directly with no intermediate image. Best when the question "
                 "is simple, single-view, or none of the transformations above fits."),
}
ROUTE_TOKENS = {"panorama": "pano", "pano": "pano", "top_down": "td", "topdown": "td",
                "top-down": "td", "point_matching": "pm", "point-matching": "pm",
                "pointmatching": "pm", "no_think": "no_think", "no-think": "no_think",
                "nothink": "no_think"}


def build_prompt(question, allowed):
    strategies = "\n".join(f"- {DESC[r]}" for r in allowed)
    names = ", ".join(sorted({k for k, v in ROUTE_TOKENS.items() if v in allowed} &
                             {"panorama", "top_down", "point_matching", "no_think"}))
    return (
        "You route spatial-reasoning questions to the best visual-thinking strategy.\n"
        "The answering model sees camera views of a scene and may first generate one "
        "intermediate 'thinking image' before answering.\n\n"
        f"Available strategies:\n{strategies}\n\n"
        f"Question to route:\n{question}\n\n"
        f"Reply with exactly one word among: {names}."
    )


def parse_route(text, allowed):
    if not text:
        return None
    t = text.strip().lower()
    for token, route in ROUTE_TOKENS.items():
        if route in allowed and re.search(rf"\b{re.escape(token)}\b", t):
            return route
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3-flash-preview")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--benchmarks", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="per-benchmark cap for smoke tests")
    args = ap.parse_args()

    from google import genai
    client = genai.Client(api_key=os.environ["VisualCoT_GEMINI"])
    lock = threading.Lock()

    benches = args.benchmarks or sorted(
        f.split("router_task_")[1][:-5] for f in os.listdir(E0DIR) if f.startswith("router_task_"))

    summary = []
    for bench in benches:
        with open(f"{E0DIR}/router_task_{bench}.json") as f:
            rows = json.load(f)
        if args.limit:
            rows = rows[:args.limit]
        allowed = sorted(rows[0]["correct_by_expert"])
        cache_fp = f"{E0DIR}/router_decisions_{bench}.jsonl"
        cache = {}
        if os.path.exists(cache_fp):
            with open(cache_fp) as f:
                for line in f:
                    r = json.loads(line)
                    cache[r["sample_id"]] = r["route"]
        todo = [r for r in rows if r["sample_id"] not in cache]

        def route_one(row):
            prompt = build_prompt(str(row["question"])[:4000], allowed)
            last_err = None
            for _ in range(4):
                try:
                    resp = client.models.generate_content(model=args.model, contents=prompt)
                    route = parse_route(resp.text, allowed)
                    if route:
                        with lock, open(cache_fp, "a") as f:
                            f.write(json.dumps({"sample_id": row["sample_id"], "route": route}) + "\n")
                        return row["sample_id"], route
                    last_err = f"unparseable: {resp.text[:60]!r}"
                except Exception as e:
                    last_err = str(e)[:120]
            return row["sample_id"], f"FAIL {last_err}"

        if todo:
            with ThreadPoolExecutor(args.workers) as ex:
                for sid, route in ex.map(route_one, todo):
                    if not str(route).startswith("FAIL"):
                        cache[sid] = route

        # score: routed accuracy; unrouted samples fall back to the best-single expert
        best_single = max(allowed, key=lambda e: sum(r["correct_by_expert"][e] for r in rows))
        routed, fallback, dist = 0, 0, {}
        for r in rows:
            route = cache.get(r["sample_id"])
            if route not in allowed:
                route = best_single
                fallback += 1
            dist[route] = dist.get(route, 0) + 1
            routed += bool(r["correct_by_expert"][route])
        n = len(rows)
        summary.append({
            "bench": bench, "n": n, "allowed": allowed,
            "best_single_expert": best_single,
            "best_single": 100.0 * sum(r["correct_by_expert"][best_single] for r in rows) / n,
            "routed": 100.0 * routed / n,
            "oracle": 100.0 * sum(any(r["correct_by_expert"].values()) for r in rows) / n,
            "fallbacks": fallback, "route_distribution": dist,
        })
        print(f"{bench}: routed {summary[-1]['routed']:.2f} vs best-single "
              f"{summary[-1]['best_single']:.2f} ({best_single}), oracle {summary[-1]['oracle']:.2f}, "
              f"dist {dist}, fallbacks {fallback}")

    with open(f"{E0DIR}/e1_router_report.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nreport: {E0DIR}/e1_router_report.json")


if __name__ == "__main__":
    main()
