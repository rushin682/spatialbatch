"""Build ``results/report.html`` from the files the pipeline wrote to ``results/``.

Everything the page states is read from those files (and from ``slides.py``, and git for the commit), so running the
pipeline again and then this script gives the page again. Each table and chart names the file it comes from.

Usage: ``make_report.py`` (``HESCAPE_SUFFIX=_v2`` is the collections with the origin ordering; the first version is the
one without a suffix).
"""

from __future__ import annotations

import base64
import datetime
import html
import json
import math
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from slides import (
    BLOCK_PX,
    EXCLUDED,
    HE_ALIGNED,
    IMAGE_CHUNK,
    IMAGE_SHARD,
    OUT_ROOT,
    TABLE,
    TILE_PX,
    TRAIN_COHORT,
    VAL_SLIDES,
)

RESULTS = OUT_ROOT / "results"
REPO = Path(__file__).resolve().parents[2]
V1, V2 = "", "_v2"  # collections ordered by tile center (first version) and by tile origin
SPLITS = ("train", "val", "test")

# Chart geometry: one viewBox, scaled. The palette is validated by the dataviz validator (3 categorical slots, light and
# dark, all pairs); the series colors are the CSS tokens --s1..--s3.
WIDTH, HEIGHT = 640, 300
LEFT, RIGHT, TOP, BOTTOM = 56, 16, 28, 44
BAR_MAX_PX, BAR_GAP_PX, BAR_RADIUS_PX = 24, 2, 4


def read_json(name: str):
    """A result file, or ``None`` if the pipeline has not written it."""
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def read_jsonl(name: str) -> list[dict]:
    """One row per line of a result file, empty if the pipeline has not written it."""
    path = RESULTS / name
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


def git(*args: str) -> str:
    """The output of a git command in the repository."""
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False).stdout.strip()


def esc(value: object) -> str:
    """``value`` as text that is safe in HTML, in an attribute too."""
    return html.escape(str(value), quote=True)


def compact(value: float) -> str:
    """``2236`` as ``2.2k``: the value labels at the top of a column are 24 px wide at most."""
    return f"{value / 1000:.1f}k" if value >= 1000 else f"{value:.0f}"


def nice_ceiling(value: float, n_ticks: int = 4) -> tuple[float, float]:
    """A round axis maximum and the step between ticks, for a axis that starts at 0."""
    raw = value / n_ticks
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    return math.ceil(value / step) * step, step


def table(head: list[str], rows: list[list[object]], numeric_from: int = 1, caption: str = "") -> str:
    """An HTML table; the columns from ``numeric_from`` on are numbers, right aligned and tabular."""
    header = "".join(f"<th{' class=num' if i >= numeric_from else ''}>{esc(h)}</th>" for i, h in enumerate(head))
    body = "".join(
        "<tr>"
        + "".join(f"<td{' class=num' if i >= numeric_from else ''}>{esc(c)}</td>" for i, c in enumerate(row))
        + "</tr>"
        for row in rows
    )
    cap = f"<caption>{esc(caption)}</caption>" if caption else ""
    return f'<div class="scroll"><table>{cap}<thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>'


def grouped_columns(
    categories: list[str], series: list[tuple[str, list[float | None]]], y_label: str, fmt=compact
) -> str:
    """Columns grouped by category, one per series, from a baseline at 0 (``None`` draws nothing).

    A column is at most 24 px thick with a 4 px rounded top and a 2 px gap to its neighbor, its value is printed at the
    cap, and hovering or focusing it shows the series, the category and the value.
    """
    top = max(v for _, values in series for v in values if v is not None)
    ceiling, step = nice_ceiling(top * 1.12)
    plot_w, plot_h = WIDTH - LEFT - RIGHT, HEIGHT - TOP - BOTTOM
    y = lambda v: TOP + plot_h * (1 - v / ceiling)
    parts = [f'<svg viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="{esc(y_label)}">']
    tick = 0.0
    while tick <= ceiling + 1e-9:
        parts.append(f'<line class="grid" x1="{LEFT}" x2="{WIDTH - RIGHT}" y1="{y(tick):.1f}" y2="{y(tick):.1f}"/>')
        parts.append(f'<text class="axis" x="{LEFT - 8}" y="{y(tick) + 4:.1f}" text-anchor="end">{fmt(tick)}</text>')
        tick += step
    parts.append(f'<text class="axis" x="{LEFT}" y="{TOP - 12}">{esc(y_label)}</text>')
    group_w = plot_w / len(categories)
    n = len(series)
    bar_w = min(BAR_MAX_PX, (group_w * 0.8 - BAR_GAP_PX * (n - 1)) / n)
    for c, category in enumerate(categories):
        left = LEFT + c * group_w + (group_w - (n * bar_w + (n - 1) * BAR_GAP_PX)) / 2
        for s, (name, values) in enumerate(series):
            value = values[c]
            if value is None:
                continue
            x0, x1, base, cap = left + s * (bar_w + BAR_GAP_PX), left + s * (bar_w + BAR_GAP_PX) + bar_w, y(0), y(value)
            r = min(BAR_RADIUS_PX, (base - cap) / 2)
            path = f"M{x0:.1f},{base:.1f}V{cap + r:.1f}Q{x0:.1f},{cap:.1f} {x0 + r:.1f},{cap:.1f}H{x1 - r:.1f}Q{x1:.1f},{cap:.1f} {x1:.1f},{cap + r:.1f}V{base:.1f}Z"
            tip = f"{name}, {category}: {value:,.0f}"
            parts.append(f'<path class="mark s{s + 1}" d="{path}" tabindex="0" data-tip="{esc(tip)}"/>')
            parts.append(
                f'<text class="value" x="{(x0 + x1) / 2:.1f}" y="{cap - 5:.1f}" text-anchor="middle">{fmt(value)}</text>'
            )
        parts.append(
            f'<text class="axis" x="{LEFT + (c + 0.5) * group_w:.1f}" y="{HEIGHT - BOTTOM + 20}" text-anchor="middle">{esc(category)}</text>'
        )
    parts.append(f'<line class="base" x1="{LEFT}" x2="{WIDTH - RIGHT}" y1="{y(0):.1f}" y2="{y(0):.1f}"/></svg>')
    return legend([name for name, _ in series]) + f'<div class="scroll"><div class="chart">{"".join(parts)}</div></div>'


def lines(series: list[tuple[str, list[float]]], x_label: str, y_label: str) -> str:
    """Lines over the same x (a step), 2 px thick with an 8 px end dot; a crosshair column per step shows the values."""
    n_points = len(series[0][1])
    top = max(max(values) for _, values in series)
    ceiling, step = nice_ceiling(top * 1.05)
    plot_w, plot_h = WIDTH - LEFT - RIGHT, HEIGHT - TOP - BOTTOM
    x = lambda i: LEFT + plot_w * i / (n_points - 1)
    y = lambda v: TOP + plot_h * (1 - v / ceiling)
    parts = [f'<svg viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="{esc(y_label)} by {esc(x_label)}">']
    tick = 0.0
    while tick <= ceiling + 1e-9:
        parts.append(f'<line class="grid" x1="{LEFT}" x2="{WIDTH - RIGHT}" y1="{y(tick):.1f}" y2="{y(tick):.1f}"/>')
        parts.append(f'<text class="axis" x="{LEFT - 8}" y="{y(tick) + 4:.1f}" text-anchor="end">{tick:g}</text>')
        tick += step
    for i in range(0, n_points, max(1, n_points // 6)):
        parts.append(f'<text class="axis" x="{x(i):.1f}" y="{HEIGHT - BOTTOM + 20}" text-anchor="middle">{i}</text>')
    parts.append(f'<text class="axis" x="{LEFT}" y="{TOP - 12}">{esc(y_label)}</text>')
    parts.append(f'<text class="axis" x="{WIDTH - RIGHT}" y="{HEIGHT - 6}" text-anchor="end">{esc(x_label)}</text>')
    for s, (_, values) in enumerate(series):
        points = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values))
        parts.append(f'<polyline class="line s{s + 1}" points="{points}"/>')
        parts.append(f'<circle class="dot s{s + 1}" cx="{x(n_points - 1):.1f}" cy="{y(values[-1]):.1f}" r="4"/>')
    cell = plot_w / (n_points - 1)
    for i in range(n_points):
        tip = f"step {i}: " + ", ".join(f"{name} {values[i]:.2f}" for name, values in series)
        parts.append(
            f'<rect class="hit" x="{x(i) - cell / 2:.1f}" y="{TOP}" width="{cell:.1f}" height="{plot_h}" tabindex="0" data-tip="{esc(tip)}"/>'
        )
    parts.append("</svg>")
    return legend([name for name, _ in series]) + f'<div class="scroll"><div class="chart">{"".join(parts)}</div></div>'


def legend(names: list[str]) -> str:
    """The legend of a chart: one swatch per series, none for a single series (its title names it)."""
    items = "".join(f'<span><i class="sw s{i + 1}"></i>{esc(name)}</span>' for i, name in enumerate(names))
    return f'<div class="legend">{items}</div>' if len(names) > 1 else ""


def rolling(values: list[float], window: int = 10) -> list[float]:
    """The mean of the last ``window`` values at each position, to read a noisy curve."""
    return [
        sum(values[max(0, i - window + 1) : i + 1]) / len(values[max(0, i - window + 1) : i + 1])
        for i in range(len(values))
    ]


def section(identifier: str, title: str, *content: str, source: str = "") -> str:
    """A titled section of the page, ending with the files its numbers come from."""
    origin = f'<p class="source">Source: {source}</p>' if source else ""
    return f'<section id="{identifier}"><h2>{esc(title)}</h2>{"".join(content)}{origin}</section>'


def summary_section() -> str:
    """What the measurements show, in a few sentences read off the result files."""
    items = []
    simulated = read_json("simulate_order_train.json") or []
    cost = {(r["order_by"], r["chunk_size"]): r["read_kb_per_spot"] for r in simulated if r["block_px"] == BLOCK_PX}
    if ("center", 32) in cost:
        items.append(
            f"Ordering the rows by tile origin instead of tile center cuts the pixels read per spot by "
            f"{1 - cost[('start', 32)] / cost[('center', 32)]:.0%} ({cost[('center', 32)]:.0f} to {cost[('start', 32)]:.0f} KB at chunk size 32)."
        )
    rows = read_jsonl("benchmark.jsonl")
    old = {(r["world_size"], r["step_ms"]): r for r in rows if "prefetch" not in r and r["mode"] == "bare"}
    new = {(r["tag"], r["world_size"], r["step_ms"], r["prefetch"]): r for r in rows if "prefetch" in r}
    keys = [("warm", g, 300, True) for g in (1, 2, 4)]
    if all(k in new for k in keys) and all((k[1], 300) in old for k in keys):
        items.append(
            "With 300 ms of compute per step and a global batch of 1024, the new Loader with prefetch keeps the GPUs waiting "
            + ", ".join(
                f"{new[k]['stall_fraction']:.0%} of the time on {k[1]} GPU{'s' if k[1] > 1 else ''}" for k in keys
            )
            + " (first Loader: "
            + ", ".join(f"{old[(k[1], 300)]['stall_fraction']:.0%}" for k in keys)
            + ")."
        )
    cold, warm = new.get(("cold", 4, 0, False)), new.get(("warm", 4, 0, False))
    if cold and warm:
        items.append(
            f"The first epoch is slower than later ones: from a cold page cache (4 GPUs, no simulated compute, no prefetch) the node reads "
            f"{cold['spots_per_s_node'] / warm['spots_per_s_node']:.0%} of the warm rate."
        )
    tutorials = [
        r
        for r in (read_json(f"tutorial_{m}{V2}.json") or read_json(f"tutorial_{m}.json") for m in ("resize", "crop"))
        if r
    ]
    if tutorials:
        items.append(
            f"The multimodal tutorial runs on {tutorials[0]['world_size']} GPUs at a global batch of {tutorials[0]['global_batch']}; "
            f"its training loss falls from {sum(tutorials[0]['train_loss'][:10]) / 10:.1f} to {sum(tutorials[0]['train_loss'][-10:]) / 10:.1f} "
            "in 150 steps."
        )
    return section("summary", "Summary", "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>") if items else ""


def collections_section() -> str:
    """The three collections: slides, spots, genes and size on disk."""
    rows = []
    for split in SPLITS:
        first, second = read_json(f"collection_{split}.json"), read_json(f"collection_{split}{V2}.json")
        info = second or first
        if info is None:
            continue
        datasets = info["datasets"]
        rows.append(
            [
                split,
                len(datasets),
                f"{sum(d['n_spots'] for d in datasets):,}",
                len(info["genes"]),
                f"{first['bytes'] / 1e9:.1f}" if first else "-",
                f"{second['bytes'] / 1e9:.1f}" if second else "-",
            ]
        )
    cohorts = "".join(f"<li><b>{esc(s)}</b>: {esc(', '.join(sorted(c)))}</li>" for s, c in slide_cohorts().items())
    return section(
        "collections",
        "The three collections",
        f"<p>Each dataset of a collection is one slide: its in-tissue 55 µm spots (<code>{esc(TABLE)}</code>, restricted to the "
        f"genes shared by every slide, in one order) and <code>{esc(HE_ALIGNED)}</code>, the H&amp;E image resampled onto the "
        f"pixel grid of <code>global</code>. Tiles are {TILE_PX} × {TILE_PX} px ({TILE_PX * 0.2125:.1f} µm), centered on the spot.</p>",
        table(
            ["split", "slides", "spots", "genes", "GB, first version", "GB, origin order (_v2)"],
            rows,
            caption="Collections written to /p/scratch/hai_1240/datasets/spatialbatch_testing/hescape/multi_tissue/",
        ),
        f"<p>The split is by slide: tiles of neighboring spots overlap, so splitting the spots of one slide would put the same pixels in train and validation. {esc(TRAIN_COHORT)} is train and "
        f"validation (validation: {esc(', '.join(sorted(VAL_SLIDES)))}); the other cohorts are test.</p>",
        f"<ul class=plain>{cohorts}</ul>",
        source="results/collection_*.json (written by build_collections.py), slides.py",
    )


def slide_cohorts() -> dict[str, list[str]]:
    """The cohorts of each split with their number of slides."""
    report = read_json(f"verification{V2}.json") or read_json("verification.json") or {}
    return {
        split: [f"{c} ({n} slide{'s' if n > 1 else ''})" for c, n in sorted(count_by_cohort(info["datasets"]).items())]
        for split, info in report.items()
    }


def count_by_cohort(datasets: list[dict]) -> dict[str, int]:
    """The number of datasets of each cohort."""
    counts: dict[str, int] = {}
    for dataset in datasets:
        counts[dataset["cohort"]] = counts.get(dataset["cohort"], 0) + 1
    return counts


def verification_section() -> str:
    """What was checked on the written collections."""
    report = read_json(f"verification{V2}.json") or read_json("verification.json")
    if report is None:
        return ""
    rows = [
        [
            split,
            f"{info['n_spots']:,}",
            info["n_genes"],
            f"{info['tiles_checked']} / {info['tiles_checked']}",
            f"{max(d['max_center_offset_px'] for d in info['datasets']):.2f}",
        ]
        for split, info in report.items()
    ]
    slide_rows = [
        [s, d["cohort"], d["slide"], f"{d['n_spots']:,}"] for s, info in report.items() for d in info["datasets"]
    ]
    return section(
        "verification",
        "Checks on the written collections",
        "<p>Every dataset has as many rows as its slide has spots in tissue, the genes of every split are the same list, "
        f"every tile is {TILE_PX} px, and no slide is in two splits. A batch read through the Loader with shuffling is compared pixel "
        "by pixel with the source image at each tile's recorded origin, and the tile center with the spot coordinate "
        "(half a pixel is the rounding of the tile corner).</p>",
        table(["split", "spots", "genes", "tiles equal to source", "max center offset (px)"], rows),
        "<details><summary>Every slide</summary>"
        + table(["split", "cohort", "slide", "spots in tissue"], slide_rows, numeric_from=3)
        + "</details>",
        source=f"results/verification{V2}.json (verify_collections.py)",
    )


def figure_section() -> str:
    """The figure of tiles with their spots and top genes, embedded in the page."""
    path = RESULTS / f"tiles_with_spots{V2}.png"
    if not path.exists():
        path = RESULTS / "tiles_with_spots.png"
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode()
    return section(
        "tiles",
        "Tiles with their spots",
        "<p>Eight tiles of one training batch with the outline of the 55 µm spot they are centered on and the three most "
        "expressed genes of that spot. The genes agree with the tissue: PECAM1 and ACTA2 (endothelium, smooth muscle) in kidney, "
        "TCL1A and MDM2 in the lymphoid leukemia section, OGN in bone.</p>",
        f'<img class="figure" alt="Eight H&amp;E tiles with the hexagonal outline of their spot and top genes" src="data:image/png;base64,{data}">',
        source=f"results/{path.name} (multimodal_hescape.py)",
    )


def read_cost_section() -> str:
    """Bytes read per spot by row order, simulated and measured on disk."""
    simulated = read_json("simulate_order_train.json")
    profile = [r for r in read_jsonl("profile_request.jsonl") if r["tag"] in ("v1-center-order", "v2-origin-order")]
    if simulated is None:
        return ""
    chunk_sizes = sorted({r["chunk_size"] for r in simulated})
    cost = {(r["order_by"], r["chunk_size"]): r["read_kb_per_spot"] for r in simulated if r["block_px"] == BLOCK_PX}
    series = [
        ("rows ordered by tile center", [cost[("center", c)] for c in chunk_sizes]),
        ("rows ordered by tile origin", [cost[("start", c)] for c in chunk_sizes]),
    ]
    saved = 1 - cost[("start", 32)] / cost[("center", 32)]
    blocks = {(r["order_by"], r["block_px"]): r["read_kb_per_spot"] for r in simulated if r["chunk_size"] == 32}
    rows = [
        [
            r["tag"],
            r["chunk_size"],
            f"{r['windows_per_request']:.0f}",
            f"{r['read_kb_per_spot']:.0f}",
            f"{r['decode_s']:.2f}",
            f"{r['decode_warm_s']:.2f}",
            f"{r['pixels_s']:.2f}",
            f"{r['split_s']:.2f}",
        ]
        for r in profile
        if "decode_warm_s" in r
    ]
    real = {(r["tag"], r["chunk_size"]): r for r in profile if "decode_warm_s" in r}
    claim = ""
    if ("v1-center-order", 32) in real and ("v2-origin-order", 32) in real:
        a, b = real[("v1-center-order", 32)], real[("v2-origin-order", 32)]
        claim = (
            f" Read from disk at chunk size 32, a request of 4096 spots touched {a['windows_per_request']:.0f} windows and "
            f"{a['read_kb_per_spot']:.0f} KB per spot with the first version, and {b['windows_per_request']:.0f} windows and "
            f"{b['read_kb_per_spot']:.0f} KB per spot with the origin order. Reading the windows again from the page cache took "
            f"{a['decode_warm_s']:.2f} s and {b['decode_warm_s']:.2f} s."
        )
    return section(
        "order",
        "Row order and bytes read per spot",
        "<p>The Loader reads one window of the image per cell of 1024 px of the tile <em>origins</em> in a request, and crops "
        "the tiles out of it. The collections were first written with their rows sorted along a Z-order curve of the tile "
        "<em>centers</em>, and a block of centers straddles several windows. Sorting by origin makes a run of consecutive rows fall in "
        f"one window. Replaying the sampler on the 13 training slides, the origin order reads {saved:.0%} fewer bytes per spot at chunk size 32 "
        f"({cost[('center', 32)]:.0f} to {cost[('start', 32)]:.0f} KB).{claim}</p>",
        grouped_columns([f"chunk {c}" for c in chunk_sizes], series, "KB read per spot, decoded"),
        f"<p>Block size of the ordering grid (chunk size 32): centers read {blocks[('center', 512)]:.0f}, {blocks[('center', 1024)]:.0f} and "
        f"{blocks[('center', 2048)]:.0f} KB per spot at 512, 1024 and 2048 px, origins {blocks[('start', 512)]:.0f}, {blocks[('start', 1024)]:.0f} and "
        f"{blocks[('start', 2048)]:.0f}. With origin order a 2048 px grid reads {blocks[('start', 2048)] / blocks[('start', 1024)] - 1:.0%} more than 1024 px; "
        "512 and 1024 read the same, and 1024 is the window the Loader reads.</p>",
        table(
            [
                "collection",
                "chunk size",
                "windows / request",
                "KB / spot",
                "read, cold (s)",
                "read, warm (s)",
                "pixels (s)",
                "shuffle copy (s)",
            ],
            rows,
            numeric_from=1,
            caption="One request of 4096 spots, 128 threads, median of 4 requests",
        )
        if rows
        else "",
        source="results/simulate_order_train.json (simulate_order.py, replay of the sampler, no image IO), results/profile_request.jsonl (profile_request.py, real reads)",
    )


def throughput_notes(old: dict, new: dict) -> str:
    """Sentences on the benchmark that are read off its rows: the gains, the cold cache and the one regression."""
    notes = []
    gains = [
        f"{g} GPU{'s' if g > 1 else ''}: {new[('warm', g, 300, True)]['spots_per_s_node'] / old[(g, 300)]['spots_per_s_node'] - 1:+.0%}"
        for g in (1, 2, 4)
        if ("warm", g, 300, True) in new and (g, 300) in old
    ]
    if gains:
        notes.append(
            "With 300 ms steps the new Loader with prefetch against the first Loader: " + ", ".join(gains) + "."
        )
    cold, warm = new.get(("cold", 4, 0, False)), new.get(("warm", 4, 0, False))
    if cold and warm:
        notes.append(
            f"With a cold page cache (4 GPUs, no compute, no prefetch) the node reads {cold['spots_per_s_node']:,.0f} spots/s against "
            f"{warm['spots_per_s_node']:,.0f} warm."
        )
    slow, fast = new.get(("warm", 4, 0, True)), new.get(("warm", 4, 0, False))
    if slow and fast and slow["spots_per_s_node"] < fast["spots_per_s_node"]:
        notes.append(
            f"One result goes the other way: with 4 GPUs and no simulated compute, prefetch is slower ({slow['spots_per_s_node']:,.0f} "
            f"against {fast['spots_per_s_node']:,.0f} spots/s). The cause was not investigated; with a 300 ms step it is not "
            "visible."
        )
    return "".join(f"<p>{n}</p>" for n in notes)


def throughput_section() -> str:
    """Loader throughput of one node for every benchmark run."""
    rows = read_jsonl("benchmark.jsonl")
    if not rows:
        return ""
    old = {(r["world_size"], r["step_ms"]): r for r in rows if "prefetch" not in r and r["mode"] == "bare"}
    new = {(r["tag"], r["world_size"], r["step_ms"], r["prefetch"]): r for r in rows if "prefetch" in r}
    gpus = [1, 2, 4]
    series = [
        (
            "first Loader, first collections",
            [old[(g, 300)]["spots_per_s_node"] if (g, 300) in old else None for g in gpus],
        ),
        (
            "new Loader, no prefetch",
            [
                new[("warm", g, 300, False)]["spots_per_s_node"] if ("warm", g, 300, False) in new else None
                for g in gpus
            ],
        ),
        (
            "new Loader, prefetch",
            [new[("warm", g, 300, True)]["spots_per_s_node"] if ("warm", g, 300, True) in new else None for g in gpus],
        ),
    ]
    table_rows = []
    for r in rows:
        table_rows.append(
            [
                ("new Loader" if "prefetch" in r else "first Loader") + (", prefetch" if r.get("prefetch") else ""),
                r["tag"] if "prefetch" in r else "not controlled",
                r["mode"],
                r["world_size"],
                f"{r['step_ms']:.0f}",
                f"{r['spots_per_s_node']:,.0f}",
                f"{r['stall_fraction']:.2f}",
                f"{r['rss_gb_max_rank']:.1f}",
            ]
        )
    return section(
        "throughput",
        "Loader throughput on one node",
        "<p>Spots per second the node consumes at a global batch of 1024 (batch per GPU = 1024 / GPUs), 224 px model input "
        "resized on the GPU from the 512 px tile, and 300 ms of simulated compute per step. Each cell is one run of 60 steps, "
        "after the page cache was filled with the collection (the rows tagged warm; the first Loader rows were measured with no explicit "
        "warm-up, on the first collections). There are no repeats, so differences of a few percent are not resolved. The benchmark also "
        "copies the batch to the GPU and resizes it, so the step takes longer than 300 ms and the ceiling is below "
        f"1024 / 0.3 s = 3413 spots/s.</p>{throughput_notes(old, new)}",
        grouped_columns([f"{g} GPU{'s' if g > 1 else ''}" for g in gpus], series, "spots/s, whole node, 300 ms step"),
        table(
            [
                "loader",
                "cache",
                "mode",
                "GPUs",
                "step (ms)",
                "spots/s (node)",
                "stall fraction",
                "peak RSS per rank (GB)",
            ],
            table_rows,
            numeric_from=3,
            caption="Every benchmark run. Stall fraction: share of the time the GPU waited for a batch. The first Loader rows are on the first "
            "collections; in workers mode a torch DataLoader with 2 workers per rank wraps the Loader.",
        ),
        source="results/benchmark.jsonl (benchmark_loader.py)",
    )


def tutorial_section() -> str:
    """The loss of the multimodal tutorial on the collections and its validation error."""
    runs = {
        (label, mode): read_json(f"tutorial_{mode}{suffix}.json")
        for label, suffix in (("first collections", V1), ("origin order (_v2)", V2))
        for mode in ("resize", "crop")
    }
    runs = {k: v for k, v in runs.items() if v}
    if not runs:
        return ""
    chosen = {m: runs.get(("origin order (_v2)", m)) or runs.get(("first collections", m)) for m in ("resize", "crop")}
    series = [(f"{m} to 224 px", rolling(r["train_loss"])) for m, r in chosen.items() if r]
    rows = [
        [
            label,
            mode,
            r["world_size"],
            r["global_batch"],
            f"{sum(r['train_loss'][:10]) / 10:.2f}",
            f"{sum(r['train_loss'][-10:]) / 10:.2f}",
            "-" if r["val_mse"] is None else f"{r['val_mse']:.2f}",
        ]
        for (label, mode), r in runs.items()
    ]
    return section(
        "tutorial",
        "The multimodal tutorial on the collections",
        "<p>The small CNN of the tutorial predicts the log-normalised expression of the spot from its tile. It trains for 150 steps on "
        "<code>train</code> with DistributedDataParallel on 4 GPUs (global batch 1024), then is scored on 20 batches of <code>val</code>, "
        "whose slides it has not seen. A 224 px center crop and a resize of the whole 512 px tile are compared. This is a test of the "
        "pipeline: 150 steps of a model this small do not separate the two inputs.</p>",
        lines(series, "step", "training loss (MSE), mean of 10 steps"),
        table(
            [
                "collections",
                "input",
                "GPUs",
                "global batch",
                "train loss, first 10 steps",
                "train loss, last 10 steps",
                "validation MSE",
            ],
            rows,
            numeric_from=2,
        ),
        source="results/tutorial_*.json (multimodal_hescape.py)",
    )


def caveats_section() -> str:
    """What is missing from the collections and what was not measured."""
    excluded = "".join(f"<li><code>{esc(k)}</code>: {esc(v)}</li>" for k, v in EXCLUDED.items())
    return section(
        "caveats",
        "What is not in the collections",
        "<ul>"
        "<li><b>No morphology image.</b> <code>morphology_focus</code> exists in no store and <code>morphology_mip</code> is missing in all UKER and "
        "LMUP stores and in three TENX stores, so the collections hold only the H&amp;E image and the spot table.</li>"
        "<li><b>TENX slides without an H&amp;E image</b> (<code>hLung_cancer_section</code>, <code>hLymphNode_nondiseased_section</code>, "
        "<code>hPancreas_nondiseased_section</code>) are ignored.</li>"
        f"{excluded}"
        "<li><b>Downsampled slides.</b> The H&amp;E of TENX <code>hPancreas_Cancer_Add_on_FFPE</code> and UKER <code>MIER175_Bl_B</code>, "
        "<code>MIER217_Bl_B</code> is finer than the Xenium grid (0.58 to 0.64 of the scale), so resampling onto the grid is nearest-neighbor "
        "downsampling and aliases.</li>"
        "<li><b>Genes.</b> The panels have 377 to 477 genes; the 377 genes shared by all slides are kept.</li>"
        "<li><b>Not measured.</b> Throughput was measured with simulated compute and one run per cell. The slow parity test against "
        "<code>XeniumDataset</code> needs the Xenium stores of the tutorials and was skipped.</li>"
        "</ul>",
        source="slides.py, scripts/hescape_multi_tissue/README.md",
    )


def reproduce_section() -> str:
    """The commit and the commands that give this page again."""
    commit, branch = git("rev-parse", "--short", "HEAD"), git("rev-parse", "--abbrev-ref", "HEAD")
    dirty = " (with uncommitted changes)" if git("status", "--porcelain", "--", "src", "scripts", "tests") else ""
    commands = f"""git checkout {branch}            # {commit}{dirty}
uv sync --group test --extra torch
# every command below runs from the repository root

sbatch --array=0-25 scripts/hescape_multi_tissue/slurm/align.sbatch      # he_image -> he_image_aligned, in place
HESCAPE_SUFFIX=_v2 sbatch scripts/hescape_multi_tissue/slurm/build.sbatch train   # also val and test
HESCAPE_SUFFIX=_v2 .venv/bin/python scripts/hescape_multi_tissue/verify_collections.py

HESCAPE_SUFFIX=_v2 sbatch scripts/hescape_multi_tissue/slurm/tutorial.sbatch
HESCAPE_SUFFIX=_v2 sbatch scripts/hescape_multi_tissue/slurm/bench_prefetch.sbatch
sbatch scripts/hescape_multi_tissue/slurm/profile.sbatch train --threads 128 --chunk-sizes 32 64 --repeats 4 --tag v1-center-order
HESCAPE_SUFFIX=_v2 sbatch scripts/hescape_multi_tissue/slurm/profile.sbatch train --threads 128 --chunk-sizes 32 64 --repeats 4 --tag v2-origin-order
.venv/bin/python scripts/hescape_multi_tissue/simulate_order.py train

.venv/bin/python scripts/hescape_multi_tissue/make_report.py     # writes results/report.html"""
    return section(
        "reproduce",
        "Reproduce",
        f"<p>Branch <code>{esc(branch)}</code>, commit <code>{esc(commit)}</code>{esc(dirty)}. All jobs use "
        "<code>--partition=dc-hwai --account=hai_1240</code>. The first collections (no suffix) have the rows ordered by tile center; "
        "<code>HESCAPE_SUFFIX=_v2</code> writes the origin order next to them.</p>",
        f"<pre><code>{esc(commands)}</code></pre>",
        "<p>This page is the output of <code>make_report.py</code>: it reads <code>results/*.json*</code> and <code>*.png</code> and nothing else.</p>",
    )


def parameters_section() -> str:
    """The write and read parameters with their reasons."""
    rows = [
        [
            "tile",
            f"{TILE_PX} px ({TILE_PX * 0.2125:.1f} µm)",
            "A 55 µm spot is about 259 px across, so a 224 px tile (47.6 µm) is smaller than the spot. 512 px allows both a 224 px center crop and a resize to 224 px.",
        ],
        ["image chunk", f"{IMAGE_CHUNK} px", f"Divides the {BLOCK_PX} px window of the Loader."],
        [
            "image shard",
            f"{IMAGE_SHARD} px",
            f"{(IMAGE_SHARD // IMAGE_CHUNK) ** 2} chunks of about 49 KB: about 50 MB per file before compression.",
        ],
        [
            "row ordering grid",
            f"{BLOCK_PX} px",
            "The window the Loader reads. Rows are Z-ordered by tile origin (the tile center in the first collections).",
        ],
        [
            "chunk size, preload",
            "32 spots, 128 chunks",
            "A request is 4096 spots per rank. At 256 spots per GPU a batch draws from about 8 chunks.",
        ],
        [
            "global batch",
            "1024",
            "Per rank 1024 / GPUs. Each rank reads a fixed contiguous shard of the rows (annbatch DistributedSampler).",
        ],
    ]
    return section(
        "parameters",
        "Parameters",
        table(["parameter", "value", "reason"], rows, numeric_from=9),
        source="slides.py, slurm/*.sbatch",
    )


CSS = """
:root {
  --bg: #fbfbfa; --fg: #1a1a1a; --muted: #5f5f5c; --rule: #dcdcd8; --surface: #ffffff; --code: #f1f1ee;
  --accent: #1757ac; --grid: #ececea;
  --s1: #257de0; --s2: #f47230; --s3: #00b894;
  --font: "Inter", system-ui, -apple-system, "Segoe UI", sans-serif;
  --head: "Space Grotesk", "Inter", system-ui, sans-serif;
  --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #151514; --fg: #ececea; --muted: #a2a29d; --rule: #353533; --surface: #1c1c1b; --code: #232321;
    --accent: #6aabf1; --grid: #2a2a28;
    --s1: #338ced; --s2: #e76624; --s3: #00a888; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #151514; --fg: #ececea; --muted: #a2a29d; --rule: #353533; --surface: #1c1c1b; --code: #232321;
  --accent: #6aabf1; --grid: #2a2a28;
  --s1: #338ced; --s2: #e76624; --s3: #00a888; color-scheme: dark;
}
body { background: var(--bg); color: var(--fg); font-family: var(--font); line-height: 1.55; margin: 0;
  padding-block: 2.5rem 4rem; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
main { max-width: 52rem; margin: 0 auto; }
header { border-bottom: 1px solid var(--rule); padding-bottom: 1.5rem; margin-bottom: 0.5rem; }
h1 { font-family: var(--head); font-size: clamp(1.7rem, 4vw, 2.3rem); line-height: 1.15; margin: 0 0 0.6rem; text-wrap: balance; }
h2 { font-family: var(--head); font-size: 1.3rem; margin: 0 0 0.75rem; text-wrap: balance; }
section { margin-top: 2.6rem; }
p { max-width: 65ch; margin: 0 0 0.9rem; }
.lede { font-size: 1.08rem; color: var(--fg); }
.meta { color: var(--muted); font-size: 0.85rem; margin: 0; }
nav { display: flex; flex-wrap: wrap; gap: 0.4rem 1.1rem; margin-top: 1rem; font-size: 0.88rem; }
nav a { color: var(--accent); text-decoration: none; } nav a:hover { text-decoration: underline; }
.source { color: var(--muted); font-size: 0.8rem; margin-top: 0.6rem; }
code { font-family: var(--mono); font-size: 0.86em; background: var(--code); padding: 0.1em 0.3em; border-radius: 3px; }
pre { background: var(--code); padding: 1rem; border-radius: 6px; overflow-x: auto; font-size: 0.8rem; line-height: 1.5; }
pre code { background: none; padding: 0; }
ul { padding-left: 1.2rem; max-width: 65ch; } li { margin-bottom: 0.45rem; } ul.plain { list-style: none; padding: 0; }
.scroll { overflow-x: auto; max-width: 100%; margin: 0.6rem 0 0.9rem; }
table { border-collapse: collapse; font-size: 0.85rem; width: 100%; min-width: 34rem; }
caption { text-align: left; color: var(--muted); font-size: 0.8rem; padding-bottom: 0.4rem; }
th, td { text-align: left; padding: 0.4rem 0.7rem 0.4rem 0; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { font-weight: 600; font-size: 0.78rem; color: var(--muted); } td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; padding-right: 0.8rem; }
details { margin: 0.4rem 0 1rem; } summary { cursor: pointer; color: var(--accent); font-size: 0.88rem; }
img.figure { max-width: 100%; height: auto; border-radius: 4px; border: 1px solid var(--rule); background: var(--surface); }
.chart { min-width: 34rem; } .chart svg { width: 100%; height: auto; display: block; }
.legend { display: flex; flex-wrap: wrap; gap: 0.3rem 1.2rem; font-size: 0.82rem; color: var(--muted); margin-top: 0.8rem; }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 0.4rem; }
.s1 { fill: var(--s1); stroke: var(--s1); background: var(--s1); } .s2 { fill: var(--s2); stroke: var(--s2); background: var(--s2); }
.s3 { fill: var(--s3); stroke: var(--s3); background: var(--s3); }
svg .grid { stroke: var(--grid); stroke-width: 1; } svg .base { stroke: var(--rule); stroke-width: 1; }
svg .axis { fill: var(--muted); font-size: 11px; font-family: var(--font); } svg .value { fill: var(--fg); font-size: 10px; font-family: var(--font); }
svg .line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; } svg .dot { stroke: var(--bg); stroke-width: 2; }
svg .mark { stroke: none; } svg .mark:hover, svg .mark:focus { opacity: 0.8; outline: none; }
svg .hit { fill: transparent; } svg .hit:hover, svg .hit:focus { fill: var(--grid); opacity: 0.6; outline: none; }
#tip { position: fixed; pointer-events: none; background: var(--fg); color: var(--bg); font-size: 0.78rem; padding: 0.3rem 0.55rem; border-radius: 4px; z-index: 20; max-width: 18rem; }
@media (prefers-reduced-motion: no-preference) { svg .mark { transition: opacity 0.12s; } }
"""

SCRIPT = """
(function () {
  var tip = document.getElementById('tip');
  function show(el, x, y) { tip.textContent = el.getAttribute('data-tip'); tip.hidden = false;
    tip.style.left = Math.min(x + 12, window.innerWidth - tip.offsetWidth - 8) + 'px'; tip.style.top = (y + 14) + 'px'; }
  document.addEventListener('mouseover', function (e) { var t = e.target.closest('[data-tip]'); if (t) show(t, e.clientX, e.clientY); });
  document.addEventListener('mousemove', function (e) { var t = e.target.closest('[data-tip]'); if (t) show(t, e.clientX, e.clientY); else tip.hidden = true; });
  document.addEventListener('focusin', function (e) { var t = e.target.closest('[data-tip]'); if (t) { var r = t.getBoundingClientRect(); show(t, r.left, r.top - 40); } });
  document.addEventListener('focusout', function () { tip.hidden = true; });
})();
"""


def summary_line() -> str:
    """The headline numbers of the page, from the same files as the sections."""
    totals = {}
    for split in SPLITS:
        info = read_json(f"collection_{split}{V2}.json") or read_json(f"collection_{split}.json")
        if info:
            totals[split] = (len(info["datasets"]), sum(d["n_spots"] for d in info["datasets"]))
    parts = [f"{totals[s][1]:,} spots in {totals[s][0]} slides for {s}" for s in SPLITS if s in totals]
    return (
        (
            "Paired H&amp;E tiles and 55 µm spot expression from the multi-tissue Xenium cohorts, written as three spatialbatch "
            "collections: " + ", ".join(parts) + "."
        )
        if parts
        else ""
    )


def main() -> None:
    """Write the page and check that every chart is well-formed SVG."""
    body = "".join(
        [
            summary_section(),
            collections_section(),
            parameters_section(),
            figure_section(),
            verification_section(),
            read_cost_section(),
            throughput_section(),
            tutorial_section(),
            caveats_section(),
            reproduce_section(),
        ]
    )
    nav = "".join(
        f'<a href="#{i}">{t}</a>'
        for i, t in [
            ("summary", "Summary"),
            ("collections", "Collections"),
            ("parameters", "Parameters"),
            ("tiles", "Tiles"),
            ("verification", "Checks"),
            ("order", "Row order"),
            ("throughput", "Throughput"),
            ("tutorial", "Tutorial"),
            ("caveats", "Not included"),
            ("reproduce", "Reproduce"),
        ]
    )
    page = (
        '<title>HESCAPE Multi-Tissue Collections</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Space+Grotesk:wght@500;600&display=swap">\n'
        f"<style>{CSS}</style>\n<main><header><h1>HESCAPE Multi-Tissue Collections</h1>"
        f'<p class="lede">{summary_line()}</p>'
        f'<p class="meta">Generated {datetime.date.today().isoformat()} by make_report.py, commit {esc(git("rev-parse", "--short", "HEAD"))} '
        f"of {esc(git('rev-parse', '--abbrev-ref', 'HEAD'))}.</p><nav>{nav}</nav></header>{body}</main>\n"
        f'<div id="tip" hidden></div>\n<script>{SCRIPT}</script>\n'
    )
    for svg in page.split("<svg")[1:]:
        ET.fromstring("<svg" + svg.split("</svg>")[0] + "</svg>")  # raises if a chart is not well-formed
    (RESULTS / "report.html").write_text(page)
    print(f"wrote {RESULTS / 'report.html'} ({len(page) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
