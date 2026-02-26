"""
Export trace JSON into presentation-ready GIF / MP4 files.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple


def _path_edge_set(path: List[str]) -> set:
    s = set()
    if not path or len(path) < 2:
        return s
    for i in range(len(path) - 1):
        s.add((path[i], path[i + 1]))
        s.add((path[i + 1], path[i]))
    return s


def _node_positions_circular(nodes: List[str], width: int, height: int) -> Dict[str, Tuple[float, float]]:
    import math

    if not nodes:
        return {}
    cx = width / 2.0
    cy = height / 2.0
    radius = min(width, height) * 0.28
    positions = {}
    for i, node in enumerate(nodes):
        angle = (2 * math.pi * i) / max(1, len(nodes)) - math.pi / 2
        positions[node] = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
    return positions


def _node_positions_flow(
    nodes: List[str],
    edges: List[Dict[str, Any]],
    source: str,
    target: str,
    width: int,
    height: int,
) -> Dict[str, Tuple[float, float]]:
    from collections import deque

    if not nodes:
        return {}

    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    for e in edges:
        u = e.get("from")
        v = e.get("to")
        if u in adj and v in adj:
            adj[u].append(v)
            adj[v].append(u)

    depth: Dict[str, int] = {}
    start = source if source in adj else nodes[0]
    depth[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in depth:
                depth[v] = depth[u] + 1
                q.append(v)

    max_depth = 0
    for n in nodes:
        if n not in depth:
            depth[n] = max_depth + 1
        max_depth = max(max_depth, depth[n])

    if target in depth:
        max_depth = max(max_depth, depth[target])

    buckets: Dict[int, List[str]] = {}
    for n in nodes:
        buckets.setdefault(depth[n], []).append(n)
    for d in buckets:
        buckets[d].sort()

    left, right = 90.0, float(width - 90)
    top, bottom = 60.0, float(height - 60)
    usable_w = max(1.0, right - left)
    usable_h = max(1.0, bottom - top)

    positions: Dict[str, Tuple[float, float]] = {}
    for n in nodes:
        d = depth[n]
        layer = buckets[d]
        idx = layer.index(n)
        x = left + (0.5 if max_depth == 0 else d / max_depth) * usable_w
        y = top + ((idx + 1) / (len(layer) + 1)) * usable_h
        positions[n] = (x, y)

    if source in positions:
        _, y = positions[source]
        positions[source] = (left, y)
    if target in positions:
        _, y = positions[target]
        positions[target] = (right, y)
    return positions


def _render_frame_image(
    trace: Dict[str, Any],
    frame: Dict[str, Any],
    width: int = 1280,
    height: int = 720,
    layout: str = "flow",
):
    import matplotlib.pyplot as plt
    import numpy as np

    nodes = trace.get("graph", {}).get("nodes", [])
    edges = trace.get("graph", {}).get("edges", [])
    best_path = frame.get("best_path") or trace.get("result", {}).get("path") or []
    path_edges = _path_edge_set(best_path)
    visited = set(frame.get("visited", []))
    current = frame.get("current_node")
    highlight_edge = frame.get("edge_highlight")

    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100, constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    ax_text = fig.add_subplot(gs[0, 1])

    ax.set_facecolor("#f8fbff")
    ax_text.set_facecolor("#f8fbff")
    ax.axis("off")
    ax_text.axis("off")

    req = trace.get("metadata", {}).get("request", {})
    src = req.get("token_in", "")
    dst = req.get("token_out", "")
    if layout == "circular":
        pos = _node_positions_circular(nodes, width=900, height=600)
    else:
        pos = _node_positions_flow(nodes, edges, src, dst, width=900, height=600)

    # Draw edges
    for e in edges:
        u = e.get("from")
        v = e.get("to")
        if u not in pos or v not in pos:
            continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        color = "#9caac0"
        lw = 1.3
        if (u, v) in path_edges:
            color = "#0b7f7a"
            lw = 3.0
        if highlight_edge and highlight_edge.get("from") == u and highlight_edge.get("to") == v:
            color = "#f28f3b"
            lw = 3.5
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, alpha=0.9)

    # Draw nodes
    for node in nodes:
        x, y = pos[node]
        fill = "#ffffff"
        edge = "#9caac0"
        if node in visited:
            fill = "#2a6fdb"
            edge = "#2a6fdb"
        if node == current:
            fill = "#f28f3b"
            edge = "#f28f3b"
        if node in best_path and node != current:
            fill = "#0b7f7a"
            edge = "#0b7f7a"
        circle = plt.Circle((x, y), 20, facecolor=fill, edgecolor=edge, linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, str(node), ha="center", va="center", color="#ffffff", fontsize=10, fontweight="bold")

    ax.set_xlim(0, 900)
    ax.set_ylim(0, 600)
    ax.invert_yaxis()

    meta = trace.get("metadata", {})
    text_lines = [
        f"Algorithm: {meta.get('algorithm', '-')}",
        f"Type: {meta.get('trace_kind', '-')}",
        "",
        f"Step: {frame.get('step', '-')}",
        f"Event: {frame.get('event', '-')}",
        "",
        "Description:",
        frame.get("description", "-"),
        "",
        "Key data:",
    ]
    if "estimated_amount_out" in frame:
        text_lines.append(f"estimated_out: {frame.get('estimated_amount_out')}")
    if "total_amount_out" in frame:
        text_lines.append(f"total_out: {frame.get('total_amount_out')}")
    if "iteration" in frame:
        text_lines.append(f"iteration: {frame.get('iteration')}")
    if "winner_pool_id" in frame:
        text_lines.append(f"winner_pool: {frame.get('winner_pool_id')}")

    ax_text.text(
        0.02,
        0.98,
        "\n".join(text_lines),
        transform=ax_text.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        color="#213047",
    )

    fig.canvas.draw()
    # `print_to_buffer` works across backends (including macOS FigureCanvasMac).
    buffer, (buf_w, buf_h) = fig.canvas.print_to_buffer()
    rgba = np.frombuffer(buffer, dtype=np.uint8).reshape((buf_h, buf_w, 4))
    img = rgba[:, :, :3].copy()
    plt.close(fig)
    return img


def _safe_slug(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


def export_trace_media_bundle(
    traces: List[Dict[str, Any]],
    output_dir: str,
    fps: int = 2,
    width: int = 1280,
    height: int = 720,
    layout: str = "flow",
) -> Dict[str, Any]:
    """
    Export each trace to GIF and MP4.

    `traces` format:
      [{"name": "...", "trace": {...}}, ...]
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        import imageio.v2 as imageio  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency gate
        return {
            "ok": False,
            "reason": (
                f"imageio import failed: {exc}. "
                "Install with: pip install imageio imageio-ffmpeg matplotlib numpy"
            ),
            "files": [],
        }

    files = []
    for item in traces:
        name = item["name"]
        trace = item["trace"]
        frames = trace.get("frames", [])
        if not frames:
            continue

        images = [
            _render_frame_image(trace, frame, width=width, height=height, layout=layout)
            for frame in frames
        ]
        slug = _safe_slug(name)
        gif_path = os.path.join(output_dir, f"{slug}.gif")
        mp4_path = os.path.join(output_dir, f"{slug}.mp4")

        # GIF
        imageio.mimsave(gif_path, images, fps=fps)
        file_record = {"name": name, "gif": gif_path, "mp4": None, "mp4_error": None}

        # MP4 (best effort)
        try:
            with imageio.get_writer(mp4_path, fps=fps, codec="libx264", quality=8) as writer:
                for img in images:
                    writer.append_data(img)
            file_record["mp4"] = mp4_path
        except Exception as exc:  # pragma: no cover - depends on ffmpeg runtime
            file_record["mp4_error"] = (
                f"mp4 export skipped: {exc}. "
                "Install ffmpeg or imageio-ffmpeg for MP4 encoding."
            )
            if os.path.exists(mp4_path):
                try:
                    os.remove(mp4_path)
                except OSError:
                    pass

        files.append(file_record)

    return {"ok": True, "files": files, "reason": None}
