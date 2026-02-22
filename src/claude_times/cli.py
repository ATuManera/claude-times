# -*- coding: utf-8 -*-
"""
claude-times - Extrae tiempos de trabajo, tokens y costos de Claude Code.

Analiza el historial de conversaciones JSONL de Claude Code para generar
reportes detallados de tiempo, consumo de tokens, uso de herramientas
y costos estimados por instruccion, sesion y modelo.

Copyright (c) 2026 J. Fernando Gallarday V.
Innova a Tu Manera Soluciones Digitales SAC (A Tu Manera Digital) -

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

SPDX-License-Identifier: AGPL-3.0-or-later

## 📥 Installation

### Install with pip (recommended)

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .

Uso:
  claude-times                         # Ultima sesion del proyecto actual
  claude-times --all                   # Todas las sesiones del proyecto
  claude-times --days 30               # Sesiones de los ultimos 30 dias
  claude-times --days 90               # Ultimos 90 dias
  claude-times --days 0                # Todo el historial
  claude-times --project lego          # Proyecto por nombre
  claude-times --list                  # Lista todos los proyectos
  claude-times --detail                # Timeline mensaje a mensaje
  claude-times --csv > tiempos.csv     # Exportar a CSV
  claude-times --json > tiempos.json   # Exportar a JSON
"""

import json
import os
import sys
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict

CLAUDE_DIR = Path.home() / ".claude" / "projects"

# Precios por millon de tokens (USD) — Fuente: platform.claude.com/docs/en/about-claude/pricing
MODEL_PRICING = {
    "claude-opus-4-6": {
        "label": "Opus 4.6",
        "input": 5.0,
        "output": 25.0,
        "cache_read": 0.50,
        "cache_write": 6.25,
    },
    "claude-opus-4-5-20251101": {
        "label": "Opus 4.5",
        "input": 5.0,
        "output": 25.0,
        "cache_read": 0.50,
        "cache_write": 6.25,
    },
    "claude-sonnet-4-5-20250929": {
        "label": "Sonnet 4.5",
        "input": 3.0,
        "output": 15.0,
        "cache_read": 0.30,
        "cache_write": 3.75,
    },
    "claude-haiku-4-5-20251001": {
        "label": "Haiku 4.5",
        "input": 1.0,
        "output": 5.0,
        "cache_read": 0.10,
        "cache_write": 1.25,
    },
}
# Fallback for unknown models
DEFAULT_PRICING = {
    "label": "Unknown",
    "input": 5.0,
    "output": 25.0,
    "cache_read": 0.50,
    "cache_write": 6.25,
}


def detect_tz_offset():
    now = datetime.now()
    utc_now = datetime.now(timezone.utc).replace(tzinfo=None)
    return timedelta(seconds=round((now - utc_now).total_seconds()))


TZ_OFFSET = detect_tz_offset()


def parse_ts(ts):
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def fmt_dur(secs):
    if secs < 60:
        return f"{secs:.0f}s"
    if secs < 3600:
        return f"{int(secs//60)}m {int(secs%60)}s"
    return f"{int(secs//3600)}h {int((secs%3600)//60)}m"


def fmt_tokens(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def get_pricing(model_id):
    """Get pricing dict for a model, with fallback."""
    if model_id in MODEL_PRICING:
        return MODEL_PRICING[model_id]
    # Try partial match
    for key, val in MODEL_PRICING.items():
        if key in model_id or model_id in key:
            return val
    return DEFAULT_PRICING


def calc_cost_for_model(model_id, inp, cache_read, cache_write, out):
    p = get_pricing(model_id)
    return (
        inp * p["input"] / 1_000_000
        + cache_read * p["cache_read"] / 1_000_000
        + cache_write * p["cache_write"] / 1_000_000
        + out * p["output"] / 1_000_000
    )


def get_tool_name(block):
    """Extract tool name from a tool_use content block."""
    name = block.get("name", "?")
    # Normalize tool names to categories
    tool_map = {
        "Read": "Read",
        "Write": "Write",
        "Edit": "Edit",
        "Bash": "Bash",
        "Grep": "Grep",
        "Glob": "Glob",
        "Task": "Task",
        "TaskCreate": "Task",
        "TaskUpdate": "Task",
        "TaskGet": "Task",
        "TaskList": "Task",
        "TaskStop": "Task",
        "WebFetch": "Web",
        "WebSearch": "Web",
        "NotebookEdit": "Edit",
        "AskUserQuestion": "Ask",
    }
    return tool_map.get(name, name), name


def get_summary(msg, max_len=80):
    content = msg.get("message", {}).get("content", "")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                return block.get("text", "")[:max_len].replace("\n", " ").strip()
            if block.get("type") == "tool_use":
                cat, name = get_tool_name(block)
                inp = block.get("input", {})
                if name in ("Write", "Edit", "Read"):
                    return f"[{name}: {inp.get('file_path','?').split('/')[-1]}]"
                if name == "Bash":
                    return f"[Bash: {inp.get('command','?')[:50]}]"
                if name in ("Grep", "Glob"):
                    return f"[{name}: {inp.get('pattern','?')[:40]}]"
                if name == "Task":
                    return f"[Task: {inp.get('description','?')[:40]}]"
                if name == "TaskCreate":
                    return f"[TaskCreate: {inp.get('subject','?')[:40]}]"
                return f"[{name}]"
    elif isinstance(content, str):
        return content[:max_len].replace("\n", " ").strip()
    return ""


def extract_tools(msg):
    """Extract all tool names from a message's content blocks."""
    tools = []
    content = msg.get("message", {}).get("content", "")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                cat, name = get_tool_name(block)
                tools.append(cat)
    return tools


def extract_tokens(msg):
    """Extract token usage from a message."""
    usage = msg.get("message", {}).get("usage", {})
    if not usage:
        return None
    return {
        "input": usage.get("input_tokens", 0),
        "cache_read": usage.get("cache_read_input_tokens", 0),
        "cache_write": usage.get("cache_creation_input_tokens", 0),
        "output": usage.get("output_tokens", 0),
        "model": msg.get("message", {}).get("model", "?"),
    }


def load_session(filepath):
    msgs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msgs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return msgs


def analyze_session(messages):
    events = []
    for msg in messages:
        msg_type = msg.get("type")
        if msg_type not in ("user", "assistant"):
            continue
        ts = parse_ts(msg.get("timestamp"))
        if ts is None:
            continue
        local_ts = ts + TZ_OFFSET
        is_error = msg.get("isApiErrorMessage", False)
        summary = get_summary(msg)
        tools = extract_tools(msg) if msg_type == "assistant" else []
        tokens = extract_tokens(msg) if msg_type == "assistant" else None
        events.append(
            {
                "ts": ts,
                "local_ts": local_ts,
                "type": msg_type,
                "is_error": is_error,
                "summary": summary,
                "tools": tools,
                "tokens": tokens,
            }
        )

    if not events:
        return None

    # Build phases (each user message = new phase)
    phases = []
    current = None
    for evt in events:
        if evt["type"] == "user" and evt["summary"]:
            if current:
                current["end"] = evt["ts"]
                phases.append(current)
            current = {
                "start": evt["ts"],
                "local_start": evt["local_ts"],
                "prompt": evt["summary"],
                "end": evt["ts"],
                "local_end": evt["local_ts"],
                "assistant_events": [],
                "errors": 0,
                "tool_counts": defaultdict(int),
                "tokens": {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0},
                "model_tokens": defaultdict(
                    lambda: {
                        "input": 0,
                        "cache_read": 0,
                        "cache_write": 0,
                        "output": 0,
                        "calls": 0,
                    }
                ),
            }
        elif evt["type"] == "assistant" and current:
            current["end"] = evt["ts"]
            current["local_end"] = evt["local_ts"]
            current["assistant_events"].append(evt)
            if evt["is_error"]:
                current["errors"] += 1
            for t in evt["tools"]:
                current["tool_counts"][t] += 1
            if evt["tokens"]:
                for k in ("input", "cache_read", "cache_write", "output"):
                    current["tokens"][k] += evt["tokens"][k]
                model = evt["tokens"].get("model", "unknown")
                if model and model != "<synthetic>":
                    for k in ("input", "cache_read", "cache_write", "output"):
                        current["model_tokens"][model][k] += evt["tokens"][k]
                    current["model_tokens"][model]["calls"] += 1

    if current:
        phases.append(current)

    # Totals
    claude_secs = 0
    total_tokens = {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0}
    total_tools = defaultdict(int)
    model_stats = defaultdict(
        lambda: {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0, "calls": 0}
    )
    for p in phases:
        if p["assistant_events"]:
            first = p["assistant_events"][0]["ts"]
            last = p["assistant_events"][-1]["ts"]
            claude_secs += (last - first).total_seconds()
        for k in total_tokens:
            total_tokens[k] += p["tokens"][k]
        for t, c in p["tool_counts"].items():
            total_tools[t] += c
        for model, mtk in p["model_tokens"].items():
            for k in total_tokens:
                model_stats[model][k] += mtk[k]
            model_stats[model]["calls"] += mtk.get("calls", 0)

    return {
        "events": events,
        "phases": phases,
        "session_start": events[0]["local_ts"],
        "session_end": events[-1]["local_ts"],
        "total_duration": (events[-1]["ts"] - events[0]["ts"]).total_seconds(),
        "claude_work_secs": claude_secs,
        "total_events": len(events),
        "total_errors": sum(p["errors"] for p in phases),
        "total_tokens": total_tokens,
        "total_tools": dict(total_tools),
        "model_stats": dict(model_stats),
    }


# ─── OUTPUT FORMATS ────────────────────────────────────────────────


def print_summary(analysis, session_name=""):
    if not analysis:
        print("  (sin datos)")
        return

    phases = analysis["phases"]
    tk = analysis["total_tokens"]
    ms = analysis.get("model_stats", {})

    # Calculate total cost per model
    total_cost = 0
    for model, mtk in ms.items():
        total_cost += calc_cost_for_model(
            model, mtk["input"], mtk["cache_read"], mtk["cache_write"], mtk["output"]
        )
    if total_cost == 0 and any(tk[k] for k in tk):
        total_cost = calc_cost_for_model(
            "unknown", tk["input"], tk["cache_read"], tk["cache_write"], tk["output"]
        )

    print()
    print("=" * 78)
    title = (
        f"REPORTE DE TIEMPOS — {session_name}" if session_name else "REPORTE DE TIEMPOS"
    )
    print(f"  {title}")
    print("=" * 78)
    print()
    print(
        f"  Inicio:          {analysis['session_start'].strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print(f"  Fin:             {analysis['session_end'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Duracion total:  {fmt_dur(analysis['total_duration'])}")
    print(f"  Trabajo Claude:  {fmt_dur(analysis['claude_work_secs'])}")
    print(f"  Errores API:     {analysis['total_errors']}")
    print(f"  Costo estimado:  ${total_cost:.2f} USD")
    print()

    # Per-model token breakdown
    if ms:
        print("  CONSUMO POR MODELO")
        print("  " + "-" * 74)
        print(
            f"  {'Modelo':20s}  {'Calls':>5}  {'Input':>8}  {'CacheR':>8}  {'CacheW':>8}  {'Output':>8}  {'Costo':>9}"
        )
        print("  " + "-" * 74)
        for model in sorted(ms.keys()):
            mtk = ms[model]
            p = get_pricing(model)
            label = p["label"]
            c = calc_cost_for_model(
                model,
                mtk["input"],
                mtk["cache_read"],
                mtk["cache_write"],
                mtk["output"],
            )
            print(
                f"  {label:20s}  {mtk['calls']:>5}  {fmt_tokens(mtk['input']):>8}  {fmt_tokens(mtk['cache_read']):>8}  "
                f"{fmt_tokens(mtk['cache_write']):>8}  {fmt_tokens(mtk['output']):>8}  ${c:>8.2f}"
            )
        print("  " + "-" * 74)
        print(
            f"  {'TOTAL':20s}  {sum(m['calls'] for m in ms.values()):>5}  {fmt_tokens(tk['input']):>8}  "
            f"{fmt_tokens(tk['cache_read']):>8}  {fmt_tokens(tk['cache_write']):>8}  "
            f"{fmt_tokens(tk['output']):>8}  ${total_cost:>8.2f}"
        )
    else:
        print(f"  Tokens entrada:  {fmt_tokens(tk['input']):>8}")
        print(f"  Tokens cache R:  {fmt_tokens(tk['cache_read']):>8}")
        print(f"  Tokens cache W:  {fmt_tokens(tk['cache_write']):>8}")
        print(f"  Tokens salida:   {fmt_tokens(tk['output']):>8}")
    print()

    # Phase table
    print("  DESGLOSE POR INSTRUCCION")
    hdr = f"  {'#':>3}  {'Inicio':8}  {'Dur':>7}  {'Claude':>7}  {'Tools':>5}  {'TkIn':>7}  {'Cache%':>6}  {'TkOut':>7}  {'Costo':>8}  Instruccion"
    print("  " + "-" * (len(hdr) - 2))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for i, phase in enumerate(phases):
        dur = (phase["end"] - phase["start"]).total_seconds()
        claude_dur = 0
        if phase["assistant_events"]:
            claude_dur = (
                phase["assistant_events"][-1]["ts"] - phase["assistant_events"][0]["ts"]
            ).total_seconds()
        total_t = sum(phase["tool_counts"].values())
        ptk = phase["tokens"]
        tk_in = ptk["input"] + ptk["cache_read"] + ptk["cache_write"]
        cache_pct = (ptk["cache_read"] / tk_in * 100) if tk_in > 0 else 0
        tk_out = ptk["output"]
        # Cost per instruction using per-model pricing
        phase_cost = 0
        for model, mtk in phase["model_tokens"].items():
            phase_cost += calc_cost_for_model(
                model,
                mtk["input"],
                mtk["cache_read"],
                mtk["cache_write"],
                mtk["output"],
            )
        prompt = phase["prompt"][:30]
        err = " ERR" if phase["errors"] else ""
        start_str = phase["local_start"].strftime("%H:%M:%S")
        print(
            f"  {i+1:>3}  {start_str}  {fmt_dur(dur):>7}  {fmt_dur(claude_dur):>7}  {total_t:>5}  {fmt_tokens(tk_in):>7}  {cache_pct:>5.1f}%  {fmt_tokens(tk_out):>7}  ${phase_cost:>7.2f}  {prompt}{err}"
        )

    print("  " + "-" * (len(hdr) - 2))

    # Tool breakdown
    tools = analysis["total_tools"]
    total_tool_count = sum(tools.values())
    print()
    print(f"  HERRAMIENTAS: {total_tool_count} total")
    print("  " + "-" * 50)
    # Define display order and descriptions
    tool_legend = {
        "Read": "Leer archivos",
        "Write": "Crear archivos nuevos",
        "Edit": "Modificar archivos existentes",
        "Bash": "Comandos de terminal (git, npm...)",
        "Grep": "Buscar texto en archivos",
        "Glob": "Buscar archivos por patron",
        "Task": "Sub-agentes y tareas",
        "Web": "Busquedas web / fetch URLs",
        "Ask": "Preguntas al usuario",
    }
    for tool_name in [
        "Read",
        "Write",
        "Edit",
        "Bash",
        "Grep",
        "Glob",
        "Task",
        "Web",
        "Ask",
    ]:
        count = tools.get(tool_name, 0)
        if count == 0:
            continue
        desc = tool_legend.get(tool_name, "")
        bar = "#" * min(count, 40)
        print(f"    {tool_name:6s}  {count:>4}  {bar}  {desc}")
    # Any others not in the standard list
    for tool_name, count in sorted(tools.items()):
        if tool_name not in tool_legend:
            bar = "#" * min(count, 40)
            print(f"    {tool_name:6s}  {count:>4}  {bar}")
    print()

    # Cost legend
    print("  LEYENDA")
    print("  " + "-" * 74)
    print("  Columnas del desglose:")
    print("    TkIn   = contexto total de entrada (input + cache_read + cache_write)")
    print("    Cache% = porcentaje del contexto leido desde cache (ahorro ~90%)")
    print("    TkOut  = tokens generados por Claude")
    print()
    print("  Tipos de tokens:")
    print("    Input       = tokens nuevos enviados (tu prompt + contexto no cacheado)")
    print("    Cache Read  = tokens leidos del cache (90% descuento vs input)")
    print("    Cache Write = tokens escritos al cache (25% mas caro vs input)")
    print("    Output      = tokens generados por Claude (5x mas caro que input)")
    print()
    print("  Precios por millon de tokens (USD):")
    print(
        f"  {'Modelo':20s}  {'Input':>7}  {'CacheR':>7}  {'CacheW':>7}  {'Output':>7}"
    )
    print("  " + "-" * 52)
    for model_id in sorted(MODEL_PRICING.keys()):
        p = MODEL_PRICING[model_id]
        print(
            f"  {p['label']:20s}  ${p['input']:>5.2f}  ${p['cache_read']:>5.2f}  "
            f"${p['cache_write']:>5.2f}  ${p['output']:>5.2f}"
        )
    print()
    print("  Factores que afectan el costo:")
    print("    - Modelo usado (Haiku < Sonnet < Opus)")
    print("    - Tokens de salida son lo mas caro (5x vs input)")
    print("    - Cache reduce ~90% el costo de contexto repetido")
    print("    - Contexto largo (>200K tokens) tiene recargo automatico")
    print("    - Errores API consumen tokens sin producir resultado util")
    print("    - Fuente: platform.claude.com/docs/en/about-claude/pricing")
    print()
    print("  NOTA SOBRE COSTOS Y PLANES DE SUSCRIPCION")
    print("  " + "-" * 74)
    print("  Los costos mostrados son precios de API (pago por token). Si usas Claude")
    print("  Code con un plan de suscripcion, el costo real es diferente:")
    print()
    print(
        f"  {'Plan':15s}  {'Precio/mes':>12}  {'Tokens':20s}  {'Costo real/instruccion'}"
    )
    print("  " + "-" * 74)
    print(
        f"  {'API':15s}  {'pay-per-use':>12}  {'N/A':20s}  El que muestra este script"
    )
    print(
        f"  {'Pro ($20)':15s}  {'$20 fijo':>12}  {'Limitados (rate limit)':20s}  No se publica"
    )
    print(
        f"  {'Max 5x ($100)':15s}  {'$100 fijo':>12}  {'5x mas que Pro':20s}  No se publica"
    )
    print(
        f"  {'Max 20x ($200)':15s}  {'$200 fijo':>12}  {'20x mas que Pro':20s}  No se publica"
    )
    print("  " + "-" * 74)
    print("  En planes Pro/Max, los costos API sirven como referencia comparativa")
    print("  para identificar que instrucciones consumen mas recursos.")
    print()


def print_detail(analysis):
    if not analysis:
        return
    events = analysis["events"]
    print()
    print("  TIMELINE DETALLADO")
    print("  " + "=" * 108)
    print(
        f"  {'Hora':8}  {'Tipo':9}  {'Delta':>9}  {'TkIn':>7}  {'Cache%':>6}  {'TkOut':>7}  Detalle"
    )
    print("  " + "-" * 108)

    prev_ts = None
    for evt in events:
        delta = ""
        if prev_ts:
            diff = (evt["ts"] - prev_ts).total_seconds()
            delta = f"+{diff:.0f}s" if diff < 60 else f"+{diff/60:.1f}m"
        err = " **ERR**" if evt["is_error"] else ""
        tk_in_s = ""
        cache_s = ""
        tk_out_s = ""
        if evt["tokens"]:
            t = evt["tokens"]
            tk_in = t["input"] + t["cache_read"] + t["cache_write"]
            cache_pct = (t["cache_read"] / tk_in * 100) if tk_in > 0 else 0
            tk_in_s = fmt_tokens(tk_in)
            cache_s = f"{cache_pct:.0f}%"
            tk_out_s = fmt_tokens(t["output"])
        time_str = evt["local_ts"].strftime("%H:%M:%S")
        print(
            f"  {time_str}  [{evt['type']:9s}]  {delta:>9}  {tk_in_s:>7}  {cache_s:>6}  {tk_out_s:>7}  {evt['summary'][:50]}{err}"
        )
        prev_ts = evt["ts"]

    print("  " + "-" * 108)
    print()


def print_multi_summary(all_analyses):
    """Print aggregated summary across multiple sessions."""
    total_dur = sum(a["total_duration"] for a in all_analyses)
    total_claude = sum(a["claude_work_secs"] for a in all_analyses)
    total_errors = sum(a["total_errors"] for a in all_analyses)
    total_phases = sum(len(a["phases"]) for a in all_analyses)
    agg_tokens = {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0}
    agg_tools = defaultdict(int)
    agg_models = defaultdict(
        lambda: {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0, "calls": 0}
    )
    for a in all_analyses:
        for k in agg_tokens:
            agg_tokens[k] += a["total_tokens"][k]
        for t, c in a["total_tools"].items():
            agg_tools[t] += c
        for model, mtk in a.get("model_stats", {}).items():
            for k in agg_tokens:
                agg_models[model][k] += mtk[k]
            agg_models[model]["calls"] += mtk.get("calls", 0)

    total_cost = 0
    for model, mtk in agg_models.items():
        total_cost += calc_cost_for_model(
            model, mtk["input"], mtk["cache_read"], mtk["cache_write"], mtk["output"]
        )

    print()
    print("=" * 78)
    print(f"  RESUMEN AGREGADO — {len(all_analyses)} sesiones")
    print("=" * 78)
    print()
    print(
        f"  Periodo:         {all_analyses[0]['session_start'].strftime('%Y-%m-%d')} a {all_analyses[-1]['session_end'].strftime('%Y-%m-%d')}"
    )
    print(f"  Sesiones:        {len(all_analyses)}")
    print(f"  Instrucciones:   {total_phases}")
    print(f"  Duracion total:  {fmt_dur(total_dur)}")
    print(f"  Trabajo Claude:  {fmt_dur(total_claude)}")
    print(f"  Errores API:     {total_errors}")
    print(f"  Costo estimado:  ${total_cost:.2f} USD")
    print()

    # Per-model breakdown
    if agg_models:
        print("  CONSUMO POR MODELO")
        print("  " + "-" * 74)
        print(
            f"  {'Modelo':20s}  {'Calls':>5}  {'Input':>8}  {'CacheR':>8}  {'CacheW':>8}  {'Output':>8}  {'Costo':>9}"
        )
        print("  " + "-" * 74)
        for model in sorted(agg_models.keys()):
            mtk = agg_models[model]
            p = get_pricing(model)
            c = calc_cost_for_model(
                model,
                mtk["input"],
                mtk["cache_read"],
                mtk["cache_write"],
                mtk["output"],
            )
            print(
                f"  {p['label']:20s}  {mtk['calls']:>5}  {fmt_tokens(mtk['input']):>8}  {fmt_tokens(mtk['cache_read']):>8}  "
                f"{fmt_tokens(mtk['cache_write']):>8}  {fmt_tokens(mtk['output']):>8}  ${c:>8.2f}"
            )
        print("  " + "-" * 74)
        print(
            f"  {'TOTAL':20s}  {sum(m['calls'] for m in agg_models.values()):>5}  "
            f"{fmt_tokens(agg_tokens['input']):>8}  {fmt_tokens(agg_tokens['cache_read']):>8}  "
            f"{fmt_tokens(agg_tokens['cache_write']):>8}  {fmt_tokens(agg_tokens['output']):>8}  ${total_cost:>8.2f}"
        )
    print()

    total_tool_count = sum(agg_tools.values())
    tool_legend = {
        "Read": "Leer archivos",
        "Write": "Crear archivos",
        "Edit": "Modificar archivos",
        "Bash": "Comandos terminal",
        "Grep": "Buscar texto",
        "Glob": "Buscar archivos",
        "Task": "Sub-agentes",
        "Web": "Busquedas web",
        "Ask": "Preguntas",
    }
    print(f"  HERRAMIENTAS: {total_tool_count} total")
    print("  " + "-" * 50)
    for tn in ["Read", "Write", "Edit", "Bash", "Grep", "Glob", "Task", "Web", "Ask"]:
        c = agg_tools.get(tn, 0)
        if c == 0:
            continue
        bar = "#" * min(c, 40)
        print(f"    {tn:6s}  {c:>4}  {bar}  {tool_legend.get(tn, '')}")
    for tn, c in sorted(agg_tools.items()):
        if tn not in tool_legend:
            print(f"    {tn:6s}  {c:>4}  {'#' * min(c, 40)}")
    print()


def print_csv(analysis):
    if not analysis:
        return
    print(
        "phase,start,end,duration_secs,claude_secs,tools,reads,writes,edits,bash,grep,glob,"
        "tokens_in,tokens_cache_r,tokens_cache_w,tokens_out,cost_usd,errors,prompt"
    )
    for i, phase in enumerate(analysis["phases"]):
        dur = (phase["end"] - phase["start"]).total_seconds()
        claude_dur = 0
        if phase["assistant_events"]:
            claude_dur = (
                phase["assistant_events"][-1]["ts"] - phase["assistant_events"][0]["ts"]
            ).total_seconds()
        tc = phase["tool_counts"]
        tk = phase["tokens"]
        # Use first model found in phase, or fallback
        phase_model = "unknown"
        for mt_model in phase["model_tokens"]:
            phase_model = mt_model
            break
        cost = calc_cost_for_model(
            phase_model, tk["input"], tk["cache_read"], tk["cache_write"], tk["output"]
        )
        prompt = phase["prompt"].replace('"', '""')[:80]
        s = phase["local_start"].strftime("%Y-%m-%d %H:%M:%S")
        e = phase["local_end"].strftime("%Y-%m-%d %H:%M:%S")
        print(
            f'{i+1},"{s}","{e}",{dur:.0f},{claude_dur:.0f},'
            f'{sum(tc.values())},{tc.get("Read",0)},{tc.get("Write",0)},'
            f'{tc.get("Edit",0)},{tc.get("Bash",0)},{tc.get("Grep",0)},{tc.get("Glob",0)},'
            f'{tk["input"]},{tk["cache_read"]},{tk["cache_write"]},{tk["output"]},'
            f'{cost:.2f},{phase["errors"]},"{prompt}"'
        )


def print_json_export(analysis):
    if not analysis:
        return
    ms = analysis.get("model_stats", {})
    total_cost = (
        sum(
            calc_cost_for_model(
                m, mt["input"], mt["cache_read"], mt["cache_write"], mt["output"]
            )
            for m, mt in ms.items()
        )
        if ms
        else 0
    )
    model_export = {}
    for m, mt in ms.items():
        p = get_pricing(m)
        model_export[p["label"]] = {
            "model_id": m,
            "calls": mt["calls"],
            "tokens": {
                k: mt[k] for k in ("input", "cache_read", "cache_write", "output")
            },
            "cost_usd": calc_cost_for_model(
                m, mt["input"], mt["cache_read"], mt["cache_write"], mt["output"]
            ),
        }
    export = {
        "session_start": analysis["session_start"].isoformat(),
        "session_end": analysis["session_end"].isoformat(),
        "total_duration_secs": analysis["total_duration"],
        "claude_work_secs": analysis["claude_work_secs"],
        "total_errors": analysis["total_errors"],
        "total_tokens": analysis["total_tokens"],
        "total_tools": analysis["total_tools"],
        "models": model_export,
        "cost_usd": total_cost,
        "phases": [],
    }
    for i, phase in enumerate(analysis["phases"]):
        dur = (phase["end"] - phase["start"]).total_seconds()
        claude_dur = 0
        if phase["assistant_events"]:
            claude_dur = (
                phase["assistant_events"][-1]["ts"] - phase["assistant_events"][0]["ts"]
            ).total_seconds()
        tc = phase["tool_counts"]
        export["phases"].append(
            {
                "number": i + 1,
                "start": phase["local_start"].isoformat(),
                "end": phase["local_end"].isoformat(),
                "duration_secs": dur,
                "claude_work_secs": claude_dur,
                "tools": dict(tc),
                "tokens": phase["tokens"],
                "errors": phase["errors"],
                "prompt": phase["prompt"][:120],
            }
        )
    print(json.dumps(export, indent=2, ensure_ascii=False))


# ─── PROJECT / SESSION DISCOVERY ───────────────────────────────────


def find_project_dir(query=None):
    if not CLAUDE_DIR.exists():
        print(f"Error: {CLAUDE_DIR} no existe.", file=sys.stderr)
        sys.exit(1)

    if query:
        matches = [
            d
            for d in CLAUDE_DIR.iterdir()
            if d.is_dir() and query.lower() in d.name.lower()
        ]
        if not matches:
            print(f"No se encontro proyecto con '{query}'.", file=sys.stderr)
            sys.exit(1)
        real = [
            m
            for m in matches
            if "-worktrees-" not in m.name and "--claude-" not in m.name
        ]
        if len(real) == 1:
            return real[0]
        remaining = real if real else matches
        if len(remaining) > 1:
            print(f"Multiples coincidencias para '{query}':")
            for i, m in enumerate(remaining):
                wt = " (worktree interno)" if "-worktrees-" in m.name else ""
                print(f"  [{i+1}] {m.name}{wt}")
            print("Usa un nombre mas especifico o --list para ver todos.")
            sys.exit(1)
        return remaining[0]
    else:
        cwd = os.getcwd()
        cwd_slug = cwd.replace("/", "-")
        if not cwd_slug.startswith("-"):
            cwd_slug = "-" + cwd_slug
        exact = CLAUDE_DIR / cwd_slug
        if exact.exists():
            return exact
        cwd_norm = cwd_slug.replace(" ", "").lower()
        for d in CLAUDE_DIR.iterdir():
            if d.is_dir() and cwd_norm == d.name.replace(" ", "").lower():
                return d
        cwd_parts = [p for p in cwd.split("/") if p]
        for d in CLAUDE_DIR.iterdir():
            if d.is_dir():
                dn = d.name.lower().replace("-", "")
                if all(cp.replace(" ", "").lower() in dn for cp in cwd_parts[-3:]):
                    return d
        last_dir = Path(cwd).name.replace(" ", "")
        for d in CLAUDE_DIR.iterdir():
            if d.is_dir() and last_dir.lower() in d.name.lower().replace("-", ""):
                return d
        print(f"Proyecto no encontrado para: {cwd}", file=sys.stderr)
        print("Usa --list o --project NOMBRE.", file=sys.stderr)
        sys.exit(1)


def list_projects():
    if not CLAUDE_DIR.exists():
        print("No se encontraron proyectos.")
        return
    print()
    print("  PROYECTOS DE CLAUDE CODE")
    print("  " + "=" * 68)
    for d in sorted(CLAUDE_DIR.iterdir()):
        if not d.is_dir():
            continue
        sessions = [
            s
            for s in d.glob("*.jsonl")
            if "subagent" not in str(s) and "agent-" not in s.name
        ]
        if not sessions:
            continue
        is_wt = "-worktrees-" in d.name
        name = d.name.replace("-", "/")
        size = sum(s.stat().st_size for s in sessions)
        latest = datetime.fromtimestamp(max(s.stat().st_mtime for s in sessions))
        wt_tag = " [worktree]" if is_wt else ""
        print(f"  {name}{wt_tag}")
        print(
            f"    Sesiones: {len(sessions)}  |  {size//1024}KB  |  Ultima: {latest.strftime('%Y-%m-%d %H:%M')}"
        )
        print()


def get_sessions(project_dir, days=None):
    sessions = [
        s
        for s in project_dir.glob("*.jsonl")
        if "subagent" not in str(s) and "agent-" not in s.name
    ]
    sessions.sort(key=lambda s: s.stat().st_mtime)
    if days is not None and days > 0:
        cutoff = datetime.now().timestamp() - days * 86400
        sessions = [s for s in sessions if s.stat().st_mtime >= cutoff]
    return sessions


# ─── MAIN ──────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Extrae tiempos de trabajo y consumo de tokens de Claude Code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--list", action="store_true", help="Lista todos los proyectos")
    parser.add_argument("--project", "-p", type=str, help="Buscar proyecto por nombre")
    parser.add_argument("--session", "-s", type=str, help="ID de sesion especifica")
    parser.add_argument("--all", "-a", action="store_true", help="Todas las sesiones")
    parser.add_argument("--days", type=int, help="Filtrar ultimos N dias (0=todo)")
    parser.add_argument(
        "--detail", "-d", action="store_true", help="Timeline detallado"
    )
    parser.add_argument("--csv", action="store_true", help="Exportar CSV")
    parser.add_argument("--json", action="store_true", help="Exportar JSON")
    args = parser.parse_args()

    if args.list:
        list_projects()
        return

    project_dir = find_project_dir(args.project)

    # Determine which sessions to load
    days_filter = None
    if args.days is not None:
        days_filter = args.days if args.days > 0 else None  # 0 = all
        sessions = get_sessions(project_dir, days_filter)
    elif args.all:
        sessions = get_sessions(project_dir)
    else:
        sessions = get_sessions(project_dir)

    if not sessions:
        print("No se encontraron sesiones.")
        return

    if args.session:
        matches = [s for s in sessions if args.session in s.stem]
        if not matches:
            print(f"Sesion '{args.session}' no encontrada. Disponibles:")
            for s in sessions:
                print(f"  {s.stem}")
            return
        sessions = matches

    # Default: only last session unless --all or --days specified
    if not args.all and args.days is None and not args.session:
        sessions = [sessions[-1]]

    # Analyze all sessions
    all_analyses = []
    for sf in sessions:
        msgs = load_session(sf)
        analysis = analyze_session(msgs)
        if analysis:
            all_analyses.append(analysis)

    if not all_analyses:
        print("No hay datos para mostrar.")
        return

    # Output
    if args.csv:
        for a in all_analyses:
            print_csv(a)
    elif args.json:
        for a in all_analyses:
            print_json_export(a)
    else:
        for a in all_analyses:
            name = ""
            for sf in sessions:
                name = sf.stem[:12] + "..."
            print_summary(a, name)
            if args.detail:
                print_detail(a)

        if len(all_analyses) > 1:
            print_multi_summary(all_analyses)


def run():
    main()


if __name__ == "__main__":
    run()
