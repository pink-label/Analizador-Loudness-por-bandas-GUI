#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================
ANALIZADOR DE LOUDNESS POR BANDAS DE FRECUENCIA
=========================================================
Software con interfaz gráfica basado en los scripts de
análisis del proyecto "Entendiendo el fonograma".

Pipeline completo:
  1) Análisis por canción: LUFS integrado, LRA (EBU Tech 3342)
     y True Peak, global y por bandas ISO.
  2) Promedios del grupo con media recortada configurable.
  3) Gráfico raincloud del conjunto de canciones.
  4) Pestaña extra: comparación de varios CSV de promedios
     (raincloud con nombres / leyenda).

Todo el resultado gráfico se exporta a UN SOLO PDF
multipágina. Los datos se guardan además en CSV.

Estética unificada: todas las páginas de gráficos tienen
exactamente el mismo tamaño y la misma área de ploteo
(los ejes ocupan siempre el mismo rectángulo, con el
espacio de la barra de colores reservado aunque esté
oculta), para poder comparar gráficas superponiéndolas.

LRA: se calcula igual que pyloudnorm (EBU Tech 3342) pero
extrayendo además los percentiles absolutos P10 y P95, de
modo que la barra de LRA pueda graficarse en su posición
real de la medición y no centrada sobre el LUFS integrado.

Formatos soportados nativamente (libsndfile >= 1.2):
  WAV, FLAC, OGG/Vorbis, Opus, MP3, AIFF, CAF, W64, etc.
Fallback automático a ffmpeg (si está disponible) para
formatos no soportados (m4a, aac, wma...).
=========================================================
"""

import os
import sys
import csv
import glob
import queue
import shutil
import tempfile
import threading
import traceback
import subprocess
import datetime

import numpy as np
import pandas as pd
import soundfile as sf
import pyloudnorm as pyln
from scipy import stats
from scipy.signal import butter, sosfilt, resample_poly
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use("Agg")  # render sin pantalla; la GUI es tkinter puro
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages

APP_NAME = "Analizador de Loudness por Bandas"
APP_VERSION = "1.1"

# =========================================================
# BANDAS ISO POR DEFECTO (editables desde la interfaz)
# (f_low, fc, f_high)
# =========================================================
DEFAULT_BANDS = [
    (22.09708691,   31.25,   44.19417382),
    (44.19417382,   62.5,    88.38834765),
    (88.38834765,   125,     176.7766953),
    (176.7766953,   250,     353.5533906),
    (353.5533906,   500,     707.1067812),
    (707.1067812,   1000,    1414.213562),
    (1414.213562,   2000,    2828.427125),
    (2828.427125,   4000,    5656.854249),
    (5656.854249,   8000,    11313.7085),
    (11313.7085,    16000,   20000.0),
]

AUDIO_EXTS = (".wav", ".flac", ".aif", ".aiff", ".aifc", ".mp3", ".ogg",
              ".oga", ".opus", ".caf", ".w64", ".au", ".m4a", ".aac",
              ".wma", ".mp4", ".wv")

CSV_COLUMNS = [
    "song", "samplerate",
    "global_lufs", "global_lra", "global_lra_low", "global_lra_high", "global_tp",
    "band_fc_hz",
    "band_lufs", "band_lra", "band_lra_low", "band_lra_high", "band_tp",
]


# =========================================================
# CARGA DE AUDIO
# soundfile nativo + fallback a ffmpeg para formatos raros
# =========================================================
def _find_ffmpeg():
    """Busca ffmpeg en el PATH o junto al ejecutable/script."""
    exe_dir = os.path.dirname(os.path.abspath(
        sys.executable if getattr(sys, "frozen", False) else __file__))
    candidates = [
        os.path.join(exe_dir, "ffmpeg.exe"),
        os.path.join(exe_dir, "ffmpeg"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return shutil.which("ffmpeg")


def load_audio(path):
    """
    Devuelve (audio, sr) con audio de forma (samples, channels), float.
    Intenta soundfile; si el formato no es soportado prueba con ffmpeg.
    """
    try:
        audio, sr = sf.read(path, always_2d=True, dtype="float64")
        return audio, sr
    except Exception as e_sf:
        ffmpeg = _find_ffmpeg()
        if not ffmpeg:
            raise RuntimeError(
                f"No se pudo leer '{os.path.basename(path)}' con libsndfile "
                f"({e_sf}). Para formatos como m4a/aac/wma colocá ffmpeg.exe "
                f"junto al programa o en el PATH."
            )
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            creationflags = 0x08000000 if os.name == "nt" else 0  # sin consola
            result = subprocess.run(
                [ffmpeg, "-y", "-i", path, "-map", "0:a:0",
                 "-c:a", "pcm_f32le", tmp.name],
                capture_output=True, creationflags=creationflags
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg no pudo decodificar '{os.path.basename(path)}': "
                    f"{result.stderr.decode(errors='replace')[-300:]}"
                )
            audio, sr = sf.read(tmp.name, always_2d=True, dtype="float64")
            return audio, sr
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


# =========================================================
# TRUE PEAK
# Sobremuestreo con resample_poly para estimar True Peak.
# =========================================================
def true_peak_dbfs(x, oversample=4):
    if x.ndim == 2:
        x = np.max(np.abs(x), axis=1)
    x_os = resample_poly(x, oversample, 1)
    peak = np.max(np.abs(x_os))
    return 20 * np.log10(peak + 1e-12)


# =========================================================
# FILTRO PASA-BANDA BUTTERWORTH (-3 dB)
# =========================================================
def bandpass_filter_3db(audio, sr, f_low, f_high, order=16):
    nyq = sr / 2.0
    f_high = min(f_high, nyq * 0.999)
    if f_low >= f_high:
        return None
    sos = butter(order, [f_low, f_high], btype="band", fs=sr, output="sos")
    return sosfilt(sos, audio, axis=0)


# =========================================================
# LRA ABSOLUTO (EBU TECH 3342)
# Réplica del algoritmo de pyloudnorm.Meter.loudness_range
# pero devolviendo también los percentiles absolutos:
#   LRA = P95 - P10 (idéntico al valor de pyloudnorm)
#   (lra, p10, p95)
# =========================================================
def loudness_range_absolute(meter, data):
    orig_bs = meter.block_size
    orig_ov = meter.overlap
    try:
        meter.block_size = 3.0
        meter.overlap = 0.97
        data2 = meter._append_silence(data, silence_duration_sec=1.5)
        meter.integrated_loudness(data2)
        stl = getattr(meter, "blockwise_loudness", None)
        if stl is None or len(stl) == 0:
            return (np.nan, np.nan, np.nan)

        ABS_THRES, REL_THRES = -70.0, -20.0
        abs_gated = [x for x in stl if x >= ABS_THRES]
        if not abs_gated:
            return (np.nan, np.nan, np.nan)

        n = len(abs_gated)
        stl_power = np.sum(np.power(10.0, np.divide(abs_gated, 10.0))) / n
        stl_integrated = 10 * np.log10(stl_power)
        rel_gated = [x for x in abs_gated if x >= stl_integrated + REL_THRES]
        if not rel_gated:
            return (np.nan, np.nan, np.nan)

        p_low = float(np.percentile(rel_gated, 10))
        p_high = float(np.percentile(rel_gated, 95))
        return (p_high - p_low, p_low, p_high)
    except Exception:
        # Fallback: usar el LRA estándar sin posición absoluta
        try:
            meter.block_size = orig_bs
            meter.overlap = orig_ov
            lra = meter.loudness_range(data)
            return (lra, np.nan, np.nan)
        except Exception:
            return (np.nan, np.nan, np.nan)
    finally:
        meter.block_size = orig_bs
        meter.overlap = orig_ov


def _measure(meter, audio, oversample):
    """LUFS, LRA (+P10/P95 absolutos) y TP de una señal."""
    lufs = meter.integrated_loudness(audio)
    lra, lra_low, lra_high = loudness_range_absolute(meter, audio)
    tp = true_peak_dbfs(audio, oversample)
    return {"LUFS": lufs, "LRA": lra,
            "LRA_LOW": lra_low, "LRA_HIGH": lra_high, "TP": tp}


# =========================================================
# ANALISIS DE UN ARCHIVO
# =========================================================
def analyze_file(audio_path, params, log=print):
    """
    Devuelve dict con:
      song, sr, global (dict), bands {fc: dict}
    """
    song_name = os.path.basename(audio_path)
    audio, sr = load_audio(audio_path)

    dur = audio.shape[0] / sr
    if dur < 1.0:
        raise RuntimeError(f"'{song_name}' es demasiado corto ({dur:.2f} s).")

    meter = pyln.Meter(sr)
    oversample = params["oversample"]

    g = _measure(meter, audio, oversample)

    bands = {}
    for f_low, fc, f_high in params["bands"]:
        band_audio = bandpass_filter_3db(audio, sr, f_low, f_high,
                                         order=params["order"])
        if band_audio is None:
            log(f"   · Banda {fc:g} Hz omitida (fuera del rango de Nyquist "
                f"para sr={sr}).")
            continue
        bands[fc] = _measure(meter, band_audio, oversample)

    return {"song": song_name, "sr": sr, "global": g, "bands": bands}


# =========================================================
# CSV
# =========================================================
def _fmt(v):
    if v is None:
        return ""
    try:
        if not np.isfinite(v):
            return ""
        return round(float(v), 4)
    except (TypeError, ValueError):
        return ""


def write_analysis_csv(csv_path, results):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for r in results:
            g = r["global"]
            for fc, m in sorted(r["bands"].items()):
                w.writerow([
                    r["song"], r["sr"],
                    _fmt(g["LUFS"]), _fmt(g["LRA"]),
                    _fmt(g["LRA_LOW"]), _fmt(g["LRA_HIGH"]), _fmt(g["TP"]),
                    fc,
                    _fmt(m["LUFS"]), _fmt(m["LRA"]),
                    _fmt(m["LRA_LOW"]), _fmt(m["LRA_HIGH"]), _fmt(m["TP"]),
                ])


def results_to_dataframe(results):
    rows = []
    for r in results:
        g = r["global"]
        for fc, m in sorted(r["bands"].items()):
            rows.append({
                "song": r["song"], "samplerate": r["sr"],
                "global_lufs": g["LUFS"], "global_lra": g["LRA"],
                "global_lra_low": g["LRA_LOW"], "global_lra_high": g["LRA_HIGH"],
                "global_tp": g["TP"],
                "band_fc_hz": float(fc),
                "band_lufs": m["LUFS"], "band_lra": m["LRA"],
                "band_lra_low": m["LRA_LOW"], "band_lra_high": m["LRA_HIGH"],
                "band_tp": m["TP"],
            })
    df = pd.DataFrame(rows)
    for col in df.columns:
        if col not in ("song",):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    return df


def load_csv_dataframe(path):
    """Carga un CSV (formato nuevo o el viejo sin lra_low/high)."""
    df = pd.read_csv(path)
    for col in ["global_lra_low", "global_lra_high",
                "band_lra_low", "band_lra_high"]:
        if col not in df.columns:
            df[col] = np.nan
    df["band_fc_hz"] = pd.to_numeric(df["band_fc_hz"], errors="coerce")
    for col in df.columns:
        if col != "song":
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    return df


# =========================================================
# MEDIA RECORTADA
# Recorta la fracción indicada en cada extremo (0 a 0.25)
# antes de promediar, para eliminar valores extremos.
# =========================================================
def trimmed_mean(series, trim):
    clean = pd.Series(series).dropna().values
    if len(clean) == 0:
        return np.nan
    if trim <= 0 or len(clean) < 3:
        return float(np.mean(clean))
    return float(stats.trim_mean(clean, trim))


def compute_averages(df, trim):
    """Promedios (media recortada) por banda y globales."""
    grouped = df.groupby("band_fc_hz")
    freqs = np.array(sorted(grouped.groups.keys()))

    def col_means(col):
        return np.array([trimmed_mean(grouped[col].get_group(f), trim)
                         for f in freqs])

    out = {
        "freqs": freqs,
        "lufs": col_means("band_lufs"),
        "lra": col_means("band_lra"),
        "lra_low": col_means("band_lra_low"),
        "lra_high": col_means("band_lra_high"),
        "tp": col_means("band_tp"),
    }

    per_song = df.groupby("song")[
        ["global_lufs", "global_lra", "global_lra_low",
         "global_lra_high", "global_tp", "samplerate"]
    ].first()
    out["g_lufs"] = trimmed_mean(per_song["global_lufs"], trim)
    out["g_lra"] = trimmed_mean(per_song["global_lra"], trim)
    out["g_lra_low"] = trimmed_mean(per_song["global_lra_low"], trim)
    out["g_lra_high"] = trimmed_mean(per_song["global_lra_high"], trim)
    out["g_tp"] = trimmed_mean(per_song["global_tp"], trim)
    sr_mode = per_song["samplerate"].mode()
    out["g_sr"] = int(sr_mode[0]) if len(sr_mode) else 0
    out["n_songs"] = len(per_song)
    return out


def write_averages_csv(csv_path, avg, group_name):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for i, fc in enumerate(avg["freqs"]):
            w.writerow([
                f"PROMEDIO_{group_name}", avg["g_sr"],
                _fmt(avg["g_lufs"]), _fmt(avg["g_lra"]),
                _fmt(avg["g_lra_low"]), _fmt(avg["g_lra_high"]),
                _fmt(avg["g_tp"]),
                fc,
                _fmt(avg["lufs"][i]), _fmt(avg["lra"][i]),
                _fmt(avg["lra_low"][i]), _fmt(avg["lra_high"][i]),
                _fmt(avg["tp"][i]),
            ])


# =========================================================
# ESTILO GRAFICO UNIFICADO
# Todas las páginas de gráficos comparten:
#   - mismo tamaño de figura (FIG_W x FIG_H)
#   - mismo rectángulo de ejes (AXES_RECT), con el espacio
#     de la barra de colores SIEMPRE reservado a la derecha
#     (se muestre o no), para que el área de ploteo sea
#     idéntica en píxeles en todas las páginas.
#   - misma paleta y grosores (estética estilo raincloud:
#     líneas finas, puntos con borde blanco, mediana cálida)
# =========================================================
FIG_W, FIG_H = 11, 6
AXES_RECT = [0.070, 0.095, 0.820, 0.795]   # [izq, abajo, ancho, alto]
CBAR_RECT = [0.905, 0.095, 0.016, 0.795]

COL_LINE   = "#4C72B0"   # línea/puntos LUFS
COL_LRA    = "#4C72B0"   # barra LRA
COL_TP     = "#E76F51"   # marcadores True Peak
COL_MEDIAN = "#F4A261"   # mediana en rainclouds
COL_CLOUD  = "steelblue" # nube KDE
COL_ACCENT = "#4285F4"   # acento general (cajas, portada)
COL_BOX_BG = "#EAF1FB"   # fondo caja de info global

Y_MIN, Y_MAX = -70, 10
X_MIN, X_MAX = 20, 20000


def new_fig():
    """Figura + ejes con geometría y estilo unificados."""
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes(AXES_RECT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="both", alpha=0.30, linewidth=0.5)
    ax.tick_params(labelsize=8)
    return fig, ax


def _finite(v):
    return v is not None and np.isfinite(v)


def _lra_bar_limits(lufs, lra, lra_low, lra_high, absolute):
    """Extremos de la barra de LRA según el modo elegido."""
    if absolute and _finite(lra_low) and _finite(lra_high):
        return lra_low, lra_high
    if _finite(lufs) and _finite(lra):
        return lufs - lra / 2.0, lufs + lra / 2.0
    return None, None


def _draw_band_plot(ax, freqs, lufs, lra, lra_low, lra_high, tp,
                    absolute_lra, label_prefix=""):
    """Gráfico de bandas (estética raincloud: trazos finos)."""
    fl, ll, tl = [], [], []
    for i, f in enumerate(freqs):
        b, t = _lra_bar_limits(lufs[i], lra[i], lra_low[i], lra_high[i],
                               absolute_lra)
        if b is not None:
            ax.vlines(f, b, t, linewidth=4, alpha=0.45, color=COL_LRA,
                      zorder=3)
        if _finite(lufs[i]):
            fl.append(f); ll.append(lufs[i]); tl.append(tp[i])

    if fl:
        ax.semilogx(fl, ll, "-", linewidth=1.1, color=COL_LINE,
                    alpha=0.8, zorder=4)
        ax.scatter(fl, ll, s=36, color=COL_LINE, zorder=5,
                   edgecolors="white", linewidths=0.4,
                   label=f"LUFS {label_prefix}".strip())
        tp_f = [f for f, t in zip(fl, tl) if _finite(t)]
        tp_v = [t for t in tl if _finite(t)]
        ax.scatter(tp_f, tp_v, s=42, marker="^", color=COL_TP, zorder=5,
                   edgecolors="white", linewidths=0.4,
                   label=f"True Peak {label_prefix}".strip())

    for i, f in enumerate(freqs):
        if _finite(lufs[i]):
            ax.text(f, lufs[i] + 1.5, f"{lufs[i]:.1f}", ha="center",
                    va="bottom", fontsize=7.5,
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor="white", edgecolor="#bbbbbb",
                              linewidth=0.4, alpha=0.75))
        if _finite(lra[i]) and _finite(lufs[i]):
            ax.text(f, lufs[i] - 2.5, f"LRA:{lra[i]:.1f}", ha="center",
                    va="top", fontsize=6.5, color="#444444")
        if _finite(tp[i]):
            ax.text(f, tp[i] + 1.0, f"{tp[i]:.1f}", ha="center",
                    va="bottom", fontsize=7.5, color=COL_TP)

    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_xlabel("Frecuencia (Hz)", fontsize=9)
    ax.set_ylabel("Nivel (LUFS / dBTP)", fontsize=9)


def _global_box(ax, title, lufs, lra, tp):
    txt = (f"{title}\n"
           f"LUFS: {lufs:.2f}\n"
           f"LRA:  {lra:.2f}\n"
           f"TP:   {tp:.2f} dBTP")
    ax.text(0.015, 0.97, txt, transform=ax.transAxes, fontsize=8.5,
            verticalalignment="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.45", facecolor=COL_BOX_BG,
                      edgecolor=COL_ACCENT, linewidth=1.0, alpha=0.92))


def fig_song(result, absolute_lra):
    fig, ax = new_fig()
    bands = result["bands"]
    freqs = sorted(bands.keys())
    lufs = [bands[f]["LUFS"] for f in freqs]
    lra = [bands[f]["LRA"] for f in freqs]
    lra_lo = [bands[f]["LRA_LOW"] for f in freqs]
    lra_hi = [bands[f]["LRA_HIGH"] for f in freqs]
    tp = [bands[f]["TP"] for f in freqs]

    _draw_band_plot(ax, freqs, lufs, lra, lra_lo, lra_hi, tp,
                    absolute_lra, "banda")
    g = result["global"]
    _global_box(ax, "GLOBAL", g["LUFS"], g["LRA"], g["TP"])

    modo = "LRA absoluto (P10–P95)" if absolute_lra else "LRA centrado en LUFS"
    ax.set_title(f"LUFS por banda + LRA — {modo}\n{result['song']}",
                 fontsize=10)
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.85)
    return fig


def fig_promedio(avg, group_name, trim, absolute_lra):
    fig, ax = new_fig()
    _draw_band_plot(ax, list(avg["freqs"]), list(avg["lufs"]),
                    list(avg["lra"]), list(avg["lra_low"]),
                    list(avg["lra_high"]), list(avg["tp"]),
                    absolute_lra, "promedio")
    _global_box(ax, f"PROMEDIO GLOBAL ({avg['n_songs']} canciones)",
                avg["g_lufs"], avg["g_lra"], avg["g_tp"])
    modo = "LRA absoluto (P10–P95)" if absolute_lra else "LRA centrado en LUFS"
    ax.set_title(f"Promedio de Loudness por Banda — {group_name}\n"
                 f"(media recortada ±{trim * 100:.1f}%)  ·  {modo}",
                 fontsize=10)
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.85)
    return fig


def _names_legend(ax, names, color_map, title=None):
    """Leyenda con los nombres de cada canción/serie."""
    patches = [
        mpatches.Patch(color=color_map[n], alpha=0.9,
                       label=(str(n).replace("PROMEDIO_", "")[:42]))
        for n in names
    ]
    ncol = 1 if len(names) <= 8 else (2 if len(names) <= 22 else 3)
    ax.legend(handles=patches, loc="lower left",
              fontsize=6.5, ncol=ncol,
              title=title, title_fontsize=7, framealpha=0.85)


def fig_raincloud(df, title, show_names=False, show_colorbar=True):
    """Raincloud del conjunto (puntos coloreados por Global LUFS)."""
    songs = list(df["song"].unique())
    freqs = sorted(df["band_fc_hz"].dropna().unique())
    n_songs = len(songs)

    per_song_lufs = df.groupby("song")["global_lufs"].first().reindex(songs)
    lufs_min, lufs_max = per_song_lufs.min(), per_song_lufs.max()

    def lufs_to_color(v):
        if np.isnan(v) or lufs_max == lufs_min:
            return plt.cm.plasma(0.5)
        return plt.cm.plasma((v - lufs_min) / (lufs_max - lufs_min))

    color_map = {s: lufs_to_color(per_song_lufs[s]) for s in songs}

    fig, ax = new_fig()

    VIOLIN_MAX_FACTOR = 0.55
    DOT_X_FACTOR = 0.82
    BOX_HALF_W_FACTOR = 0.05

    rng = np.random.RandomState(42)
    song_x_jitter = {s: rng.uniform(-0.018, 0.018) for s in songs}

    for fc in freqs:
        subset = df[df["band_fc_hz"] == fc]
        vals = subset["band_lufs"].dropna().values
        if len(vals) < 2:
            continue

        y_min, y_max = vals.min(), vals.max()
        y_pad = max((y_max - y_min) * 0.5, 3.0)

        # Nube (KDE) a la derecha
        try:
            y_range = np.linspace(y_min - y_pad, y_max + y_pad, 300)
            kde = gaussian_kde(vals, bw_method=0.4)
            density = kde(y_range)
            density_norm = density / density.max()
            x_violin = fc * (1 + VIOLIN_MAX_FACTOR * density_norm)
            ax.fill_betweenx(y_range, fc, x_violin, alpha=0.20,
                             color=COL_CLOUD)
            ax.plot(x_violin, y_range, color=COL_CLOUD, alpha=0.45,
                    linewidth=0.9)
        except Exception:
            pass  # valores idénticos → KDE singular: se omite la nube

        # Caja (IQR) centrada en fc
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        iqr = q3 - q1
        w_lo = max(y_min, q1 - 1.5 * iqr)
        w_hi = min(y_max, q3 + 1.5 * iqr)
        bw = fc * BOX_HALF_W_FACTOR
        ax.vlines(fc, w_lo, w_hi, color="black", linewidth=1.0, zorder=4)
        rect = mpatches.FancyBboxPatch((fc - bw, q1), 2 * bw, iqr,
                                       boxstyle="square,pad=0",
                                       facecolor="white", edgecolor="black",
                                       linewidth=1.2, zorder=5)
        ax.add_patch(rect)
        ax.hlines(med, fc - bw, fc + bw, color=COL_MEDIAN, linewidth=2.5,
                  zorder=6)

        # Lluvia: puntos con x fija por canción
        for _, row in subset.iterrows():
            if not np.isnan(row["band_lufs"]):
                x_dot = fc * DOT_X_FACTOR + fc * song_x_jitter[row["song"]]
                ax.scatter(x_dot, row["band_lufs"],
                           color=color_map[row["song"]],
                           s=28 if n_songs > 20 else 40,
                           zorder=7, alpha=0.85,
                           edgecolors="white", linewidths=0.3)

    # Líneas por canción conectando los puntos
    for song in songs:
        sub = df[df["song"] == song].sort_values("band_fc_hz")
        sub_v = sub[~sub["band_lufs"].isna()]
        if len(sub_v) < 2:
            continue
        x_pts = sub_v["band_fc_hz"] * DOT_X_FACTOR \
            + sub_v["band_fc_hz"] * song_x_jitter[song]
        ax.plot(x_pts, sub_v["band_lufs"], color=color_map[song],
                linewidth=0.8, alpha=0.55, zorder=6)

    per_song = df.groupby("song")[["global_lufs", "global_lra",
                                   "global_tp"]].first()
    _global_box(ax, f"PROMEDIO ({n_songs} canciones)",
                per_song["global_lufs"].mean(),
                per_song["global_lra"].mean(),
                per_song["global_tp"].mean())

    if show_colorbar:
        cax = fig.add_axes(CBAR_RECT)
        sm = cm.ScalarMappable(
            cmap="plasma",
            norm=plt.Normalize(vmin=lufs_min, vmax=lufs_max))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Global LUFS por canción", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    if show_names:
        _names_legend(ax, songs, color_map)

    ax.set_xscale("log")
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel("Frecuencia (Hz)", fontsize=9)
    ax.set_ylabel("Nivel (LUFS)", fontsize=9)
    ax.set_title(
        f"LUFS por banda — {title}  ({n_songs} canciones)\n"
        "Nube = KDE  ·  Caja = IQR + mediana  ·  "
        "Puntos = cada canción (color = Global LUFS)", fontsize=9)
    return fig


def fig_raincloud_nombres(df, title, show_names=True):
    """Raincloud comparativo (cada 'song' = una serie de color)."""
    periodos = list(df["song"].unique())
    freqs = sorted(df["band_fc_hz"].dropna().unique())
    n_per = len(periodos)

    cmap = plt.cm.turbo
    if n_per > 1:
        colors = [cmap(i / (n_per - 1)) for i in range(n_per)]
    else:
        colors = [cmap(0.5)]
    color_map = dict(zip(periodos, colors))

    fig, ax = new_fig()

    VIOLIN_MAX_FACTOR = 0.55
    DOT_LEFT_FACTOR = 0.02
    BOX_HALF_W_FACTOR = 0.05

    for fc in freqs:
        subset = df[df["band_fc_hz"] == fc].sort_values("song")
        vals = subset["band_lufs"].dropna().values
        if len(vals) < 2:
            continue

        y_min, y_max = vals.min(), vals.max()
        y_pad = max((y_max - y_min) * 0.5, 3.0)

        try:
            y_range = np.linspace(y_min - y_pad, y_max + y_pad, 300)
            kde = gaussian_kde(vals, bw_method=0.6)
            density = kde(y_range)
            density_norm = density / density.max()
            x_violin = fc * (1 + VIOLIN_MAX_FACTOR * density_norm)
            ax.fill_betweenx(y_range, fc, x_violin, alpha=0.40,
                             color=COL_CLOUD)
            ax.plot(x_violin, y_range, color=COL_CLOUD, alpha=0.45,
                    linewidth=0.9)
        except Exception:
            pass

        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        iqr = q3 - q1
        w_lo = max(y_min, q1 - 1.5 * iqr)
        w_hi = min(y_max, q3 + 1.5 * iqr)
        bw = fc * BOX_HALF_W_FACTOR
        ax.vlines(fc, w_lo, w_hi, color="black", linewidth=1.0, zorder=4)
        rect = mpatches.FancyBboxPatch((fc - bw, q1), 2 * bw, iqr,
                                       boxstyle="square,pad=0",
                                       facecolor="white", edgecolor="black",
                                       linewidth=1.2, zorder=5)
        ax.add_patch(rect)
        ax.hlines(med, fc - bw, fc + bw, color=COL_MEDIAN, linewidth=2.5,
                  zorder=6)

        n = len(vals)
        x_dots = fc * np.linspace(1 - DOT_LEFT_FACTOR, 1 - 0.06, n)
        rng = np.random.RandomState(42)
        x_dots = x_dots + fc * rng.uniform(-0.015, 0.015, n)

        j = 0
        for _, row in subset.iterrows():
            if not np.isnan(row["band_lufs"]):
                ax.scatter(x_dots[j], row["band_lufs"],
                           color=color_map[row["song"]],
                           s=36, zorder=7, alpha=0.92,
                           edgecolors="white", linewidths=0.4)
                j += 1

    for periodo in periodos:
        sub = df[df["song"] == periodo].sort_values("band_fc_hz")
        valid = ~sub["band_lufs"].isna()
        ax.plot(sub.loc[valid, "band_fc_hz"], sub.loc[valid, "band_lufs"],
                "-", color=color_map[periodo], linewidth=1.0, alpha=0.45)

    per_song = df.groupby("song")[["global_lufs", "global_lra",
                                   "global_tp"]].first()
    _global_box(ax, "PROMEDIO TOTAL",
                per_song["global_lufs"].mean(),
                per_song["global_lra"].mean(),
                per_song["global_tp"].mean())

    if show_names:
        _names_legend(ax, periodos, color_map, title=title)

    ax.set_xscale("log")
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel("Frecuencia (Hz)", fontsize=9)
    ax.set_ylabel("Nivel (LUFS)", fontsize=9)
    ax.set_title(f"LUFS por banda — {title}\n"
                 "Nube = distribución KDE  ·  Caja = IQR + mediana",
                 fontsize=9)
    return fig


def fig_portada(group_name, results, params, absolute_lra):
    """Portada del PDF: parámetros + tabla resumen (mismo tamaño de página)."""
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # Banda de color superior
    fig.patches.append(mpatches.Rectangle(
        (0, 0.86), 1, 0.14, transform=fig.transFigure,
        facecolor=COL_ACCENT, zorder=0))
    fig.text(0.5, 0.945, "Informe de Loudness por Bandas de Frecuencia",
             color="white", fontsize=15, fontweight="bold",
             ha="center", va="center")
    fig.text(0.5, 0.885, group_name, color="white", fontsize=10,
             ha="center", va="center", style="italic")

    fecha = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    modo = "Absoluto (P10–P95)" if absolute_lra else "Centrado en LUFS"
    info = (
        f"Fecha: {fecha}        Canciones: {len(results)}\n"
        f"Media recortada: ±{params['trim'] * 100:.1f}%   ·   "
        f"Filtro Butterworth orden {params['order']}   ·   "
        f"True Peak x{params['oversample']}   ·   "
        f"{len(params['bands'])} bandas   ·   LRA: {modo}\n"
        f"Métricas: LUFS integrado (BS.1770) · LRA (EBU Tech 3342) · "
        f"True Peak (sobremuestreo)   ·   "
        f"{APP_NAME} v{APP_VERSION}"
    )
    fig.text(0.06, 0.825, info, fontsize=8, va="top", family="monospace",
             color="#333333", linespacing=1.7)

    # Tabla resumen
    ax = fig.add_axes([0.05, 0.03, 0.90, 0.62])
    ax.axis("off")

    max_rows = 14
    shown = results[:max_rows]
    col_labels = ["Canción", "SR (Hz)", "LUFS", "LRA",
                  "LRA P10", "LRA P95", "TP (dBTP)"]
    cell_text = []
    for r in shown:
        g = r["global"]
        name = r["song"] if len(r["song"]) <= 48 else r["song"][:45] + "..."

        def s(v, nd=2):
            return f"{v:.{nd}f}" if _finite(v) else "—"
        cell_text.append([name, str(r["sr"]), s(g["LUFS"]), s(g["LRA"]),
                          s(g["LRA_LOW"]), s(g["LRA_HIGH"]), s(g["TP"])])

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     loc="upper center", cellLoc="center",
                     colWidths=[0.38, 0.09, 0.09, 0.09, 0.10, 0.10, 0.11])
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.3)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row == 0:
            cell.set_facecolor(COL_ACCENT)
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F4F8FE")
        if col == 0:
            cell.set_text_props(ha="left")
            cell.PAD = 0.02

    if len(results) > max_rows:
        ax.text(0.5, 0.0,
                f"(+{len(results) - max_rows} canciones más; "
                f"ver CSV para el detalle completo)",
                ha="center", fontsize=8, transform=ax.transAxes,
                color="#555555")
    return fig


# =========================================================
# PIPELINE DE ANALISIS COMPLETO
# =========================================================
def _unique_path(path):
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 1
    while os.path.exists(f"{base}_{i}{ext}"):
        i += 1
    return f"{base}_{i}{ext}"


def run_pipeline(files, outdir, group_name, params, options,
                 log=print, progress=None, cancel=None):
    """
    Ejecuta análisis completo y genera:
      datos_<grupo>.csv, promedios_<grupo>.csv, informe_<grupo>.pdf
    Devuelve la ruta del PDF.
    """
    results = []
    errors = []
    total = len(files)

    for i, path in enumerate(files):
        if cancel is not None and cancel.is_set():
            log("✖ Análisis cancelado por el usuario.")
            return None
        log(f"[{i + 1}/{total}] Analizando: {os.path.basename(path)}")
        try:
            results.append(analyze_file(path, params, log))
            log("   ✔ OK")
        except Exception as e:
            errors.append((os.path.basename(path), str(e)))
            log(f"   ✖ Error: {e}")
        if progress:
            progress((i + 1) / (total + 1))

    if not results:
        raise RuntimeError("Ningún archivo pudo analizarse correctamente.")

    safe_group = "".join(c if c.isalnum() or c in "-_ " else "_"
                         for c in group_name).strip().replace(" ", "_")
    df = results_to_dataframe(results)

    csv_datos = csv_promedios = None
    if options.get("save_csv", True):
        csv_datos = _unique_path(os.path.join(outdir,
                                              f"datos_{safe_group}.csv"))
        write_analysis_csv(csv_datos, results)
        log(f"✔ CSV de datos: {os.path.basename(csv_datos)}")

    avg = compute_averages(df, params["trim"])
    if options.get("save_csv", True):
        csv_promedios = _unique_path(
            os.path.join(outdir, f"promedios_{safe_group}.csv"))
        write_averages_csv(csv_promedios, avg, safe_group)
        log(f"✔ CSV de promedios: {os.path.basename(csv_promedios)}")

    pdf_path = _unique_path(os.path.join(outdir,
                                         f"informe_{safe_group}.pdf"))
    absolute_lra = options.get("absolute_lra", True)

    log("Generando PDF…")
    with PdfPages(pdf_path) as pdf:
        fig = fig_portada(group_name, results, params, absolute_lra)
        pdf.savefig(fig); plt.close(fig)

        if options.get("page_per_song", True):
            for r in results:
                fig = fig_song(r, absolute_lra)
                pdf.savefig(fig); plt.close(fig)

        if options.get("page_average", True):
            fig = fig_promedio(avg, group_name, params["trim"], absolute_lra)
            pdf.savefig(fig); plt.close(fig)

        if options.get("page_raincloud", True) and len(results) >= 2:
            fig = fig_raincloud(
                df, group_name,
                show_names=options.get("rain_names", False),
                show_colorbar=options.get("rain_colorbar", True))
            pdf.savefig(fig); plt.close(fig)
        elif options.get("page_raincloud", True):
            log("· Raincloud omitido (se necesitan al menos 2 canciones).")

        meta = pdf.infodict()
        meta["Title"] = f"Informe de loudness — {group_name}"
        meta["Creator"] = f"{APP_NAME} v{APP_VERSION}"

    if progress:
        progress(1.0)
    log(f"✔ PDF generado: {os.path.basename(pdf_path)}")
    if errors:
        log(f"⚠ {len(errors)} archivo(s) con error:")
        for name, msg in errors:
            log(f"   - {name}: {msg}")
    return pdf_path


def run_comparison(csv_files, outdir, comp_name, show_names=True, log=print):
    """Combina varios CSV y genera el raincloud comparativo en PDF."""
    dfs = []
    for p in csv_files:
        log(f"Leyendo: {os.path.basename(p)}")
        dfs.append(load_csv_dataframe(p))
    df = pd.concat(dfs, ignore_index=True)

    safe = "".join(c if c.isalnum() or c in "-_ " else "_"
                   for c in comp_name).strip().replace(" ", "_")

    csv_out = _unique_path(os.path.join(outdir, f"combinado_{safe}.csv"))
    df.to_csv(csv_out, index=False, encoding="utf-8")
    log(f"✔ CSV combinado: {os.path.basename(csv_out)}")

    pdf_path = _unique_path(os.path.join(outdir, f"comparacion_{safe}.pdf"))
    with PdfPages(pdf_path) as pdf:
        fig = fig_raincloud_nombres(df, comp_name, show_names=show_names)
        pdf.savefig(fig); plt.close(fig)
        meta = pdf.infodict()
        meta["Title"] = f"Comparación — {comp_name}"
        meta["Creator"] = f"{APP_NAME} v{APP_VERSION}"
    log(f"✔ PDF generado: {os.path.basename(pdf_path)}")
    return pdf_path


# =========================================================
# INTERFAZ GRAFICA (tkinter)
# =========================================================
def launch_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    ACCENT = COL_ACCENT
    ACCENT_DARK = "#2F6AD9"
    BG_HEADER = "#1E3A5F"

    root = tk.Tk()
    root.title(f"{APP_NAME}  v{APP_VERSION}")
    root.geometry("900x720")
    root.minsize(780, 620)

    style = ttk.Style()
    try:
        style.theme_use("vista" if os.name == "nt" else "clam")
    except tk.TclError:
        pass
    style.configure("TLabelframe.Label", foreground=BG_HEADER,
                    font=("Segoe UI", 9, "bold"))
    style.configure("TNotebook.Tab", padding=(14, 6))

    log_queue = queue.Queue()
    cancel_event = threading.Event()
    custom_bands = {"bands": list(DEFAULT_BANDS)}

    # --- Encabezado con onda ---
    header = tk.Frame(root, bg=BG_HEADER)
    header.pack(fill="x")
    tk.Label(header, text="♪  Analizador de Loudness por Bandas",
             bg=BG_HEADER, fg="white",
             font=("Segoe UI", 14, "bold")).pack(anchor="w",
                                                 padx=14, pady=(10, 0))
    tk.Label(header,
             text="LUFS · LRA (EBU Tech 3342) · True Peak — informe PDF "
                  "multipágina por bandas de frecuencia",
             bg=BG_HEADER, fg="#A8C4E8",
             font=("Segoe UI", 9)).pack(anchor="w", padx=14, pady=(0, 10))

    def accent_button(parent, text, command=None):
        return tk.Button(parent, text=text, command=command,
                         bg=ACCENT, fg="white",
                         activebackground=ACCENT_DARK,
                         activeforeground="white",
                         font=("Segoe UI", 10, "bold"),
                         relief="flat", padx=18, pady=4,
                         cursor="hand2",
                         disabledforeground="#cfdcf5")

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=8, pady=8)

    # =====================================================
    # PESTAÑA 1: ANALISIS
    # =====================================================
    tab1 = ttk.Frame(notebook)
    notebook.add(tab1, text="  Análisis  ")

    # --- Selección de archivos ---
    frm_files = ttk.LabelFrame(tab1, text=" Archivos de audio ")
    frm_files.pack(fill="both", expand=True, padx=8, pady=(8, 4))

    lst_frame = ttk.Frame(frm_files)
    lst_frame.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
    listbox = tk.Listbox(lst_frame, selectmode="extended", height=8)
    sb = ttk.Scrollbar(lst_frame, orient="vertical", command=listbox.yview)
    listbox.configure(yscrollcommand=sb.set)
    listbox.pack(side="left", fill="both", expand=True)
    sb.pack(side="right", fill="y")

    file_paths = []

    def add_files():
        types = [("Audio", " ".join(f"*{e}" for e in AUDIO_EXTS)),
                 ("Todos los archivos", "*.*")]
        for p in filedialog.askopenfilenames(title="Seleccionar archivos",
                                             filetypes=types):
            if p not in file_paths:
                file_paths.append(p)
                listbox.insert("end", os.path.basename(p))
        update_default_names()

    def add_folder():
        d = filedialog.askdirectory(title="Seleccionar carpeta")
        if not d:
            return
        found = 0
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(AUDIO_EXTS):
                p = os.path.join(d, f)
                if p not in file_paths:
                    file_paths.append(p)
                    listbox.insert("end", f)
                    found += 1
        if found == 0:
            messagebox.showinfo("Sin audios",
                                "No se encontraron archivos de audio "
                                "compatibles en esa carpeta.")
        update_default_names()

    def remove_selected():
        for i in reversed(listbox.curselection()):
            listbox.delete(i)
            del file_paths[i]
        update_default_names()

    def clear_all():
        listbox.delete(0, "end")
        file_paths.clear()
        update_default_names()

    btns = ttk.Frame(frm_files)
    btns.pack(side="right", fill="y", padx=6, pady=6)
    ttk.Button(btns, text="Agregar archivos…",
               command=add_files).pack(fill="x", pady=2)
    ttk.Button(btns, text="Agregar carpeta…",
               command=add_folder).pack(fill="x", pady=2)
    ttk.Button(btns, text="Quitar selección",
               command=remove_selected).pack(fill="x", pady=2)
    ttk.Button(btns, text="Limpiar todo",
               command=clear_all).pack(fill="x", pady=2)

    # --- Parámetros ---
    frm_par = ttk.LabelFrame(tab1, text=" Parámetros ")
    frm_par.pack(fill="x", padx=8, pady=4)

    var_trim = tk.DoubleVar(value=5.0)
    var_order = tk.IntVar(value=16)
    var_oversample = tk.IntVar(value=4)
    var_abs_lra = tk.BooleanVar(value=True)
    var_pg_song = tk.BooleanVar(value=True)
    var_pg_avg = tk.BooleanVar(value=True)
    var_pg_rain = tk.BooleanVar(value=True)
    var_csv = tk.BooleanVar(value=True)
    var_rain_names = tk.BooleanVar(value=False)
    var_rain_cbar = tk.BooleanVar(value=True)

    row1 = ttk.Frame(frm_par); row1.pack(fill="x", padx=6, pady=(6, 2))
    ttk.Label(row1, text="Media recortada (%):").pack(side="left")
    ttk.Spinbox(row1, from_=0, to=25, increment=0.5, width=6,
                textvariable=var_trim).pack(side="left", padx=(4, 16))
    ttk.Label(row1, text="Orden del filtro:").pack(side="left")
    ttk.Spinbox(row1, from_=2, to=32, increment=2, width=5,
                textvariable=var_order).pack(side="left", padx=(4, 16))
    ttk.Label(row1, text="Oversample TP:").pack(side="left")
    ttk.Spinbox(row1, from_=2, to=16, increment=1, width=5,
                textvariable=var_oversample).pack(side="left", padx=(4, 16))

    def edit_bands():
        win = tk.Toplevel(root)
        win.title("Editar bandas (f_low  fc  f_high)")
        win.geometry("420x360")
        win.transient(root)
        txt = tk.Text(win, font=("Consolas", 10))
        txt.pack(fill="both", expand=True, padx=8, pady=8)
        for fl, fc, fh in custom_bands["bands"]:
            txt.insert("end", f"{fl:.6f}\t{fc:g}\t{fh:.6f}\n")

        def restore():
            txt.delete("1.0", "end")
            for fl, fc, fh in DEFAULT_BANDS:
                txt.insert("end", f"{fl:.6f}\t{fc:g}\t{fh:.6f}\n")

        def save():
            new_bands = []
            for ln, line in enumerate(txt.get("1.0", "end").splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.replace(",", " ").split()
                if len(parts) != 3:
                    messagebox.showerror(
                        "Error", f"Línea {ln}: se esperan 3 valores "
                        f"(f_low, fc, f_high).", parent=win)
                    return
                try:
                    fl, fc, fh = (float(x) for x in parts)
                except ValueError:
                    messagebox.showerror("Error",
                                         f"Línea {ln}: valores no numéricos.",
                                         parent=win)
                    return
                if not (0 < fl < fc < fh):
                    messagebox.showerror(
                        "Error", f"Línea {ln}: debe cumplirse "
                        f"0 < f_low < fc < f_high.", parent=win)
                    return
                new_bands.append((fl, fc, fh))
            if not new_bands:
                messagebox.showerror("Error", "No hay bandas definidas.",
                                     parent=win)
                return
            custom_bands["bands"] = new_bands
            lbl_bands.config(text=f"Bandas: {len(new_bands)}")
            win.destroy()

        bb = ttk.Frame(win); bb.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(bb, text="Restaurar ISO por defecto",
                   command=restore).pack(side="left")
        ttk.Button(bb, text="Guardar", command=save).pack(side="right")
        ttk.Button(bb, text="Cancelar",
                   command=win.destroy).pack(side="right", padx=4)

    ttk.Button(row1, text="Editar bandas…",
               command=edit_bands).pack(side="left")
    lbl_bands = ttk.Label(row1, text=f"Bandas: {len(DEFAULT_BANDS)}")
    lbl_bands.pack(side="left", padx=6)

    row2 = ttk.Frame(frm_par); row2.pack(fill="x", padx=6, pady=2)
    ttk.Checkbutton(row2, text="LRA en posición absoluta (P10–P95)",
                    variable=var_abs_lra).pack(side="left", padx=(0, 14))
    ttk.Checkbutton(row2, text="Página por canción",
                    variable=var_pg_song).pack(side="left", padx=(0, 14))
    ttk.Checkbutton(row2, text="Promedios",
                    variable=var_pg_avg).pack(side="left", padx=(0, 14))
    ttk.Checkbutton(row2, text="Raincloud",
                    variable=var_pg_rain).pack(side="left", padx=(0, 14))
    ttk.Checkbutton(row2, text="Guardar CSV",
                    variable=var_csv).pack(side="left")

    row3 = ttk.Frame(frm_par); row3.pack(fill="x", padx=6, pady=(2, 6))
    ttk.Label(row3, text="Opciones del raincloud:").pack(side="left",
                                                         padx=(0, 8))
    ttk.Checkbutton(row3, text="Nombres de canciones (leyenda)",
                    variable=var_rain_names).pack(side="left", padx=(0, 14))
    ttk.Checkbutton(row3, text="Escala de colores (Global LUFS)",
                    variable=var_rain_cbar).pack(side="left")

    # --- Salida ---
    frm_out = ttk.LabelFrame(tab1, text=" Salida ")
    frm_out.pack(fill="x", padx=8, pady=4)

    var_group = tk.StringVar()
    var_outdir = tk.StringVar()

    def update_default_names():
        if file_paths and not var_group.get():
            var_group.set(os.path.basename(os.path.dirname(file_paths[0]))
                          or "analisis")
        if file_paths and not var_outdir.get():
            var_outdir.set(os.path.dirname(file_paths[0]))

    ro1 = ttk.Frame(frm_out); ro1.pack(fill="x", padx=6, pady=(6, 2))
    ttk.Label(ro1, text="Nombre del grupo:").pack(side="left")
    ttk.Entry(ro1, textvariable=var_group, width=34).pack(side="left", padx=4)

    ro2 = ttk.Frame(frm_out); ro2.pack(fill="x", padx=6, pady=(2, 6))
    ttk.Label(ro2, text="Carpeta de salida:").pack(side="left")
    ttk.Entry(ro2, textvariable=var_outdir).pack(side="left", padx=4,
                                                 fill="x", expand=True)
    ttk.Button(ro2, text="Examinar…",
               command=lambda: var_outdir.set(
                   filedialog.askdirectory() or var_outdir.get())
               ).pack(side="left")

    # --- Ejecución / progreso / log ---
    frm_run = ttk.Frame(tab1)
    frm_run.pack(fill="x", padx=8, pady=4)
    progress_var = tk.DoubleVar(value=0.0)
    pb = ttk.Progressbar(frm_run, variable=progress_var, maximum=1.0)
    pb.pack(side="left", fill="x", expand=True, padx=(0, 8))

    btn_run = accent_button(frm_run, "▶  ANALIZAR")
    btn_run.pack(side="left")
    btn_cancel = ttk.Button(frm_run, text="Cancelar", state="disabled",
                            command=lambda: cancel_event.set())
    btn_cancel.pack(side="left", padx=4)

    frm_log = ttk.LabelFrame(tab1, text=" Registro ")
    frm_log.pack(fill="both", expand=True, padx=8, pady=(4, 8))
    txt_log = tk.Text(frm_log, height=9, state="disabled",
                      font=("Consolas", 9), bg="#FAFBFD")
    sb2 = ttk.Scrollbar(frm_log, orient="vertical", command=txt_log.yview)
    txt_log.configure(yscrollcommand=sb2.set)
    txt_log.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
    sb2.pack(side="right", fill="y")

    def gui_log(msg):
        log_queue.put(str(msg))

    def poll_log():
        try:
            while True:
                msg = log_queue.get_nowait()
                txt_log.configure(state="normal")
                txt_log.insert("end", msg + "\n")
                txt_log.see("end")
                txt_log.configure(state="disabled")
        except queue.Empty:
            pass
        root.after(150, poll_log)

    def set_running(running):
        btn_run.configure(state="disabled" if running else "normal")
        btn_cancel.configure(state="normal" if running else "disabled")
        btn_run2.configure(state="disabled" if running else "normal")

    def start_analysis():
        if not file_paths:
            messagebox.showwarning("Sin archivos",
                                   "Agregá archivos o una carpeta primero.")
            return
        outdir = var_outdir.get().strip()
        if not outdir or not os.path.isdir(outdir):
            messagebox.showwarning("Carpeta inválida",
                                   "Elegí una carpeta de salida válida.")
            return
        group = var_group.get().strip() or "analisis"
        try:
            trim = float(var_trim.get()) / 100.0
            order = int(var_order.get())
            oversample = int(var_oversample.get())
        except (tk.TclError, ValueError):
            messagebox.showwarning("Parámetros",
                                   "Revisá los valores numéricos.")
            return
        trim = min(max(trim, 0.0), 0.25)

        params = {"trim": trim, "order": order, "oversample": oversample,
                  "bands": list(custom_bands["bands"])}
        options = {"absolute_lra": var_abs_lra.get(),
                   "page_per_song": var_pg_song.get(),
                   "page_average": var_pg_avg.get(),
                   "page_raincloud": var_pg_rain.get(),
                   "save_csv": var_csv.get(),
                   "rain_names": var_rain_names.get(),
                   "rain_colorbar": var_rain_cbar.get()}

        cancel_event.clear()
        progress_var.set(0.0)
        set_running(True)
        gui_log("=" * 55)
        gui_log(f"Inicio del análisis — grupo '{group}' "
                f"({len(file_paths)} archivos)")

        def progress_cb(v):
            root.after(0, lambda: progress_var.set(v))

        def work():
            try:
                pdf = run_pipeline(list(file_paths), outdir, group, params,
                                   options, log=gui_log,
                                   progress=progress_cb, cancel=cancel_event)
                if pdf:
                    gui_log("Terminado correctamente.")
                    root.after(0, lambda: messagebox.showinfo(
                        "Listo", f"Informe generado:\n{pdf}"))
            except Exception as e:
                gui_log(f"✖ ERROR: {e}")
                gui_log(traceback.format_exc(limit=3))
                root.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                root.after(0, lambda: set_running(False))

        threading.Thread(target=work, daemon=True).start()

    btn_run.configure(command=start_analysis)

    # =====================================================
    # PESTAÑA 2: COMPARAR PROMEDIOS
    # =====================================================
    tab2 = ttk.Frame(notebook)
    notebook.add(tab2, text="  Comparar promedios  ")

    frm_csv = ttk.LabelFrame(
        tab2, text=" Archivos CSV (de promedios o de datos) ")
    frm_csv.pack(fill="both", expand=True, padx=8, pady=(8, 4))

    lst2_frame = ttk.Frame(frm_csv)
    lst2_frame.pack(side="left", fill="both", expand=True,
                    padx=(6, 0), pady=6)
    listbox2 = tk.Listbox(lst2_frame, selectmode="extended", height=8)
    sb3 = ttk.Scrollbar(lst2_frame, orient="vertical",
                        command=listbox2.yview)
    listbox2.configure(yscrollcommand=sb3.set)
    listbox2.pack(side="left", fill="both", expand=True)
    sb3.pack(side="right", fill="y")

    csv_paths = []

    def add_csvs():
        for p in filedialog.askopenfilenames(
                title="Seleccionar CSV", filetypes=[("CSV", "*.csv")]):
            if p not in csv_paths:
                csv_paths.append(p)
                listbox2.insert("end", os.path.basename(p))
        if csv_paths and not var_outdir2.get():
            var_outdir2.set(os.path.dirname(csv_paths[0]))

    def remove_csvs():
        for i in reversed(listbox2.curselection()):
            listbox2.delete(i)
            del csv_paths[i]

    btns2 = ttk.Frame(frm_csv)
    btns2.pack(side="right", fill="y", padx=6, pady=6)
    ttk.Button(btns2, text="Agregar CSV…",
               command=add_csvs).pack(fill="x", pady=2)
    ttk.Button(btns2, text="Quitar selección",
               command=remove_csvs).pack(fill="x", pady=2)
    ttk.Button(btns2, text="Limpiar todo",
               command=lambda: (listbox2.delete(0, "end"),
                                csv_paths.clear())).pack(fill="x", pady=2)

    frm_out2 = ttk.LabelFrame(tab2, text=" Salida ")
    frm_out2.pack(fill="x", padx=8, pady=4)
    var_comp = tk.StringVar(value="comparacion")
    var_outdir2 = tk.StringVar()
    var_comp_names = tk.BooleanVar(value=True)

    co1 = ttk.Frame(frm_out2); co1.pack(fill="x", padx=6, pady=(6, 2))
    ttk.Label(co1, text="Nombre de la comparación:").pack(side="left")
    ttk.Entry(co1, textvariable=var_comp, width=30).pack(side="left", padx=4)
    ttk.Checkbutton(co1, text="Leyenda con nombres",
                    variable=var_comp_names).pack(side="left", padx=12)

    co2 = ttk.Frame(frm_out2); co2.pack(fill="x", padx=6, pady=(2, 6))
    ttk.Label(co2, text="Carpeta de salida:").pack(side="left")
    ttk.Entry(co2, textvariable=var_outdir2).pack(side="left", padx=4,
                                                  fill="x", expand=True)
    ttk.Button(co2, text="Examinar…",
               command=lambda: var_outdir2.set(
                   filedialog.askdirectory() or var_outdir2.get())
               ).pack(side="left")

    frm_run2 = ttk.Frame(tab2)
    frm_run2.pack(fill="x", padx=8, pady=(4, 8))
    btn_run2 = accent_button(frm_run2, "▶  GENERAR COMPARACIÓN")
    btn_run2.pack(side="left")
    ttk.Label(frm_run2,
              text="Cada 'song' del CSV se grafica como una serie con su "
                   "color.\nIdeal para combinar archivos promedios_*.csv "
                   "de distintos grupos.",
              foreground="#555").pack(side="left", padx=12)

    def start_comparison():
        if len(csv_paths) < 1:
            messagebox.showwarning("Sin CSV", "Agregá al menos un CSV.")
            return
        outdir = var_outdir2.get().strip()
        if not outdir or not os.path.isdir(outdir):
            messagebox.showwarning("Carpeta inválida",
                                   "Elegí una carpeta de salida válida.")
            return
        comp = var_comp.get().strip() or "comparacion"
        show_names = var_comp_names.get()
        set_running(True)
        gui_log("=" * 55)
        gui_log(f"Comparación '{comp}' ({len(csv_paths)} CSV)")

        def work():
            try:
                pdf = run_comparison(list(csv_paths), outdir, comp,
                                     show_names=show_names, log=gui_log)
                gui_log("Terminado correctamente.")
                root.after(0, lambda: messagebox.showinfo(
                    "Listo", f"Comparación generada:\n{pdf}"))
            except Exception as e:
                gui_log(f"✖ ERROR: {e}")
                root.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                root.after(0, lambda: set_running(False))

        threading.Thread(target=work, daemon=True).start()

    btn_run2.configure(command=start_comparison)

    # Barra de estado
    status = ttk.Label(
        root, anchor="w", foreground="#666",
        text=f"{APP_NAME} v{APP_VERSION}  ·  "
             f"Formatos: WAV, FLAC, OGG, Opus, MP3, AIFF y más"
             + ("  ·  ffmpeg detectado (m4a/aac/wma habilitados)"
                if _find_ffmpeg() else
                "  ·  ffmpeg no detectado (m4a/aac/wma deshabilitados)"))
    status.pack(fill="x", padx=8, pady=(0, 6))

    poll_log()
    gui_log("Listo. Agregá archivos o una carpeta y presioná ANALIZAR.")
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
