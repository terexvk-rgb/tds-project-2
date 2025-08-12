# app/tools.py
import io, base64, requests, duckdb,os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup
from typing import Any, Dict

# ---------- web scrape ----------
def web_scrape(url: str) -> str:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.text

# ---------- read uploaded bytes ----------
def read_uploaded_file(file_bytes: bytes, filename: str):
    name = filename.lower()
    buf = io.BytesIO(file_bytes)
    if name.endswith(".csv"):
        return pd.read_csv(buf)
    if name.endswith(".parquet"):
        return pd.read_parquet(buf)
    if name.endswith(".json"):
        return pd.read_json(buf, lines=True) if is_lines(buf) else pd.read_json(buf)
    # fallback try
    try:
        buf.seek(0)
        return pd.read_csv(buf)
    except Exception:
        return None

def is_lines(bio):
    try:
        bio.seek(0)
        line = bio.readline()
        return b'\n' in line or b'{' in line
    except:
        return False

# ---------- duckdb query (local or S3 via httpfs) ----------
def duckdb_query(sql: str, httpfs_config: Dict[str,str]=None):
    """
    Run DuckDB SQL and return a pandas DataFrame.
    If S3/HTTPFS is needed, user should include INSTALL httpfs; LOAD httpfs; and query read_parquet('s3://...') in SQL.
    """
    con = duckdb.connect()  # ephemeral
    try:
        # if httpfs_config provided, set environment (not required if SQL uses parameters in URL)
        if httpfs_config:
            # Example usage not included for credentials; rely on open/public access or environment
            pass
        df = con.execute(sql).df()
        return df
    finally:
        con.close()

# ---------- safe correlation ----------
def safe_correlation(df: pd.DataFrame, a: str, b: str):
    if a not in df.columns or b not in df.columns:
        raise KeyError(f"Columns not found: {a}, {b}")
    s = df[[a,b]].dropna()
    if len(s) < 2:
        raise ValueError("Not enough data points for correlation")
    return float(s[a].corr(s[b]))

# ---------- plotting helper (ensures under size limit) ----------
def plot_scatter_with_regression(df, xcol, ycol, dotted=True, color_line='red', max_size_bytes=100000):
    x = pd.to_numeric(df[xcol], errors='coerce').dropna()
    y = pd.to_numeric(df[ycol], errors='coerce').dropna()
    # align indices
    joined = pd.concat([x, y], axis=1).dropna()
    if len(joined) < 2:
        raise ValueError("Not enough points to plot")
    x = joined.iloc[:,0].astype(float)
    y = joined.iloc[:,1].astype(float)

    coeffs = np.polyfit(x, y, 1)
    m, c = float(coeffs[0]), float(coeffs[1])

    fig, ax = plt.subplots(figsize=(6,4), dpi=100)
    ax.scatter(x, y)
    ax.set_xlabel(str(xcol))
    ax.set_ylabel(str(ycol))
    ax.set_title(f"{xcol} vs {ycol}")
    xs = np.linspace(x.min(), x.max(), 100)
    if dotted:
        ax.plot(xs, m*xs + c, linestyle=':', color=color_line)
    else:
        ax.plot(xs, m*xs + c, color=color_line)

    buf = io.BytesIO()
    # iterative compression
    for dpi_try in [150,120,100,80,60,40]:
        buf.seek(0); buf.truncate(0)
        fig.savefig(buf, format='png', dpi=dpi_try, bbox_inches='tight', optimize=True)
        if buf.tell() <= max_size_bytes:
            break
    if buf.tell() > max_size_bytes:
        for scale in [0.9,0.8,0.7,0.6,0.5]:
            buf.seek(0); buf.truncate(0)
            fig.set_size_inches(6*scale, 4*scale)
            fig.savefig(buf, format='png', dpi=80, bbox_inches='tight', optimize=True)
            if buf.tell() <= max_size_bytes:
                break

    data = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{data}", m, c, buf.tell()
