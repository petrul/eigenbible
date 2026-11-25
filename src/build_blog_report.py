"""Synthesizes all PCA, kPCA, HDBSCAN, and BERTopic runs across every local FAISS
dataset into a standalone, strict-XML-valid interactive `.xhtml` blog article
(matching scriptorium.ro/blog typography & aesthetic) with live Plotly.js 3D/2D
visualizations, cross-lingual eigenvector alignment heatmaps, and full run tables.
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import KernelPCA, PCA
from umap import UMAP

from eigenbible.bibles import COLLECTION_TO_BIBLE
from eigenbible.chapter_text_resolver import ChapterTextResolver, InlineTextResolver
from eigenbible.kpca_labeler import _row_ref
from eigenbible.label_results_store import JsonlResultsStore
from eigenbible.settings import BIBLIOTECA_BASE_URL
from eigenbible.vector_collection_reader import FaissCollectionReader

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_XHTML = PROJECT_ROOT / "eigenbooks_synthesis.xhtml"

logger = logging.getLogger(__name__)

_DARBY_BOOK_ONLY = {
    "les_psaumes",
    "abdias",
    "epitre_a_philemon",
    "epitre_de_jude",
    "seconde_epitre_de_jean",
    "troisieme_epitre_de_jean",
}

_CORPUS_ROOT_URLS: dict[str, str] = {
    "gesta_hungarorum": f"{BIBLIOTECA_BASE_URL}/trad/gesta_hungarorum_la",
    "ureche_letopisetul": f"{BIBLIOTECA_BASE_URL}/ureche/letopisetul_tarii_moldovei",
    "miron_letopisetul": f"{BIBLIOTECA_BASE_URL}/costin/letopisetul_tarii_moldovei",
    "biblia_ortodoxa_subcapitole": f"{BIBLIOTECA_BASE_URL}/trad/biblia_ortodoxa_ro",
    "biblia_darby_fr_subcapitole": f"{BIBLIOTECA_BASE_URL}/trad/la_bible_ancien_testament_fr",
    "creanga_amintiri": f"{BIBLIOTECA_BASE_URL}/creanga/amintiri_din_copilarie",
    "london_white_fang": f"{BIBLIOTECA_BASE_URL}/london/white_fang",
    "hugo_notre_dame": f"{BIBLIOTECA_BASE_URL}/hugo/notre_dame_de_paris",
    "hugo_notre_dame_en": f"{BIBLIOTECA_BASE_URL}/hugo/notre_dame_de_paris_2",
}


def _row_biblioteca_url(corpus_key: str, row: dict[str, Any]) -> str:
    """Resolve any dataset row to its live biblioteca.scriptorium.ro URL."""
    raw_url = row.get("url")
    if isinstance(raw_url, str) and raw_url.startswith("http"):
        return raw_url

    file_name = str(row.get("file_name", ""))
    parts = file_name.split("/")
    if corpus_key == "biblia_ortodoxa_subcapitole" and len(parts) >= 2:
        testament_map = {"vechiul": "vechiul_testament", "noul": "noul_testament"}
        testament = testament_map.get(parts[0], parts[0])
        book = parts[1]
        return f"{BIBLIOTECA_BASE_URL}/trad/biblia_ortodoxa_ro/{testament}/{book}"

    if corpus_key == "biblia_darby_fr_subcapitole" and len(parts) >= 3:
        opus_map = {
            "ancien": "trad/la_bible_ancien_testament_fr",
            "nouveau": "trad/la_bible_nouveau_testament_fr",
        }
        opus = opus_map.get(parts[0], "trad/la_bible_ancien_testament_fr")
        book = parts[1]
        stem = Path(parts[-1]).stem  # e.g. le_second_livre_des_rois-17
        chap_num = stem.rsplit("-", 1)[-1] if "-" in stem else ""
        if book in _DARBY_BOOK_ONLY or not chap_num.isdigit():
            return f"{BIBLIOTECA_BASE_URL}/{opus}/{book}"
        return f"{BIBLIOTECA_BASE_URL}/{opus}/{book}/chapitre_{chap_num}"

    return _CORPUS_ROOT_URLS.get(corpus_key, BIBLIOTECA_BASE_URL)


def _link_ref(ref: str, ref_urls: dict[str, str], css_class: str = "ref-link") -> str:
    if not ref:
        return ""
    url = ref_urls.get(ref)
    if not url:
        return _esc(ref)
    return (
        f"<a class=\"{css_class}\" href=\"{_esc(url)}\" "
        f"target=\"_blank\" rel=\"noopener noreferrer\">{_esc(ref)}</a>"
    )

# Ordered catalog of works to synthesize, with editorial metadata
CORPUS_CATALOG: list[dict[str, Any]] = [
    {
        "key": "gesta_hungarorum",
        "title": "Gesta Hungarorum",
        "author": "Anonymus (Notarius of King Béla)",
        "lang": "Latin",
        "era": "c. 1200",
        "group": "Chronicles",
        "unit": "paragraphs",
        "color": "#f59e0b",
        "summary": (
            "The earliest surviving narrative chronicle of the Magyar conquest of the Carpathian Basin "
            "(Honfoglalás), composed in medieval Latin around 1200. Its 122 paragraphs span the Scythian "
            "origin myth, Duke Álmos and Árpád's marches, treaties with the Ruthenians and Cumans, "
            "Menumorout in Bihar, and Tuhutum's campaign against Gelou in Transylvania (terra ultra siluana)."
        ),
    },
    {
        "key": "ureche_letopisetul",
        "title": "Letopisețul Țării Moldovei (Ureche)",
        "author": "Grigore Ureche",
        "lang": "Romanian",
        "era": "c. 1642–1647",
        "group": "Chronicles",
        "unit": "paragraphs",
        "color": "#10b981",
        "summary": (
            "The foundational vernacular chronicle of Moldavia (1359–1594), written in vigorous 17th-century "
            "Romanian and preserved with interpolations by Simion Dascălul, Misail Călugărul, and Axinte Uricariul. "
            "Covers the founding of Moldavia, court offices (dregătorii), Stephen the Great's wars, Despot Vodă, "
            "and comparative critiques of Polish and Wallachian chronicles."
        ),
    },
    {
        "key": "miron_letopisetul",
        "title": "Letopisețul Țării Moldovei (Miron Costin)",
        "author": "Miron Costin",
        "lang": "Romanian",
        "era": "1675",
        "group": "Chronicles",
        "unit": "paragraphs",
        "color": "#06b6d4",
        "summary": (
            "Continues Grigore Ureche's chronicle from 1595 to 1661 in a more reflective, humanist, and "
            "rhetorically elaborate register. Interweaves Turco-Tatar and Polish campaigns, Mihai Viteazul, "
            "Vasile Lupu vs. Matei Basarab, Transylvanian alliances, and moral meditations on state decay."
        ),
    },
    {
        "key": "biblia_ortodoxa_subcapitole",
        "title": "Biblia Ortodoxă (Synodal)",
        "author": "Orthodox Canon (Old & New Testament)",
        "lang": "Romanian",
        "era": "Canonical (OT + Deuterocanon + NT)",
        "group": "Scripture",
        "unit": "chapters",
        "color": "#8b5cf6",
        "summary": (
            "Complete Romanian Orthodox Bible embedded at chapter granularity (1,345 chapters across 77 books, "
            "including the Deuterocanonical/Anagignoskomena books). Spans wildly distinct genres: genealogies, "
            "Levitical ritual law, royal chronicles, Psalms, Wisdom literature, Prophets, Gospels, Epistles, and Apocalypse."
        ),
    },
    {
        "key": "biblia_darby_fr_subcapitole",
        "title": "La Bible (Traduction Darby)",
        "author": "J. N. Darby (Old & New Testament)",
        "lang": "French",
        "era": "1859–1885 (66 books)",
        "group": "Scripture",
        "unit": "chapters",
        "color": "#ec4899",
        "summary": (
            "Literal 19th-century French translation of the 66-book Protestant canon (1,173 chapters). Paired with "
            "the Romanian Orthodox Bible in the exact same multilingual bge-m3 space, it provides a direct test of "
            "whether spectral eigen-directions capture deep canonical genre structure across languages."
        ),
    },
    {
        "key": "creanga_amintiri",
        "title": "Amintiri din copilărie",
        "author": "Ion Creangă",
        "lang": "Romanian",
        "era": "1881–1892",
        "group": "Literature",
        "unit": "paragraphs",
        "color": "#f97316",
        "summary": (
            "The quintessential Romanian literary memoir of rural childhood in Humulești (401 paragraphs across "
            "four parts). Blends oral Moldavian folklore, nostalgic village scenes, schoolboy pranks, "
            "Trăsnea's hilariously literal Romanian grammar drills in Fălticeni, and the poignant cart journey to Iași."
        ),
    },
    {
        "key": "london_white_fang",
        "title": "White Fang",
        "author": "Jack London",
        "lang": "English",
        "era": "1906",
        "group": "Literature",
        "unit": "paragraphs",
        "color": "#38bdf8",
        "summary": (
            "A Darwinian wilderness novel (1,069 paragraphs) tracing a wolf-dog's arc from the frozen Yukon "
            "trail (Bill & Henry besieged by wolves), cave cubhood and Kiche, Grey Beaver's camp and sled-pack "
            "dominance, Beauty Smith's brutal bulldog arena, to domestication under Weedon Scott ('the Love Master')."
        ),
    },
    {
        "key": "hugo_notre_dame",
        "title": "Notre-Dame de Paris (Français)",
        "author": "Victor Hugo",
        "lang": "French",
        "era": "1831",
        "group": "Literature",
        "unit": "paragraphs",
        "color": "#a855f7",
        "summary": (
            "Victor Hugo's monumental Gothic novel in its original French (11 books). Famously alternates between "
            "sweeping essays on medieval architecture, Parisian topography, and printing ('Ceci tuera cela') and "
            "high Romantic drama around Quasimodo, Esmeralda, Claude Frollo, Pierre Gringoire, and the Cour des Miracles."
        ),
    },
    {
        "key": "hugo_notre_dame_en",
        "title": "Notre-Dame de Paris (English)",
        "author": "Victor Hugo (English translation)",
        "lang": "English",
        "era": "1831 / 19th-c. trans.",
        "group": "Literature",
        "unit": "paragraphs",
        "color": "#14b8a6",
        "summary": (
            "Full English translation of Victor Hugo's Notre-Dame de Paris from the same Qdrant collection. "
            "Paired directly against the French original in the shared 1024-D bge-m3 embedding space to measure "
            "language invariance of PCA subspaces, scree decay, and BERTopic clusters."
        ),
    },
]


def _esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return JsonlResultsStore.read_all(path)


def _pick_kpca_file(results_dir: Path) -> tuple[str, list[dict[str, Any]]]:
    for fname, label in [
        ("kpca_rbf_g2.jsonl", "RBF kPCA (γ=2.0)"),
        ("kpca_rbf_top_projection.jsonl", "RBF kPCA"),
        ("kpca_cosine_top_projection.jsonl", "Cosine kPCA"),
    ]:
        rows = _read_jsonl_if_exists(results_dir / fname)
        if rows:
            return label, rows
    return "kPCA", []


def _load_corpus_bundle(spec: dict[str, Any]) -> dict[str, Any] | None:
    ds_dir = DATA_DIR / spec["key"]
    if not (ds_dir / "index.faiss").exists():
        logger.warning("Skipping missing dataset: %s", ds_dir)
        return None

    logger.info("Building visualization & summary bundle for '%s' ...", spec["key"])
    reader = FaissCollectionReader(ds_dir)
    ds = reader.fetch_dataset()
    X = ds.vectors
    n, d = X.shape

    has_inline = bool(ds.records and "text" in ds.records[0])
    default_bible = COLLECTION_TO_BIBLE.get(ds.name)
    resolver = InlineTextResolver() if has_inline else ChapterTextResolver(PROJECT_ROOT, default_bible=default_bible)

    # 1. Full top-30 PCA for scree comparison + 3D coords + eigenvectors for cross-lingual alignment
    n_pca = min(30, n - 1, d)
    pca = PCA(n_components=n_pca, random_state=42)
    proj_pca = pca.fit_transform(X)
    evr = [round(float(v), 5) for v in pca.explained_variance_ratio_]

    # 2. Non-linear RBF kPCA (gamma=2.0) 3D coords
    kpca_rbf = KernelPCA(n_components=min(30, n - 1), kernel="rbf", gamma=2.0)
    proj_kpca = kpca_rbf.fit_transform(X)
    kpca_eigs = [round(float(v), 4) for v in kpca_rbf.eigenvalues_]

    # 3. UMAP 3D (cosine) for visual manifold inspection
    eff_neighbors = max(5, min(12, n - 1))
    proj_umap = UMAP(
        n_components=3,
        n_neighbors=eff_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        compatibility_layout=True,
    ).fit_transform(X)

    results_dir = ds_dir / "results"
    pca_rows = _read_jsonl_if_exists(results_dir / "pca_top_projection.jsonl")
    kpca_cosine_rows = _read_jsonl_if_exists(results_dir / "kpca_cosine_top_projection.jsonl")
    kpca_rbf_label, kpca_rows = _pick_kpca_file(results_dir)
    hdbscan_rows = _read_jsonl_if_exists(results_dir / "hdbscan.jsonl")
    bertopic_rows = _read_jsonl_if_exists(results_dir / "bertopic.jsonl")

    # Map every row_idx -> cluster_id and cluster label (from BERTopic, fallback HDBSCAN)
    cluster_source = bertopic_rows or hdbscan_rows
    row_cluster = [-1] * n
    cluster_labels: dict[int, str] = {-1: "Noise / Outliers"}
    cluster_keywords: dict[int, str] = {-1: ""}
    for rec in cluster_source:
        cid = int(rec["cluster_id"])
        cluster_labels[cid] = rec.get("label") or f"Cluster {cid}"
        cluster_keywords[cid] = ", ".join((rec.get("keywords") or [])[:5])
        for idx in rec.get("member_indices") or []:
            if 0 <= idx < n:
                row_cluster[idx] = cid

    # Compact per-point arrays for Plotly (including live biblioteca.scriptorium.ro URLs!)
    refs = [_row_ref(r) for r in ds.records]
    urls = [_row_biblioteca_url(spec["key"], r) for r in ds.records]
    ref_urls = dict(zip(refs, urls))
    excerpts = []
    for r in ds.records:
        try:
            txt = resolver.read_row(r).replace("\n", " ").strip()
        except Exception:
            txt = ""
        excerpts.append((txt[:115] + "…") if len(txt) > 115 else txt)

    # For very large corpora (>2500 pts), keep all points in arrays (rounded to 3 decimals = super compact!)
    def _round_cols(mat: np.ndarray) -> list[list[float]]:
        return [[round(float(mat[i, c]), 3) for i in range(n)] for c in range(min(3, mat.shape[1]))]

    return {
        "spec": spec,
        "n_rows": n,
        "dimension": d,
        "meta": ds.meta,
        "ref_urls": ref_urls,
        "root_url": _CORPUS_ROOT_URLS.get(spec["key"], BIBLIOTECA_BASE_URL),
        "pca_components_matrix": pca.components_[:12],  # (12, 1024) for cross-lingual cosine matrix
        "mean_vector": pca.mean_,
        "evr": evr,
        "kpca_eigs": kpca_eigs,
        "pca_rows": pca_rows,
        "kpca_cosine_rows": kpca_cosine_rows,
        "kpca_label": kpca_rbf_label,
        "kpca_rows": kpca_rows,
        "hdbscan_rows": hdbscan_rows,
        "bertopic_rows": bertopic_rows,
        "viz": {
            "key": spec["key"],
            "title": spec["title"],
            "lang": spec["lang"],
            "n": n,
            "evr": evr,
            "kpca_eigs": kpca_eigs,
            "pca3": _round_cols(proj_pca),
            "kpca3": _round_cols(proj_kpca),
            "umap3": _round_cols(proj_umap),
            "refs": refs,
            "urls": urls,
            "excerpts": excerpts,
            "clusters": row_cluster,
            "cluster_labels": {str(k): v for k, v in cluster_labels.items()},
            "cluster_keywords": {str(k): v for k, v in cluster_keywords.items()},
        },
    }


def _cross_lingual_alignment(bundle_a: dict[str, Any], bundle_b: dict[str, Any], title: str) -> dict[str, Any]:
    """Compute |V_A^T V_B| (8x8 principal axis cosine alignment) + mean-vector cosine."""
    va = bundle_a["pca_components_matrix"]
    vb = bundle_b["pca_components_matrix"]
    sim = np.abs(va @ vb.T)
    ma = bundle_a["mean_vector"]
    mb = bundle_b["mean_vector"]
    mean_cos = float(np.dot(ma, mb) / (np.linalg.norm(ma) * np.linalg.norm(mb)))
    # Singular values of V_A @ V_B.T are the cosines of the canonical/principal angles between the two 8D subspaces!
    principal_cosines = np.linalg.svd(va @ vb.T, compute_uv=False)
    return {
        "title": title,
        "label_a": f"{bundle_a['spec']['title']} ({bundle_a['spec']['lang']})",
        "label_b": f"{bundle_b['spec']['title']} ({bundle_b['spec']['lang']})",
        "matrix": [[round(float(val), 3) for val in row] for row in sim],
        "mean_cosine": round(mean_cos, 4),
        "principal_cosines": [round(float(c), 3) for c in principal_cosines],
        "subspace_overlap_mean": round(float(np.mean(principal_cosines)), 3),
    }


def _render_axes_table(rows: list[dict[str, Any]], heading: str, ref_urls: dict[str, str]) -> str:
    if not rows:
        return ""
    by_comp: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        by_comp.setdefault(int(r["component_index"]), []).append(r)

    trs = []
    for comp_idx in sorted(by_comp):
        entries = {e.get("sign", "+"): e for e in by_comp[comp_idx]}
        pos = entries.get("+", {})
        neg = entries.get("-", {})
        score = float((pos or neg).get("eigenvalue", 0.0))
        score_name = (pos or neg).get("score_name", "eigenvalue")
        score_str = f"{score * 100:.2f}%" if score_name == "explained_variance_ratio" else f"{score:.3f}"
        pos_lbl = _esc(pos.get("label", "—"))
        pos_anc = _link_ref(pos.get("anchor_id", ""), ref_urls)
        pos_nbs = ", ".join(_link_ref(nref, ref_urls, "ref-link-subtle") for nref in (pos.get("neighbour_ids") or [])[1:4])
        neg_lbl = _esc(neg.get("label", "—"))
        neg_anc = _link_ref(neg.get("anchor_id", ""), ref_urls)
        neg_nbs = ", ".join(_link_ref(nref, ref_urls, "ref-link-subtle") for nref in (neg.get("neighbour_ids") or [])[1:4])
        pos_nb_html = f" <span class=\"muted\">· {pos_nbs}</span>" if pos_nbs else ""
        neg_nb_html = f" <span class=\"muted\">· {neg_nbs}</span>" if neg_nbs else ""
        trs.append(
            f"<tr>"
            f"<td class=\"mono accent\">C{comp_idx + 1}</td>"
            f"<td class=\"mono\">{score_str}</td>"
            f"<td><span class=\"pole pos\">+</span> {pos_lbl}<div class=\"anchor\">{pos_anc}{pos_nb_html}</div></td>"
            f"<td><span class=\"pole neg\">−</span> {neg_lbl}<div class=\"anchor\">{neg_anc}{neg_nb_html}</div></td>"
            f"</tr>"
        )
    metric_col = "Variance" if rows[0].get("score_name") == "explained_variance_ratio" else "Eigenvalue"
    return (
        f"<div class=\"result-block\">"
        f"<h4>{_esc(heading)}</h4>"
        f"<div class=\"table-wrap\"><table class=\"data-table\">"
        f"<thead><tr><th>Axis</th><th>{metric_col}</th><th>Positive Pole (+)</th><th>Negative Pole (−)</th></tr></thead>"
        f"<tbody>{''.join(trs)}</tbody>"
        f"</table></div></div>"
    )


def _render_cluster_table(rows: list[dict[str, Any]], heading: str, ref_urls: dict[str, str]) -> str:
    if not rows:
        return ""
    non_noise = [r for r in rows if int(r["cluster_id"]) != -1]
    noise = [r for r in rows if int(r["cluster_id"]) == -1]
    noise_cnt = int(noise[0]["size"]) if noise else 0
    total = sum(int(r["size"]) for r in rows)
    trs = []
    for r in non_noise:
        cid = int(r["cluster_id"])
        size = int(r["size"])
        pct = (100.0 * size / total) if total else 0.0
        lbl = _esc(r.get("label") or "(unlabelled)")
        kws = ", ".join(r.get("keywords") or [])
        reps = r.get("representatives") or []
        rep_links = ", ".join(_link_ref(rep.get("ref", ""), ref_urls) for rep in reps[:4] if rep.get("ref"))
        kw_html = f"<div class=\"kw-line\">{_esc(kws)}</div>" if kws else ""
        trs.append(
            f"<tr>"
            f"<td class=\"mono accent\">#{cid}</td>"
            f"<td class=\"mono\">{size} <span class=\"muted\">({pct:.1f}%)</span></td>"
            f"<td>{lbl}{kw_html}</td>"
            f"<td class=\"mono small-cell\">{rep_links}</td>"
            f"</tr>"
        )
    return (
        f"<div class=\"result-block\">"
        f"<h4>{_esc(heading)} — <span class=\"mono\">{len(non_noise)} clusters, {noise_cnt} noise ({100.0 * noise_cnt / max(1, total):.1f}%)</span></h4>"
        f"<div class=\"table-wrap\"><table class=\"data-table\">"
        f"<thead><tr><th>ID</th><th>Size</th><th>Theme Label &amp; c-TF-IDF Keywords</th><th>Centroid Anchors (biblioteca.scriptorium.ro)</th></tr></thead>"
        f"<tbody>{''.join(trs)}</tbody>"
        f"</table></div></div>"
    )


def _render_corpus_section(bundle: dict[str, Any], idx: int, active: bool = False) -> str:
    spec = bundle["spec"]
    ref_urls = bundle["ref_urls"]
    evr_sum = sum(bundle["evr"][:8]) * 100.0
    n_hdb = sum(1 for r in bundle["hdbscan_rows"] if int(r["cluster_id"]) != -1)
    n_bt = sum(1 for r in bundle["bertopic_rows"] if int(r["cluster_id"]) != -1)

    tabs_html = []
    panels_html = []
    sub_items = [
        ("pca", "Linear PCA", _render_axes_table(bundle["pca_rows"], "Linear PCA — Top 8 Eigen-Axes (Two-Pole Contrast)", ref_urls)),
        ("kpca", bundle["kpca_label"], _render_axes_table(bundle["kpca_rows"], f"{bundle['kpca_label']} — Top 8 Kernel Axes", ref_urls)),
        ("bertopic", f"BERTopic ({n_bt})", _render_cluster_table(bundle["bertopic_rows"], "BERTopic Clusters (Precomputed bge-m3 + c-TF-IDF)", ref_urls)),
        ("hdbscan", f"HDBSCAN ({n_hdb})", _render_cluster_table(bundle["hdbscan_rows"], "HDBSCAN Density Clusters", ref_urls)),
    ]
    if bundle["kpca_cosine_rows"] and bundle["kpca_label"] != "Cosine kPCA":
        sub_items.insert(
            2,
            ("kpca_cos", "Cosine kPCA", _render_axes_table(bundle["kpca_cosine_rows"], "Cosine Kernel PCA (Unit-Sphere Identity Check)", ref_urls)),
        )

    for s_idx, (sid, slabel, content) in enumerate(sub_items):
        if not content:
            continue
        sub_active = " active" if s_idx == 0 else ""
        panel_id = f"panel-{spec['key']}-{sid}"
        tabs_html.append(
            f"<button type=\"button\" class=\"sub-tab{sub_active}\" data-group=\"{spec['key']}\" data-algo=\"{sid}\" data-target=\"{panel_id}\">{_esc(slabel)}</button>"
        )
        panels_html.append(
            f"<div id=\"{panel_id}\" class=\"sub-panel{sub_active}\" data-group=\"{spec['key']}\" data-algo=\"{sid}\">{content}</div>"
        )

    card_active = " active" if active else ""
    return f"""
    <section id="work-{spec['key']}" class="work-card{card_active}" data-work="{spec['key']}">
      <div class="work-header">
        <div>
          <span class="badge" style="border-color: {spec['color']}; color: {spec['color']}">{_esc(spec['group'])} · {_esc(spec['lang'])} · {_esc(spec['era'])}</span>
          <h3>{idx}. <a class="work-title-link" href="{_esc(bundle['root_url'])}" target="_blank" rel="noopener noreferrer">{_esc(spec['title'])}</a></h3>
          <div class="work-author">{_esc(spec['author'])} · <a class="ref-link" href="{_esc(bundle['root_url'])}" target="_blank" rel="noopener noreferrer">Open on biblioteca.scriptorium.ro ↗</a></div>
        </div>
        <div class="stat-pills">
          <div class="stat-pill"><span class="stat-val">{bundle['n_rows']:,}</span><span class="stat-lbl">{_esc(spec['unit'])}</span></div>
          <div class="stat-pill"><span class="stat-val">{evr_sum:.1f}%</span><span class="stat-lbl">Top-8 PCA Var</span></div>
          <div class="stat-pill"><span class="stat-val">{n_bt}</span><span class="stat-lbl">BERTopic Clusters</span></div>
        </div>
      </div>
      <p class="work-summary">{_esc(spec['summary'])}</p>
      <div class="sub-tabs">{''.join(tabs_html)}</div>
      {''.join(panels_html)}
    </section>
    """


def build_xhtml(output_path: Path = OUTPUT_XHTML, bundles: list[dict[str, Any]] | None = None) -> Path:
    if bundles is None:
        bundles = []
        for spec in CORPUS_CATALOG:
            bundle = _load_corpus_bundle(spec)
            if bundle is not None:
                bundles.append(bundle)
    by_key: dict[str, dict[str, Any]] = {b["spec"]["key"]: b for b in bundles}

    if not bundles:
        raise RuntimeError("No FAISS datasets found under data/")

    # Compute cross-lingual subspace alignments
    alignments: list[dict[str, Any]] = []
    if "biblia_ortodoxa_subcapitole" in by_key and "biblia_darby_fr_subcapitole" in by_key:
        alignments.append(
            _cross_lingual_alignment(
                by_key["biblia_ortodoxa_subcapitole"],
                by_key["biblia_darby_fr_subcapitole"],
                "Romanian Orthodox Bible vs. French Darby Bible",
            )
        )
    if "hugo_notre_dame" in by_key and "hugo_notre_dame_en" in by_key:
        alignments.append(
            _cross_lingual_alignment(
                by_key["hugo_notre_dame"],
                by_key["hugo_notre_dame_en"],
                "Notre-Dame de Paris: Original French vs. English Translation",
            )
        )

    total_docs = sum(b["n_rows"] for b in bundles)
    viz_payload = {
        "corpora": [b["viz"] for b in bundles],
        "alignments": alignments,
    }
    # Escape any accidental ']]>' inside JSON so XML CDATA stays 100% well-formed
    viz_json = json.dumps(viz_payload, ensure_ascii=False, separators=(",", ":")).replace("]]>", "]]\\u003e")

    # Overview table rows
    overview_rows = []
    for b in bundles:
        s = b["spec"]
        evr1 = b["evr"][0] * 100.0
        evr8 = sum(b["evr"][:8]) * 100.0
        n_bt = sum(1 for r in b["bertopic_rows"] if int(r["cluster_id"]) != -1)
        overview_rows.append(
            f"<tr>"
            f"<td><a href=\"#work-{s['key']}\"><strong>{_esc(s['title'])}</strong></a><div class=\"anchor\">{_esc(s['author'])}</div></td>"
            f"<td>{_esc(s['lang'])}</td>"
            f"<td class=\"mono\">{b['n_rows']:,} {_esc(s['unit'])}</td>"
            f"<td class=\"mono\">{evr1:.2f}%</td>"
            f"<td class=\"mono\">{evr8:.1f}%</td>"
            f"<td class=\"mono\">{n_bt}</td>"
            f"</tr>"
        )

    corpus_buttons = "\n".join(
        f"<button type=\"button\" class=\"corpus-btn{' active' if i == 0 else ''}\" data-key=\"{b['spec']['key']}\">{_esc(b['spec']['title'])} <span class=\"btn-sub\">({b['spec']['lang']}, {b['n_rows']:,})</span></button>"
        for i, b in enumerate(bundles)
    )

    alignment_cards = []
    for a_idx, al in enumerate(alignments):
        pc_list = ", ".join(f"{c:.3f}" for c in al["principal_cosines"][:5])
        alignment_cards.append(
            f"<div class=\"align-card\">"
            f"<h4>{_esc(al['title'])}</h4>"
            f"<p class=\"align-meta\">Mean-vector cosine $\\cos(\\mu_A, \\mu_B) = {al['mean_cosine']:.4f}$ · "
            f"Top-8 subspace canonical overlap (mean principal cosine) ${al['subspace_overlap_mean']:.3f}$ "
            f"(top-5: $[{pc_list}]$)</p>"
            f"<div id=\"align-heatmap-{a_idx}\" class=\"plot-box-sm\"></div>"
            f"</div>"
        )

    work_tab_buttons = "\n".join(
        f"<button type=\"button\" class=\"work-tab{' active' if i == 0 else ''}\" data-work=\"{b['spec']['key']}\">"
        f"<span class=\"work-dot\" style=\"background:{b['spec']['color']}\"></span>"
        f"{_esc(b['spec']['title'])} <span class=\"btn-sub\">({b['spec']['lang']}, {b['n_rows']:,})</span></button>"
        for i, b in enumerate(bundles)
    )
    work_sections = "\n".join(_render_corpus_section(b, i, active=(i == 1)) for i, b in enumerate(bundles, start=1))

    xhtml = rf"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Eigenbooks: Spectral Decomposition &amp; Semantic Clustering Across 9 Literary Corpora</title>
  <meta name="description" content="PCA, Kernel PCA (Cosine &amp; RBF), HDBSCAN, and BERTopic over 1024-D bge-m3 embeddings of medieval chronicles, parallel Bibles, Ion Creangă, Jack London, and bilingual Victor Hugo." />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin="anonymous" />
  <link href="https://fonts.googleapis.com/css2?family=Alegreya:ital,wght@0,400;0,700;1,400&amp;family=JetBrains+Mono:wght@400;500;700&amp;family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;0,6..72,700;1,6..72,400&amp;family=Plus+Jakarta+Sans:wght@400;500;600;700;800&amp;display=swap" rel="stylesheet" />
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css" crossorigin="anonymous" />
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js" crossorigin="anonymous"></script>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
  <style type="text/css">
    :root {{
      --bg: #09090b;
      --bg-card: #111115;
      --bg-elev: #18181f;
      --border: #27272f;
      --text: #e4e4e7;
      --text-muted: #a1a1aa;
      --heading: #fafafa;
      --emerald: #34d399;
      --indigo: #818cf8;
      --amber: #fbbf24;
      --rose: #fb7185;
      --cyan: #22d3ee;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      background: radial-gradient(circle at 18% 0%, rgba(99,102,241,0.12), transparent 36%),
                  radial-gradient(circle at 86% 12%, rgba(16,185,129,0.10), transparent 32%),
                  var(--bg);
      color: var(--text);
      font-family: 'Newsreader', 'Alegreya', Georgia, serif;
      font-size: 18px;
      line-height: 1.72;
      -webkit-font-smoothing: antialiased;
    }}
    .container {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 2.5rem 1.5rem 5rem;
    }}
    header.hero {{
      border-bottom: 1px solid var(--border);
      padding-bottom: 2.2rem;
      margin-bottom: 2.5rem;
    }}
    .eyebrow {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: var(--emerald);
      margin-bottom: 0.75rem;
    }}
    h1 {{
      font-family: 'Newsreader', Georgia, serif;
      font-size: clamp(2.1rem, 4.4vw, 3.4rem);
      line-height: 1.12;
      color: var(--heading);
      margin: 0 0 1rem;
      font-weight: 700;
      letter-spacing: -0.02em;
    }}
    .subtitle {{
      font-size: 1.2rem;
      color: var(--text-muted);
      max-width: 860px;
      margin: 0 0 1.6rem;
    }}
    .hero-stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 0.85rem;
      margin-top: 1.4rem;
    }}
    .hero-stat {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 0.95rem 1.15rem;
    }}
    .hero-stat strong {{
      display: block;
      font-family: 'Plus Jakarta Sans', sans-serif;
      font-size: 1.45rem;
      color: var(--heading);
      line-height: 1.2;
    }}
    .hero-stat span {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.72rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    nav.toc {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin: 1.5rem 0 2.5rem;
      padding: 0.85rem 1rem;
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 14px;
    }}
    nav.toc a {{
      font-family: 'Plus Jakarta Sans', sans-serif;
      font-size: 0.82rem;
      font-weight: 600;
      color: var(--text);
      text-decoration: none;
      padding: 0.35rem 0.72rem;
      border-radius: 8px;
      background: var(--bg-elev);
      border: 1px solid transparent;
      transition: border-color 0.15s, color 0.15s;
    }}
    nav.toc a:hover {{
      border-color: var(--emerald);
      color: var(--emerald);
    }}
    h2 {{
      font-family: 'Plus Jakarta Sans', sans-serif;
      font-size: 1.65rem;
      color: var(--heading);
      margin: 3.2rem 0 1rem;
      padding-bottom: 0.45rem;
      border-bottom: 1px solid var(--border);
      letter-spacing: -0.015em;
    }}
    h3 {{
      font-family: 'Newsreader', Georgia, serif;
      font-size: 1.45rem;
      color: var(--heading);
      margin: 0.3rem 0;
    }}
    h4 {{
      font-family: 'Plus Jakarta Sans', sans-serif;
      font-size: 0.98rem;
      color: var(--heading);
      margin: 0 0 0.75rem;
    }}
    p {{ margin: 0 0 1.05rem; }}
    a {{ color: var(--indigo); }}
    code, .mono {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.84em;
    }}
    code {{
      background: var(--bg-elev);
      padding: 0.14rem 0.4rem;
      border-radius: 6px;
      color: #c4b5fd;
    }}
    .math-box {{
      margin: 1.6rem 0;
      padding: 1.4rem 1.6rem;
      border-radius: 16px;
      background: linear-gradient(145deg, #101017, #151720);
      border: 1px solid #2a2c38;
      box-shadow: 0 16px 40px rgba(0, 0, 0, 0.35);
    }}
    .math-box-head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid #272733;
      padding-bottom: 0.65rem;
      margin-bottom: 1rem;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }}
    .math-box-head .tag-l {{ color: var(--emerald); }}
    .math-box-head .tag-r {{ color: var(--text-muted); }}
    .formula-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(290px, 1fr));
      gap: 1rem;
    }}
    .formula-item {{
      background: rgba(9, 9, 11, 0.55);
      border: 1px solid #242430;
      border-radius: 12px;
      padding: 0.9rem 1.05rem;
    }}
    .formula-title {{
      font-family: 'Plus Jakarta Sans', sans-serif;
      font-size: 0.84rem;
      font-weight: 700;
      color: var(--heading);
      margin-bottom: 0.35rem;
    }}
    .formula-eq {{
      font-size: 0.96rem;
      color: var(--emerald);
      margin: 0.35rem 0 0.5rem;
      overflow-x: auto;
    }}
    .formula-eq .katex-display {{
      margin: 0.25rem 0;
      text-align: left;
    }}
    .formula-note {{
      font-family: 'Plus Jakarta Sans', sans-serif;
      font-size: 0.78rem;
      color: var(--text-muted);
      line-height: 1.45;
      margin: 0;
    }}
    .callout {{
      margin: 1.5rem 0;
      padding: 1.1rem 1.35rem;
      border-left: 3px solid var(--amber);
      background: rgba(251, 191, 36, 0.06);
      border-radius: 0 12px 12px 0;
      font-size: 0.96rem;
    }}
    .callout.emerald {{
      border-left-color: var(--emerald);
      background: rgba(52, 211, 153, 0.06);
    }}
    .lab-panel {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 1.4rem;
      margin: 1.8rem 0;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
    }}
    .corpus-selector {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.45rem;
      margin-bottom: 1rem;
    }}
    .corpus-btn {{
      font-family: 'Plus Jakarta Sans', sans-serif;
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--text);
      background: var(--bg-elev);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 0.45rem 0.78rem;
      cursor: pointer;
      transition: all 0.15s;
    }}
    .corpus-btn:hover {{ border-color: var(--indigo); }}
    .corpus-btn.active {{
      background: rgba(99, 102, 241, 0.2);
      border-color: var(--indigo);
      color: #fff;
    }}
    .btn-sub {{
      font-weight: 400;
      color: var(--text-muted);
      font-size: 0.74rem;
    }}
    .controls-bar {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 0.9rem;
      padding: 0.65rem 0.85rem;
      background: var(--bg-elev);
      border-radius: 12px;
      border: 1px solid var(--border);
    }}
    .mode-group {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.4rem;
    }}
    .mode-btn {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.76rem;
      padding: 0.36rem 0.7rem;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: var(--bg-card);
      color: var(--text-muted);
      cursor: pointer;
    }}
    .mode-btn.active {{
      background: rgba(52, 211, 153, 0.16);
      border-color: var(--emerald);
      color: var(--emerald);
      font-weight: 700;
    }}
    .search-input {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.8rem;
      background: var(--bg);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.42rem 0.75rem;
      min-width: 240px;
    }}
    .search-input:focus {{ outline: none; border-color: var(--emerald); }}
    .plot-box {{ width: 100%; height: 560px; }}
    .plot-box-md {{ width: 100%; height: 400px; }}
    .plot-box-sm {{ width: 100%; height: 370px; }}
    .align-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(440px, 1fr));
      gap: 1.2rem;
      margin: 1.4rem 0;
    }}
    .align-card {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 1.2rem;
    }}
    .align-meta {{
      font-family: 'Plus Jakarta Sans', sans-serif;
      font-size: 0.82rem;
      color: var(--text-muted);
      margin-bottom: 0.75rem;
    }}
    .work-tabs {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.45rem;
      margin: 1rem 0 1.2rem;
    }}
    .work-tab {{
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
      font-family: 'Plus Jakarta Sans', sans-serif;
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--text);
      background: var(--bg-elev);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 0.45rem 0.78rem;
      cursor: pointer;
      transition: all 0.15s;
    }}
    .work-tab:hover {{ border-color: var(--indigo); }}
    .work-tab.active {{
      background: rgba(99, 102, 241, 0.22);
      border-color: var(--indigo);
      color: #fff;
    }}
    .work-dot {{
      width: 8px;
      height: 8px;
      border-radius: 999px;
      flex-shrink: 0;
    }}
    .work-card {{
      display: none;
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 1.5rem 1.6rem;
      margin: 1.2rem 0 1.8rem;
    }}
    .work-card.active {{ display: block; }}
    .work-header {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: flex-start;
      gap: 1rem;
      margin-bottom: 0.9rem;
    }}
    .badge {{
      display: inline-block;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      padding: 0.18rem 0.55rem;
      border: 1px solid;
      border-radius: 999px;
      margin-bottom: 0.4rem;
    }}
    .work-author {{
      font-family: 'Plus Jakarta Sans', sans-serif;
      font-size: 0.88rem;
      color: var(--text-muted);
    }}
    .stat-pills {{ display: flex; gap: 0.6rem; flex-wrap: wrap; }}
    .stat-pill {{
      background: var(--bg-elev);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 0.42rem 0.75rem;
      text-align: center;
      min-width: 92px;
    }}
    .stat-val {{
      display: block;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.98rem;
      font-weight: 700;
      color: var(--heading);
    }}
    .stat-lbl {{
      font-family: 'Plus Jakarta Sans', sans-serif;
      font-size: 0.68rem;
      color: var(--text-muted);
      text-transform: uppercase;
    }}
    .work-summary {{ color: #d4d4d8; font-size: 0.98rem; margin-bottom: 1.1rem; }}
    .sub-tabs {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.4rem;
      border-bottom: 1px solid var(--border);
      padding-bottom: 0.65rem;
      margin-bottom: 1rem;
    }}
    .sub-tab {{
      font-family: 'Plus Jakarta Sans', sans-serif;
      font-size: 0.8rem;
      font-weight: 600;
      padding: 0.35rem 0.75rem;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: var(--bg-elev);
      color: var(--text-muted);
      cursor: pointer;
    }}
    .sub-tab.active {{
      background: rgba(52, 211, 153, 0.16);
      border-color: var(--emerald);
      color: var(--emerald);
    }}
    .sub-panel {{ display: none; }}
    .sub-panel.active {{ display: block; }}
    .table-wrap {{ overflow-x: auto; }}
    table.data-table {{
      width: 100%;
      border-collapse: collapse;
      font-family: 'Plus Jakarta Sans', sans-serif;
      font-size: 0.84rem;
      line-height: 1.48;
    }}
    table.data-table th {{
      text-align: left;
      padding: 0.6rem 0.7rem;
      border-bottom: 1px solid #32323e;
      color: var(--text-muted);
      font-size: 0.73rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    table.data-table td {{
      padding: 0.58rem 0.7rem;
      border-bottom: 1px solid #1e1e27;
      vertical-align: top;
    }}
    table.data-table tr:hover td {{ background: rgba(255, 255, 255, 0.02); }}
    .accent {{ color: var(--emerald); font-weight: 700; }}
    .muted {{ color: var(--text-muted); font-size: 0.78em; }}
    .pole {{
      display: inline-block;
      width: 1.25rem;
      text-align: center;
      border-radius: 4px;
      font-family: 'JetBrains Mono', monospace;
      font-weight: 700;
      font-size: 0.76rem;
      margin-right: 0.25rem;
    }}
    .pole.pos {{ background: rgba(52, 211, 153, 0.18); color: var(--emerald); }}
    .pole.neg {{ background: rgba(251, 113, 133, 0.18); color: var(--rose); }}
    .anchor {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.72rem;
      color: var(--text-muted);
      margin-top: 0.2rem;
    }}
    .ref-link {{
      color: #93c5fd;
      text-decoration: none;
      border-bottom: 1px dotted rgba(147, 197, 253, 0.45);
    }}
    .ref-link:hover {{
      color: var(--emerald);
      border-bottom-color: var(--emerald);
    }}
    .ref-link-subtle {{
      color: #a1a1aa;
      text-decoration: none;
    }}
    .ref-link-subtle:hover {{
      color: var(--emerald);
      text-decoration: underline;
    }}
    .work-title-link {{
      color: var(--heading);
      text-decoration: none;
    }}
    .work-title-link:hover {{
      color: var(--emerald);
    }}
    .kw-line {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.73rem;
      color: var(--cyan);
      margin-top: 0.22rem;
    }}
    .small-cell {{ font-size: 0.74rem; color: var(--text-muted); max-width: 240px; word-break: break-word; }}
    .grid-2 {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 1.1rem;
      margin: 1.2rem 0;
    }}
    .info-card {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 1.15rem 1.3rem;
    }}
    .info-card h4 {{ color: var(--emerald); margin-bottom: 0.45rem; }}
    .info-card p {{ font-size: 0.92rem; margin-bottom: 0.5rem; }}
    footer {{
      margin-top: 4rem;
      padding-top: 1.5rem;
      border-top: 1px solid var(--border);
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.78rem;
      color: var(--text-muted);
    }}
  </style>
</head>
<body>
  <article class="container">
    <header class="hero">
      <div class="eyebrow">Computational Philology · Spectral Linear Algebra · Manifold Learning</div>
      <h1>Eigenbooks: Uncovering Latent Semantic Axes &amp; Narrative Clusters Across Nine Literary Corpora</h1>
      <p class="subtitle">
        A unified numerical experiment applying <strong>Linear PCA</strong>, <strong>Kernel PCA (Cosine &amp; RBF)</strong>,
        <strong>HDBSCAN</strong>, and <strong>BERTopic</strong> to 1024-dimensional <code>bge-m3</code> embeddings of medieval chronicles,
        parallel Romanian &amp; French Bibles, Ion Creangă, Jack London, and bilingual Victor Hugo — stored locally in FAISS
        and interpreted via LLM-labelled spectral poles.
      </p>
      <div class="hero-stats">
        <div class="hero-stat"><strong>{len(bundles)} Works</strong><span>4 Languages (LA · RO · FR · EN)</span></div>
        <div class="hero-stat"><strong>{total_docs:,}</strong><span>Indexed Passages / Chapters</span></div>
        <div class="hero-stat"><strong>1,024-D</strong><span>bge-m3 Unit-Sphere Embeddings</span></div>
        <div class="hero-stat"><strong>4 Algorithms</strong><span>PCA · kPCA · HDBSCAN · BERTopic</span></div>
      </div>
    </header>

    <nav class="toc" aria-label="Article sections">
      <a href="#interactive-lab">1. Interactive 3D Lab</a>
      <a href="#scree-comparison">2. Eigenvalue Spectra</a>
      <a href="#cross-lingual">3. Cross-Lingual Alignment</a>
      <a href="#mathematics">4. Mathematical Foundations</a>
      <a href="#labelling-method">5. How Axes Were Labelled</a>
      <a href="#corpus-results">6. Detailed Per-Work Results</a>
      <a href="#js-viz-guide">7. Best JS Viz Libraries</a>
      <a href="#future-directions">8. Future Directions</a>
    </nav>

    <section id="overview">
      <h2>0. Corpus Overview &amp; Experimental Setup</h2>
      <p>
        Can unsupervised numerical linear algebra recover the <em>intellectual skeleton</em> of a book — separating
        Victor Hugo's architectural essays from Esmeralda's drama, Ion Creangă's grammar drills from Humulești nostalgia,
        or Old Testament genealogies from Pauline ethics — without ever telling the algorithm what a chapter, genre,
        or character is?
      </p>
      <p>
        To find out, we assembled <strong>{len(bundles)} complete works ({total_docs:,} passages)</strong> in a single shared
        embedding geometry: <code>bge-m3:567m-fp16</code> (1024 dimensions, unit $\ell_2$-normalized onto the hypersphere
        $S^{{1023}}$). Seven works were pulled directly from Qdrant (<code>biblioteca_paragraphs_bge_m3</code>) with
        paragraph texts fetched from <code>biblioteca.scriptorium.ro</code> and verified bit-for-bit against Qdrant's
        SHA-256 hashes; the two parallel Bible translations were embedded at chapter granularity using the exact same
        <code>bge-m3:567m-fp16</code> model in Ollama. Every corpus is persisted locally as an exact flat FAISS index
        (<code>IndexFlatIP</code>) paired with <code>records.jsonl</code>, giving instant cosine kNN search alongside
        lossless <code>float32</code> matrix reconstruction (<code>index.reconstruct_n(0, N)</code>) for NumPy and scikit-learn.
      </p>
      <div class="table-wrap">
        <table class="data-table">
          <thead>
            <tr><th>Work</th><th>Language</th><th>Size</th><th>PC1 Variance</th><th>Top-8 PCA Var</th><th>BERTopic Clusters</th></tr>
          </thead>
          <tbody>{''.join(overview_rows)}</tbody>
        </table>
      </div>
    </section>

    <section id="interactive-lab">
      <h2>1. Interactive 3D Embedding &amp; Manifold Explorer</h2>
      <p>
        Explore every passage of all {len(bundles)} works directly in your browser. Switch between <strong>Linear PCA (3D)</strong>,
        <strong>Non-Linear RBF Kernel PCA ($\gamma = 2.0$, 3D)</strong>, and <strong>UMAP (3D cosine manifold)</strong>, colored by
        unsupervised BERTopic/HDBSCAN clusters. Hover any point to read its chapter/section reference, cluster theme,
        and text excerpt — or type a name/word in the filter box (e.g., <code>gelou</code>, <code>humulești</code>,
        <code>cherokee</code>, <code>quasimodo</code>, <code>psalmii</code>) to spotlight matching passages in 3D space.
      </p>
      <div class="lab-panel">
        <div class="corpus-selector" id="corpus-selector">
          {corpus_buttons}
        </div>
        <div class="controls-bar">
          <div class="mode-group" id="mode-selector">
            <button type="button" class="mode-btn active" data-mode="umap3">UMAP 3D (Cluster Manifold)</button>
            <button type="button" class="mode-btn" data-mode="pca3">Linear PCA 3D</button>
            <button type="button" class="mode-btn" data-mode="kpca3">RBF kPCA (γ=2.0) 3D</button>
          </div>
          <input type="text" id="point-search" class="search-input" placeholder="Highlight keyword (e.g. quasimodo, arpad)..." />
        </div>
        <div id="scatter-3d" class="plot-box"></div>
        <div id="lab-status" class="anchor" style="margin-top:0.5rem;"></div>
      </div>
    </section>

    <section id="scree-comparison">
      <h2>2. Intrinsic Dimensionality: Eigenvalue &amp; Scree Comparison</h2>
      <p>
        Comparing the top-15 PCA explained variance curves across all {len(bundles)} works immediately reveals how
        <em>structurally concentrated</em> vs. <em>thematically sprawling</em> each book is. Short, formulaic or
        bipartite corpora concentrate variance sharply in PC1–PC3 (e.g., <em>Gesta Hungarorum</em> puts <strong>34.2%</strong>
        into its first 8 components, and both Bibles put ~9–10% into PC1 alone — cleanly separating Old Testament narrative/law
        from New Testament ethics/epistles). By contrast, richly picaresque prose like Creangă's <em>Amintiri</em> spreads
        semantic variance across many orthogonal local episodes.
      </p>
      <div class="lab-panel">
        <div class="controls-bar">
          <div class="mode-group" id="scree-mode">
            <button type="button" class="mode-btn active" data-scree="individual">Per-Component Variance Ratio</button>
            <button type="button" class="mode-btn" data-scree="cumulative">Cumulative Explained Variance</button>
          </div>
          <span class="mono muted">Top 15 linear PCA axes across all corpora (1024-D bge-m3)</span>
        </div>
        <div id="scree-plot" class="plot-box-md"></div>
      </div>
    </section>

    <section id="cross-lingual">
      <h2>3. Cross-Lingual Eigen-Axis Alignment (Language Invariance Test)</h2>
      <p>
        Because <code>bge-m3</code> maps Romanian, French, English, and Latin into the <strong>same 1024-dimensional
        Euclidean space</strong>, we can directly test how much PCA axes depend on language versus underlying meaning.
        For both of our parallel pairs — <strong>Biblia Ortodoxă (RO) vs. La Bible Darby (FR)</strong> and
        <strong>Notre-Dame de Paris (FR) vs. Notre-Dame de Paris (EN)</strong> — we compute the $8 \times 8$
        absolute cosine-similarity matrix $|V_A V_B^\top|$ between their top-8 principal eigenvectors, together
        with the canonical principal angles (singular values of $V_A V_B^\top$) measuring how tightly the two
        8-dimensional semantic subspaces overlap.
      </p>
      <div class="align-grid">
        {''.join(alignment_cards)}
      </div>
      <div class="callout emerald">
        <strong>What the heatmaps prove:</strong> A crisp bright diagonal ($|\cos \theta| \approx 0.70\text{{–}}0.95$) on the leading
        components demonstrates that the <em>dominant eigen-directions of meaning are largely language-invariant</em> —
        Romanian and French Bibles, and French and English <em>Notre-Dame de Paris</em>, span nearly the same top
        semantic subspace inside <code>bge-m3</code>, even when translated into a different language and segmented into
        different paragraph counts! Off-diagonal swaps (e.g., $\mathrm{{PC}}_3 \leftrightarrow \mathrm{{PC}}_4$) occur where consecutive eigenvalues are nearly
        degenerate, causing small translation or canon differences (such as the Orthodox Deuterocanon) to rotate two
        adjacent axes.
      </div>
    </section>

    <section id="mathematics">
      <h2>4. Mathematical Foundations &amp; Key Geometric Insights</h2>
      <div class="math-box">
        <div class="math-box-head">
          <span class="tag-l">Mathematical Formulation</span>
          <span class="tag-r">Hypersphere $S^{{1023}}$ · RKHS · Density Trees</span>
        </div>
        <div class="formula-grid">
          <div class="formula-item">
            <div class="formula-title">1. Linear PCA on Centered Embeddings</div>
            <div class="formula-eq">$$\tilde{{X}} = X - \mathbf{{1}}\mu^\top, \quad C = \frac{{1}}{{N-1}}\tilde{{X}}^\top\tilde{{X}} = V\Lambda V^\top$$</div>
            <p class="formula-note">Extracts orthogonal axes $v_k$ maximizing global variance. Projection score $z_{{ik}} = \tilde{{x}}_i \cdot v_k$ orders every passage from the $-$ pole to the $+$ pole.</p>
          </div>
          <div class="formula-item">
            <div class="formula-title">2. Cosine Kernel PCA Identity on $S^{{d-1}}$</div>
            <div class="formula-eq">$$\|x_i\|_2 = 1 \;\Rightarrow\; K_{{\cos}}(x_i, x_j) = x_i^\top x_j \;\Rightarrow\; \tilde{{K}}_{{\cos}} = \tilde{{X}}\tilde{{X}}^\top$$</div>
            <p class="formula-note">Because <code>bge-m3</code> vectors satisfy $\|x_i\|_2 = 1$, Cosine kPCA shares the exact same eigenspace as Linear PCA (up to sign flip), with $\lambda_{{\mathrm{{kpca}}}} = (N-1)\,\mathrm{{Var}}_k$.</p>
          </div>
          <div class="formula-item">
            <div class="formula-title">3. Non-Linear RBF Kernel PCA &amp; The $\gamma$-Scale Trap</div>
            <div class="formula-eq">$$K_{{\mathrm{{RBF}}}}(x_i, x_j) = \exp\!\left(-\gamma\|x_i - x_j\|_2^2\right) = \exp\!\left(-2\gamma(1 - x_i^\top x_j)\right)$$</div>
            <p class="formula-note">Crucial insight: sklearn's default $\gamma = 1/1024 \approx 0.001$ makes $e^u \approx 1 + u$ 99.8% linear! Setting $\gamma \approx 2.0$ activates genuine non-linear RKHS curvature and exponentially damps distant outliers (like Wikisource colophons).</p>
          </div>
          <div class="formula-item">
            <div class="formula-title">4. UMAP + HDBSCAN (<code>eom</code> vs. <code>leaf</code>)</div>
            <div class="formula-eq">$$d_{{\mathrm{{mreach}}}}(a, b) = \max\!\bigl(\mathrm{{core}}_k(a),\, \mathrm{{core}}_k(b),\, d(a, b)\bigr)$$</div>
            <p class="formula-note">Pre-reducing 1024-D cosine neighborhoods to 5-D via UMAP lets HDBSCAN find crisp density peaks. When a corpus contains distant metadata/title outliers (<em>Gesta</em>, <em>White Fang</em>), <code>eom</code> collapses body text into one mass, whereas <code>leaf</code> selection cleanly splits narrative episodes.</p>
          </div>
          <div class="formula-item">
            <div class="formula-title">5. BERTopic Class-Based TF-IDF</div>
            <div class="formula-eq">$$W_{{t,c}} = \mathrm{{tf}}_{{t,c}} \cdot \log\!\left(1 + \frac{{A}}{{\mathrm{{tf}}_t}}\right)$$</div>
            <p class="formula-note">Reuses our precomputed <code>bge-m3</code> embeddings directly (<code>embedding_model=None</code>), treats each HDBSCAN cluster as one mega-document, and extracts distinctive n-gram vocabulary (e.g., <code>tuhutum</code>, <code>gelou</code>, <code>menumorout</code>) without re-embedding.</p>
          </div>
          <div class="formula-item">
            <div class="formula-title">6. Cross-Lingual Subspace Overlap</div>
            <div class="formula-eq">$$\sigma_k = \mathrm{{SVD}}\!\left(V_A V_B^\top\right)_k = \cos\theta_k$$</div>
            <p class="formula-note">The singular values of $V_A V_B^\top$ are the cosines of the canonical principal angles $\theta_k$ between two corpora's top-$k$ eigen-subspaces in the shared multilingual embedding space.</p>
          </div>
        </div>
      </div>

      <div class="callout">
        <strong>The Two Biggest Empirical Lessons from Running Real Books:</strong>
        (1) <strong>Default RBF $\gamma = 1/d$ collapses to linear PCA on unit-normalized LLM embeddings</strong> — verifying
        <code>Gesta Hungarorum</code> with $\gamma = 1/1024$ vs. $\gamma = 2.0$ showed that only at
        $\gamma = 2.0$ did the Gaussian kernel push the Wikisource <code>about/</code> colophon down from $\mathrm{{PC}}_1/\mathrm{{PC}}_3$
        to Component 6 and reveal Transylvania (<code>de_terra_ultra_siluana</code>) on its own axis.
        (2) <strong>HDBSCAN's <code>eom</code> vs. <code>leaf</code></strong>: whenever a scraped book contains a small
        pocket of structural boilerplate (7 Wikisource colophon paragraphs in <em>Gesta</em>, or 24 bare chapter headings
        in <em>White Fang</em>), <code>eom</code> splits the book into just 2 clusters (boilerplate vs. everything else);
        switching to <code>leaf</code> immediately unpacks 8–19 razor-sharp narrative episodes!
      </div>
    </section>

    <section id="labelling-method">
      <h2>5. How Axes &amp; Clusters Were Labelled (and Honest Caveats)</h2>
      <p>
        An eigenvector in $\mathbb{{R}}^{{1024}}$ or an integer cluster ID is mathematically meaningful but human-opaque.
        To translate geometry back into philology, our refactored pipeline separates <em>linear algebra</em> from
        <em>semantic interpretation</em> through three pluggable stages:
      </p>
      <div class="grid-2">
        <div class="info-card">
          <h4>Stage 1 · Neighbourhood Selection</h4>
          <p>
            For PCA and kPCA we use <strong><code>top-projection</code></strong>: for each axis $j$, we sort
            all passages by their projection coordinate $z_{{ij}}$ and take both the <strong>$K = 6$ highest ($+$)</strong>
            and <strong>$K = 6$ lowest ($-$)</strong> passages. Because every principal axis is a <em>contrast</em> between
            two poles, labelling both ends exposes the underlying semantic tension (e.g., <em>Scythian origins</em> vs.
            <em>Árpád's military marches</em>; <em>OT genealogies</em> vs. <em>NT ethics</em>). An alternative
            <strong><code>anchor-knn</code></strong> strategy picks the single furthest point and queries FAISS for its
            exact cosine neighbours in 1024-D.
          </p>
          <p>
            For HDBSCAN and BERTopic, each cluster's representatives are the <strong>$K = 5$ passages with highest cosine
            similarity to the cluster centroid in the original 1024-D embedding space</strong> (not UMAP space, avoiding
            projection distortion).
          </p>
        </div>
        <div class="info-card">
          <h4>Stage 2 · Prompting &amp; LLM Discipline</h4>
          <p>
            The <code>ChapterSummarizer</code> concatenates the opening <code>600</code> characters of each representative
            passage separated by <code>---</code> and asks <code>qwen3.5:4b</code> on Ollama (with <code>think=False</code>
            so reasoning tokens don't exhaust the generation budget) to name the shared theme in one concise English phrase.
            An alternative <code>ShuffledLinesSummarizer</code> pools and shuffles individual lines across passages to test
            vocabulary-level stability independent of narrative order.
          </p>
          <p>
            <strong>Honest caveats:</strong> A compact 4B model occasionally slips on archaic 17th-century Romanian or
            medieval Latin proper nouns (e.g., calling Ion Creangă's boarding-house mate <em>Mogorogea</em> a "dog", or
            attributing <em>Gesta</em>'s 10th-century Duke Árpád to "King Stephen"). Pairing LLM labels with
            <strong>BERTopic's exact c-TF-IDF keywords</strong> and <strong>anchor passage references</strong> gives a
            self-checking audit trail.
          </p>
        </div>
      </div>
    </section>

    <section id="corpus-results">
      <h2>6. Detailed Run Results Across All {len(bundles)} Corpora</h2>
      <p>
        Select a corpus tab below, then switch between <strong>Linear PCA</strong>, <strong>Kernel PCA</strong>,
        <strong>BERTopic</strong> (with c-TF-IDF keywords), and <strong>HDBSCAN</strong> inside the work panel.
        The selected algorithm tab stays active as you flip between works.
      </p>
      <div class="work-tabs" id="work-tabs">{work_tab_buttons}</div>
      {work_sections}
    </section>

    <section id="js-viz-guide">
      <h2>7. Best JavaScript Visualization Libraries for Embedding &amp; Manifold Exploration</h2>
      <p>
        Choosing the right browser visualization stack depends primarily on point count <code>N</code>, whether you need
        2D vs. 3D rotation, and how tightly you want text inspection coupled with WebGL rendering:
      </p>
      <div class="table-wrap">
        <table class="data-table">
          <thead>
            <tr><th>Library</th><th>Rendering</th><th>Sweet Spot (N)</th><th>Strengths for Eigenbooks / Embeddings</th><th>Trade-offs</th></tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Plotly.js</strong> <span class="muted">(used here)</span></td>
              <td class="mono">WebGL + SVG</td>
              <td class="mono">100 – 25,000</td>
              <td>Drop-in <code>scatter3d</code>, <code>scattergl</code>, scree lines, heatmaps, rich hover tooltips, legend filtering, zero build step via CDN.</td>
              <td>Heavy bundle (~3.5 MB); custom shaders/animations less flexible than Three.js.</td>
            </tr>
            <tr>
              <td><strong>regl-scatterplot</strong></td>
              <td class="mono">Raw WebGL</td>
              <td class="mono">10k – 2,000,000+</td>
              <td>Purpose-built for 2D UMAP/t-SNE/PCA embedding clouds; silky 60fps lasso selection, zoom, and point transitions.</td>
              <td>2D only; requires pairing with your own tooltip/sidebar UI.</td>
            </tr>
            <tr>
              <td><strong>Three.js</strong></td>
              <td class="mono">WebGL / WebGPU</td>
              <td class="mono">1k – 500,000+</td>
              <td>Full 3D camera control, instanced point clouds, animated trajectories along narrative time <code>t</code>, custom GLSL shaders (great for RBF surfaces &amp; manifolds).</td>
              <td>More boilerplate for axes, legends, and raycast hover tooltips.</td>
            </tr>
            <tr>
              <td><strong>deck.gl</strong></td>
              <td class="mono">WebGL2 / WebGPU</td>
              <td class="mono">10k – 1,000,000+</td>
              <td>Layered <code>ScatterplotLayer</code> + <code>TextLayer</code> + <code>PathLayer</code> (ideal for drawing chapter-to-chapter narrative paths over cluster hulls).</td>
              <td>Larger ecosystem footprint; shines best in React/TypeScript apps.</td>
            </tr>
            <tr>
              <td><strong>D3.js / Observable Plot</strong></td>
              <td class="mono">SVG / Canvas</td>
              <td class="mono">50 – 5,000</td>
              <td>Unrivaled typographic polish for 2D editorial charts: scree plots, dendrograms (HDBSCAN condensed trees!), Sankey/alluvial cross-book flows, and custom interactive matrices.</td>
              <td>SVG slows down above ~5,000 points; no native 3D WebGL.</td>
            </tr>
            <tr>
              <td><strong>Apache ECharts</strong></td>
              <td class="mono">Canvas / WebGL</td>
              <td class="mono">100 – 100,000</td>
              <td>Polished dark-mode dashboards, built-in brushing/linking between cluster bar charts, 3D GL scatter, and heatmaps.</td>
              <td>Less idiomatic for bespoke mathematical layouts than D3.</td>
            </tr>
            <tr>
              <td><strong>Apple Embedding Atlas</strong></td>
              <td class="mono">WebGPU / WASM</td>
              <td class="mono">10k – 5,000,000+</td>
              <td>Automatic density contours, real-time nearest-neighbour lookup, and dynamic zoom-level topic labels out of the box.</td>
              <td>Standalone widget/app model rather than lightweight inline blog script.</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section id="future-directions">
      <h2>8. Future Directions</h2>
      <div class="grid-2">
        <div class="info-card">
          <h4>1. Supervised Metric Learning (NCA &amp; LMNN)</h4>
          <p>
            Now that every work sits in a uniform local FAISS + <code>VectorDataset</code> format, we can move beyond
            unsupervised variance (PCA/kPCA) to <strong>Neighborhood Components Analysis (NCA)</strong> and
            <strong>Large Margin Nearest Neighbor (LMNN)</strong>: learn a linear projection $L$ supervised by
            known labels (book, testament, author, historical century, or protagonist) to ask <em>which directions in
            embedding space best separate Grigore Ureche's voice from Miron Costin's</em>, or <em>Synoptic Gospels from
            John</em>.
          </p>
        </div>
        <div class="info-card">
          <h4>2. Narrative Trajectories in Eigen-Space</h4>
          <p>
            Because <code>qdrant_downloader.py</code> preserves exact reading order <code>(section, paragraph_num)</code>,
            we can plot a smoothed curve $t \mapsto (\mathrm{{PC}}_1(t),\, \mathrm{{PC}}_2(t),\, \mathrm{{PC}}_3(t))$ tracing how <em>White Fang</em> moves
            continuously from the Yukon survival pole to the domestication pole, or how <em>Notre-Dame de Paris</em>
            oscillates between Gothic architectural digressions and street-level melodrama.
          </p>
        </div>
        <div class="info-card">
          <h4>3. Boilerplate Cleaning &amp; Procrustes Alignment</h4>
          <p>
            Filtering out Wikisource <code>about/</code> colophons and single-line chapter headers before fitting will
            remove structural noise from $\mathrm{{PC}}_1/\mathrm{{PC}}_3$. On the bilingual pairs (RO/FR Bibles, FR/EN <em>Notre-Dame</em>, and
            Latin/Hungarian <em>Gesta</em>), Orthogonal Procrustes ($\min_R \|X_A R - X_B\|_F$) and removing
            the language mean shift $\mu_{{\mathrm{{FR}}}} - \mu_{{\mathrm{{EN}}}}$ will cleanly isolate pure translation drift from semantic
            structure.
          </p>
        </div>
        <div class="info-card">
          <h4>4. Stronger Philological Labellers</h4>
          <p>
            Swapping <code>qwen3.5:4b</code> for <code>qwen3.5:9b</code> or <code>gemma3:12b</code> (already available on
            <code>zmeu</code>), feeding BERTopic's c-TF-IDF keywords directly into the summarizer prompt, and filtering
            stop-words per language (Latin, Romanian, French, English) will make both cluster keywords and archaic-text
            labels dramatically sharper.
          </p>
        </div>
      </div>
    </section>

    <footer>
      Generated programmatically from local FAISS datasets &amp; JSONL experiment logs via
      <code>src/build_blog_report.py</code> · Embeddings: <code>bge-m3:567m-fp16</code> (1024-D) ·
      Labeller: <code>qwen3.5:4b</code> (Ollama on <code>zmeu</code>) · Vector Store: Qdrant + FAISS.
    </footer>
  </article>

  <script id="eigenbooks-data" type="application/json"><![CDATA[
{viz_json}
]]></script>
  <script type="text/javascript">//<![CDATA[
  (function () {{
    var raw = document.getElementById('eigenbooks-data').textContent;
    raw = raw.replace(/^\s*(?:\/\/)?\s*(?:<!\[CDATA\[)?\s*/, '').replace(/\s*(?:\/\/)?\s*(?:\]\]>)?\s*$/, '');
    var DATA = JSON.parse(raw);
    var byKey = {{}};
    DATA.corpora.forEach(function (c) {{ byKey[c.key] = c; }});

    var PALETTE = [
      '#34d399', '#818cf8', '#fbbf24', '#fb7185', '#22d3ee',
      '#f97316', '#a78bfa', '#4ade80', '#f472b6', '#38bdf8',
      '#facc15', '#c084fc', '#2dd4bf', '#fb923c', '#60a5fa',
      '#e879f9', '#86efac', '#fda4af', '#67e8f9', '#fde047'
    ];

    var state = {{
      key: DATA.corpora[0].key,
      mode: 'umap3',
      query: '',
      screeMode: 'individual'
    }};

    function renderScatter3D() {{
      var c = byKey[state.key];
      if (!c) return;
      var coords = c[state.mode];
      var xs = coords[0], ys = coords[1], zs = coords[2];
      var q = state.query.trim().toLowerCase();

      var byCluster = {{}};
      var matchCount = 0;
      for (var i = 0; i < c.n; i++) {{
        var cid = c.clusters[i];
        var hay = (c.refs[i] + ' ' + c.excerpts[i] + ' ' + (c.cluster_labels[String(cid)] || '')).toLowerCase();
        var isMatch = !q || hay.indexOf(q) !== -1;
        if (q && isMatch) matchCount++;
        var bucket = q ? (isMatch ? 'MATCH' : 'DIM') : String(cid);
        if (!byCluster[bucket]) {{
          byCluster[bucket] = {{ x: [], y: [], z: [], text: [], customdata: [] }};
        }}
        byCluster[bucket].x.push(xs[i]);
        byCluster[bucket].y.push(ys[i]);
        byCluster[bucket].z.push(zs[i]);
        byCluster[bucket].customdata.push({{ ref: c.refs[i], url: c.urls[i] }});
        var clbl = c.cluster_labels[String(cid)] || ('Cluster ' + cid);
        var kw = c.cluster_keywords[String(cid)];
        var tip = '<b>' + c.refs[i] + '</b> <span style="color:#93c5fd">(click dot to open on biblioteca.scriptorium.ro)</span><br>' +
          '<span style="color:#34d399">' + (cid === -1 ? 'Noise (-1)' : ('Cluster #' + cid + ': ' + clbl)) + '</span>' +
          (kw ? '<br><span style="color:#22d3ee">Keywords: ' + kw + '</span>' : '') +
          '<br><span style="color:#d4d4d8">' + c.excerpts[i] + '</span>';
        byCluster[bucket].text.push(tip);
      }}

      var traces = [];
      if (q) {{
        if (byCluster.DIM) {{
          traces.push({{
            type: 'scatter3d', mode: 'markers', name: 'Other (' + byCluster.DIM.x.length + ')',
            x: byCluster.DIM.x, y: byCluster.DIM.y, z: byCluster.DIM.z, text: byCluster.DIM.text,
            customdata: byCluster.DIM.customdata,
            hovertemplate: '%{{text}}<extra></extra>',
            marker: {{ size: 2.8, color: '#3f3f46', opacity: 0.2 }}
          }});
        }}
        if (byCluster.MATCH) {{
          traces.push({{
            type: 'scatter3d', mode: 'markers', name: 'Matches: "' + state.query + '" (' + byCluster.MATCH.x.length + ')',
            x: byCluster.MATCH.x, y: byCluster.MATCH.y, z: byCluster.MATCH.z, text: byCluster.MATCH.text,
            customdata: byCluster.MATCH.customdata,
            hovertemplate: '%{{text}}<extra></extra>',
            marker: {{ size: 6.2, color: '#fbbf24', opacity: 0.96, line: {{ color: '#ffffff', width: 0.6 }} }}
          }});
        }}
      }} else {{
        var keys = Object.keys(byCluster).sort(function (a, b) {{
          if (a === '-1') return 1;
          if (b === '-1') return -1;
          return Number(a) - Number(b);
        }});
        keys.forEach(function (k, idx) {{
          var b = byCluster[k];
          var isNoise = k === '-1';
          var shortLbl = (c.cluster_labels[k] || ('Cluster ' + k));
          if (shortLbl.length > 40) shortLbl = shortLbl.slice(0, 39) + '…';
          traces.push({{
            type: 'scatter3d',
            mode: 'markers',
            name: isNoise ? ('Noise (' + b.x.length + ')') : ('#' + k + ' · ' + shortLbl + ' (' + b.x.length + ')'),
            x: b.x, y: b.y, z: b.z, text: b.text,
            customdata: b.customdata,
            hovertemplate: '%{{text}}<extra></extra>',
            marker: {{
              size: isNoise ? 2.6 : (c.n > 1500 ? 3.2 : 4.6),
              color: isNoise ? '#52525b' : PALETTE[idx % PALETTE.length],
              opacity: isNoise ? 0.3 : 0.88
            }}
          }});
        }});
      }}

      var axisPrefix = state.mode === 'pca3' ? 'PC' : (state.mode === 'kpca3' ? 'kPC' : 'UMAP');
      var layout = {{
        paper_bgcolor: '#111115',
        plot_bgcolor: '#111115',
        font: {{ family: 'Plus Jakarta Sans, sans-serif', color: '#e4e4e7', size: 11 }},
        margin: {{ l: 0, r: 0, t: 32, b: 0 }},
        title: {{
          text: c.title + ' — ' + (state.mode === 'pca3' ? 'Linear PCA 3D' : (state.mode === 'kpca3' ? 'RBF Kernel PCA (γ=2.0) 3D' : 'UMAP 3D Manifold')),
          font: {{ size: 14, color: '#fafafa' }}
        }},
        legend: {{
          bgcolor: 'rgba(17,17,21,0.85)',
          bordercolor: '#27272f',
          borderwidth: 1,
          font: {{ size: 10 }}
        }},
        scene: {{
          bgcolor: '#0b0b0f',
          xaxis: {{ title: axisPrefix + '1', gridcolor: '#27272f', zerolinecolor: '#3f3f46' }},
          yaxis: {{ title: axisPrefix + '2', gridcolor: '#27272f', zerolinecolor: '#3f3f46' }},
          zaxis: {{ title: axisPrefix + '3', gridcolor: '#27272f', zerolinecolor: '#3f3f46' }}
        }}
      }};
      var plotEl = document.getElementById('scatter-3d');
      Plotly.newPlot(plotEl, traces, layout, {{ responsive: true, displaylogo: false }});
      if (plotEl.removeAllListeners) {{ plotEl.removeAllListeners('plotly_click'); }}
      plotEl.on('plotly_click', function (ev) {{
        if (!ev || !ev.points || !ev.points.length) return;
        var cd = ev.points[0].customdata;
        if (!cd || !cd.url) return;
        var st = document.getElementById('lab-status');
        st.textContent = '';
        var lbl = document.createTextNode('Selected passage: ');
        var link = document.createElement('a');
        link.href = cd.url;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        link.className = 'ref-link';
        link.textContent = cd.ref + ' (' + cd.url + ') ↗';
        st.appendChild(lbl);
        st.appendChild(link);
        window.open(cd.url, '_blank', 'noopener');
      }});
      var status = document.getElementById('lab-status');
      status.textContent = q
        ? ('Showing ' + matchCount + ' matching passages out of ' + c.n + ' in ' + c.title + '. Click any dot to open it on biblioteca.scriptorium.ro.')
        : ('Showing all ' + c.n + ' passages of ' + c.title + ' (' + c.lang + '). Drag to rotate, scroll to zoom, click any dot to open the passage on biblioteca.scriptorium.ro.');
    }}

    function renderScree() {{
      var traces = DATA.corpora.map(function (c, idx) {{
        var ys = [];
        var acc = 0;
        for (var i = 0; i < c.evr.length; i++) {{
          acc += c.evr[i];
          ys.push(state.screeMode === 'cumulative' ? +(acc * 100).toFixed(2) : +(c.evr[i] * 100).toFixed(2));
        }}
        var xs = ys.map(function (_, i) {{ return 'PC' + (i + 1); }});
        return {{
          type: 'scatter',
          mode: 'lines+markers',
          name: c.title + ' (' + c.lang + ')',
          x: xs,
          y: ys,
          line: {{ width: 2.2, color: PALETTE[idx % PALETTE.length] }},
          marker: {{ size: 6 }}
        }};
      }});
      var layout = {{
        paper_bgcolor: '#111115',
        plot_bgcolor: '#0b0b0f',
        font: {{ family: 'Plus Jakarta Sans, sans-serif', color: '#e4e4e7', size: 11 }},
        margin: {{ l: 55, r: 20, t: 25, b: 45 }},
        xaxis: {{ gridcolor: '#23232c', title: 'Principal Component' }},
        yaxis: {{
          gridcolor: '#23232c',
          title: state.screeMode === 'cumulative' ? 'Cumulative Variance Explained (%)' : 'Explained Variance Ratio (%)'
        }},
        legend: {{ bgcolor: 'rgba(17,17,21,0.85)', bordercolor: '#27272f', borderwidth: 1 }}
      }};
      Plotly.newPlot('scree-plot', traces, layout, {{ responsive: true, displaylogo: false }});
    }}

    function renderAlignments() {{
      DATA.alignments.forEach(function (al, idx) {{
        var axes = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8'];
        var textGrid = al.matrix.map(function (row) {{
          return row.map(function (v) {{ return v.toFixed(2); }});
        }});
        var trace = {{
          type: 'heatmap',
          z: al.matrix,
          x: axes,
          y: axes,
          text: textGrid,
          texttemplate: '%{{text}}',
          textfont: {{ family: 'JetBrains Mono, monospace', size: 10 }},
          zmin: 0,
          zmax: 1,
          colorscale: [
            [0.0, '#09090b'],
            [0.25, '#1e1b4b'],
            [0.55, '#4f46e5'],
            [0.8, '#10b981'],
            [1.0, '#a7f3d0']
          ]
        }};
        var layout = {{
          paper_bgcolor: '#111115',
          plot_bgcolor: '#111115',
          font: {{ family: 'Plus Jakarta Sans, sans-serif', color: '#e4e4e7', size: 11 }},
          margin: {{ l: 55, r: 20, t: 20, b: 55 }},
          xaxis: {{ title: al.label_b }},
          yaxis: {{ title: al.label_a, autorange: 'reversed' }}
        }};
        Plotly.newPlot('align-heatmap-' + idx, [trace], layout, {{ responsive: true, displaylogo: false }});
      }});
    }}

    // Wire corpus selector buttons
    document.querySelectorAll('.corpus-btn').forEach(function (btn) {{
      btn.addEventListener('click', function () {{
        document.querySelectorAll('.corpus-btn').forEach(function (b) {{ b.classList.remove('active'); }});
        btn.classList.add('active');
        state.key = btn.getAttribute('data-key');
        renderScatter3D();
      }});
    }});

    // Wire 3D projection mode buttons
    document.querySelectorAll('#mode-selector .mode-btn').forEach(function (btn) {{
      btn.addEventListener('click', function () {{
        document.querySelectorAll('#mode-selector .mode-btn').forEach(function (b) {{ b.classList.remove('active'); }});
        btn.classList.add('active');
        state.mode = btn.getAttribute('data-mode');
        renderScatter3D();
      }});
    }});

    // Wire keyword spotlight filter
    var searchEl = document.getElementById('point-search');
    if (searchEl) {{
      searchEl.addEventListener('input', function () {{
        state.query = searchEl.value;
        renderScatter3D();
      }});
    }}

    // Wire scree chart mode toggle
    document.querySelectorAll('#scree-mode .mode-btn').forEach(function (btn) {{
      btn.addEventListener('click', function () {{
        document.querySelectorAll('#scree-mode .mode-btn').forEach(function (b) {{ b.classList.remove('active'); }});
        btn.classList.add('active');
        state.screeMode = btn.getAttribute('data-scree');
        renderScree();
      }});
    }});

    // Wire outer work tabs + inner per-work algorithm sub-tabs
    var activeAlgo = 'pca';
    function selectWorkTab(workKey, scrollToResults) {{
      document.querySelectorAll('.work-tab').forEach(function (t) {{
        t.classList.toggle('active', t.getAttribute('data-work') === workKey);
      }});
      document.querySelectorAll('.work-card').forEach(function (c) {{
        c.classList.toggle('active', c.getAttribute('data-work') === workKey);
      }});
      var activeCard = document.querySelector('.work-card[data-work="' + workKey + '"]');
      if (activeCard) {{
        var wanted = activeCard.querySelector('.sub-tab[data-algo="' + activeAlgo + '"]') || activeCard.querySelector('.sub-tab');
        if (wanted) wanted.click();
      }}
      if (scrollToResults) {{
        var sec = document.getElementById('corpus-results');
        if (sec) sec.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
      }}
    }}
    document.querySelectorAll('.work-tab').forEach(function (tab) {{
      tab.addEventListener('click', function () {{
        selectWorkTab(tab.getAttribute('data-work'), false);
      }});
    }});
    document.querySelectorAll('a[href^="#work-"]').forEach(function (link) {{
      link.addEventListener('click', function (ev) {{
        var workKey = link.getAttribute('href').slice('#work-'.length);
        if (document.querySelector('.work-card[data-work="' + workKey + '"]')) {{
          ev.preventDefault();
          selectWorkTab(workKey, true);
        }}
      }});
    }});
    document.querySelectorAll('.sub-tab').forEach(function (tab) {{
      tab.addEventListener('click', function () {{
        var group = tab.getAttribute('data-group');
        var target = tab.getAttribute('data-target');
        activeAlgo = tab.getAttribute('data-algo') || activeAlgo;
        document.querySelectorAll('.sub-tab[data-group="' + group + '"]').forEach(function (t) {{
          t.classList.remove('active');
        }});
        document.querySelectorAll('.sub-panel[data-group="' + group + '"]').forEach(function (p) {{
          p.classList.remove('active');
        }});
        tab.classList.add('active');
        var panel = document.getElementById(target);
        if (panel) panel.classList.add('active');
      }});
    }});

    renderScatter3D();
    renderScree();
    renderAlignments();
    if (typeof renderMathInElement === 'function') {{
      renderMathInElement(document.body, {{
        delimiters: [
          {{ left: '$$', right: '$$', display: true }},
          {{ left: '$', right: '$', display: false }}
        ],
        ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
        throwOnError: false
      }});
    }}
  }})();
  //]]></script>
</body>
</html>
"""

    # Validate strict XML well-formedness before writing
    ET.fromstring(xhtml.encode("utf-8"))
    output_path.write_text(xhtml, encoding="utf-8")
    logger.info("Wrote & XML-validated %s (%.1f KB, %d corpora)", output_path, len(xhtml.encode("utf-8")) / 1024.0, len(bundles))
    return output_path


def _render_cms_axes_table(rows: list[dict[str, Any]], heading: str, ref_urls: dict[str, str], open_by_default: bool = True) -> str:
    if not rows:
        return ""
    by_comp: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        by_comp.setdefault(int(r["component_index"]), []).append(r)

    trs = []
    for comp_idx in sorted(by_comp):
        entries = {e.get("sign", "+"): e for e in by_comp[comp_idx]}
        pos = entries.get("+", {})
        neg = entries.get("-", {})
        score = float((pos or neg).get("eigenvalue", 0.0))
        score_name = (pos or neg).get("score_name", "eigenvalue")
        score_str = f"{score * 100:.2f}%" if score_name == "explained_variance_ratio" else f"{score:.3f}"
        pos_lbl = _esc(pos.get("label", "—"))
        pos_anc = _link_ref(pos.get("anchor_id", ""), ref_urls, "text-indigo-600 hover:underline font-mono text-xs")
        pos_nbs = ", ".join(
            _link_ref(nref, ref_urls, "text-slate-500 hover:text-indigo-600 font-mono text-[11px]")
            for nref in (pos.get("neighbour_ids") or [])[1:4]
        )
        neg_lbl = _esc(neg.get("label", "—"))
        neg_anc = _link_ref(neg.get("anchor_id", ""), ref_urls, "text-indigo-600 hover:underline font-mono text-xs")
        neg_nbs = ", ".join(
            _link_ref(nref, ref_urls, "text-slate-500 hover:text-indigo-600 font-mono text-[11px]")
            for nref in (neg.get("neighbour_ids") or [])[1:4]
        )
        pos_nb_html = f" <span class=\"text-slate-400\">·</span> {pos_nbs}" if pos_nbs else ""
        neg_nb_html = f" <span class=\"text-slate-400\">·</span> {neg_nbs}" if neg_nbs else ""
        trs.append(
            f"<tr class=\"border-b border-slate-200/80 hover:bg-slate-50/70\">"
            f"<td class=\"py-2.5 px-3 font-mono font-bold text-emerald-700 align-top\">C{comp_idx + 1}</td>"
            f"<td class=\"py-2.5 px-3 font-mono text-xs text-slate-600 align-top whitespace-nowrap\">{score_str}</td>"
            f"<td class=\"py-2.5 px-3 align-top\"><span class=\"inline-block px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-800 font-mono text-xs font-bold mr-1\">+</span>{pos_lbl}<div class=\"mt-1\">{pos_anc}{pos_nb_html}</div></td>"
            f"<td class=\"py-2.5 px-3 align-top\"><span class=\"inline-block px-1.5 py-0.5 rounded bg-rose-100 text-rose-800 font-mono text-xs font-bold mr-1\">−</span>{neg_lbl}<div class=\"mt-1\">{neg_anc}{neg_nb_html}</div></td>"
            f"</tr>"
        )
    metric_col = "Variance" if rows[0].get("score_name") == "explained_variance_ratio" else "Eigenvalue"
    open_attr = " open=\"open\"" if open_by_default else ""
    return (
        f"<details{open_attr} class=\"rounded-2xl border border-slate-200 bg-white overflow-hidden\">"
        f"<summary class=\"cursor-pointer px-4 py-3 bg-slate-50 font-sans text-sm font-bold text-slate-800 hover:bg-slate-100\">{_esc(heading)}</summary>"
        f"<div class=\"overflow-x-auto\"><table class=\"w-full text-left text-sm font-sans border-collapse\">"
        f"<thead class=\"bg-slate-50/80 text-xs uppercase font-mono text-slate-500 border-b border-slate-200\">"
        f"<tr><th class=\"py-2 px-3\">Axis</th><th class=\"py-2 px-3\">{metric_col}</th><th class=\"py-2 px-3\">Positive Pole (+)</th><th class=\"py-2 px-3\">Negative Pole (−)</th></tr>"
        f"</thead><tbody>{''.join(trs)}</tbody></table></div></details>"
    )


def _render_cms_cluster_table(rows: list[dict[str, Any]], heading: str, ref_urls: dict[str, str], open_by_default: bool = False) -> str:
    if not rows:
        return ""
    non_noise = [r for r in rows if int(r["cluster_id"]) != -1]
    noise = [r for r in rows if int(r["cluster_id"]) == -1]
    noise_cnt = int(noise[0]["size"]) if noise else 0
    total = sum(int(r["size"]) for r in rows)
    trs = []
    for r in non_noise:
        cid = int(r["cluster_id"])
        size = int(r["size"])
        pct = (100.0 * size / total) if total else 0.0
        lbl = _esc(r.get("label") or "(unlabelled)")
        kws = ", ".join(r.get("keywords") or [])
        reps = r.get("representatives") or []
        rep_links = ", ".join(
            _link_ref(rep.get("ref", ""), ref_urls, "text-indigo-600 hover:underline font-mono text-xs")
            for rep in reps[:4]
            if rep.get("ref")
        )
        kw_html = f"<div class=\"mt-1 font-mono text-xs text-teal-700\">{_esc(kws)}</div>" if kws else ""
        trs.append(
            f"<tr class=\"border-b border-slate-200/80 hover:bg-slate-50/70\">"
            f"<td class=\"py-2.5 px-3 font-mono font-bold text-indigo-700 align-top\">#{cid}</td>"
            f"<td class=\"py-2.5 px-3 font-mono text-xs text-slate-600 align-top whitespace-nowrap\">{size} ({pct:.1f}%)</td>"
            f"<td class=\"py-2.5 px-3 align-top\">{lbl}{kw_html}</td>"
            f"<td class=\"py-2.5 px-3 align-top break-words max-w-[240px]\">{rep_links}</td>"
            f"</tr>"
        )
    open_attr = " open=\"open\"" if open_by_default else ""
    return (
        f"<details{open_attr} class=\"rounded-2xl border border-slate-200 bg-white overflow-hidden\">"
        f"<summary class=\"cursor-pointer px-4 py-3 bg-slate-50 font-sans text-sm font-bold text-slate-800 hover:bg-slate-100\">"
        f"{_esc(heading)} — <span class=\"font-mono text-xs text-slate-500\">{len(non_noise)} clusters, {noise_cnt} noise ({100.0 * noise_cnt / max(1, total):.1f}%)</span>"
        f"</summary>"
        f"<div class=\"overflow-x-auto\"><table class=\"w-full text-left text-sm font-sans border-collapse\">"
        f"<thead class=\"bg-slate-50/80 text-xs uppercase font-mono text-slate-500 border-b border-slate-200\">"
        f"<tr><th class=\"py-2 px-3\">ID</th><th class=\"py-2 px-3\">Size</th><th class=\"py-2 px-3\">Theme &amp; c-TF-IDF Keywords</th><th class=\"py-2 px-3\">Centroid Anchors</th></tr>"
        f"</thead><tbody>{''.join(trs)}</tbody></table></div></details>"
    )


_LANG_CODES: dict[str, str] = {
    "Latin": "LA",
    "Romanian": "RO",
    "French": "FR",
    "English": "EN",
}

_SHORT_TITLES: dict[str, str] = {
    "gesta_hungarorum": "Gesta Hungarorum",
    "ureche_letopisetul": "Ureche",
    "miron_letopisetul": "Miron Costin",
    "biblia_ortodoxa_subcapitole": "Biblia Ortodoxă",
    "biblia_darby_fr_subcapitole": "La Bible Darby",
    "creanga_amintiri": "Creangă (Amintiri)",
    "london_white_fang": "White Fang",
    "hugo_notre_dame": "Notre-Dame (FR)",
    "hugo_notre_dame_en": "Notre-Dame (EN)",
}


def _serialize_axes_tab(tab_id: str, label: str, heading: str, rows: list[dict[str, Any]], ref_urls: dict[str, str]) -> dict[str, Any] | None:
    if not rows:
        return None
    by_comp: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        by_comp.setdefault(int(r["component_index"]), []).append(r)

    def _ref_obj(ref: str) -> dict[str, str] | None:
        if not ref:
            return None
        return {"ref": ref, "url": ref_urls.get(ref, BIBLIOTECA_BASE_URL)}

    axes: list[dict[str, Any]] = []
    for comp_idx in sorted(by_comp):
        entries = {e.get("sign", "+"): e for e in by_comp[comp_idx]}
        pos = entries.get("+", {})
        neg = entries.get("-", {})
        score = float((pos or neg).get("eigenvalue", 0.0))
        score_name = str((pos or neg).get("score_name", "eigenvalue"))
        axes.append(
            {
                "component_index": comp_idx,
                "score": round(score, 6),
                "score_name": score_name,
                "pos_label": str(pos.get("label", "")),
                "pos_anchor": _ref_obj(str(pos.get("anchor_id", ""))),
                "pos_neighbours": [
                    obj for ref in (pos.get("neighbour_ids") or [])[1:30] if (obj := _ref_obj(str(ref))) is not None
                ],
                "neg_label": str(neg.get("label", "")),
                "neg_anchor": _ref_obj(str(neg.get("anchor_id", ""))),
                "neg_neighbours": [
                    obj for ref in (neg.get("neighbour_ids") or [])[1:30] if (obj := _ref_obj(str(ref))) is not None
                ],
            }
        )
    metric_col = "Variance" if rows[0].get("score_name") == "explained_variance_ratio" else "Eigenvalue"
    return {
        "id": tab_id,
        "label": label,
        "heading": heading,
        "kind": "axes",
        "metric_col": metric_col,
        "axes": axes,
    }


def _serialize_cluster_tab(tab_id: str, label: str, heading: str, rows: list[dict[str, Any]], ref_urls: dict[str, str]) -> dict[str, Any] | None:
    if not rows:
        return None
    non_noise = [r for r in rows if int(r["cluster_id"]) != -1]
    noise = [r for r in rows if int(r["cluster_id"]) == -1]
    noise_cnt = int(noise[0]["size"]) if noise else 0
    total = sum(int(r["size"]) for r in rows)
    out_rows = []
    for r in non_noise:
        size = int(r["size"])
        reps = [
            {"ref": str(rep["ref"]), "url": ref_urls.get(str(rep["ref"]), BIBLIOTECA_BASE_URL)}
            for rep in (r.get("representatives") or [])[:30]
            if rep.get("ref")
        ]
        out_rows.append(
            {
                "cluster_id": int(r["cluster_id"]),
                "size": size,
                "pct": round((100.0 * size / total) if total else 0.0, 1),
                "label": str(r.get("label") or ""),
                "keywords": [str(k) for k in (r.get("keywords") or [])],
                "representatives": reps,
            }
        )
    return {
        "id": tab_id,
        "label": label,
        "heading": heading,
        "kind": "clusters",
        "clusters": {
            "total": total,
            "n_clusters": len(non_noise),
            "noise_count": noise_cnt,
            "noise_pct": round(100.0 * noise_cnt / max(1, total), 1),
            "rows": out_rows,
        },
    }


def _serialize_results_payload(bundles: list[dict[str, Any]]) -> dict[str, Any]:
    corpora = []
    for idx, b in enumerate(bundles, start=1):
        s = b["spec"]
        ref_urls = b["ref_urls"]
        algos = [
            _serialize_axes_tab("pca", "Linear PCA", "Linear PCA — Top 30 Eigen-Axes (+ vs. − Poles)", b["pca_rows"], ref_urls),
            _serialize_axes_tab("kpca", b["kpca_label"], f"{b['kpca_label']} — Top 30 Kernel Axes", b["kpca_rows"], ref_urls),
            _serialize_cluster_tab("bertopic", "BERTopic", "BERTopic Clusters (Precomputed bge-m3 + c-TF-IDF)", b["bertopic_rows"], ref_urls),
            _serialize_cluster_tab("hdbscan", "HDBSCAN", "HDBSCAN Density Clusters", b["hdbscan_rows"], ref_urls),
        ]
        if b["kpca_cosine_rows"] and b["kpca_label"] != "Cosine kPCA":
            algos.insert(
                2,
                _serialize_axes_tab(
                    "kpca_cos",
                    "Cosine kPCA",
                    "Cosine Kernel PCA (Unit-Sphere Identity Check)",
                    b["kpca_cosine_rows"],
                    ref_urls,
                ),
            )
        corpora.append(
            {
                "key": s["key"],
                "index": idx,
                "title": s["title"],
                "short_title": _SHORT_TITLES.get(s["key"], s["title"]),
                "author": s["author"],
                "lang": s["lang"],
                "lang_code": _LANG_CODES.get(s["lang"], s["lang"][:2].upper()),
                "era": s["era"],
                "group": s["group"],
                "unit": s["unit"],
                "color": s["color"],
                "summary": s["summary"],
                "root_url": b["root_url"],
                "n_rows": b["n_rows"],
                "evr1_pct": round(b["evr"][0] * 100.0, 2),
                "evr8_pct": round(sum(b["evr"][:8]) * 100.0, 1),
                "evr30_pct": round(sum(b["evr"][:30]) * 100.0, 1),
                "n_bertopic": sum(1 for r in b["bertopic_rows"] if int(r["cluster_id"]) != -1),
                "algorithms": [a for a in algos if a is not None],
            }
        )
    return {"corpora": corpora}


def export_to_blog_cms(
    cms_dir: Path,
    slug: str = "eigenbooks-spectral-decomposition-clustering",
    date_prefix: str = "2026-11-25",
    bundles: list[dict[str, Any]] | None = None,
) -> tuple[Path, Path, Path]:
    """Export CMS-ready DOMPurify-safe XHTML, post metadata JSON, and static Plotly/results JSON."""
    if bundles is None:
        bundles = []
        for spec in CORPUS_CATALOG:
            bundle = _load_corpus_bundle(spec)
            if bundle is not None:
                bundles.append(bundle)
    by_key: dict[str, dict[str, Any]] = {b["spec"]["key"]: b for b in bundles}

    alignments: list[dict[str, Any]] = []
    if "biblia_ortodoxa_subcapitole" in by_key and "biblia_darby_fr_subcapitole" in by_key:
        alignments.append(
            _cross_lingual_alignment(
                by_key["biblia_ortodoxa_subcapitole"],
                by_key["biblia_darby_fr_subcapitole"],
                "Romanian Orthodox Bible vs. French Darby Bible",
            )
        )
    if "hugo_notre_dame" in by_key and "hugo_notre_dame_en" in by_key:
        alignments.append(
            _cross_lingual_alignment(
                by_key["hugo_notre_dame"],
                by_key["hugo_notre_dame_en"],
                "Notre-Dame de Paris: Original French vs. English Translation",
            )
        )

    # 1. Write static visualization JSON and structured per-work tabbed results JSON under public/data/
    viz_path = cms_dir / "public" / "data" / "eigenbooks-viz.json"
    results_json_path = cms_dir / "public" / "data" / "eigenbooks-results.json"
    viz_path.parent.mkdir(parents=True, exist_ok=True)
    viz_payload = {
        "title": f"Explore All {len(bundles)} Literary Corpora in 3D Embedding Space",
        "subtitle": "Click any dot to open the passage on biblioteca.scriptorium.ro",
        "source_label": "on biblioteca.scriptorium.ro",
        "kpca_label": "RBF kPCA (γ=2.0)",
        "corpora": [b["viz"] for b in bundles],
        "alignments": alignments,
    }
    viz_path.write_text(json.dumps(viz_payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    results_payload = {
        "label": f"Select Corpus ({len(bundles)} Works)",
        **_serialize_results_payload(bundles),
    }
    results_json_path.write_text(json.dumps(results_payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    total_docs = sum(b["n_rows"] for b in bundles)
    total_topics = sum(sum(1 for r in b["bertopic_rows"] if int(r["cluster_id"]) != -1) for b in bundles)
    max_evr1 = max((b["evr"][0] * 100.0 for b in bundles), default=10.0)
    max_evr30 = max((sum(b["evr"][:30]) * 100.0 for b in bundles), default=65.0)

    group_badge_cls = {
        "Chronicles": "bg-amber-50 text-amber-800 border-amber-200",
        "Scripture": "bg-violet-50 text-violet-800 border-violet-200",
        "Literature": "bg-sky-50 text-sky-800 border-sky-200",
    }

    overview_trs = []
    for idx, b in enumerate(bundles, start=1):
        s = b["spec"]
        evr1 = b["evr"][0] * 100.0
        evr30 = sum(b["evr"][:30]) * 100.0
        evr1_bar = min(100.0, max(8.0, (evr1 / max(1.0, max_evr1)) * 100.0))
        evr30_bar = min(100.0, max(10.0, (evr30 / max(1.0, max_evr30)) * 100.0))
        n_bt = sum(1 for r in b["bertopic_rows"] if int(r["cluster_id"]) != -1)
        g_cls = group_badge_cls.get(s["group"], "bg-slate-100 text-slate-700 border-slate-200")
        row_bg = "bg-white" if idx % 2 == 1 else "bg-slate-50/50"
        overview_trs.append(
            f"<tr class=\"{row_bg} hover:bg-indigo-50/40 transition-colors border-b border-slate-200/70 last:border-b-0\">"
            f"<td class=\"border-0 py-3.5 px-4 align-middle\">"
            f"<div class=\"flex items-start gap-2.5\">"
            f"<span class=\"mt-1.5 w-2.5 h-2.5 rounded-full shrink-0\" style=\"background:{s['color']}\"></span>"
            f"<div class=\"min-w-0\">"
            f"<div class=\"flex flex-wrap items-center gap-x-2 gap-y-0.5\">"
            f"<a href=\"#work-{s['key']}\" class=\"font-serif text-[15px] font-bold text-slate-900 hover:text-indigo-700 no-underline hover:underline\">{_esc(s['title'])}</a>"
            f"<a href=\"{_esc(b['root_url'])}\" target=\"_blank\" rel=\"noopener noreferrer\" title=\"Open on biblioteca.scriptorium.ro\" class=\"font-mono text-[11px] text-indigo-600 hover:text-indigo-800 no-underline\">↗</a>"
            f"</div>"
            f"<div class=\"text-xs text-slate-500 font-sans mt-0.5\">{_esc(s['author'])} · <span class=\"text-slate-400\">{_esc(s['era'])}</span></div>"
            f"</div></div></td>"
            f"<td class=\"border-0 py-3.5 px-3 align-middle whitespace-nowrap\">"
            f"<span class=\"inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full border font-mono text-[11px] font-semibold {g_cls}\">{_esc(s['lang'])}</span>"
            f"<div class=\"text-[10px] font-mono uppercase tracking-wider text-slate-400 mt-1 pl-1\">{_esc(s['group'])}</div>"
            f"</td>"
            f"<td class=\"border-0 py-3.5 px-3 align-middle whitespace-nowrap\">"
            f"<div class=\"font-mono text-sm font-bold text-slate-900\">{b['n_rows']:,}</div>"
            f"<div class=\"text-[10px] font-mono uppercase text-slate-400\">{_esc(s['unit'])}</div>"
            f"</td>"
            f"<td class=\"border-0 py-3.5 px-3 align-middle min-w-[110px]\">"
            f"<div class=\"font-mono text-xs font-bold text-slate-800\">{evr1:.2f}%</div>"
            f"<div class=\"mt-1.5 h-1.5 w-full rounded-full bg-slate-200/80 overflow-hidden\">"
            f"<div class=\"h-full rounded-full bg-gradient-to-r from-indigo-500 to-violet-500\" style=\"width:{evr1_bar:.0f}%\"></div>"
            f"</div></td>"
            f"<td class=\"border-0 py-3.5 px-3 align-middle min-w-[115px]\">"
            f"<div class=\"font-mono text-xs font-bold text-emerald-700\">{evr30:.1f}%</div>"
            f"<div class=\"mt-1.5 h-1.5 w-full rounded-full bg-slate-200/80 overflow-hidden\">"
            f"<div class=\"h-full rounded-full bg-gradient-to-r from-emerald-500 to-teal-400\" style=\"width:{evr30_bar:.0f}%\"></div>"
            f"</div></td>"
            f"<td class=\"border-0 py-3.5 px-4 align-middle text-right whitespace-nowrap\">"
            f"<span class=\"inline-flex items-center justify-center min-w-[2.2rem] px-2.5 py-1 rounded-xl bg-indigo-50 border border-indigo-200/80 font-mono text-xs font-bold text-indigo-800\">{n_bt}</span>"
            f"</td>"
            f"</tr>"
        )

    alignment_summary_items = []
    for al in alignments:
        pcs = ", ".join(f"{v:.3f}" for v in al["principal_cosines"][:6])
        alignment_summary_items.append(
            f"<li><strong>{_esc(al['title'])}</strong>: mean-vector cosine $\\cos(\\mu_A, \\mu_B) = {al['mean_cosine']:.4f}$, "
            f"mean top-12 subspace overlap ${al['subspace_overlap_mean']:.3f}$, "
            f"top-6 canonical principal cosines $[{pcs}]$.</li>"
        )

    cms_xhtml = rf"""<h1>Eigenbooks: Spectral Decomposition and Semantic Clustering Across Nine Literary Corpora</h1>

<div class="prose prose-slate max-w-none space-y-6">
  <p class="text-lg leading-relaxed font-serif">
    Can unsupervised numerical linear algebra recover the <em>intellectual skeleton</em> of a book — separating
    Victor Hugo's Gothic architectural essays from Esmeralda's drama, Ion Creangă's Fălticeni grammar drills from
    Humulești nostalgia, or Old Testament genealogies from Pauline ethics — without ever telling the algorithm what a
    chapter, genre, or protagonist is?
  </p>

  <p>
    To test this across languages and centuries, we assembled <strong>{len(bundles)} complete works ({total_docs:,} passages)</strong>
    in a single shared 1024-dimensional embedding geometry: <code>bge-m3:567m-fp16</code> (unit $\ell_2$-normalized onto the
    hypersphere $S^{{1023}}$). Seven works were pulled directly from Qdrant (<code>biblioteca_paragraphs_bge_m3</code>)
    with paragraph texts fetched from <a href="https://biblioteca.scriptorium.ro" target="_blank" rel="noopener noreferrer">biblioteca.scriptorium.ro</a>
    and verified bit-for-bit against Qdrant's SHA-256 digests — then filtered to remove section headings (<code>_1</code>),
    <code>about/</code> digital-edition colophons, and fragments shorter than 100 characters so that every point represents
    substantive prose rather than rubric titles or 1-word exclamations. The Romanian Orthodox and French Darby Bibles were
    embedded at chapter granularity (all 2,518 chapters exceed 130 characters) with the exact same
    <code>bge-m3:567m-fp16</code> model in Ollama. Every corpus is stored locally as a flat FAISS index
    (<code>IndexFlatIP</code>) paired with <code>records.jsonl</code>, giving instant cosine kNN search alongside exact
    <code>float32</code> matrix reconstruction for NumPy, scikit-learn, UMAP, HDBSCAN, and BERTopic.
  </p>

  <div class="not-prose my-8 rounded-3xl border border-slate-200/90 bg-white shadow-sm overflow-hidden">
    <div class="px-5 py-3.5 bg-gradient-to-r from-slate-900 via-slate-900 to-indigo-950 text-white flex flex-wrap items-center justify-between gap-2">
      <div class="flex items-center gap-2">
        <span class="w-2 h-2 rounded-full bg-emerald-400"></span>
        <span class="font-mono text-xs uppercase tracking-widest text-emerald-300 font-bold">Corpus Matrix · {len(bundles)} Literary Works</span>
      </div>
      <span class="font-mono text-[11px] text-slate-300">bge-m3 (1024-D) · qwen3.5:9b · K = 30 reps</span>
    </div>
    <div class="overflow-x-auto">
      <table class="table w-full text-left font-sans border-collapse m-0 text-sm">
        <thead>
          <tr class="bg-slate-50 border-b border-slate-200 text-[11px] uppercase tracking-wider font-mono text-slate-500">
            <th class="border-0 bg-slate-50 py-3 px-4 font-semibold">Work &amp; Author</th>
            <th class="border-0 bg-slate-50 py-3 px-3 font-semibold">Language</th>
            <th class="border-0 bg-slate-50 py-3 px-3 font-semibold">Indexed Units</th>
            <th class="border-0 bg-slate-50 py-3 px-3 font-semibold">PC1 Variance</th>
            <th class="border-0 bg-slate-50 py-3 px-3 font-semibold">Top-30 PCA</th>
            <th class="border-0 bg-slate-50 py-3 px-4 text-right font-semibold">Topics</th>
          </tr>
        </thead>
        <tbody>{''.join(overview_trs)}</tbody>
        <tfoot>
          <tr class="bg-slate-900 text-slate-200 font-mono text-xs">
            <td class="border-0 py-3 px-4 font-bold text-white" colspan="2">Total · {len(bundles)} works across LA · RO · FR · EN</td>
            <td class="border-0 py-3 px-3 font-bold text-emerald-300">{total_docs:,} units</td>
            <td class="border-0 py-3 px-3 text-slate-400" colspan="2">Click any work title to jump to its tabbed breakdown ↓</td>
            <td class="border-0 py-3 px-4 text-right font-bold text-indigo-300">{total_topics} topics</td>
          </tr>
        </tfoot>
      </table>
    </div>
  </div>

  <h2>1. Interactive 3D Embedding Lab, Eigenvalue Spectra &amp; Cross-Lingual Alignment</h2>
  <p>
    Below is the live WebGL laboratory powered by Plotly.js (streamed dynamically on demand from
    <code>data/eigenbooks-viz.json</code>, adding 0 KB to the blog's landing bundle). Switch between all {len(bundles)} works,
    toggle <strong>UMAP 3D</strong>, <strong>Linear PCA 3D</strong>, and <strong>RBF Kernel PCA ($\gamma = 2.0$) 3D</strong>,
    spotlight any keyword in 3D space, and click any dot to jump straight to that passage on
    <code>biblioteca.scriptorium.ro</code>.
  </p>

  <div data-widget="embedding-explorer" data-src="data/eigenbooks-viz.json" class="my-8 not-prose"></div>

  <h2>2. Cross-Lingual Language-Invariance Findings</h2>
  <p>
    Because <code>bge-m3</code> projects Romanian, French, English, and Latin into the exact same 1024-D Euclidean space,
    the two bilingual pairs let us measure directly whether spectral eigen-directions reflect language or underlying meaning
    via the $12 \times 12$ eigenvector alignment matrix $|V_A V_B^\top|$ and the canonical principal angles
    $\cos\theta_k = \sigma_k(V_A V_B^\top)$:
  </p>
  <ul>{''.join(alignment_summary_items)}</ul>
  <p>
    Even across different languages and different paragraph segmentations ({by_key['hugo_notre_dame']['n_rows']:,} French vs. {by_key['hugo_notre_dame_en']['n_rows']:,} English body paragraphs in
    <em>Notre-Dame de Paris</em>), the leading PCA subspaces align tightly in the shared multilingual embedding space —
    demonstrating that the dominant eigen-directions capture narrative and thematic structure rather than surface language.
  </p>

  <h2>3. Mathematical Foundations &amp; Key Geometric Insights</h2>
  <div class="my-8 p-6 rounded-2xl bg-zinc-900 text-zinc-100 border border-zinc-800 shadow-xl not-prose">
    <div class="flex items-center justify-between border-b border-zinc-800 pb-3 mb-5">
      <span class="text-xs uppercase tracking-widest font-mono text-emerald-400">Mathematical Formulation</span>
      <span class="text-xs text-zinc-400 font-mono">Hypersphere $S^{{1023}}$ · RKHS · Density Trees</span>
    </div>
    <div class="space-y-5 text-sm leading-relaxed">
      <div>
        <div class="font-sans font-semibold text-zinc-100 mb-1">1. Linear PCA on Centered Unit Embeddings</div>
        <div class="pl-2 text-emerald-300 overflow-x-auto">$$\tilde{{X}} = X - \mathbf{{1}}\mu^\top, \qquad C = \frac{{1}}{{N-1}}\tilde{{X}}^\top\tilde{{X}} = V\Lambda V^\top, \qquad z_{{ik}} = \tilde{{x}}_i \cdot v_k$$</div>
        <p class="pl-2 text-xs text-zinc-400 font-sans mt-1">Extracts orthogonal axes $v_k \in \mathbb{{R}}^{{1024}}$ maximizing global variance; projection coordinates $z_{{ik}}$ order every passage from the $-$ pole to the $+$ pole.</p>
      </div>
      <div>
        <div class="font-sans font-semibold text-zinc-100 mb-1">2. Cosine Kernel PCA Identity on $S^{{d-1}}$</div>
        <div class="pl-2 text-emerald-300 overflow-x-auto">$$\|x_i\|_2 = 1 \;\Longrightarrow\; K_{{\cos}}(x_i, x_j) = x_i^\top x_j \;\Longrightarrow\; \tilde{{K}}_{{\cos}} = \tilde{{X}}\tilde{{X}}^\top$$</div>
        <p class="pl-2 text-xs text-zinc-400 font-sans mt-1">On unit-normalized embeddings, Cosine kPCA recovers the exact same eigenspace as Linear PCA (up to sign flip), with kernel eigenvalues $\lambda_{{\mathrm{{kpca}}}} = (N - 1)\,\mathrm{{Var}}_k$.</p>
      </div>
      <div>
        <div class="font-sans font-semibold text-zinc-100 mb-1">3. Non-Linear RBF Kernel PCA &amp; the $\gamma$-Scale Trap</div>
        <div class="pl-2 text-emerald-300 overflow-x-auto">$$K_{{\mathrm{{RBF}}}}(x_i, x_j) = \exp\!\left(-\gamma\|x_i - x_j\|_2^2\right) = \exp\!\left(-2\gamma(1 - x_i^\top x_j)\right)$$</div>
        <p class="pl-2 text-xs text-zinc-400 font-sans mt-1">With scikit-learn's default $\gamma = 1/1024 \approx 0.001$, the Taylor expansion $e^u \approx 1 + u$ is 99.8% linear! Setting $\gamma \approx 2.0$ unlocks genuine RKHS curvature and exponentially damps distant outliers (pushing <em>Gesta</em>'s Wikisource colophon from $\mathrm{{PC}}_1/\mathrm{{PC}}_3$ down to Component 6).</p>
      </div>
      <div>
        <div class="font-sans font-semibold text-zinc-100 mb-1">4. UMAP + HDBSCAN (<code class="bg-zinc-800 text-emerald-300 px-1.5 py-0.5 rounded">eom</code> vs. <code class="bg-zinc-800 text-emerald-300 px-1.5 py-0.5 rounded">leaf</code>) &amp; BERTopic c-TF-IDF</div>
        <div class="pl-2 text-emerald-300 overflow-x-auto">$$d_{{\mathrm{{mreach}}}}(a, b) = \max\!\bigl(\mathrm{{core}}_k(a),\, \mathrm{{core}}_k(b),\, d(a, b)\bigr), \qquad W_{{t,c}} = \mathrm{{tf}}_{{t,c}} \cdot \log\!\left(1 + \frac{{A}}{{\mathrm{{tf}}_t}}\right)$$</div>
        <p class="pl-2 text-xs text-zinc-400 font-sans mt-1">When a book contains a tiny pocket of distant metadata/headings (<em>Gesta</em>'s 7 Wikisource colophons or <em>White Fang</em>'s 24 chapter titles), HDBSCAN's default <code class="bg-zinc-800 text-emerald-300 px-1.5 py-0.5 rounded">eom</code> merges all body paragraphs into one mass; switching to <code class="bg-zinc-800 text-emerald-300 px-1.5 py-0.5 rounded">leaf</code> cleanly splits 8–19 sharp narrative episodes.</p>
      </div>
      <div>
        <div class="font-sans font-semibold text-zinc-100 mb-1">5. Cross-Lingual Subspace Overlap (Canonical Principal Angles)</div>
        <div class="pl-2 text-emerald-300 overflow-x-auto">$$\sigma_k = \mathrm{{SVD}}\!\left(V_A V_B^\top\right)_k = \cos\theta_k, \qquad \overline{{\cos\theta}} = \frac{{1}}{{m}}\sum_{{k=1}}^{{m}}\sigma_k$$</div>
        <p class="pl-2 text-xs text-zinc-400 font-sans mt-1">The singular values of $V_A V_B^\top$ measure how tightly the top-$m$ eigen-subspaces of two languages align inside the shared multilingual geometry.</p>
      </div>
    </div>
  </div>

  <h2>4. How Axes &amp; Clusters Were Labelled</h2>
  <p>
    Every PCA and Kernel PCA axis in $\mathbb{{R}}^{{1024}}$ is a <strong>contrast between two opposing poles</strong>. Using our
    <code>top-projection</code> strategy, we rank all passages along component $j$ by projection score $z_{{ij}}$, extract the
    <strong>$K = 30$ highest-scoring ($+$)</strong> and <strong>$K = 30$ lowest-scoring ($-$)</strong> passages (so labels reflect
    the broad pole rather than a few idiosyncratic outliers), and ask <code>qwen3.5:9b</code> on Ollama (with <code>think=False</code>
    so reasoning traces don't starve the concise label) to name the shared theme of each pole. For HDBSCAN and BERTopic,
    representatives are the <strong>$K = 30$ passages closest to each cluster centroid in the original 1024-D embedding
    space</strong>, paired with BERTopic's exact c-TF-IDF n-gram keywords.
  </p>
  <p>
    <strong>Transparency &amp; verification:</strong> Even a strong 9B model can occasionally slip on archaic 17th-century
    Romanian or medieval Latin proper nouns. Pairing every LLM theme label with up to 30 clickable
    <code>biblioteca.scriptorium.ro</code> passage anchors (with instant hover previews) and BERTopic's c-TF-IDF keywords
    lets any reader audit every axis and cluster in one click.
  </p>

  <h2>5. Detailed Results Across All {len(bundles)} Corpora</h2>
  <p>
    Select a corpus tab below, then switch between <strong>Linear PCA</strong>, <strong>RBF / Cosine Kernel PCA</strong>,
    <strong>BERTopic</strong>, and <strong>HDBSCAN</strong> inside the work panel (your chosen algorithm tab stays active
    as you switch works for instant side-by-side comparison). Every anchor and centroid reference links directly to its
    live passage on <code>biblioteca.scriptorium.ro</code>.
  </p>
  <div data-widget="tabbed-analysis" data-src="data/eigenbooks-results.json" data-title="Select Corpus (9 Works)" class="my-8 not-prose"></div>

  <h2>6. Best JavaScript Visualization Libraries for High-Dimensional Embeddings</h2>
  <ul>
    <li><strong>Plotly.js</strong> <em>(used here)</em>: Best zero-boilerplate WebGL <code>scatter3d</code>, scree plots, and heatmaps up to ~25,000 points with rich hover cards and legend filtering.</li>
    <li><strong>regl-scatterplot</strong>: Purpose-built raw WebGL for 2D UMAP/PCA clouds from 10k to 2M+ points with 60fps lasso selection.</li>
    <li><strong>Three.js</strong>: Unbeatable for bespoke 3D manifolds, animated narrative trajectories $t \mapsto (\mathrm{{PC}}_1(t),\, \mathrm{{PC}}_2(t),\, \mathrm{{PC}}_3(t))$, and custom GLSL shaders.</li>
    <li><strong>deck.gl</strong>: Ideal for layered <code>ScatterplotLayer</code> + <code>PathLayer</code> + <code>TextLayer</code> views over large multi-book corpora.</li>
    <li><strong>D3.js / Observable Plot</strong>: Best typographic precision for 2D editorial charts, HDBSCAN condensed-tree dendrograms, and cross-corpus Sankey diagrams.</li>
    <li><strong>Apple Embedding Atlas</strong>: WebGPU/WASM explorer with automatic density contours and zoom-adaptive topic labels for millions of vectors.</li>
  </ul>

  <h2>7. Future Directions</h2>
  <ul>
    <li><strong>Supervised Metric Learning (NCA &amp; LMNN)</strong>: Learn linear projections $L$ supervised by known labels (author, testament, historical century, protagonist) to find the exact directions separating Ureche from Miron Costin or the Synoptic Gospels from John.</li>
    <li><strong>Narrative Trajectories in Eigen-Space</strong>: Because reading order <code>(section, paragraph_num)</code> is preserved in FAISS, plot smoothed trajectories $t \mapsto (\mathrm{{PC}}_1(t),\, \mathrm{{PC}}_2(t),\, \mathrm{{PC}}_3(t))$ tracing how <em>White Fang</em> travels from Yukon survival to domestication or how <em>Notre-Dame de Paris</em> oscillates between Gothic architecture essays and melodrama.</li>
    <li><strong>Boilerplate Filtering &amp; Procrustes Alignment</strong>: Strip Wikisource <code>about/</code> colophons and single-line headings before fitting, and apply Orthogonal Procrustes alignment ($\min_R \|X_A R - X_B\|_F$) after removing the language mean shift $\mu_{{\mathrm{{FR}}}} - \mu_{{\mathrm{{EN}}}}$.</li>
    <li><strong>Language-Specific Stopwords &amp; Multi-Model Audits</strong>: Add custom Latin/archaic-Romanian/French stopword lists to BERTopic's c-TF-IDF vectorizer and cross-check <code>qwen3.5:9b</code> labels against larger reasoning models.</li>
  </ul>
</div>
"""

    # Validate strict XML well-formedness by wrapping fragment in a root element
    ET.fromstring(f"<root>{cms_xhtml}</root>".encode("utf-8"))

    storage_slug = f"{date_prefix}_{slug}"
    posts_dir = cms_dir / ".content-work" / "@petru" / "content" / "posts"
    meta_dir = posts_dir / "meta"
    posts_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    post_path = posts_dir / f"{storage_slug}.xhtml"
    meta_path = meta_dir / f"{storage_slug}.json"
    post_path.write_text(cms_xhtml, encoding="utf-8")

    meta_obj = {
        "id": f"post-{slug}",
        "slug": slug,
        "title": "Eigenbooks: Spectral Decomposition & Semantic Clustering Across Nine Literary Corpora",
        "subtitle": f"PCA, Cosine & RBF Kernel PCA, HDBSCAN, and BERTopic over {total_docs:,} bge-m3 passages from medieval chronicles, parallel Bibles, Creangă, Jack London, and bilingual Victor Hugo",
        "excerpt": (
            "Applying Linear PCA, Cosine & non-linear RBF Kernel PCA, HDBSCAN, and BERTopic to 1024-D bge-m3 "
            "embeddings of nine works across Latin, Romanian, French, and English — with live 3D WebGL visualizations "
            "and cross-lingual subspace alignment heatmaps."
        ),
        "format": "xhtml",
        "language": "en",
        "authorUsername": "petru",
        "publishedAt": f"{date_prefix}T16:00:00.000Z",
        "updatedAt": f"{date_prefix}T16:00:00.000Z",
        "isPublished": True,
        "featured": True,
        "coverImage": "https://images.unsplash.com/photo-1507842217343-583bb7270b66?w=1200&auto=format&fit=crop&q=80",
        "tags": ["Mate", "NLP", "Literatură", "PCA"],
        "readingTimeMinutes": 18,
        "requiredPackages": ["plotly", "katex"],
        "seo": {
            "metaTitle": "Eigenbooks: PCA, Kernel PCA, HDBSCAN & BERTopic Across 9 Literary Corpora | Scriptorium",
            "metaDescription": (
                "Interactive 3D spectral decomposition and topic clustering over bge-m3 embeddings of Gesta Hungarorum, "
                "Ureche, Miron Costin, Romanian & French Bibles, Ion Creangă, Jack London, and bilingual Notre-Dame de Paris."
            ),
            "keywords": [
                "PCA", "Kernel PCA", "RBF Kernel", "HDBSCAN", "BERTopic", "FAISS", "bge-m3",
                "Digital Humanities", "Gesta Hungarorum", "Notre-Dame de Paris", "Eigenbible",
            ],
        },
    }
    meta_path.write_text(json.dumps(meta_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info(
        "Exported CMS post (%s, %.1f KB), meta (%s), viz JSON (%s, %.1f KB), results JSON (%s, %.1f KB)",
        post_path,
        len(cms_xhtml.encode("utf-8")) / 1024.0,
        meta_path,
        viz_path,
        viz_path.stat().st_size / 1024.0,
        results_json_path,
        results_json_path.stat().st_size / 1024.0,
    )
    return post_path, meta_path, viz_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cms-dir", type=Path, default=None, help="also export CMS-ready XHTML, meta JSON, and public/data/*.json into scriptorium-blog-cms")
    parser.add_argument("--skip-standalone", action="store_true", help="skip regenerating standalone eigenbooks_synthesis.xhtml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    bundles = [b for spec in CORPUS_CATALOG if (b := _load_corpus_bundle(spec)) is not None]
    if not args.skip_standalone:
        out = build_xhtml(bundles=bundles)
        print(f"Generated valid XHTML report: {out}")
    if args.cms_dir is not None:
        post_path, meta_path, viz_path = export_to_blog_cms(args.cms_dir, bundles=bundles)
        print(f"Exported to CMS: {post_path}, {meta_path}, {viz_path}")


if __name__ == "__main__":
    main()
