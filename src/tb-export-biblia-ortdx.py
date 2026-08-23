#!/usr/bin/env python3
"""
Export a Bible translation from textbase.scriptorium.ro into per-chapter markdown files.

Site structure (a book's table of contents can nest to any depth - e.g. the French
Darby Psalms are split into "Livre premier".."Livre quatrieme" before the individual
psalms - so discovery walks the tree breadth-first until it hits pages with no
<ul id="toc">):

    <testament root>                      (has a <ul id="toc"> of books)
      <book_slug>                          (has a <ul id="toc"> of chapters, or
                                             of further subdivisions - e.g. les_psaumes)
        <chapter_slug>                     (leaf page; also served as <chapter_slug>.txt)
        <subdivision_slug>                 (has its own <ul id="toc"> of chapters)
          <chapter_slug>

The plain-text export (append ".txt" to a chapter URL) gives one line per header and
then the verses. Two verse layouts are supported:

  - one verse per line, "N. text" (biblia_ortodoxa):
        1. Dacă vreun suflet va păcătui prin aceea că ...

  - a verse's text wraps across several raw lines, "N text", with inline footnotes
    in brackets that always close on the line they open on (la_bible_ancien/
    nouveau_testament, i.e. the French Darby translation):
        1 Au commencement Dieu  [ — v. 1 : en hébreu : Élohim...]
          créa les cieux et la terre.

This script walks the whole tree and writes, per book, one markdown file per chapter
with verse numbers and footnotes stripped, one verse per line:

    output/vechiul/facerea/facerea-1.md
    output/vechiul/facerea/facerea-2.md
    ...
    output/ancien/les_psaumes/les_psaumes-1.md
    ...
"""
import argparse
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Testament:
    path: str      # path below the site root, e.g. "/anon/biblia_ortodoxa/vechiul_testament"
    dirname: str    # output subdirectory name, e.g. "vechiul"


@dataclass(frozen=True)
class BibleSource:
    key: str
    testaments: list


BIBLES = {
    "ortodox": BibleSource("ortodox", [
        Testament("/anon/biblia_ortodoxa/vechiul_testament", "vechiul"),
        Testament("/anon/biblia_ortodoxa/noul_testament", "noul"),
    ]),
    "darby": BibleSource("darby", [
        Testament("/anon/la_bible_ancien_testament", "ancien"),
        Testament("/anon/la_bible_nouveau_testament", "nouveau"),
    ]),
}


@dataclass(frozen=True)
class Chapter:
    href: str
    label: str


@dataclass(frozen=True)
class Book:
    slug: str
    name: str
    chapters: list  # list[Chapter]


class TextbaseClient:
    """Fetches pages from textbase.scriptorium.ro and parses their table-of-contents lists."""

    BASE = "https://textbase.scriptorium.ro"
    USER_AGENT = "Mozilla/5.0 (compatible; biblia-export-script/1.0)"

    TOC_RE = re.compile(r'id="toc"\s*>(.*?)</ul>', re.S)
    TOC_ITEM_RE = re.compile(r'href="([^"]+)">\s*<span>([^<]*)</span>')

    def fetch(self, path: str, retries: int = 3, timeout: int = 20) -> str:
        url = f"{self.BASE}{path}"
        req = urllib.request.Request(url, headers={"User-Agent": self.USER_AGENT})
        last_exc = None
        for attempt in range(1, retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return resp.read().decode("utf-8")
            except (urllib.error.URLError, TimeoutError) as exc:
                last_exc = exc
                time.sleep(0.5 * attempt)
        raise RuntimeError(f"failed to fetch {url}: {last_exc}")

    def parse_toc(self, html: str):
        """Return [(href, label), ...] from the page's <ul id="toc">, or [] if it has none."""
        m = self.TOC_RE.search(html)
        if not m:
            return []
        return self.TOC_ITEM_RE.findall(m.group(1))

    @staticmethod
    def slug_of(href: str) -> str:
        return href.rstrip("/").split("/")[-1]


class ChapterTextParser:
    """Parses a chapter's .txt export into (chapter_number, [verse_text, ...])."""

    CHAPTER_NUM_RE = re.compile(r'(\d+)')
    VERSE_START_RE = re.compile(r'^\s*(\d+)\.?\s+(\S.*)$')
    # Footnotes ("— v. 12: ..."), whether inline in brackets or on their own line -
    # as opposed to plain "[supplied word]" brackets, which are part of the verse
    # text itself (a translator's-insertion convention) and must be kept.
    FOOTNOTE_MARK = r'—\s*v\.\s*\d+'
    FOOTNOTE_LINE_RE = re.compile(r'^' + FOOTNOTE_MARK)
    FOOTNOTE_BRACKET_RE = re.compile(r'\s*\[\s*' + FOOTNOTE_MARK + r'.*?\]')
    FOOTNOTE_ARTIFACT_RE = re.compile(r'^\*+$')  # stray reference marker left on its own line
    ASTERISK_MARKER_RE = re.compile(r'\*+')  # footnote reference markers, glued or space-separated
    SPACE_BEFORE_PUNCT_RE = re.compile(r'\s+([,.])')
    MULTI_SPACE_RE = re.compile(r' {2,}')

    def parse(self, txt: str, fallback_num: int):
        lines = txt.split("\n")
        chapter_num = fallback_num
        if lines:
            m = self.CHAPTER_NUM_RE.search(lines[0])
            if m:
                chapter_num = int(m.group(1))

        verses = []
        current = None  # fragments of the verse currently being accumulated
        for raw_line in lines:
            stripped = raw_line.replace("\xa0", " ").strip()
            if not stripped or self.FOOTNOTE_LINE_RE.match(stripped) or self.FOOTNOTE_ARTIFACT_RE.match(stripped):
                continue
            line = self.FOOTNOTE_BRACKET_RE.sub("", stripped)
            m = self.VERSE_START_RE.match(line)
            if m:
                if current is not None:
                    verses.append(self._join(current))
                current = [self.ASTERISK_MARKER_RE.sub("", m.group(2)).strip()]
            elif current is not None:
                current.append(self.ASTERISK_MARKER_RE.sub("", line).strip())
        if current is not None:
            verses.append(self._join(current))
        return chapter_num, verses

    def _join(self, fragments) -> str:
        text = " ".join(f for f in fragments if f)
        text = self.SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
        text = self.MULTI_SPACE_RE.sub(" ", text)
        return text.strip()

    def looks_like_chapter(self, txt: str) -> bool:
        """True if txt is a chapter's plain-text export (a header line, then - after however
        many blank lines - verse 1). Used to tell a single-chapter book's own page - which some
        single-chapter books (e.g. Obadiah, Philemon, Jude) serve directly instead of nesting a
        separate chapter page - apart from a non-chapter stub page such as an "About this
        edition" appendix."""
        lines = txt.split("\n")
        for line in lines[1:]:
            stripped = line.replace("\xa0", " ").strip()
            if not stripped:
                continue
            m = self.VERSE_START_RE.match(stripped)
            return bool(m) and m.group(1) == "1"
        return False


class BibleExporter:
    def __init__(self, client: TextbaseClient, parser: ChapterTextParser, out_root: Path, workers: int):
        self.client = client
        self.parser = parser
        self.out_root = out_root
        self.workers = workers

    def export(self, source: BibleSource):
        for testament in source.testaments:
            print(f"== {testament.path} ==")
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                books = self._collect_books(testament.path, pool)
                out_dir = self.out_root / testament.dirname

                tasks = []
                for book in books:
                    if not book.chapters:
                        print(f"  ! skipping {book.name!r} ({book.slug}): no chapters found")
                        continue
                    print(f"  {book.name} ({book.slug}): {len(book.chapters)} chapters")
                    for position, chapter in enumerate(book.chapters, start=1):
                        tasks.append((book.slug, chapter, position))

                futures = {
                    pool.submit(self._export_chapter, out_dir, book_slug, chapter, position): (book_slug, chapter.href)
                    for book_slug, chapter, position in tasks
                }
                done = 0
                for future in futures:
                    book_slug, chapter_href = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        print(f"  ! failed {chapter_href}: {exc}", file=sys.stderr)
                        continue
                    done += 1
                    if done % 50 == 0:
                        print(f"  ... {done}/{len(tasks)} chapters written")
            print(f"  done: {len(tasks)} chapters")

        print(f"Export complete -> {self.out_root}/")

    def _collect_books(self, testament_path: str, pool: ThreadPoolExecutor):
        """Resolve a testament's books and, for each, its leaf chapters - walking any
        intermediate subdivision pages (e.g. the "Livre premier" split within Psalms)
        breadth-first so nesting depth doesn't need to be known in advance."""
        html = self.client.fetch(testament_path)
        book_items = self.client.parse_toc(html)

        book_htmls = list(pool.map(lambda item: self.client.fetch(item[0]), book_items))

        chapters_by_slug = {}
        frontier = []  # (href, label, book_slug)
        single_chapter_candidates = []  # (href, name, slug) - book pages with no sub-toc
        for (href, name), book_html in zip(book_items, book_htmls):
            slug = self.client.slug_of(href)
            items = self.client.parse_toc(book_html)
            if items:
                chapters_by_slug[slug] = []
                frontier.extend((sub_href, sub_label, slug) for sub_href, sub_label in items)
            else:
                single_chapter_candidates.append((href, name, slug))

        if single_chapter_candidates:
            # A book with no sub-toc is either a single-chapter book that serves its one chapter
            # directly at its own URL (e.g. Obadiah, Philemon), or a non-chapter stub page (e.g.
            # an "About this edition" appendix) - fetch its .txt to tell the two apart.
            txts = pool.map(lambda e: self.client.fetch(f"{e[0]}.txt"), single_chapter_candidates)
            for (href, name, slug), txt in zip(single_chapter_candidates, txts):
                chapters_by_slug[slug] = [Chapter(href, name)] if self.parser.looks_like_chapter(txt) else []

        while frontier:
            htmls = list(pool.map(lambda e: self.client.fetch(e[0]), frontier))
            next_frontier = []
            for (href, label, slug), html in zip(frontier, htmls):
                items = self.client.parse_toc(html)
                if items:
                    next_frontier.extend((sub_href, sub_label, slug) for sub_href, sub_label in items)
                else:
                    chapters_by_slug[slug].append(Chapter(href, label))
            frontier = next_frontier

        return [
            Book(self.client.slug_of(href), name, chapters_by_slug[self.client.slug_of(href)])
            for href, name in book_items
        ]

    def _export_chapter(self, out_dir: Path, book_slug: str, chapter: Chapter, position: int):
        txt = self.client.fetch(f"{chapter.href}.txt")
        chapter_num, verses = self.parser.parse(txt, fallback_num=position)
        book_dir = out_dir / book_slug
        book_dir.mkdir(parents=True, exist_ok=True)
        out_file = book_dir / f"{book_slug}-{chapter_num}.md"
        out_file.write_text("\n".join(verses) + "\n", encoding="utf-8")
        return out_file, len(verses)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-b", "--bible", choices=sorted(BIBLES), default="ortodox",
                         help="which bible/translation to export (default: ortodox)")
    parser.add_argument("-o", "--output", default="output", help="output directory (default: output)")
    parser.add_argument("-w", "--workers", type=int, default=8, help="number of concurrent downloads (default: 8)")
    args = parser.parse_args()

    exporter = BibleExporter(
        client=TextbaseClient(),
        parser=ChapterTextParser(),
        out_root=Path(args.output),
        workers=args.workers,
    )
    exporter.export(BIBLES[args.bible])


if __name__ == "__main__":
    main()
