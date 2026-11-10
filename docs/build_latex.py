# -*- coding: utf-8 -*-
"""
Typeset docs/DDA-manual.pdf from docs/DDA-manual.md with LATEX (classic
textbook layout: Times/Palatino-like serif, numbered TOC, running heads).

    mamba activate DDA
    python docs/build_latex.py             # -> docs/DDA-manual.pdf (xelatex)

Pipeline (no hand-maintained .tex — the Markdown stays the single source):
  1. preprocess the markdown (see process_markdown()):
     - the H1/version lines become the LaTeX title/subtitle/date,
     - the markdown TOC table is dropped (LaTeX \tableofcontents replaces it),
     - display equations carrying \tag{...} become raw \begin{equation} blocks
       (a \tag inside a plain \[...\] display is an amsmath error),
     - degrees signs INSIDE math become ^\circ (unicode-math safe),
     - <br> inside table cells become \newline,
     - box-drawing characters (repo tree, §0.2) become ASCII (Latin Modern
       Mono has no U+2500 glyphs; ASCII trees are the classic style anyway),
     - mermaid blocks are pre-rendered to PNG via @mermaid-js/mermaid-cli
       (cached in figures/mermaid_png/, same content-hash keying as
       build_html.py; scale 4 ≈ 300 dpi at \textwidth) and included as images.
  2. pandoc -> LaTeX (pypandoc_binary, installed in the DDA env) with the
     classic preamble below (docs/latex/manual-preamble.tex).
  3. latexmk -xelatex; the PDF lands at docs/DDA-manual.pdf.

Requirements: TeX Live (xelatex, latexmk; TeX Gyre Termes fonts ship with it),
pypandoc_binary (pip, in the DDA env), node/npx for mermaid-cli, and a Chrome
binary found automatically (PATH, /Applications, or the puppeteer cache — same
logic as build_html.py). HTML remains built separately: docs/build_html.py.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'DDA-manual.md')
PDF = os.path.join(HERE, 'DDA-manual.pdf')
TEX = os.path.join(HERE, 'DDA-manual-latex.tex')
PREAMBLE = os.path.join(HERE, 'latex', 'manual-preamble.tex')
PDFDIR = os.path.join(HERE, 'figures', 'mermaid_pdf')

try:
    import pypandoc
except ImportError:
    sys.exit('pypandoc not found — pip install pypandoc_binary (DDA env)')

# ---------------------------------------------------------------- fonts ---
# File-name font lookup via kpathsea (robust on this machine: name-based
# lookup needs a fresh luaotfload/fontspec name DB which is not guaranteed).
PREAMBLE_DEFAULT = r"""\usepackage{amsmath,amssymb}
\usepackage{fontspec}%
\usepackage{unicode-math}
\setmainfont{texgyretermes}[Extension=.otf, UprightFont=*-regular,
  BoldFont=*-bold, ItalicFont=*-italic, BoldItalicFont=*-bolditalic]
\setsansfont{texgyreheros}[Extension=.otf, UprightFont=*-regular,
  BoldFont=*-bold, ItalicFont=*-italic, BoldItalicFont=*-bolditalic]
\setmathfont{texgyretermes-math.otf}
% classic book convention: typewriter text set slightly smaller than body
\setmonofont{lmmono10-regular}[Extension=.otf, UprightFont=*,
  ItalicFont=lmmono10-italic, BoldFont=lmmonolt10-bold,
  BoldItalicFont=lmmonolt10-boldoblique, Scale=0.9]
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{adjustbox}
\usepackage{url}
\usepackage{fvextra}
\fvset{breaklines=true,breakanywhere=false}
\usepackage{hyperref}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{newunicodechar}

% ---- unicode chars that occur in PROSE (math-mode-safe mappings) ----
\newunicodechar{→}{\ensuremath{\rightarrow}}
\newunicodechar{←}{\ensuremath{\leftarrow}}
\newunicodechar{⇒}{\ensuremath{\Rightarrow}}
\newunicodechar{≈}{\ensuremath{\approx}}
\newunicodechar{≤}{\ensuremath{\le}}
\newunicodechar{≥}{\ensuremath{\ge}}
\newunicodechar{≲}{\ensuremath{\lesssim}}
\newunicodechar{≡}{\ensuremath{\equiv}}
\newunicodechar{∝}{\ensuremath{\propto}}
\newunicodechar{∞}{\ensuremath{\infty}}
\newunicodechar{−}{\ensuremath{-}}
\newunicodechar{×}{\ensuremath{\times}}
\newunicodechar{∈}{\ensuremath{\in}}
\newunicodechar{⚠}{\textbf{(!)}\,}
\newunicodechar{✓}{\ensuremath{\checkmark}}

% ---- classic book layout ----
\setcounter{tocdepth}{2}
\setcounter{secnumdepth}{0}          % section numbers are written INTO the
                                     % headings in the manual source
\raggedbottom
\linespread{1.03}
% H1 = "Part" level: own page, centered, large
\newcommand{\sectionbreak}{\clearpage}
\titleformat{\section}[block]{\normalfont\LARGE\bfseries\filcenter}{}{0em}{}
\titlespacing{\section}{0pt}{6ex plus 1fill}{3.5ex}
\titleformat{\subsection}[hang]{\normalfont\Large\bfseries}{}{0em}{}
\titlespacing{\subsection}{0pt}{3.5ex plus .5ex}{1.2ex}
\titleformat{\subsubsection}[hang]{\normalfont\large\bfseries}{}{0em}{}
\titlespacing{\subsubsection}{0pt}{2.8ex plus .3ex}{0.8ex}
% running heads (classic: short title left, current subsection right, page centered)
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0.4pt}
\fancyhead[L]{\small DDA Scattering Pipeline --- Project Manual}
\fancyhead[R]{\small\nouppercase{\leftmark}}
\fancyfoot[C]{\small\thepage}
\fancypagestyle{plain}{\fancyhf{}\renewcommand{\headrulewidth}{0pt}%
  \fancyfoot[C]{\small\thepage}}
\renewcommand{\sectionmark}[1]{\markboth{#1}{}}
\renewcommand{\subsectionmark}[1]{\markright{\thesection#1}}
% tables: airy but compact; pandoc uses booktabs
\renewcommand{\arraystretch}{1.25}
\usepackage{etoolbox}
\AtBeginEnvironment{longtable}{\small\sloppy\emergencystretch=4.5em}
\AtBeginEnvironment{tabular}{\small\sloppy\emergencystretch=4.5em}
\setlength{\parskip}{0.4em}
\setlength{\parindent}{0pt}
\setlength{\emergencystretch}{2.5em}
\hypersetup{hidelinks}
"""

MER_CFG = {
    'themeVariables': {'fontSize': '18px',
                       'fontFamily': '"Times New Roman", "TeX Gyre Termes", serif'},
    'flowchart': {'nodeSpacing': 20, 'rankSpacing': 30, 'padding': 6},
}
MER_VERSION = 'pdf-v2'      # cache key (vector PDF route; independent of HTML SVGs)


def find_chrome():
    for c in [os.environ.get('CHROME_BIN'),
              '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
              '/Applications/Chromium.app/Contents/MacOS/Chromium']:
        if c and os.path.exists(c):
            return c
    import glob
    for g in sorted(
            glob.glob(os.path.expanduser(
                '~/.cache/puppeteer/chrome-headless-shell/*/*-*/'
                'chrome-headless-shell')) +
            glob.glob(os.path.expanduser(
                '~/.cache/puppeteer/chrome/*/*/chrome-mac*/Google Chrome for '
                'Testing.app/Contents/MacOS/Google Chrome for Testing')),
            reverse=True):
        if os.path.exists(g):
            return g
    return None


def render_mermaid(block):
    """Pre-render one mermaid block to a VECTOR PDF (cached), via
    mermaid-cli SVG + Chrome print-to-pdf with the @page sized exactly to
    the diagram. (PNG rasterization clipped wide diagrams at the puppeteer
    viewport and printed blurry; vector pages cannot clip and stay sharp.)
    Returns the cached file's path relative to docs/, or None on failure."""
    os.makedirs(PDFDIR, exist_ok=True)
    digest = hashlib.sha256((MER_VERSION + block).encode()).hexdigest()[:12]
    out = os.path.join(PDFDIR, 'mmd-%s.pdf' % digest)
    if not os.path.exists(out):
        chrome = find_chrome()
        if chrome is None:
            return None
        with tempfile.TemporaryDirectory() as td:
            mmd = os.path.join(td, 'd.mmd')
            svg = os.path.join(td, 'd.svg')
            pptr = os.path.join(td, 'pptr.json')
            cfg = os.path.join(td, 'cfg.json')
            page = os.path.join(td, 'page.html')
            open(mmd, 'w').write(block)
            with open(cfg, 'w') as fh:
                json.dump(MER_CFG, fh)
            with open(pptr, 'w') as fh:
                fh.write('{"executablePath": "%s", "headless": true, '
                         '"args": ["--no-sandbox"]}' % chrome.replace('\\', '/'))
            try:
                subprocess.run(
                    ['npx', '-y', '@mermaid-js/mermaid-cli',
                     '-i', mmd, '-o', svg, '-p', pptr, '-c', cfg],
                    check=True, capture_output=True, timeout=180)
                s = open(svg, encoding='utf-8').read()[:4000]
                # mermaid-cli emits width="100%" + style="max-width:Wpx" +
                # viewBox="minx miny W H" — the viewBox extent is the exact
                # intrinsic size; keep a plain width=/height= fallback.
                m = (re.search(r'viewBox="[-\d.]+ [-\d.]+ ([\d.]+) ([\d.]+)"', s)
                     or re.search(r'width="([\d.]+)(?:px)?"\s+height="([\d.]+)'
                                  r'(?:px)?"', s))
                if not m:
                    raise RuntimeError('no size found in mermaid SVG')
                w, h = m.group(1), m.group(2)
                # @page exactly diagram-sized, zero margins -> tight PDF page
                open(page, 'w').write(
                    '<style>@page{size:%spx %spx;margin:0}'
                    'html,body{margin:0;padding:0;overflow:hidden}'
                    'img{display:block}</style>'
                    '<img src="file://%s" width="%s" height="%s">'
                    % (w, h, svg, w, h))
                subprocess.run(
                    [chrome, '--headless=new', '--disable-gpu', '--no-sandbox',
                     '--no-pdf-header-footer',
                     '--print-to-pdf=%s' % out, 'file://%s' % page],
                    check=True, capture_output=True, timeout=120)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                    FileNotFoundError, RuntimeError) as e:
                msg = (e.stderr or b'') if isinstance(e,
                                                      subprocess.CalledProcessError) else b''
                sys.stderr.write('mermaid render failed: %s %s\n'
                                 % (e, msg.decode()[:300]))
                if os.path.exists(out):
                    os.remove(out)
                return None
    # native page size of the produced PDF (pt) — decides upright vs rotated
    d = open(out, 'rb').read()
    mb = re.search(rb'MediaBox\s*\[\s*[\d.]+ [\d.]+ ([\d.]+) ([\d.]+)\]', d)
    if mb is None:
        sys.exit('mermaid cache %s has no MediaBox' % out)
    return (os.path.relpath(out, HERE), float(mb.group(1)), float(mb.group(2)))


def _fix_degree(s):
    # NO trailing space: pandoc's tex_math_dollars requires a NON-SPACE char
    # immediately before the closing $ — "^\circ $" would de-mathify the span
    # and emit escaped \$ raw text (the U+2218/U+1D703 missing-glyph mess).
    return s.replace('°', '^\\circ')


def _outside_code(text, func):
    """Apply func() to every part of the text OUTSIDE fenced/inline code —
    math-ish characters inside code (stray $ like $CONDA_PREFIX, or literal
    $...$ spans) must never be math-processed."""
    out = []
    for fence in re.split(r'(```.*?```|~~~.*?~~~)', text, flags=re.S):
        if fence.startswith('```') or fence.startswith('~~~'):
            out.append(fence)
            continue
        for seg in re.split(r'(`[^`\n]*`)', fence):
            out.append(seg if seg.startswith('`') else func(seg))
    return ''.join(out)


def _esc_meta(s):
    """make a plain string safe for pandoc -V metadata -> LaTeX (the date line
    carries code spans like `beta_0.9.1`: drop backticks, escape specials)."""
    s = s.replace('`', '')
    return re.sub(r'([&#%$_{}~^\\])', r'\\\1', s)


def process_markdown(text):
    """markdown in -> pandoc-ready markdown out; also returns (title,
    subtitle, date) for the LaTeX front matter."""
    # ---- front matter: H1 title, bold subtitle, italic *Version:* line ----
    m = re.match(r'# (.*)\n+\*\*(.*?)\*\*\n+\*Version: (.*?)\*\n', text, re.S)
    if not m:
        sys.exit('unexpected manual header layout — front-matter extraction '
                 'failed (H1 / **subtitle** / *Version:* paragraph)')
    title = _esc_meta(m.group(1))
    subtitle = _esc_meta(m.group(2))
    # classic title page: just the version date (the long repo-state sentence
    # from the *Version: ... · ...* line stays out of the title block)
    date = _esc_meta(m.group(3).split('·')[0].replace('\n', ' ').strip())
    text = text[m.end():]

    # ---- drop the markdown TOC table (LaTeX TOC replaces it) ----
    text = re.sub(r'(?ms)^### Table of contents\n.*?(?=^### )', '', text,
                  flags=re.M)

    # ---- mermaid blocks -> pre-rendered vector PDF figures ----
    def stash(mr):
        rel, w, h = render_mermaid(mr.group(1))
        if rel is None:
            sys.exit('cannot render mermaid block (no Chrome found)')
        # A4 text block ~459x660 pt; adjustbox applies max size AFTER angle,
        # so the slots are always 0.97tw x 0.85th = 445 x 561 pt. Choose the
        # orientation with the larger fit; a rotated figure wins when its
        # natural shape is WIDE (long side then rides the page height).
        # All mermaid figures become [p] floats: textbook-style dedicated
        # figure pages, body text flows around them (no inline white gaps).
        upright = min(445.0 / w, 561.0 / h)
        rotated = min(445.0 / h, 561.0 / w)
        angle = 'angle=90,' if rotated > 1.15 * upright else ''
        return ('```{=latex}\n'
                '\\begin{figure}[p]\\centering\\begin{adjustbox}{%s'
                'max size={0.97\\textwidth}{0.85\\textheight}}'
                '{\\includegraphics{%s}}\\end{adjustbox}\\end{figure}\n'
                '```\n' % (angle, rel))
    text = re.sub(r'```mermaid\n(.*?)```\n', stash, text, flags=re.S)

    # ---- display equations carrying \tag: $$ ... $$ -> equation env ----
    def tagged(mr):
        ind, body = mr.group(1), mr.group(2)
        if '\\tag' not in body:
            return mr.group(0)
        return '%s\\begin{equation}\n%s%s\\end{equation}\n' % (ind, body, ind)
    text = re.sub(r'(?m)^([ \t]*)\$\$\n(.*?\n)[ \t]*\$\$\n?', tagged, text,
                  flags=re.S)

    # ---- ° inside math -> ^\circ (unicode-math); display + inline spans ----
    text = re.sub(r'\\begin{equation}\n.*?\n[ \t]*\\end{equation}',
                  lambda mr: _fix_degree(mr.group(0)), text, flags=re.S)
    text = re.sub(r'(?m)^([ \t]*)\$\$\n(.*?\n)[ \t]*\$\$',
                  lambda mr: mr.group(1) + _fix_degree(mr.group(0)), text,
                  flags=re.S)
    # inline spans: mirror pandoc's tex_math_dollars delimiting rules (non-space
    # right after the opening $, non-space right before the closing $), applied
    # only OUTSIDE code spans/fences (stray $ inside code must not pair up).
    def _inline(seg):
        return re.sub(r'\$(?:[^\s$`]|[^\s$`][^$\n`]*[^\s$`])\$',
                      lambda mr: _fix_degree(mr.group(0)), seg)
    text = _outside_code(text, _inline)

    # ---- <br> in table cells -> \newline (pandoc raw_tex passes it) ----
    text = re.sub(r'<br\s*/?>', r'\\newline', text)

    # ---- box-drawing / oddities inside code (ASCII tree = classic anyway) ----
    text = (text.replace('─', '-').replace('│', '|').replace('├', '+')
                .replace('└', '+').replace('┐', '+').replace('┌', '+')
                .replace('┴', '+').replace('┬', '+').replace('┤', '+')
                .replace('f̂', 'f^'))
    return title, subtitle, date, text


def _make_tt_breakable(tex):
    """Classic typesetting lets long typewriter runs break at punctuation;
    pandoc's inline \\texttt cannot, so long paths/URLs/filenames overflow the
    column (624 pt offenders). Insert discretionary breaks after _ / . -
    inside long \\texttt bodies."""
    def ins(m):
        body = m.group(1)
        if len(body) < 12:
            return m.group(0)
        body = re.sub(r'\\_|/|(?<=\w)-(?=\w)|(?<=\w)\.(?=\w)'
                      r'|(?<=\{)(?=\w)|(?<=\})(?=\w)',
                      lambda mr: mr.group(0) + '\\allowbreak ', body)
        return '\\texttt{' + body + '}'
    # body alternatives: escaped chars (pandoc's \{ \} \_ in code), plain
    # chars, and BALANCED bare-brace groups (pandoc emits \textless{} /
    # \textgreater{} for < > inside code).
    pat = (r'\\texttt\{('
           r'(?:\\.|[^\\{}]|'
           r'\{(?:[^{}\\]|\\.)*\}'
           r')*)\}')
    tex = re.sub(pat, ins, tex)

    # pandoc's plain code fences use the KERNEL verbatim, which fvextra's
    # \fvset does not reach — promote them to fancyvrb/fvextra Verbatim with
    # line breaking (over-long tree/CLI lines then wrap at spaces).
    tex = re.sub(r'\\begin\{verbatim\}',
                 r'\\begin{Verbatim}[breaklines=true,breakanywhere=false]',
                 tex)
    tex = tex.replace('\\end{verbatim}', '\\end{Verbatim}')
    return tex


def build():
    raw = open(SRC, encoding='utf-8').read()
    title, subtitle, date, body = process_markdown(raw)

    os.makedirs(os.path.dirname(PREAMBLE), exist_ok=True)
    if not os.path.exists(PREAMBLE):
        open(PREAMBLE, 'w').write(PREAMBLE_DEFAULT)

    md = os.path.join(HERE, 'DDA-manual-latexsrc.md')
    open(md, 'w', encoding='utf-8').write(body)

    pypandoc.convert_file(
        md, 'latex', outputfile=TEX,
        # -implicit_figures: keep standalone images as inline images so the
        # imgcenter.lua filter centers/sizes them; the manual's own bold
        # "Fig. N" caption paragraphs are the captions (no float, no pandoc
        # "Figure 1..28" numbering that would clash with them).
        format='markdown+pipe_tables+raw_tex-implicit_figures',
        extra_args=[
            '--standalone', '--toc', '--toc-depth=2',
            '--resource-path', HERE,
            '--include-in-header', PREAMBLE,
            '--lua-filter', os.path.join(HERE, 'latex', 'imgcenter.lua'),
            '-V', 'documentclass=article',
            '-V', 'fontsize=11pt',
            '-V', 'geometry=paper=a4paper,left=2.4cm,right=2.4cm,top=2.6cm,bottom=2.6cm',
            '-V', 'title=' + title,
            '-V', 'subtitle=' + subtitle,
            '-V', 'date=' + date,
            '-V', 'links-as-notes=false',
            '-V', 'colorlinks=true',
            '-V', 'linkcolor=black', '-V', 'urlcolor=black',
            '-V', 'toc-title=Contents',
        ])
    src = open(TEX, encoding='utf-8').read()
    open(TEX, 'w', encoding='utf-8').write(_make_tt_breakable(src))
    print('wrote', os.path.relpath(TEX, os.path.dirname(HERE)))

    # figures/ live relative to docs/, compile there
    cmd = ['latexmk', '-xelatex', '-interaction=nonstopmode',
           '-halt-on-error', '-jobname=DDA-manual', TEX]
    subprocess.run(cmd, cwd=HERE, check=True)
    out = os.path.join(HERE, 'DDA-manual.pdf')
    print('wrote', out)


if __name__ == '__main__':
    build()
