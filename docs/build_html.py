# -*- coding: utf-8 -*-
"""
Build docs/DDA-manual.html from docs/DDA-manual.md.

    mamba activate DDA
    python docs/build_html.py            # → docs/DDA-manual.html
    python docs/build_html.py --pdf      # additionally renders a PDF

Features:
- tables, fenced code, anchors, $...$/$$..$$ math (pymdownx.arithmatex +
  MathJax v3 from CDN; the .md also renders natively on GitHub via KaTeX).
- Mermaid diagrams: pre-rendered to SVG via @mermaid-js/mermaid-cli (cached in
  docs/figures/mermaid/, keyed by content hash) and embedded as <img>. This
  makes the HTML/PDF self-contained and print-reliable. If mermaid-cli is not
  available, falls back to client-side mermaid.js.
    - puppeteer/mermaid-cli needs a Chrome binary: found automatically at a
      default macOS location OR in the puppeteer cache
      (~/.cache/puppeteer/chrome-headless-shell/…); override with
      CHROME_BIN=<path-to-chrome> (e.g. a chrome-headless-shell binary).

Requires: pip install markdown pymdown-extensions
"""
import hashlib
import json
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'DDA-manual.md')
DST = os.path.join(HERE, 'DDA-manual.html')
PDF = os.path.join(HERE, 'DDA-manual.pdf')
MERDIR = os.path.join(HERE, 'figures', 'mermaid')

CHROME_CANDIDATES = [os.environ.get('CHROME_BIN')] + [
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Chromium.app/Contents/MacOS/Chromium']
if CHROME_CANDIDATES[0] is None:
    # puppeteer-managed headless-shell cache (what this machine's mermaid-cli/PDF
    # builds actually use; newest version first) — spares maintainers the CHROME_BIN export
    import glob as _glob
    CHROME_CANDIDATES += sorted(
        _glob.glob(os.path.expanduser(
            '~/.cache/puppeteer/chrome-headless-shell/*/*-*/chrome-headless-shell'))
        + _glob.glob(os.path.expanduser(
            '~/.cache/puppeteer/chrome/*/*/chrome-mac*/Google Chrome for Testing.app/'
            'Contents/MacOS/Google Chrome for Testing')),
        reverse=True)
CHROME = next((c for c in CHROME_CANDIDATES if c and os.path.exists(c)), None)


def find_chrome():
    return CHROME


RENDER_VERSION = 'v3'  # bump when mermaid render config changes (cache key)

# themeVariables.fontSize makes mermaid *measure the layout* with that font
# (a post-layout CSS font-size would overflow the boxes). Tight spacing keeps
# the tall flowcharts compact so the print max-height shrink stays readable.
MER_CFG = {
    'themeVariables': {'fontSize': '18px'},
    'flowchart': {'nodeSpacing': 20, 'rankSpacing': 30, 'padding': 6},
}


def render_mermaid(block, idx):
    """Pre-render one mermaid block to SVG (cached). Returns rel path or None."""
    os.makedirs(MERDIR, exist_ok=True)
    digest = hashlib.sha256(
        (RENDER_VERSION + block).encode()).hexdigest()[:12]
    svg = os.path.join(MERDIR, 'mmd-%s.svg' % digest)
    if os.path.exists(svg):
        return os.path.relpath(svg, HERE)
    chrome = find_chrome()
    if chrome is None:
        return None
    with tempfile.TemporaryDirectory() as td:
        mmd = os.path.join(td, 'd.mmd')
        pptr = os.path.join(td, 'pptr.json')
        cfg = os.path.join(td, 'cfg.json')
        css = os.path.join(td, 'style.css')
        open(mmd, 'w').write(block)
        with open(cfg, 'w') as fh:
            json.dump(MER_CFG, fh)
        with open(pptr, 'w') as fh:
            fh.write('{"executablePath": "%s", "headless": true, '
                     '"args": ["--no-sandbox"]}' % chrome.replace('\\', '/'))
        try:
            subprocess.run(
                ['npx', '-y', '@mermaid-js/mermaid-cli', '-i', mmd, '-o', svg,
                 '-p', pptr, '-c', cfg, '-b', 'white'],
                check=True, capture_output=True, timeout=180)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                FileNotFoundError) as e:
            sys.stderr.write('mermaid-cli render failed for block %d: %s\n'
                             % (idx, (e.stderr or b'').decode()[:300]))
            return None
    return os.path.relpath(svg, HERE)


def build_html():
    text = open(SRC, encoding='utf-8').read()

    # 1. pull mermaid blocks out (markdown escapes them otherwise)
    blocks = []

    def stash(m):
        blocks.append(m.group(1))
        return '\nMERMAIDBLOCK%d\n' % (len(blocks) - 1)

    text = re.sub(r'```mermaid\n(.*?)```', stash, text, flags=re.S)

    body = markdown.markdown(
        text,
        extensions=['tables', 'fenced_code', 'toc', 'sane_lists',
                    'pymdownx.arithmatex'],
        extension_configs={'pymdownx.arithmatex': {
            'generic': True,
            'inline_syntax': ['dollar', 'round'],
            'block_syntax': ['dollar', 'square'],
        }},
        toc_depth=2)

    # 2. re-inject: prefer pre-rendered SVG, fall back to client-side mermaid
    client = []          # blocks that need client-side rendering

    def reinject(m):
        i = int(m.group(1))
        rel = render_mermaid(blocks[i], i)
        if rel:
            return ('<div class="mermaidfig"><img src="%s" '
                    'alt="flow diagram %d"></div>' % (rel, i + 1))
        client.append((i, blocks[i]))
        return ('<div class="mermaid" data-idx="%d">\n%s\n</div>'
                % (i, html.unescape(blocks[i]).strip()))

    body = re.sub(r'<p>MERMAIDBLOCK(\d+)</p>', reinject, body)
    body = re.sub(r'MERMAIDBLOCK(\d+)', reinject, body)

    scripts = ''
    if client:
        scripts = """
<script type="module">
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
mermaid.initialize({startOnLoad:true, securityLevel:'loose', maxTextSize:200000});
</script>"""

    page = (TEMPLATE
            .replace('%%BODY%%', body)
            .replace('%%MERMAIDSCRIPT%%', scripts))
    open(DST, 'w', encoding='utf-8').write(page)
    print('wrote', DST)


def build_pdf():
    chrome = find_chrome()
    if chrome is None:
        print('no Chrome/Chromium found — open DDA-manual.html and Print → PDF.')
        return
    cmd = [chrome, '--headless', '--disable-gpu', '--no-sandbox',
           '--virtual-time-budget=25000', '--no-pdf-header-footer',
           '--print-to-pdf=' + PDF, 'file://' + DST]
    subprocess.run(cmd, check=True)
    print('wrote', PDF)


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DDA Scattering Pipeline — Project Manual</title>
<style>
 body{font:15px/1.55 -apple-system,"Segoe UI",Helvetica,Arial,sans-serif;
      color:#1b1f23; max-width:1080px; margin:0 auto; padding:2rem 1.5rem 4rem;}
 h1,h2,h3{line-height:1.25; margin:1.6em 0 .6em;}
 h1{font-size:1.9em; border-bottom:2px solid #d0d7de; padding-bottom:.3em;}
 h2{font-size:1.45em; border-bottom:1px solid #d0d7de; padding-bottom:.25em;}
 code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;}
 pre{background:#f6f8fa; border-radius:6px; padding:.8em 1em; overflow-x:auto;}
 code{background:#f0f1f3; border-radius:4px; padding:.1em .35em;}
 pre code{background:none; padding:0;}
 table{border-collapse:collapse; margin:.8em 0; font-size:.92em;}
 th,td{border:1px solid #d0d7de; padding:.35em .7em; vertical-align:top;}
 th{background:#f6f8fa;}
 img{max-width:100%; height:auto; border:1px solid #e3e6ea; border-radius:6px;
     margin:.5em 0;}
 .mermaidfig img{border:none; padding:.4em 0;}
 .mermaidfig svg, .mermaidfig img{max-width:100%;}
 .mermaid{background:#fff; border:1px solid #e3e6ea; border-radius:6px;
          padding:.6em; margin:.8em 0; overflow-x:auto;}
 blockquote{border-left:4px solid #d0d7de; margin:.8em 0; padding:.2em 1em;
            color:#57606a;}
 a{color:#0969da; text-decoration:none;} a:hover{text-decoration:underline;}
 @media print{ @page{margin:10mm;}
               .mermaidfig,.mermaid,img{break-inside:avoid;}
               .mermaidfig img{max-height:26.5cm; max-width:18.5cm; width:auto;
                               height:auto;} body{padding:0;} }
</style>
</head>
<body>
%%BODY%%
<script>
window.MathJax = {tex: {inlineMath: [['\\(','\\)'], ['$','$']],
                        displayMath: [['\\[','\\]'], ['$$','$$']]},
                  options: {enableMenu: false,
                            ignoreHtmlClass: 'arithmatex-done|no-mathjax',
                            processHtmlClass: 'arithmatex|mathjax-math'}};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
%%MERMAIDSCRIPT%%
</body>
</html>
"""

if __name__ == '__main__':
    build_html()
    if '--pdf' in sys.argv:
        build_pdf()
