import os as _os, sys as _sys
REPO = (_os.environ.get('REPO') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *['..'] * 5))).rstrip('/') + '/'
_sys.path.insert(0, REPO + 'code'); _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
# Parse CCKLL Table 1 (arXiv:2312.16443v2 HTML, LaTeXML) and compare F/B labels with c19/out/a5_table1.txt core content.
import re, html, ast
if not _os.path.exists('cckll.html'):
    import urllib.request  # arXiv HTML of CCKLL (arXiv:2312.16443v2), not stored in the repo
    urllib.request.urlretrieve('https://arxiv.org/html/2312.16443v2', 'cckll.html')
s = open('cckll.html', encoding='utf-8').read()
caps=[m for m in re.finditer(r'<figcaption[^>]*>(.*?)</figcaption>', s, re.S) if 'Table 1:' in re.sub('<[^>]+>','',m.group(1))[:12]]
end = caps[0].start(); start = s.rfind('<figure', 0, end)
rows = re.findall(r'<tr[^>]*>(.*?)</tr>', s[start:end], re.S)
def clean(c):
    c = re.sub(r'<math[^>]*alttext="([^"]*)"[^>]*>.*?</math>', lambda m: html.unescape(m.group(1)), c, flags=re.S)
    c = re.sub('<[^>]+>', '', html.unescape(c))
    return re.sub(r'\\color\[rgb\]\{[^}]*\}', '', c).strip()
hdr = [clean(c) for c in re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', rows[0], re.S)]
paper = {}
for r in rows[1:]:
    cells = [clean(c) for c in re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', r, re.S)]
    j = int(cells[0])
    for h, c in zip(hdr[1:], cells[1:]):
        if c:
            paper.setdefault(j, {})[c] = paper.get(j, {}).get(c, 0) + (-1 if h.startswith('F') else 1)
ours = {}
for line in open(REPO + 'scripts/index/c-parity/c19/out/a5_table1.txt'):
    m = re.match(r'\s*j (\d+) (\{.*\})\s*$', line)
    if m and 'data' not in line:
        ours[int(m.group(1))] = ast.literal_eval(m.group(2))
for j in range(24, 41, 2):
    p, o = paper.get(j, {}), ours.get(j, {})
    print(j, 'paper', p, '| ours', o, '| MATCH' if p == o else '| DIFF')
