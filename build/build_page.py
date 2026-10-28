"""build_page.py -- index.html: interactive stress Hessian on five real networks (NEI), KRDS x GNU tokens, 2026-10-28.

Input   build/nets/<key>.json  from make_nets.py (MDS terminal, hop distances, preset node and path, full-Hessian check)
Output  index.html             (self-contained apart from assets/ and vendor/)
Checks  per network: stored stress reproduced, analytic node block H_ii against central finite differences.
Usage   python build/build_page.py && node build/check_katex.cjs index.html
"""
import base64, json, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ['karate', 'dolphins', 'lesmis', 'football', 'netscience']
NETS = {}
for k in ORDER:
    m = json.load(open(os.path.join(HERE, 'nets', k + '.json')))
    n, X = m['N'], np.array(m['X']); sel = m['sel']
    D = np.frombuffer(base64.b64decode(m['D']), np.uint8).reshape(n, n).astype(float)
    dd = np.linalg.norm(X[:, None] - X[None], axis=2); iu = np.triu_indices(n, 1)
    F = ((dd - D)[iu] ** 2).sum(); assert abs(F - m['stress_total']) / F < 1e-9
    Xo, Do = np.delete(X, sel, 0), np.delete(D[sel], sel)
    Fi = lambda p: ((np.linalg.norm(p - Xo, axis=1) - Do) ** 2).sum()
    v = X[sel] - Xo; r = np.linalg.norm(v, axis=1); u = v / r[:, None]; Pu = np.einsum('ka,kb->kab', u, u)
    Hii = np.einsum('kab->ab', 2 * Pu + 2 * (1 - Do / r)[:, None, None] * (np.eye(2) - Pu))
    h = 1e-4; Hfd = np.array([[(Fi(X[sel] + h * (a + b)) - Fi(X[sel] + h * (a - b)) - Fi(X[sel] - h * (a - b)) + Fi(X[sel] - h * (a + b))) / (4 * h * h)
                              for b in np.eye(2)] for a in np.eye(2)])
    fd = float(np.abs(Hfd - Hii).max() / np.abs(Hii).max()); assert fd < 1e-5
    m['check'].update(F=float(F), frac_comp=float((dd < D)[iu].mean()), fd=fd)
    # NEI ensemble: aligned terminals as int16 offsets from the reference centre (resolution ~ span / 65000)
    z = np.load(os.path.join(HERE, 'nets', k + '_ens.npz'))
    Xa = z['Xa']; c0 = X.mean(0); sc = float(np.abs(Xa - c0).max() / 32000)
    q = np.round((Xa - c0) / sc).astype('<i2')
    assert np.abs(q.astype(float) * sc + c0 - Xa).max() < sc
    m['ens'] = dict(M=int(len(Xa)), nei=float(z['nei']), c=c0.tolist(), s=sc, q=base64.b64encode(q.tobytes()).decode(),
                    Fmin=float(z['F'].min()), pct=float((z['F'] < F).mean()), Fmed=float(np.median(z['F'])), Fmax=float(z['F'].max()), gmax=float(z['gmax'].max()))
    NETS[k] = m
    print(f"{k:10s} N={n:3d} node={sel:3d} λmin(H⊥)={m['check']['lam_min']:.3f} fd={fd:.1e} " +
          (f"barrier=+{m['barrier'] - m['F0']:.2f} gain={m['F0'] - m['FB']:+.2f}" if m['B'] else 'single basin') + f" NEI={m['ens']['nei']:.4f} M={m['ens']['M']}")

HTML = r"""<!doctype html>
<html lang="ko" data-krds-mode="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The stress Hessian</title>
<meta name="description" content="Interactive note: the node-block Hessian of the stress energy on five real networks (MDS terminals), split into stiffness and prestress">
<link rel="stylesheet" href="assets/krds/primitives.css">
<link rel="stylesheet" href="assets/krds/semantic.css">
<link rel="stylesheet" href="vendor/katex/katex.min.css">
<style>
@font-face{font-family:'Noto Sans KR';src:url('assets/fonts/NotoSansKR-Variable.subset.woff2') format('woff2');font-weight:100 900;font-display:swap}
@font-face{font-family:'SUITE';src:url('assets/fonts/SUITE-Variable.woff2') format('woff2');font-weight:100 900;font-display:swap}
/* viz colours: KRDS steps of the GNU palette; graphics need ≥3:1 (step 40+) */
:root,[data-krds-mode="light"]{
  --viz-bg:#fff;
  --viz-band-lo:var(--krds-color-light-primary-30);--viz-band-hi:var(--krds-color-light-primary-5);
  --viz-iso:var(--krds-color-light-primary-60);
  --viz-edge:var(--krds-color-light-gray-90);--viz-node:var(--krds-color-light-secondary-70);--viz-node-ring:#fff;
  --viz-ten:var(--krds-color-light-primary-50);--viz-com:var(--krds-color-light-point-50);
  --viz-up:var(--krds-color-light-warning-40);--viz-down:var(--krds-color-light-primary-70);--viz-sad:var(--krds-color-light-success-50);
  --viz-mark:var(--krds-color-light-gray-90);
  --type-up:var(--krds-color-light-warning-60);--type-down:var(--gk-text-primary);--type-sad:var(--krds-color-light-success-60);
  --viz-r-lo:var(--krds-color-light-danger-40);--viz-r-mid:var(--krds-color-light-danger-10);--viz-r-hi:var(--krds-color-light-point-5);--viz-ens:var(--krds-color-light-danger-60);--viz-ens-lt:var(--krds-color-light-danger-30);
  --viz-s-lo:var(--krds-color-light-primary-40);--viz-s-mid:var(--krds-color-light-primary-10);--viz-s-hi:var(--krds-color-light-point-5);
}
[data-krds-mode="high-contrast"]{
  --viz-bg:#000;
  --viz-band-lo:var(--krds-color-high-contrast-primary-50);--viz-band-hi:var(--krds-color-high-contrast-gray-95);
  --viz-iso:var(--krds-color-high-contrast-primary-30);
  --viz-edge:var(--krds-color-high-contrast-gray-0);--viz-node:var(--krds-color-high-contrast-gray-10);--viz-node-ring:#000;
  --viz-ten:var(--krds-color-high-contrast-primary-30);--viz-com:var(--krds-color-high-contrast-point-30);
  --viz-up:var(--krds-color-high-contrast-warning-30);--viz-down:var(--krds-color-high-contrast-primary-20);--viz-sad:var(--krds-color-high-contrast-success-40);
  --viz-mark:var(--krds-color-high-contrast-gray-0);
  --type-up:var(--krds-color-high-contrast-warning-20);--type-down:var(--gk-text-primary);--type-sad:var(--krds-color-high-contrast-success-20);
  --viz-r-lo:var(--krds-color-high-contrast-danger-50);--viz-r-mid:var(--krds-color-high-contrast-danger-20);--viz-r-hi:var(--krds-color-high-contrast-point-5);--viz-ens:var(--krds-color-high-contrast-danger-30);--viz-ens-lt:var(--krds-color-high-contrast-danger-20);
  --viz-s-lo:var(--krds-color-high-contrast-primary-50);--viz-s-mid:var(--krds-color-high-contrast-primary-20);--viz-s-hi:var(--krds-color-high-contrast-point-5);
}
*{box-sizing:border-box}
html{font-size:62.5%;-webkit-text-size-adjust:100%}
body{margin:0;background:var(--gk-bg);color:var(--gk-text);font:400 var(--krds-typo-body-medium)/1.5 'Noto Sans KR','Apple SD Gothic Neo',sans-serif;word-break:keep-all}
h1,h2,h3{font-family:'SUITE','Noto Sans KR',sans-serif;line-height:1.3;margin:0}
p,ul,ol,figure{margin:0}
a{color:var(--gk-link);text-underline-offset:.3rem}
a:hover{color:var(--gk-link-hover)}
:focus-visible{outline:none;box-shadow:var(--gk-focus);border-radius:.2rem}
.wrap{max-width:124.8rem;margin:0 auto;padding:0 2.4rem}
.skip{position:absolute;left:-999rem}.skip:focus{left:1.6rem;top:1.6rem;background:var(--gk-bg);padding:.8rem 1.6rem;z-index:9}
/* header — site name as text, no logo */
.hd{border-bottom:1px solid var(--gk-divider)}
.hd .wrap{display:flex;align-items:center;justify-content:space-between;gap:1.6rem;min-height:7.2rem}
.site{font:700 2.2rem/1.5 'SUITE','Noto Sans KR',sans-serif;color:var(--gk-text);text-decoration:none}
.site small{display:block;font:500 1.3rem/1.3 'Noto Sans KR',sans-serif;color:var(--gk-text-subtle)}
.mode{min-height:4rem;padding:0 1.6rem;border:1px solid var(--gk-border);background:var(--gk-surface);color:var(--gk-text);border-radius:var(--krds-radius-medium);font:500 1.5rem 'Noto Sans KR',sans-serif;cursor:pointer}
.crumb{background:var(--gk-surface-secondary)}
.crumb ol{list-style:none;display:flex;flex-wrap:wrap;gap:.8rem;padding:1.2rem 2.4rem;margin:0 auto;max-width:124.8rem;font-size:1.5rem;color:var(--gk-text-subtle)}
.crumb li+li::before{content:'›';margin-right:.8rem}
/* title */
.title{padding:var(--krds-number-10) 0;border-bottom:1px solid var(--gk-divider)}
.eyebrow{font:700 1.5rem 'Noto Sans KR',sans-serif;color:var(--gk-text-primary);margin-bottom:.8rem}
h1{font-size:var(--krds-pc-typo-heading-xlarge);font-weight:700;letter-spacing:-.02em}
.lead{font-size:var(--krds-typo-body-large);color:var(--gk-text-subtle);margin-top:1.2rem;max-width:80rem}
.lead .katex{font-size:1.05em}
.nei-intro{max-width:92rem;margin-top:var(--krds-number-8);padding-left:var(--krds-number-7);border-left:2px solid var(--gk-border-primary);font-size:var(--krds-typo-body-medium);line-height:1.5}
.nei-intro a{white-space:nowrap}
/* page body: in-page nav + content */
.page{display:grid;grid-template-columns:16rem minmax(0,1fr);gap:var(--krds-number-10);padding:var(--krds-number-10) 0 var(--krds-number-20)}
.page>*{min-width:0}
.toc{position:sticky;top:1.6rem;align-self:start;font-size:1.5rem}
.toc b{display:block;font-size:1.3rem;color:var(--gk-text-subtle);margin-bottom:.8rem}
.toc a{display:block;padding:.8rem 1.2rem;border-left:2px solid var(--gk-divider);color:var(--gk-text);text-decoration:none}
.toc a:hover{border-color:var(--gk-border-primary);color:var(--gk-text-primary)}
section{padding:3.2rem 0;border-top:1px solid var(--gk-divider)}
section:first-child{border-top:0;padding-top:0}
h2{font-size:var(--krds-pc-typo-heading-medium);font-weight:700;margin-bottom:1.6rem}
h3{font-size:var(--krds-typo-heading-small);font-weight:700;margin:2.4rem 0 .8rem}
.c p+p{margin-top:1.2rem}
.c ul{padding-left:2.2rem}.c li+li{margin-top:.6rem}
.katex-display{overflow-x:auto;overflow-y:hidden;padding:.4rem 0;margin:1.6rem 0}
.fine{font-size:var(--krds-typo-body-small);color:var(--gk-text-subtle)}
/* key-number cards */
.keys{display:flex;flex-wrap:wrap;gap:1.2rem 0;margin:2.4rem 0 0;padding:1.2rem 0;border-top:2px solid var(--gk-text);border-bottom:1px solid var(--gk-divider)}
.keys>div{padding:0 2.4rem;border-left:1px solid var(--gk-divider)}
.keys>div:first-child{padding-left:0;border-left:0}
.keys dt{font-size:1.4rem;color:var(--gk-text-subtle)}
.keys dd{margin:.2rem 0 0;font-size:1.9rem;font-weight:700;font-variant-numeric:tabular-nums}
.netsel{display:flex;flex-wrap:wrap;align-items:center;gap:.8rem 1.6rem;margin:2rem 0 0}
.netsel .lbl{font-weight:700;font-size:1.5rem}
.seg{flex-wrap:wrap}
/* tables */
.tbl{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:1.6rem;margin-top:1.6rem}
caption{text-align:left;font-size:1.5rem;color:var(--gk-text-subtle);padding-bottom:.8rem}
th,td{border-bottom:1px solid var(--gk-divider);padding:1rem 1.2rem;text-align:left;vertical-align:top}
thead th{border-top:2px solid var(--gk-text);background:var(--gk-surface-secondary);font-weight:700}
.sw{display:inline-block;width:1.2rem;height:1.2rem;border-radius:50%;margin-right:.6rem;vertical-align:-.1rem}
/* controls */
.ctl{display:flex;flex-wrap:wrap;gap:.8rem 2rem;align-items:center;margin:1.6rem 0;padding:1.2rem 0;border-top:1px solid var(--gk-divider);border-bottom:1px solid var(--gk-divider)}
.ctl fieldset{border:0;margin:0;padding:0;display:flex;flex-wrap:wrap;gap:.4rem 1.2rem;align-items:center}
.ctl legend{float:left;font-size:1.4rem;font-weight:700;color:var(--gk-text-subtle);margin-right:.8rem}
.ctl label{display:inline-flex;gap:.6rem;align-items:center;min-height:4rem;font-size:1.5rem;cursor:pointer}
.ctl input[type=checkbox],.ctl input[type=radio]{width:1.8rem;height:1.8rem;accent-color:var(--gk-btn-primary)}
.ctl input[type=range]{width:12rem;accent-color:var(--gk-btn-primary)}
.btn{min-height:4rem;padding:0 1.6rem;border-radius:var(--krds-radius-medium);font:500 1.5rem 'Noto Sans KR',sans-serif;cursor:pointer;border:1px solid var(--gk-btn-secondary-border);background:var(--gk-btn-secondary);color:var(--gk-text-primary)}
.btn:hover{background:var(--gk-btn-secondary-hover)}
.btn.pri{background:var(--gk-btn-primary);border-color:var(--gk-btn-primary);color:var(--gk-text-inverse)}
.btn.pri:hover{background:var(--gk-btn-primary-hover)}
.seg{display:inline-flex;border:1px solid var(--gk-border);border-radius:var(--krds-radius-medium);overflow:hidden}
.seg button{min-height:4rem;padding:0 1.4rem;border:0;background:var(--gk-surface);color:var(--gk-text);font:500 1.5rem 'Noto Sans KR',sans-serif;cursor:pointer}
.seg button+button{border-left:1px solid var(--gk-border)}
.seg button[aria-pressed=true]{background:var(--gk-btn-primary);color:var(--gk-text-inverse)}
/* plan view above two matched landscape/readout columns */
.viz{display:grid;grid-template-columns:minmax(0,1fr);gap:var(--krds-number-10)}
.plan-panel{width:100%;max-width:72rem;justify-self:center;min-width:0}
.split .plan-panel{max-width:none}
.landscape-panel{min-width:0}
.landscape-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:var(--krds-number-10);align-items:start}
.landscape-column{min-width:0}
.read h3{margin:0;font-size:var(--krds-typo-body-small);line-height:1.5}
.panel-t{font:700 1.5rem/1.5 'Noto Sans KR',sans-serif;margin-bottom:.8rem;display:flex;flex-wrap:wrap;justify-content:space-between;gap:var(--krds-number-3) var(--krds-number-6)}
.panel-t .hint{font-weight:400;color:var(--gk-text-subtle)}
.panel-t{align-items:center;min-height:4rem}
.seg.sm button{min-height:4rem;padding:0 var(--krds-number-7);font-size:var(--krds-typo-body-small)}
.layers{position:relative;aspect-ratio:1/1;border:1px solid var(--gk-border);border-radius:var(--krds-radius-large);overflow:hidden;background:var(--viz-bg);touch-action:none}
.layer{position:absolute;inset:0;margin:0}
.layer canvas{width:100%;height:100%;display:block}
.layer figcaption{display:none}
.layer[hidden]{display:none}
.layers canvas{cursor:crosshair}
.split .viz{grid-template-columns:1fr}
.split .layers{aspect-ratio:auto;display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:1.2rem;border:0;overflow:visible;background:none}
.split .layer{position:relative;inset:auto}
.split .layer canvas{aspect-ratio:1/1;height:auto;border:1px solid var(--gk-border);border-radius:var(--krds-radius-medium);background:var(--viz-bg)}
.split .layer figcaption{display:block;font-size:1.4rem;margin-top:.6rem}
.split .layer figcaption b{display:block;font-size:1.5rem}
.split .layer:first-child canvas{background:none}
#surf{width:100%;height:auto;display:block;border:1px solid var(--gk-border);border-radius:var(--krds-radius-large);background:var(--viz-bg);cursor:grab;touch-action:none}
.cam{display:flex;flex-wrap:wrap;gap:.4rem 1.6rem;align-items:center;margin:1rem 0 .4rem;font-size:1.4rem}
.cam label{display:inline-flex;gap:.6rem;align-items:center;min-height:4rem}
.cam input[type=range]{width:10rem;accent-color:var(--gk-btn-primary)}
.cam input[type=checkbox]{width:1.8rem;height:1.8rem;accent-color:var(--gk-btn-primary)}
.twin{display:grid;grid-template-columns:minmax(0,1fr);gap:var(--krds-number-7)}
.twin>div{min-width:0}
.twin .red{border-top:1px solid var(--gk-divider);padding-top:var(--krds-number-7)}
.twin h3{margin:0;font-size:var(--krds-typo-body-small);line-height:1.5}
.blue h3{color:var(--viz-down)}.red h3{color:var(--viz-ens)}
.red .katex{color:var(--viz-ens)}
#surf2{width:100%;height:auto;display:block;border:1px solid var(--gk-border);border-radius:var(--krds-radius-large);background:var(--viz-bg);cursor:grab;touch-action:none}
#lamcloud{width:100%;max-width:56rem;height:auto;display:block;border:1px solid var(--gk-border);border-radius:var(--krds-radius-medium);background:var(--viz-bg)}
.legend{display:flex;flex-wrap:wrap;gap:.4rem 1.6rem;font-size:1.4rem;color:var(--gk-text-subtle);margin-top:.8rem}
.legend i{display:inline-block;width:2rem;height:.4rem;vertical-align:middle;margin-right:.6rem;border-radius:.2rem}
/* live readout */
.read{margin-top:var(--krds-number-8);padding-top:var(--krds-number-7);border-top:2px solid var(--gk-border-secondary)}
.read-head{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:var(--krds-number-3) var(--krds-number-5)}
.read-head>span{font-size:var(--krds-typo-body-small);font-weight:700;color:var(--gk-text)}
#mat,#mat2{max-width:100%;overflow-x:auto;overflow-y:hidden;font-size:var(--krds-typo-body-small)}
.read .katex-display{margin:var(--krds-number-5) 0;padding:var(--krds-number-3) 0;text-align:left}
.read .katex-display>.katex{text-align:left}
.read .katex{font-size:1em}
.verdict{display:flex;flex-wrap:wrap;gap:var(--krds-number-3) var(--krds-number-8);align-items:baseline}
#lam,#lam2{font-size:var(--krds-typo-body-small)}
.node-picker{display:flex;align-items:center;gap:var(--krds-number-5);font-size:var(--krds-typo-body-small);font-weight:500}
.node-picker input{width:7.2rem;min-height:4rem;border:var(--gk-border-width-input,1px) solid var(--gk-border-input);border-radius:var(--krds-radius-medium);background:var(--gk-surface);color:var(--gk-text);font:inherit;padding:var(--krds-number-3) var(--krds-number-5)}
.node-picker button{padding:0 var(--krds-number-6)}
.viz-help{margin-top:var(--krds-number-7);border-top:1px solid var(--gk-divider);font-size:var(--krds-typo-body-small);line-height:1.5;color:var(--gk-text-subtle)}
.viz-help summary{min-height:4rem;padding:var(--krds-number-5) 0;cursor:pointer;font-weight:700;color:var(--gk-text)}
.viz-help p{margin:var(--krds-number-5) 0}
.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip-path:inset(50%);white-space:nowrap;border:0}
.stat{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));margin-top:1.6rem;border-top:1px solid var(--gk-divider)}
.stat div{min-width:0;padding:var(--krds-number-5) var(--krds-number-3) 0;font-size:var(--krds-typo-body-xsmall);color:var(--gk-text-subtle)}
.stat b{display:block;font-size:var(--krds-typo-body-large);color:var(--gk-text);font-variant-numeric:tabular-nums}
.refs{padding-left:2.2rem;font-size:1.5rem}
code{font-size:.9em;background:var(--gk-surface-secondary);padding:.1rem .4rem;border-radius:.2rem}
.ft{background:var(--gk-surface-secondary);border-top:1px solid var(--gk-divider);padding:3.2rem 0;font-size:1.5rem;color:var(--gk-text-subtle)}
.ft strong{color:var(--gk-text)}
@media (max-width:1024px){.page{grid-template-columns:1fr}.toc{position:static;display:flex;flex-wrap:wrap;gap:.4rem}.toc b{width:100%}.toc a{min-height:4rem;border:1px solid var(--gk-divider);border-radius:var(--krds-radius-medium)}.split .layers{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:767px){.landscape-grid{grid-template-columns:minmax(0,1fr)}.viz{gap:var(--krds-number-10)}.ctl{gap:var(--krds-number-5) var(--krds-number-7)}.ctl fieldset{gap:var(--krds-number-3) var(--krds-number-7)}.panel-t .hint{width:100%}.cam{gap:var(--krds-number-3) var(--krds-number-7)}.cam label{flex:1 1 14rem;justify-content:space-between}.cam input[type=range]{width:9.6rem}.split .layers{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:768px){.wrap{padding:0 1.6rem}h1{font-size:var(--krds-mobile-typo-heading-xlarge)}h2{font-size:var(--krds-mobile-typo-heading-medium)}.stat{grid-template-columns:repeat(2,minmax(0,1fr))}.keys>div{padding:0 1.6rem 0 0;border-left:0;width:50%}}
@media print{.ctl,.mode,.toc{display:none}}
</style>
</head>
<body>
<a class="skip" href="#main">본문 바로가기</a>
<header class="hd"><div class="wrap">
  <a class="site" href="#main">NEI 연구<small>Network Embedding Inconsistency</small></a>
  <button type="button" class="mode" id="mode" aria-pressed="false">선명한 화면</button>
</div></header>
<nav class="crumb" aria-label="현재 위치"><ol><li>홈</li><li>NEI 연구</li><li aria-current="page">Stress Hessian</li></ol></nav>

<main id="main" tabindex="-1" class="wrap">
<div class="title">
  <p class="eyebrow">Interactive note · statistical physics · network embedding</p>
  <h1>The stress Hessian</h1>
  <p class="lead">모든 node 쌍을 spring 으로 이은 network embedding 에서, node 하나를 움직일 때 energy 의 curvature 를 본다. node block Hessian $\mathbf H_{ii}$ 는 언제나 positive semidefinite 인 stiffness $K$ 와, 쌍마다 기여의 부호가 tension 의 부호로 정해지는 prestress $S$ 의 합으로 정확히 나뉜다.</p>
  <p class="nei-intro"><b>NEI (Network Embedding Inconsistency)</b> 는 같은 network 를 같은 조건에서 반복 embedding 했을 때, node 쌍 거리의 상대적 분산을 평균한 값이다. 작을수록 초기조건에 따른 거리 배치의 차이가 작다. <a href="#nei">정의와 수식</a></p>
  <dl class="keys" id="keys" aria-label="Key numbers"></dl>
</div>

<div class="page">
<nav class="toc" aria-label="본문 메뉴"><b>Contents</b>
  <a href="#model">Model</a><a href="#viz">Slice viewer</a><a href="#nei">NEI ensemble</a><a href="#decomp">$K$ and $S$</a><a href="#numbers">Numbers</a><a href="#scope">Scope</a>
</nav>
<div class="c">

<section id="model">
<h2>Model</h2>
<p>graph $G$ 의 node 를 평면에 embed 하고, 모든 쌍 $(i,j)$ 를 rest length $\delta_{ij}$ (unweighted shortest-path hop distance) 의 spring 으로 잇는다. energy 는 unweighted stress 다.</p>
$$F(X)=\sum_{i<j}\bigl(d_{ij}-\delta_{ij}\bigr)^2,\qquad d_{ij}=\lVert\mathbf x_i-\mathbf x_j\rVert,\qquad X=(\mathbf x_1,\dots,\mathbf x_N)\in\mathbb R^{2N}.$$
<p>node position 은 metric MDS 로 정한다. classical (Torgerson) MDS 를 initial condition 으로 두고 SMACOF (stress majorization) 와 L-BFGS 로 $F$ 를 줄여, optimizer 가 멈춘 점을 <b>terminal</b> 이라 부른다. 아래 다섯 MDS terminal 은 projected Hessian $\mathbf H_\perp$ 이 positive definite 임을 확인했으므로 (Numbers) $F$ 의 strict local minimum 이다. constraint $N(N-1)/2$ 개가 degree of freedom $2N-3$ 보다 훨씬 많은 overconstrained framework 이고, hop distance metric 이 2차원 Euclidean metric 으로 정확히 실현되지 않는 한 ($F\gt0$; 다섯 network 모두 해당) 모든 spring 이 동시에 rest length 에 있을 수는 없다. terminal 에 남은 spring force 가 prestress 다.</p>
<p>node $i$ 하나만 움직이고 나머지를 terminal 에 고정하면 $F$ 는 2차원 slice $F_i(\mathbf x)$ 가 되고, 그 Hessian 이 full $2N\times2N$ Hessian 의 $(i,i)$ block 이다.</p>
$$F_i(\mathbf x)=\sum_{j\ne i}\bigl(\lVert\mathbf x-\mathbf x_j\rVert-\delta_{ij}\bigr)^2+\text{const},\qquad \mathbf H_{ii}=\nabla\nabla^{\mathsf T}F_i=\mathbf J(\nabla F_i).$$
<p>symmetric matrix $\mathbf H_{ii}$ 의 eigenvalue $\lambda_1\le\lambda_2$ 는 그 eigenvector 방향의 second directional derivative 이다 ($F_i$ 의 graph 곡면이 갖는 principal curvature 와는 $\nabla F_i=\mathbf 0$ 에서만 일치한다). eigenvector $\mathbf v$ 방향으로 $\epsilon$ 만큼 움직이면 $F_i(\mathbf x+\epsilon\mathbf v)=F_i(\mathbf x)+\epsilon\,\nabla F_i\cdot\mathbf v+\tfrac12\lambda\epsilon^2+O(\epsilon^3)$ 이다.</p>
<div class="tbl"><table>
<caption>eigenvalue 부호에 따른 local curvature type. stationary point ($\nabla F_i=\mathbf 0$) 가 아닌 곳에서는 그 자리의 이차 형태일 뿐이고, 음의 eigenvalue 개수가 Morse index 가 되는 것은 nondegenerate stationary point 에서뿐이다.</caption>
<thead><tr><th scope="col">type</th><th scope="col">eigenvalues</th><th scope="col">at a stationary point</th></tr></thead>
<tbody>
<tr><td><span class="sw" style="background:var(--viz-up)"></span><b>BOWL</b></td><td>$0\lt\lambda_1\le\lambda_2$</td><td>local minimum, index 0</td></tr>
<tr><td><span class="sw" style="background:var(--viz-sad)"></span><b>SADDLE</b></td><td>$\lambda_1\lt0\lt\lambda_2$</td><td>saddle point (slice barrier), index 1</td></tr>
<tr><td><span class="sw" style="background:var(--viz-down)"></span><b>DOME</b></td><td>$\lambda_1\le\lambda_2\lt0$</td><td>local maximum, index 2</td></tr>
</tbody></table></div>
</section>

<section id="viz">
<h2>Slice viewer</h2>
<div class="netsel"><span class="lbl" id="net-lbl">network</span><div class="seg" id="nets" role="group" aria-labelledby="net-lbl"></div></div>
<p><span id="vizlead"></span> Plan view 의 다섯 layer 와 같은 $F_i$ 의 3D landscape 를 비교한다. node 를 선택하거나 끌어서 slice 를 바꿀 수 있다.</p>

<div class="ctl" role="group" aria-label="그림 조절">
  <button type="button" class="btn pri" id="play">일시정지</button>
  <button type="button" class="btn" id="reset">terminal 로</button>
  <div class="seg" role="group" aria-label="layer 배치"><button type="button" id="stack" aria-pressed="true">overlay</button><button type="button" id="split" aria-pressed="false">split</button></div>
  <fieldset><legend>layer</legend>
    <label><input type="checkbox" data-layer="L1" checked>contour</label>
    <label><input type="checkbox" data-layer="L2" checked>spring stress</label>
    <label><input type="checkbox" data-layer="L3" checked>network</label>
    <label><input type="checkbox" data-layer="L4" checked>eigenvector</label>
    <label><input type="checkbox" data-layer="L5" checked>NEI ensemble</label>
  </fieldset>
  <label>landscape 강조 <input type="range" id="gamma" min="0" max="100" value="30" aria-describedby="gamma-note"></label>
</div>

<div class="viz">
  <div class="plan-panel">
    <div class="panel-t"><span>Plan view (layers)</span><div class="node-picker"><label for="node-choice">node</label><input type="number" id="node-choice" required min="0" step="1" inputmode="numeric" aria-describedby="selinfo"><button type="button" class="btn" id="node-apply">선택</button></div><span class="hint" id="selinfo"></span></div>
    <div class="layers" id="layers" tabindex="0" role="group" aria-label="Plan view. 방향키로 선택 node 이동, Home 으로 terminal 복귀">
      <figure class="layer" id="L1"><canvas width="1040" height="1040" aria-label="contour layer"></canvas><figcaption><b>① contour</b>$\log F_i$ 의 contour band</figcaption></figure>
      <figure class="layer" id="L2"><canvas width="1040" height="1040" aria-label="spring stress layer"></canvas><figcaption><b>② spring stress</b>선택 node 의 $N-1$ 개 spring</figcaption></figure>
      <figure class="layer" id="L3"><canvas width="1040" height="1040" aria-label="network layer"></canvas><figcaption><b>③ network</b>graph edge, node size ∝ √degree</figcaption></figure>
      <figure class="layer" id="L4"><canvas width="1040" height="1040" aria-label="eigenvector layer"></canvas><figcaption><b>④ eigenvector</b>$\mathbf H_{ii}$ 의 principal direction, 길이 ∝ $\lvert\lambda\rvert$</figcaption></figure>
      <figure class="layer" id="L5"><canvas width="1040" height="1040" aria-label="NEI ensemble layer"></canvas><figcaption><b>⑤ NEI ensemble</b>$M$ 개 terminal 의 node $i$ 위치, $\langle F_i\rangle_M$ contour, $\langle\mathbf H_{ii}\rangle_M$ 의 eigenvector</figcaption></figure>
    </div>
    <div class="cam" role="group" aria-label="plan view zoom">
      <button type="button" class="btn" id="pz-in" aria-label="zoom in">＋</button>
      <button type="button" class="btn" id="pz-out" aria-label="zoom out">－</button>
      <button type="button" class="btn" id="pz-node">node 로 zoom</button>
      <button type="button" class="btn" id="pz-fit">전체 보기</button>
      <span class="hint">wheel · pinch 로 zoom, Shift + drag 로 pan</span>
    </div>
    <div class="legend">
      <span><i style="background:var(--viz-edge)"></i>graph edge</span>
      <span><i style="background:var(--viz-ten)"></i>tension $d\gt\delta$</span>
      <span><i style="background:var(--viz-com)"></i>compression $d\lt\delta$</span>
      <span><i style="background:var(--viz-up)"></i>$\lambda\gt0$</span>
      <span><i style="background:var(--viz-down)"></i>$\lambda\lt0$</span>
      <span><i style="background:var(--viz-ens)"></i>NEI ensemble</span>
    </div>

  </div>
  <div class="landscape-panel">
    <div class="panel-t"><span>Landscape $F_i(\mathbf x)$</span> <div class="seg sm" role="group" aria-label="view"><button type="button" data-view="top" aria-pressed="false">top</button><button type="button" data-view="tilt" aria-pressed="true">tilt</button><button type="button" data-view="side" aria-pressed="false">side</button></div></div>
    <div class="landscape-grid" aria-label="Single terminal 과 NEI ensemble 비교" aria-describedby="slice-interpretation">
      <div class="landscape-column"><canvas id="surf" width="1040" height="760" aria-label="3D landscape of F_i, single terminal"></canvas><div class="read"><div class="blue"><div class="read-head"><h3>single terminal</h3><span id="verdict"></span></div><div id="mat" tabindex="0" aria-label="Single terminal: Hessian, stiffness, prestress"></div><div class="verdict"><span id="lam"></span></div></div></div></div>
      <div class="landscape-column"><canvas id="surf2" width="1040" height="760" aria-label="3D landscape of the ensemble mean F_i"></canvas><div class="read"><div class="red"><div class="read-head"><h3>NEI ensemble · <span id="ens-count"></span></h3><span id="verdict2"></span></div><div id="mat2" tabindex="0" aria-label="Ensemble mean: Hessian, stiffness, prestress"></div><div class="verdict"><span id="lam2"></span></div></div></div></div>
    </div>
    <p class="fine" id="slice-interpretation"><b>곡률 라벨.</b> BOWL / SADDLE / DOME 은 그 지점 Hessian 의 이차형식 모양이다. stationary point 가 아닌 곳에서는 minimum·saddle point·maximum 또는 Morse index 의 판정이 아니다.<br><b>Slice 장벽.</b> 다른 node 를 고정한 조건부 장벽이며, 모든 node 가 함께 움직이는 전체 landscape 의 최소 에너지 경로 장벽과는 다르다.</p>
    <div class="stat">
    <div>$F_i-F_i^{\rm term}$<b id="dF"></b></div>
    <div>tension pairs<b id="nT"></b></div>
    <div>compression pairs<b id="nC"></b></div>
    <div>$\lVert\nabla F_i\rVert$<b id="gN"></b></div>
  </div>
    <p class="legend"><span><i style="background:var(--viz-s-lo)"></i>single terminal $F_i$</span><span><i style="background:var(--viz-r-lo)"></i>NEI ensemble $\langle F_i\rangle_M$</span><span class="hint" id="msinfo"></span></p>
    <div class="cam" role="group" aria-label="camera">
      <label>azimuth <input type="range" id="cam-az" min="-180" max="180" step="1"></label>
      <label>elevation <input type="range" id="cam-el" min="2" max="83" step="1"></label>
      <label>zoom <input type="range" id="cam-zoom" min="60" max="300" step="5" value="100"></label>
      <label>height scale <input type="range" id="cam-h" min="30" max="250" step="5" value="100"></label>
      <label><input type="checkbox" id="cam-spin" checked>auto-rotate</label>
      <button type="button" class="btn" id="cam-reset">reset camera</button>
    </div>
    <details class="viz-help"><summary>조작 방법 · 높이 변환</summary>
      <p>Landscape 를 끌면 회전하고, wheel 이나 pinch 로 zoom 한다. Plan view 는 Shift + drag 로 pan, 방향키로 선택 node 이동, Home 으로 terminal 에 복귀한다.</p>
      <p id="gamma-note">height 는 window 안의 $F_i$ 를 saturating linear height $1-e^{-(F_i-F_{\min})/\ell}$ 와 rank (quantile) height 사이에서 섞은 값이다 (강조 0 = linear). $\ell$ 은 preset node 에서 barrier 높이의 2배, 그 밖에는 window median 까지의 높이다. 둘 다 monotone transform 이라 contour 와 stationary point 의 위치·type 은 그대로이고, 색칠한 curvature 판정은 언제나 원래 $F_i$ 의 $\mathbf H_{ii}$ 로 한다. plan view 의 점선 사각형이 landscape window 다.</p>
    </details>
  </div>
</div>

<p id="viewer-status" class="sr-only" role="status"></p>
</section>

<section id="nei">
<h2>NEI ensemble</h2>
<p>NEI 는 하나의 terminal 이 아니라 terminal 분포의 statistic 이다. 같은 graph 를 census protocol (initial condition $\mathrm{Uniform}[0,1]^{2N}$, seed $1000+m$ → SMACOF → L-BFGS) 로 $M=500$ 번 relax 하고, labeled pair distance 의 relative spread 를 평균한다.</p>
$$\mathrm{NEI}=\frac{1}{\binom N2}\sum_{i<j}\frac{\operatorname{Var}_m\bigl(d_{ij}^{(m)}\bigr)}{\bigl\langle d_{ij}^{(m)}\bigr\rangle_m^{2}},\qquad m=1,\dots,M.$$
<p>붉은 쪽의 Hessian 은 이 ensemble 위의 slice 다. terminal $m$ 을 기준 MDS terminal 에 orthogonal Procrustes (translation, rotation, reflection) 로 맞춘 좌표 $\tilde{\mathbf x}_j^{(m)}$ 로 다른 node 를 고정하고, slice energy 를 ensemble 평균한다.</p>
$$\langle F_i\rangle_M(\mathbf x)=\frac1M\sum_{m=1}^{M}\sum_{j\ne i}\bigl(\lVert\mathbf x-\tilde{\mathbf x}_j^{(m)}\rVert-\delta_{ij}\bigr)^2,\qquad \langle\mathbf H_{ii}\rangle_M=\nabla\nabla^{\mathsf T}\langle F_i\rangle_M=\langle K\rangle_M+\langle S\rangle_M .$$
<p>미분은 선형이므로 평균의 Hessian 은 Hessian 의 평균이다. $\operatorname{tr}\langle K\rangle_M=2(N-1)$ 은 그대로이므로 trace 를 바꾸는 것은 $\langle S\rangle_M$ 뿐이다 ($\langle K\rangle_M$ 의 anisotropy 는 바뀐다). 평균이 basin 구조를 지운다고 기대하기 쉽지만, 이 다섯 preset node 에서는 그렇지 않다. landscape window 의 $61\times61$ grid 에서 local minimum 개수는 단일 terminal 대 ensemble 평균이 karate 2 대 2, dolphins 2 대 2, Les Misérables 3 대 1, football 1 대 1, ca-netscience 2 대 2 였고, dome ($\lambda_1\le\lambda_2\lt0$) 면적 비율은 Les Misérables 에서만 늘고 (0.059 → 0.078) 나머지 넷에서는 조금 줄었다. 그림에서 붉은 landscape 가 더 매끈해 보인다면 높이 변환과 subsample ($M_s$) 의 영향일 수 있으니, 비교는 위 숫자로 한다. aligned 좌표에서 다른 node 의 position spread 는 Procrustes 에 의존하는 양이라 NEI (gauge 와 무관한 pair-distance spread) 와 관련은 있어도 같은 양은 아니다.</p>
<p>$\langle F_i\rangle_M$ 은 Procrustes 기준 (gauge) 에 의존한다. gauge 와 무관한 비교로, 각 terminal 이 자기 위치에서 갖는 block $\mathbf H_{ii}^{(m)}\bigl(\mathbf x_i^{(m)}\bigr)$ 의 eigenvalue 분포를 아래에 함께 그린다 (rigid motion 에 불변). ensemble terminal 은 optimizer 가 멈춘 점이며, 각각의 $\mathbf H_\perp$ 로 local minimum 인지 확인하지는 않았다.</p>
<canvas id="lamcloud" width="1040" height="640" aria-label="eigenvalues of each terminal's own node block"></canvas>
<p class="fine" id="lamcloud-note"></p>
</section>

<section id="decomp">
<h2>$K$ and $S$: curvature 는 어디서 오는가</h2>
<p>쌍 방향의 unit vector 를 $\mathbf u_{ij}=(\mathbf x_i-\mathbf x_j)/d_{ij}$, projector 를 $P_{ij}=\mathbf u_{ij}\mathbf u_{ij}^{\mathsf T}$ 라 하면 block 은 정확히 두 항으로 나뉜다.</p>
$$\mathbf H_{ii}=\underbrace{2\sum_{j\ne i}P_{ij}}_{K\ \succeq\ 0\ \text{(stiffness)}}\;+\;\underbrace{\sum_{j\ne i}\omega_{ij}\bigl(\mathbf I-P_{ij}\bigr)}_{S\ \text{(prestress)}},\qquad \omega_{ij}=\frac{2\,(d_{ij}-\delta_{ij})}{d_{ij}}.$$
<p>$K$ 는 spring 을 longitudinal 방향으로 늘이는 (material) stiffness 다. 위치와 무관하게 $\operatorname{tr}K=2(N-1)$ 이지만 $K$ 의 anisotropy 는 위치에 따라 바뀐다. $\omega_{ij}$ 는 force density (길이당 tension) 로, rigidity theory 의 stress coefficient 와 같은 양이다. $S$ 는 transverse 방향에만 작용하는 geometric (stress) stiffness 로, tension ($\omega\gt0$) 은 transverse 방향을 단단하게, compression ($\omega\lt0$) 은 무르게 만든다. 따라서</p>
$$\operatorname{tr}\mathbf H_{ii}=2(N-1)+\sum_{j\ne i}\omega_{ij},\qquad \text{DOME}\ \Rightarrow\ \sum_{j\ne i}\omega_{ij}\lt-2(N-1).$$
<p>즉 dome 의 필요조건은 그 자리에서 compression 의 force density 합이 고정된 stiffness trace 를 넘어서는 것이고, 이것은 $S$ 만으로 정해진다. 반면 saddle ($\det\mathbf H_{ii}\lt0$) 은 $K$ 의 anisotropy 와 $S$ 가 함께 정한다. 같은 graph 라도 terminal 마다 좌표와 prestress 배치가 다르므로 curvature map 도 terminal 마다 다르다.</p>
</section>

<section id="numbers">
<h2>Numbers</h2>
<p id="facts"></p>
<div class="tbl"><table>
<caption>terminal 에서의 check (build 때 numpy 로 계산)</caption>
<thead><tr><th scope="col">item</th><th scope="col">value</th></tr></thead>
<tbody id="checks"></tbody></table></div>
<noscript><p class="fine">Numbers 표의 값은 JavaScript 로 선택한 network 에 맞춰 채워진다. 수치를 보려면 JavaScript 를 활성화해야 한다.</p></noscript>
</section>

<section id="scope">
<h2>Scope</h2>
<ul>
<li><b>Slice 다.</b> barrier 와 second basin 은 다른 $N-1$ 개 node 를 고정한 conditional 값이다. 모든 node 가 함께 relax 하는 full landscape 의 minimum-energy path 와는 다른 양이며, 다른 좌표가 relax 하면 barrier 는 낮아질 수 있다.</li>
<li><b>Block 판정이다.</b> BOWL / SADDLE / DOME 은 $\mathbf H_{ii}$ 하나에 대한 것이다. terminal 전체의 stability 는 Numbers 표의 $\mathbf H_\perp$ (rigid-body mode 3개를 뺀 $2N-3$ 차원) 로 따로 확인했다.</li>
<li><b>Unweighted stress.</b> graph drawing 에서 흔한 $w_{ij}=\delta_{ij}^{-2}$ weight (Kamada–Kawai 형) 가 아니다. weight 를 넣으면 $K$ 와 $\omega_{ij}$ 에 $w_{ij}$ 가 곱해질 뿐 decomposition 의 형태는 같고, 다만 $\operatorname{tr}K=2\sum_j w_{ij}$ 가 된다.</li>
<li><b>Data.</b> 모두 unweighted. Zachary karate club ($N=34$), Lusseau dolphins ($N=62$), Les Misérables co-appearance ($N=77$, weight 제거), college football ($N=115$) 은 largest connected component 다. ca-netscience coauthorship 은 Network Repository 의 ca-netscience 를 NEI 연구에서 쓴 그대로 ($N=352$, $E=707$) 가져왔다.</li>
<li><b>Terminal.</b> classical MDS → SMACOF → L-BFGS, random seed 없음. classical MDS 에서 겹치는 structurally equivalent node 는 $10^{-3}$ 크기의 golden-angle offset 으로 떼어 놓았다.</li>
<li><b>Preset node.</b> 모든 node 의 slice 에서 terminal 이 아닌 basin (grid local minimum 을 BFGS 로 다듬고, minimax path 의 barrier 가 두 minimum 보다 0.05 이상 높은 것) 을 찾아 gain $F_i^{\rm term}-F_i^{\rm basin}$ 이 가장 큰 node 를 골랐다. gain $\gt0$ 이면 slice 안에 terminal 보다 낮은 basin 이 있다. college football 은 모든 slice 가 single basin 이라, $\lambda_1$ 이 가장 작은 (가장 soft 한) node 로 dome region 까지 다녀오는 curvature tour 를 보여 준다. 현상이 잘 보이는 node 를 고른 것이지 random sample 이 아니다.</li>
<li><b>NEI ensemble.</b> 각 network 에서 census protocol 로 $M=500$ 개 terminal 을 만들고 (build/ensemble.py), 기준 MDS terminal 에 orthogonal Procrustes 로 맞췄다. 붉은 landscape 와 contour 는 계산량 때문에 처음 $M_s$ 개 terminal 로 그리고 ($M_s$ 는 그림 아래 표시), 붉은 Hessian readout 과 eigenvalue 분포는 $M$ 전체를 쓴다. $\langle F_i\rangle_M$ 과 그 Hessian 은 Procrustes gauge 에 의존하고, 자기 위치에서 잰 $\mathbf H_{ii}^{(m)}$ 의 eigenvalue 는 의존하지 않는다.</li>
<li><b>Reproduction.</b> <code>python build/make_nets.py</code>, <code>python build/ensemble.py</code>, <code>python build/build_page.py</code>, <code>node build/check_katex.cjs index.html</code> (math compile error 0).</li>
</ul>
<h3>References</h3>
<ol class="refs">
<li>R. Connelly, Rigidity and energy, <i>Invent. Math.</i> <b>66</b>, 11 (1982) — stress matrix 와 prestress stability.</li>
<li>S. D. Guest, The stiffness of prestressed frameworks: a unifying approach, <i>Int. J. Solids Struct.</i> <b>43</b>, 842 (2006) — material + geometric stiffness.</li>
<li>T. Kamada, S. Kawai, An algorithm for drawing general undirected graphs, <i>Inf. Process. Lett.</i> <b>31</b>, 7 (1989).</li>
<li>E. R. Gansner, Y. Koren, S. North, Graph drawing by stress majorization, <i>Graph Drawing 2004</i>, LNCS 3383, 239 (2005).</li>
<li>J. de Leeuw, Applications of convex analysis to multidimensional scaling, in <i>Recent Developments in Statistics</i>, 133 (North-Holland, 1977) — SMACOF.</li>
<li>W. S. Torgerson, Multidimensional scaling: I. Theory and method, <i>Psychometrika</i> <b>17</b>, 401 (1952) — classical MDS.</li>
<li>W. W. Zachary, An information flow model for conflict and fission in small groups, <i>J. Anthropol. Res.</i> <b>33</b>, 452 (1977).</li>
<li>D. Lusseau <i>et al.</i>, The bottlenose dolphin community of Doubtful Sound features a large proportion of long-lasting associations, <i>Behav. Ecol. Sociobiol.</i> <b>54</b>, 396 (2003).</li>
<li>D. E. Knuth, <i>The Stanford GraphBase</i> (Addison-Wesley, 1993) — Les Misérables.</li>
<li>M. Girvan, M. E. J. Newman, Community structure in social and biological networks, <i>PNAS</i> <b>99</b>, 7821 (2002) — college football.</li>
<li>M. E. J. Newman, Finding community structure in networks using the eigenvectors of matrices, <i>Phys. Rev. E</i> <b>74</b>, 036104 (2006) — netscience.</li>
<li>R. A. Rossi, N. K. Ahmed, The network data repository with interactive graph analytics and visualization, <i>AAAI</i> (2015) — ca-netscience.</li>
</ol>
</section>

</div>
</div>
</main>
<footer class="ft"><div class="wrap">
<p><strong>Network Embedding Inconsistency (NEI) 연구</strong></p>
<p>디자인: KRDS 디자인 스타일 확장형 (GNU 색·서체). 최종 수정 2026-10-28.</p>
</div></footer>

<script src="vendor/katex/katex.min.js"></script>
<script src="vendor/katex/contrib/auto-render.min.js"></script>
<script>
const NETS = __NETS__;
(function(){
const $ = id => document.getElementById(id);
const cvs = ['L1', 'L2', 'L3', 'L4', 'L5'].map(id => $(id).querySelector('canvas'));
const [c1, c2, c3, c4, c5] = cvs.map(c => c.getContext('2d'));
const W = cvs[0].width;
let NET, n, X, E, Draw, span, cx, cy, S, deg, EX, M;
const delta = (i, j) => Draw[i * n + j];
const toB = (x, y) => [W / 2 + (x - cx) * S, W / 2 - (y - cy) * S];
const fromB = (a, b) => [(a - W / 2) / S + cx, -(b - W / 2) / S + cy];
// plan-view zoom and pan: canvas = W/2 + vz (base - W/2) + (vx, vy)
let vz = 1, vx = 0, vy = 0;
const toC = (x, y) => { const [a, b] = toB(x, y); return [W / 2 + vz * (a - W / 2) + vx, W / 2 + vz * (b - W / 2) + vy]; };
const fromC = (a, b) => fromB(W / 2 + (a - W / 2 - vx) / vz, W / 2 + (b - W / 2 - vy) / vz);
// an image drawn at view (z0, x0, y0) shown at the current view; redrawn sharp once zooming stops
const blit = (ctx, img, [z0, x0, y0]) => { const f = vz / z0; ctx.setTransform(f, 0, 0, f, W / 2 * (1 - f) - f * x0 + vx, W / 2 * (1 - f) - f * y0 + vy); ctx.drawImage(img, 0, 0); ctx.setTransform(1, 0, 0, 1, 0, 0); };
let redrawT = null; const viewChanged = () => { dirty = true; clearTimeout(redrawT); redrawT = setTimeout(() => { buildContour(); buildEnsLayer(); dirty = true; }, 250); };
function setNet(key) {
  NET = NETS[key]; n = NET.N; X = NET.X; E = NET.E;
  Draw = Uint8Array.from(atob(NET.D), c => c.charCodeAt(0));
  const lo = NET.lo, hi = NET.hi; span = Math.max(hi[0] - lo[0], hi[1] - lo[1]) * 1.04;
  cx = (lo[0] + hi[0]) / 2; cy = (lo[1] + hi[1]) / 2; S = W / span;
  deg = new Array(n).fill(0); for (const [a, b] of E) { deg[a]++; deg[b]++; }
  // aligned ensemble terminals, int16 relative to the centre, (M, N, 2)
  const en = NET.ens, raw = Uint8Array.from(atob(en.q), c => c.charCodeAt(0)), q = new Int16Array(raw.buffer);
  M = en.M; EX = new Float32Array(q.length); for (let k = 0; k < q.length; k++) EX[k] = en.c[k & 1] + q[k] * en.s;
}
// mean over the first Ms terminals of the node-i slice and its block Hessian
function eFi(p, i, Ms) { let f = 0; for (let m = 0; m < Ms; m++) { const o = m * n * 2; for (let j = 0; j < n; j++) { if (j === i) continue; const e = Math.hypot(p[0] - EX[o + 2 * j], p[1] - EX[o + 2 * j + 1]) - delta(i, j); f += e * e; } } return f / Ms; }
function eHess(p, i) {
  const K = [0, 0, 0], Sm = [0, 0, 0]; let f = 0;
  for (let m = 0; m < M; m++) { const o = m * n * 2;
    for (let j = 0; j < n; j++) { if (j === i) continue;
      const dx = p[0] - EX[o + 2 * j], dy = p[1] - EX[o + 2 * j + 1], d = Math.hypot(dx, dy) || 1e-9, de = delta(i, j), ux = dx / d, uy = dy / d, w = 2 * (d - de) / d;
      K[0] += 2 * ux * ux; K[1] += 2 * ux * uy; K[2] += 2 * uy * uy; Sm[0] += w * (1 - ux * ux); Sm[1] -= w * ux * uy; Sm[2] += w * (1 - uy * uy); f += (d - de) * (d - de); } }
  for (const A of [K, Sm]) for (let k = 0; k < 3; k++) A[k] /= M;
  const H = [K[0] + Sm[0], K[1] + Sm[1], K[2] + Sm[2]], [l1, l2] = eig(H);
  const ev = l => { let vx = H[1], vy = l - H[0]; if (Math.hypot(vx, vy) < 1e-12) { vx = l - H[2]; vy = H[1]; } if (Math.hypot(vx, vy) < 1e-12) { vx = 1; vy = 0; } const r = Math.hypot(vx, vy); return [vx / r, vy / r]; };
  return {H, K, S: Sm, l1, l2, v1: ev(l1), v2: ev(l2), f: f / M};
}
// each terminal's own block at its own node position (gauge free)
function ownEigs(i) { const out = []; for (let m = 0; m < M; m++) { const o = m * n * 2, p = [EX[o + 2 * i], EX[o + 2 * i + 1]]; const H = [0, 0, 0];
  for (let j = 0; j < n; j++) { if (j === i) continue; const dx = p[0] - EX[o + 2 * j], dy = p[1] - EX[o + 2 * j + 1], d = Math.hypot(dx, dy) || 1e-9, de = delta(i, j), ux = dx / d, uy = dy / d, w = 2 * (d - de) / d;
    H[0] += 2 * ux * ux + w * (1 - ux * ux); H[1] += 2 * ux * uy - w * ux * uy; H[2] += 2 * uy * uy + w * (1 - uy * uy); }
  out.push(eig(H)); } return out; }
const budget = G => Math.max(10, Math.min(M, Math.floor(4e7 / ((n - 1) * G * G))));
let C = {};
const hex = s => { s = s.trim(); if (s[0] !== '#') return [128, 128, 128]; if (s.length === 4) s = '#' + [...s.slice(1)].map(c => c + c).join(''); return [1, 3, 5].map(k => parseInt(s.slice(k, k + 2), 16)); };
const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
function readColors() { const cs = getComputedStyle(document.documentElement); for (const k of ['bg', 'band-lo', 'band-hi', 'iso', 'edge', 'node', 'node-ring', 'ten', 'com', 'up', 'down', 'sad', 'mark', 's-lo', 's-mid', 's-hi', 'r-lo', 'r-mid', 'r-hi', 'ens', 'ens-lt']) C[k] = hex(cs.getPropertyValue('--viz-' + k)); }
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const narrowReadout = matchMedia('(max-width: 900px)');
let sel, pos, t = 0, playing = !reducedMotion.matches, dragging = false, dirty = true;
let planRevision = 0, viewerVisible = true;
let gamma = 0.3, az = -0.75, el = 0.55; const cam = {zoom: 1, h: 1, spin: !reducedMotion.matches};

function Fi(p, i) { let f = 0; for (let j = 0; j < n; j++) { if (j === i) continue; const e = Math.hypot(p[0] - X[j][0], p[1] - X[j][1]) - delta(i, j); f += e * e; } return f; }
function hess(p, i) {
  const K = [0, 0, 0], Sm = [0, 0, 0], g = [0, 0]; let f = 0, nt = 0, nc = 0;
  for (let j = 0; j < n; j++) {
    if (j === i) continue;
    const dx = p[0] - X[j][0], dy = p[1] - X[j][1], d = Math.hypot(dx, dy) || 1e-9, de = delta(i, j);
    const ux = dx / d, uy = dy / d, w = 2 * (d - de) / d;
    K[0] += 2 * ux * ux; K[1] += 2 * ux * uy; K[2] += 2 * uy * uy;
    Sm[0] += w * (1 - ux * ux); Sm[1] -= w * ux * uy; Sm[2] += w * (1 - uy * uy);
    g[0] += 2 * (d - de) * ux; g[1] += 2 * (d - de) * uy; f += (d - de) * (d - de);
    if (d > de) nt++; else nc++;
  }
  const H = [K[0] + Sm[0], K[1] + Sm[1], K[2] + Sm[2]];
  const [l1, l2] = eig(H);
  const ev = l => { let vx = H[1], vy = l - H[0]; if (Math.hypot(vx, vy) < 1e-12) { vx = l - H[2]; vy = H[1]; } if (Math.hypot(vx, vy) < 1e-12) { vx = 1; vy = 0; } const r = Math.hypot(vx, vy); return [vx / r, vy / r]; };
  return {H, K, S: Sm, l1, l2, v1: ev(l1), v2: ev(l2), g, f, nt, nc};
}
function eig(M) { const tr = M[0] + M[2], det = M[0] * M[2] - M[1] * M[1], q = Math.sqrt(Math.max(tr * tr / 4 - det, 0)); return [tr / 2 - q, tr / 2 + q]; }

// ── layer 1: contour bands of log F_i over the whole map ──
let Fref = 0, win = null, surf = null, contImg = null, contView = [1, 0, 0], ensView = [1, 0, 0];
function buildContour() {
  planRevision++;
  // sample the visible region so that zooming in adds detail
  const [rx0, ry0] = fromC(0, W), rs = W / (S * vz);
  const G = 97, L = new Float64Array(G * G); let mn = Infinity, mx = -Infinity;
  for (let a = 0; a < G; a++) for (let b = 0; b < G; b++) { const v = Fi([rx0 + rs * b / (G - 1), ry0 + rs * a / (G - 1)], sel); L[a * G + b] = v; mn = Math.min(mn, v); mx = Math.max(mx, v); }
  for (let k = 0; k < G * G; k++) L[k] = Math.log(L[k] - mn + 1e-3 * (mx - mn));
  const srt = Array.from(L).sort((p, q) => p - q), NB = 22, qs = [];
  for (let k = 1; k < NB; k++) qs.push(srt[Math.floor(k / NB * (srt.length - 1))]);
  const off = document.createElement('canvas'); off.width = off.height = W; const o1 = off.getContext('2d');
  const id = o1.createImageData(W, W), px = id.data, R = 4;
  const at = (x, y) => { const bx = Math.min(G - 1.001, Math.max(0, (x - rx0) / rs * (G - 1))), by = Math.min(G - 1.001, Math.max(0, (y - ry0) / rs * (G - 1)));
    const i0 = Math.floor(by), j0 = Math.floor(bx), fy = by - i0, fx = bx - j0;
    return L[i0 * G + j0] * (1 - fx) * (1 - fy) + L[i0 * G + j0 + 1] * fx * (1 - fy) + L[(i0 + 1) * G + j0] * (1 - fx) * fy + L[(i0 + 1) * G + j0 + 1] * fx * fy; };
  for (let yy = 0; yy < W; yy += R) for (let xx = 0; xx < W; xx += R) {
    const v = at(...fromC(xx + R / 2, yy + R / 2)); let k = 0; while (k < qs.length && v > qs[k]) k++;
    const s = Math.pow(k / qs.length, 0.8), col = C['band-lo'].map((c, m) => c + (C['band-hi'][m] - c) * s);
    for (let dy = 0; dy < R; dy++) for (let dx = 0; dx < R; dx++) { const p4 = ((yy + dy) * W + xx + dx) * 4; px[p4] = col[0]; px[p4 + 1] = col[1]; px[p4 + 2] = col[2]; px[p4 + 3] = 255; }
  }
  o1.putImageData(id, 0, 0);
  const cell = rs / (G - 1);
  qs.forEach((lev, li) => {
    o1.strokeStyle = rgba(C.iso, li < 6 ? 0.85 : 0.45); o1.lineWidth = li < 6 ? 2 : 1.2; o1.beginPath();
    for (let a = 0; a < G - 1; a++) for (let b = 0; b < G - 1; b++) {
      const v = [L[a * G + b], L[a * G + b + 1], L[(a + 1) * G + b + 1], L[(a + 1) * G + b]], P = [[b, a], [b + 1, a], [b + 1, a + 1], [b, a + 1]], pts = [];
      for (let e = 0; e < 4; e++) { const u = v[e], w = v[(e + 1) % 4]; if ((u - lev) * (w - lev) < 0) { const tt = (lev - u) / (w - u); pts.push([P[e][0] + tt * (P[(e + 1) % 4][0] - P[e][0]), P[e][1] + tt * (P[(e + 1) % 4][1] - P[e][1])]); } }
      for (let k = 0; k + 1 < pts.length; k += 2) { const A = toC(rx0 + pts[k][0] * cell, ry0 + pts[k][1] * cell), B = toC(rx0 + pts[k + 1][0] * cell, ry0 + pts[k + 1][1] * cell); o1.moveTo(A[0], A[1]); o1.lineTo(B[0], B[1]); }
    }
    o1.stroke();
  });
  contImg = off; contView = [vz, vx, vy];
}
// ── landscape window and surface heights (blue: single terminal, red: ensemble mean) ──
let surf2 = null, MsS = 0, MsC = 0;
function makeSurf(F, G, ell0) {
  const srt = Array.from(F).sort((p, q) => p - q), mn = srt[0], ell = ell0 ? ell0(mn) : srt[Math.floor(0.5 * (srt.length - 1))] - mn;
  const rank = v => { let l = 0, r = srt.length - 1; while (l < r) { const m = (l + r) >> 1; if (srt[m] < v) l = m + 1; else r = m; } return l / (srt.length - 1); };
  const lin = new Float64Array(G * G), rk = new Float64Array(G * G);
  for (let k = 0; k < G * G; k++) { lin[k] = 1 - Math.exp(-(F[k] - mn) / ell); rk[k] = rank(F[k]); }
  return {G, F, lin, rk, Z: new Float64Array(G * G)};
}
function buildSurface() {
  if (sel === NET.sel) win = {x0: NET.win[0], y0: NET.win[1], s: NET.win[2]};
  else { const half = 0.2 * span; win = {x0: X[sel][0] - half, y0: X[sel][1] - half, s: 2 * half}; }
  const G = 61, F = new Float64Array(G * G), G2 = 49, F2 = new Float64Array(G2 * G2);
  for (let a = 0; a < G; a++) for (let b = 0; b < G; b++) F[a * G + b] = Fi([win.x0 + win.s * b / (G - 1), win.y0 + win.s * a / (G - 1)], sel);
  MsS = budget(G2);
  for (let a = 0; a < G2; a++) for (let b = 0; b < G2; b++) F2[a * G2 + b] = eFi([win.x0 + win.s * b / (G2 - 1), win.y0 + win.s * a / (G2 - 1)], sel, MsS);
  // linear height saturates smoothly, 1 - exp(-(F - min)/ℓ), with ℓ = 2 × (barrier - min) for the preset node and the window median otherwise
  surf = makeSurf(F, G, sel === NET.sel && NET.B ? mn => 2 * (NET.barrier - mn) : null);
  surf2 = makeSurf(F2, G2, null);
  setGamma(gamma);
}
function setGamma(gm) { gamma = gm; for (const sf of [surf, surf2]) { if (!sf) continue; const {lin, rk, Z} = sf; for (let k = 0; k < Z.length; k++) Z[k] = (1 - gm) * lin[k] + gm * Math.pow(rk[k], 1.35); } dirty = true; }
function zAtXY(sf, x, y) { const {G, Z} = sf; const b = (x - win.x0) / win.s * (G - 1), a = (y - win.y0) / win.s * (G - 1);
  if (a < 0 || b < 0 || a > G - 1 || b > G - 1) return null; const i0 = Math.min(G - 2, Math.floor(a)), j0 = Math.min(G - 2, Math.floor(b)), fy = a - i0, fx = b - j0;
  return Z[i0 * G + j0] * (1 - fx) * (1 - fy) + Z[i0 * G + j0 + 1] * fx * (1 - fy) + Z[(i0 + 1) * G + j0] * (1 - fx) * fy + Z[(i0 + 1) * G + j0 + 1] * fx * fy; }

const sv = $('surf'), sv2 = $('surf2');
function drawSurf(cv, sf, pre, cls, single) {
  const sx = cv.getContext('2d'), Wv = cv.width, Hv = cv.height, {G, Z} = sf; sx.clearRect(0, 0, Wv, Hv);
  const ca = Math.cos(az), sa = Math.sin(az), ce = Math.cos(el), se = Math.sin(el);
  // fit the rotated unit box (|x|,|y| ≤ 1/√2, 0 ≤ z ≤ 1) into the canvas for every azimuth
  let sc = Wv * 0.62, zs = Hv * 0.58 * cam.h; const k = Math.min(1, 0.9 * Wv / (1.414 * sc), 0.9 * Hv / (1.414 * se * sc + zs * ce)); sc *= k * cam.zoom; zs *= k * cam.zoom;
  const base = Hv / 2 + (zs * ce) / 2;
  const P3 = (u, v, z) => { const x = u - 0.5, y = v - 0.5, xr = x * ca - y * sa, yr = x * sa + y * ca; return [Wv / 2 + xr * sc, base - yr * se * sc - z * zs * ce, yr]; };
  const pts = new Array(G * G); for (let a = 0; a < G; a++) for (let b = 0; b < G; b++) pts[a * G + b] = P3(b / (G - 1), a / (G - 1), Z[a * G + b]);
  const quads = [];
  for (let a = 0; a < G - 1; a++) for (let b = 0; b < G - 1; b++) { const k = a * G + b; quads.push([k, pts[k][2] + pts[k + G + 1][2]]); }
  quads.sort((p, q) => q[1] - p[1]);
  const lx = -0.5, ly = 0.45, lz = 0.74, du = 1 / (G - 1);
  const ramp = z => { const A = C[pre + '-lo'], M = C[pre + '-mid'], B = C[pre + '-hi']; return z < 0.5 ? A.map((c, m) => c + (M[m] - c) * z * 2) : M.map((c, m) => c + (B[m] - c) * (z - 0.5) * 2); };
  for (const [k] of quads) {
    const z00 = Z[k], z10 = Z[k + 1], z01 = Z[k + G], z11 = Z[k + G + 1];
    const gx = (z10 + z11 - z00 - z01) / 2 / du * 0.62 * (zs / sc), gy = (z01 + z11 - z00 - z10) / 2 / du * 0.62 * (zs / sc);
    const nn = Math.hypot(gx, gy, 1), sh = Math.max(0.25, (-gx * lx - gy * ly + lz) / nn);
    const base = ramp((z00 + z11) / 2), col = base.map(c => Math.min(255, c * (0.72 + 0.38 * sh)));
    const q = [pts[k], pts[k + 1], pts[k + G + 1], pts[k + G]];
    sx.fillStyle = sx.strokeStyle = `rgb(${col[0] | 0},${col[1] | 0},${col[2] | 0})`; sx.lineWidth = 0.8;
    sx.beginPath(); sx.moveTo(q[0][0], q[0][1]); for (let m = 1; m < 4; m++) sx.lineTo(q[m][0], q[m][1]); sx.closePath(); sx.fill(); sx.stroke();
  }
  // grid lines for depth cue
  sx.strokeStyle = rgba(pre === 's' ? C.iso : C.ens, 0.28); sx.lineWidth = 1.2;
  for (let a = 0; a < G; a += 6) { sx.beginPath(); for (let b = 0; b < G; b++) { const p = pts[a * G + b]; b ? sx.lineTo(p[0], p[1]) : sx.moveTo(p[0], p[1]); } sx.stroke(); }
  for (let b = 0; b < G; b += 6) { sx.beginPath(); for (let a = 0; a < G; a++) { const p = pts[a * G + b]; a ? sx.lineTo(p[0], p[1]) : sx.moveTo(p[0], p[1]); } sx.stroke(); }
  const proj = p => { const z = zAtXY(sf, p[0], p[1]); return z === null ? null : P3((p[0] - win.x0) / win.s, (p[1] - win.y0) / win.s, z); };
  // curvature patch around the moving point
  const cc = single ? (cls === 'bowl' ? C.up : cls === 'dome' ? C.down : C.sad) : C.ens, r = 0.06 * win.s;
  sx.strokeStyle = rgba(cc, 1); sx.fillStyle = rgba(cc, 0.45); sx.lineWidth = 4; sx.beginPath();
  for (let k = 0; k <= 40; k++) { const th = k / 40 * 2 * Math.PI, p = proj([pos[0] + r * Math.cos(th), pos[1] + r * Math.sin(th)]); if (p) k ? sx.lineTo(p[0], p[1] - 3) : sx.moveTo(p[0], p[1] - 3); }
  sx.closePath(); sx.fill(); sx.stroke();
  sx.font = '700 28px "Noto Sans KR",sans-serif'; sx.textBaseline = 'bottom';
  const label = (p, txt, hgt = 46) => { const q = proj(p); if (!q) return; sx.strokeStyle = rgba(C.mark, 0.9); sx.lineWidth = 2; sx.beginPath(); sx.moveTo(q[0], q[1]); sx.lineTo(q[0], q[1] - hgt); sx.stroke(); sx.fillStyle = rgba(C.mark, 1); sx.beginPath(); sx.arc(q[0], q[1], 5, 0, 7); sx.fill();
    const tw = sx.measureText(txt).width; sx.fillStyle = rgba(C.bg, 0.92); sx.fillRect(q[0] + 2, q[1] - hgt - 38, tw + 16, 38); sx.fillStyle = rgba(C.mark, 1); sx.fillText(txt, q[0] + 10, q[1] - hgt - 4); };
  label(X[sel], single ? 'terminal' : 'reference terminal'); if (single && sel === NET.sel && NET.B) { label(NET.saddle, 'barrier', 96); label(NET.B, NET.F0 > NET.FB ? 'lower basin' : 'second basin', 146); }
  const q = proj(pos); if (q) { sx.fillStyle = rgba(C.mark, 1); sx.beginPath(); sx.arc(q[0], q[1] - 3, 11, 0, 7); sx.fill(); sx.strokeStyle = rgba(C.bg, 1); sx.lineWidth = 3; sx.stroke(); }
}

// ── layers 2–4 ──
function drawSprings() {
  c2.clearRect(0, 0, W, W); const A = toC(...pos); c2.lineWidth = 2;
  for (let j = 0; j < n; j++) { if (j === sel) continue;
    const de = delta(sel, j), st = (Math.hypot(pos[0] - X[j][0], pos[1] - X[j][1]) - de) / de;
    c2.strokeStyle = rgba(st > 0 ? C.ten : C.com, Math.min(0.85, 0.08 + 1.4 * Math.abs(st)));
    const B = toC(...X[j]); c2.beginPath(); c2.moveTo(A[0], A[1]); c2.lineTo(B[0], B[1]); c2.stroke(); }
}
const nodeR = j => (n > 200 ? 4.5 : 7) + (n > 200 ? 2.6 : 3.2) * Math.sqrt(deg[j]);
function drawNetwork() {
  c3.clearRect(0, 0, W, W); const P = j => (j === sel ? pos : X[j]);
  c3.strokeStyle = rgba(C.edge, 0.78); c3.lineWidth = n > 200 ? 2.6 : 3.4; c3.lineCap = 'round'; c3.beginPath();
  for (const [a, b] of E) { const A = toC(...P(a)), B = toC(...P(b)); c3.moveTo(A[0], A[1]); c3.lineTo(B[0], B[1]); } c3.stroke();
  for (let j = 0; j < n; j++) { if (j === sel) continue; const [a, b] = toC(...X[j]); c3.beginPath(); c3.arc(a, b, nodeR(j), 0, 7); c3.fillStyle = rgba(C.node, 1); c3.fill(); c3.lineWidth = 2; c3.strokeStyle = rgba(C['node-ring'], 1); c3.stroke(); }
  const [a, b] = toC(...pos); c3.beginPath(); c3.arc(a, b, nodeR(sel) + 2, 0, 7); c3.fillStyle = rgba(C.mark, 1); c3.fill();
}
function arrow(c, x1, y1, x2, y2, col) {
  c.strokeStyle = rgba(C.bg, 0.9); c.lineWidth = 11; c.lineCap = 'round'; c.beginPath(); c.moveTo(x1, y1); c.lineTo(x2, y2); c.stroke();
  c.strokeStyle = c.fillStyle = rgba(col, 1); c.lineWidth = 6; c.beginPath(); c.moveTo(x1, y1); c.lineTo(x2, y2); c.stroke();
  const an = Math.atan2(y2 - y1, x2 - x1), s = 20; c.beginPath(); c.moveTo(x2 + 6 * Math.cos(an), y2 + 6 * Math.sin(an));
  c.lineTo(x2 - s * Math.cos(an - 0.5), y2 - s * Math.sin(an - 0.5)); c.lineTo(x2 - s * Math.cos(an + 0.5), y2 - s * Math.sin(an + 0.5)); c.closePath(); c.fill();
}
let ensImg = null;
function buildEnsLayer() {
  planRevision++;
  // sample the visible region so that zooming in adds detail
  const [rx0, ry0] = fromC(0, W), rs = W / (S * vz);
  // static part: ensemble-mean contour lines over the map and the M aligned positions of node i
  MsC = budget(61); const G = 61, L = new Float64Array(G * G); let mn = Infinity;
  for (let a = 0; a < G; a++) for (let b = 0; b < G; b++) { const v = eFi([rx0 + rs * b / (G - 1), ry0 + rs * a / (G - 1)], sel, MsC); L[a * G + b] = v; mn = Math.min(mn, v); }
  let mx = -Infinity; for (let k = 0; k < G * G; k++) { L[k] = Math.log(L[k] - mn + 1); mx = Math.max(mx, L[k]); }
  const srt = Array.from(L).sort((p, q) => p - q), qs = []; for (let k = 1; k < 16; k++) qs.push(srt[Math.floor(k / 16 * (srt.length - 1))]);
  const off = document.createElement('canvas'); off.width = off.height = W; const o = off.getContext('2d'), cell = rs / (G - 1);
  qs.forEach((lev, li) => { o.strokeStyle = rgba(C.ens, li < 5 ? 0.9 : 0.5); o.lineWidth = li < 5 ? 2.4 : 1.4; o.setLineDash(li < 5 ? [] : [8, 6]); o.beginPath();
    for (let a = 0; a < G - 1; a++) for (let b = 0; b < G - 1; b++) {
      const v = [L[a * G + b], L[a * G + b + 1], L[(a + 1) * G + b + 1], L[(a + 1) * G + b]], P = [[b, a], [b + 1, a], [b + 1, a + 1], [b, a + 1]], pts = [];
      for (let e = 0; e < 4; e++) { const u = v[e], w = v[(e + 1) % 4]; if ((u - lev) * (w - lev) < 0) { const tt = (lev - u) / (w - u); pts.push([P[e][0] + tt * (P[(e + 1) % 4][0] - P[e][0]), P[e][1] + tt * (P[(e + 1) % 4][1] - P[e][1])]); } }
      for (let k = 0; k + 1 < pts.length; k += 2) { const A = toC(rx0 + pts[k][0] * cell, ry0 + pts[k][1] * cell), B = toC(rx0 + pts[k + 1][0] * cell, ry0 + pts[k + 1][1] * cell); o.moveTo(A[0], A[1]); o.lineTo(B[0], B[1]); } }
    o.stroke(); });
  o.setLineDash([]); o.fillStyle = rgba(C.ens, Math.min(0.6, 30 / M + 0.12));
  for (let m = 0; m < M; m++) { const o2 = m * n * 2 + 2 * sel, [a, b] = toC(EX[o2], EX[o2 + 1]); o.beginPath(); o.arc(a, b, 5, 0, 7); o.fill(); }
  ensImg = off; ensView = [vz, vx, vy];
}
function drawEns(he) {
  c5.clearRect(0, 0, W, W); if (ensImg) blit(c5, ensImg, ensView);
  const [a, b] = toC(...pos), lm = Math.max(Math.abs(he.l1), Math.abs(he.l2)) || 1;
  for (const [l, v] of [[he.l1, he.v1], [he.l2, he.v2]]) { const L = 34 + 76 * Math.abs(l) / lm; for (const sg of [1, -1]) arrow(c5, a, b, a + sg * v[0] * L, b - sg * v[1] * L, l > 0 ? C.ens : C['ens-lt']); }
}
let own = [];
function drawLamCloud() {
  const cv = $('lamcloud'), x = cv.getContext('2d'), Wc = cv.width, Hc = cv.height, pad = 90; x.clearRect(0, 0, Wc, Hc);
  const h0 = hess(X[sel], sel), all = own.concat([[h0.l1, h0.l2]]);
  const L1 = all.map(e => e[0]), L2 = all.map(e => e[1]);
  let x0 = Math.min(0, ...L1), x1 = Math.max(0, ...L1), y0 = Math.min(0, ...L2), y1 = Math.max(0, ...L2);
  const px = 0.08 * (x1 - x0 || 1), py = 0.08 * (y1 - y0 || 1); x0 -= px; x1 += px; y0 -= py; y1 += py;
  const tx = v => pad + (v - x0) / (x1 - x0) * (Wc - 1.4 * pad), ty = v => Hc - pad - (v - y0) / (y1 - y0) * (Hc - 1.6 * pad);
  x.save(); x.beginPath(); x.rect(tx(x0), ty(y1), tx(x1) - tx(x0), ty(y0) - ty(y1)); x.clip();
  x.fillStyle = rgba(C.up, 0.08); x.fillRect(tx(0), ty(y1), tx(x1) - tx(0), ty(y0) - ty(y1));
  x.strokeStyle = rgba(C.mark, 0.55); x.lineWidth = 2; x.beginPath(); x.moveTo(tx(x0), ty(0)); x.lineTo(tx(x1), ty(0)); x.moveTo(tx(0), ty(y0)); x.lineTo(tx(0), ty(y1)); x.stroke();
  x.setLineDash([6, 6]); x.beginPath(); const d0 = Math.max(x0, y0), d1 = Math.min(x1, y1); x.moveTo(tx(d0), ty(d0)); x.lineTo(tx(d1), ty(d1)); x.stroke(); x.setLineDash([]);
  x.fillStyle = rgba(C.ens, Math.min(0.7, 40 / M + 0.15)); for (const [l1, l2] of own) { x.beginPath(); x.arc(tx(l1), ty(l2), 6, 0, 7); x.fill(); }
  x.fillStyle = rgba(C.down, 1); x.strokeStyle = rgba(C.bg, 1); x.lineWidth = 3; x.beginPath(); x.arc(tx(h0.l1), ty(h0.l2), 12, 0, 7); x.fill(); x.stroke();
  x.restore();
  x.strokeStyle = rgba(C.mark, 0.4); x.lineWidth = 1.5; x.strokeRect(tx(x0), ty(y1), tx(x1) - tx(x0), ty(y0) - ty(y1));
  x.fillStyle = rgba(C.mark, 1); x.font = '500 22px "Noto Sans KR",sans-serif'; x.textAlign = 'center';
  for (const v of [x0 + px, 0, x1 - px]) x.fillText(v.toFixed(Math.abs(v) < 10 ? 1 : 0), tx(v), ty(y0) + 30);
  x.textAlign = 'right'; for (const v of [y0 + py, 0, y1 - py]) x.fillText(v.toFixed(Math.abs(v) < 10 ? 1 : 0), tx(x0) - 10, ty(v) + 8);
  x.font = '600 26px "Noto Sans KR",sans-serif'; x.textAlign = 'center'; x.fillText('λ₁', tx((x0 + x1) / 2), Hc - 14);
  x.save(); x.translate(26, ty((y0 + y1) / 2)); x.rotate(-Math.PI / 2); x.fillText('λ₂', 0, 0); x.restore(); x.textAlign = 'left';
  x.fillText('BOWL (λ₁ > 0)', Math.max(tx(0), tx(x0)) + 12, ty(y1) + 34); if (tx(0) - tx(x0) > 170) x.fillText('λ₁ < 0', tx(x0) + 12, ty(y1) + 34);
  const bowl = own.filter(e => e[0] > 0).length, med = v => v.slice().sort((p, q) => p - q)[v.length >> 1];
  setMath('lamcloud-note', `붉은 점: ensemble terminal ${M} 개 각각에서 node ${sel} 의 block $\\mathbf H_{ii}^{(m)}$ 을 자기 위치 $\\mathbf x_i^{(m)}$ 에서 잰 eigenvalue $(\\lambda_1,\\lambda_2)$. 푸른 점: 기준 MDS terminal. BOWL 비율 ${(100 * bowl / M).toFixed(1)}%, median $\\lambda_1=${med(own.map(e => e[0])).toFixed(2)}$, median $\\lambda_2=${med(own.map(e => e[1])).toFixed(2)}$.`);
}
function drawVectors(h) {
  c4.clearRect(0, 0, W, W);
  const [w0, w1] = toC(win.x0, win.y0 + win.s), [w2, w3] = toC(win.x0 + win.s, win.y0);
  c4.setLineDash([10, 8]); c4.strokeStyle = rgba(C.mark, 0.7); c4.lineWidth = 2; c4.strokeRect(w0, w1, w2 - w0, w3 - w1); c4.setLineDash([]);
  const ring = (p, txt, dash, dy = -14) => { const [a, b] = toC(...p); c4.strokeStyle = rgba(C.mark, 1); c4.lineWidth = 2.5; c4.setLineDash(dash ? [6, 6] : []); c4.beginPath(); c4.arc(a, b, 18, 0, 7); c4.stroke(); c4.setLineDash([]);
    c4.font = '700 24px "Noto Sans KR",sans-serif'; c4.lineWidth = 6; c4.strokeStyle = rgba(C.bg, 0.9); c4.strokeText(txt, a + 22, b + dy); c4.fillStyle = rgba(C.mark, 1); c4.fillText(txt, a + 22, b + dy); };
  ring(X[sel], 'terminal', false); if (sel === NET.sel && NET.B) { ring(NET.saddle, 'barrier', true, 44); ring(NET.B, NET.F0 > NET.FB ? 'lower basin' : 'second basin', true, -58); }
  const [a, b] = toC(...pos), lm = Math.max(Math.abs(h.l1), Math.abs(h.l2)) || 1;
  for (const [l, v] of [[h.l1, h.v1], [h.l2, h.v2]]) { const L = 34 + 76 * Math.abs(l) / lm, c = l > 0 ? C.up : C.down; for (const sg of [1, -1]) arrow(c4, a, b, a + sg * v[0] * L, b - sg * v[1] * L, c); }
  c4.fillStyle = rgba(C.mark, 1); c4.beginPath(); c4.arc(a, b, 11, 0, 7); c4.fill(); c4.strokeStyle = rgba(C.bg, 1); c4.lineWidth = 3; c4.stroke();
}

// ── live readout (KaTeX) ──
const f2 = v => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));
const bm = M => `\\begin{bmatrix}${f2(M[0])}&${f2(M[1])}\\\\${f2(M[1])}&${f2(M[2])}\\end{bmatrix}`;
let lastTex = '', lastTex2 = '';
const matrixTex = (h, ensemble = false) => {
  const H = ensemble ? '\\langle\\mathbf H_{ii}\\rangle_M' : '\\mathbf H_{ii}';
  const K = ensemble ? '\\langle K\\rangle' : 'K\\,\\succeq\\,0';
  const S = ensemble ? '\\langle S\\rangle' : 'S';
  const parts = `\\underbrace{${bm(h.K)}}_{${K}}+\\underbrace{${bm(h.S)}}_{${S}}`;
  return narrowReadout.matches ? `\\begin{gathered}${H}=${bm(h.H)}\\\\{}=${parts}\\end{gathered}`
    : `${H}=\\underbrace{${bm(h.H)}}_{\\text{Hessian}}=${parts}`;
};
function readout(h, cls, he) {
  const tex = matrixTex(h);
  if (tex !== lastTex) { katex.render(tex, $('mat'), {displayMode: true, throwOnError: false}); lastTex = tex;
    katex.render(`\\lambda_1=${f2(h.l1)},\\quad \\lambda_2=${f2(h.l2)}`, $('lam'), {throwOnError: false}); }
  const tex2 = matrixTex(he, true);
  if (tex2 !== lastTex2) { katex.render(tex2, $('mat2'), {displayMode: true, throwOnError: false}); lastTex2 = tex2;
    katex.render(`\\lambda_1=${f2(he.l1)},\\quad \\lambda_2=${f2(he.l2)}`, $('lam2'), {throwOnError: false}); }
  const ce = he.l1 > 0 ? 'BOWL' : he.l2 < 0 ? 'DOME' : 'SADDLE'; $('verdict2').textContent = ce; $('verdict2').style.color = 'var(--viz-ens)';
  const V = $('verdict'), col = cls === 'bowl' ? 'up' : cls === 'dome' ? 'down' : 'sad';
  V.textContent = cls.toUpperCase();
  V.title = cls === 'bowl' ? '어느 방향으로든 위로 휜다' : cls === 'dome' ? '두 방향 모두 아래로 휜다' : '한 방향은 위로, 다른 방향은 아래로 휜다';
  V.style.color = `var(--type-${col})`;
  const dF = h.f - Fref; $('dF').textContent = (dF >= 0 ? '+' : '−') + Math.abs(dF).toFixed(1);
  $('nT').textContent = h.nt; $('nC').textContent = h.nc; $('gN').textContent = Math.hypot(...h.g).toFixed(2);
}
function rebuild() {
  Fref = Fi(X[sel], sel); buildContour(); buildSurface(); buildEnsLayer(); own = ownEigs(sel); drawLamCloud();
  $('msinfo').textContent = `red grids: first ${MsS} (landscape) / ${MsC} (contour) of M = ${M} terminals; red readout uses all ${M}`;
  $('selinfo').textContent = `degree ${deg[sel]}` + (sel === NET.sel ? ' · preset path' : ' · drag to move');
  $('node-choice').max = n - 1; $('node-choice').value = sel;
  $('ens-count').textContent = `${M} terminals`;
  $('viewer-status').textContent = `${NET.name}, node ${sel}, degree ${deg[sel]}`;
  dirty = true;
}
// Physics depends on the selected graph, node and position, not on camera rotation.
let lastPos = '', lastPlanKey = '', lastH = null, lastHE = null, lastFrame = 0;
function frame(now) {
  requestAnimationFrame(frame);
  if (document.hidden || !viewerVisible) { lastFrame = 0; return; }
  if (lastFrame && now - lastFrame < 1000 / 30) return;
  const steps = lastFrame ? Math.max(1, Math.round(Math.min(now - lastFrame, 80) / (1000 / 60))) : 1;
  lastFrame = now;
  if (playing && !dragging && sel === NET.sel) { t = (t + steps) % NET.anim.length; pos = NET.anim[t].slice(); }
  if (elT !== null) { el += (elT - el) * (1 - Math.pow(0.85, steps)); if (Math.abs(elT - el) < 1e-3) { el = elT; elT = null; } dirty = true; }
  if (cam.spin && !rot) { az += 0.003 * steps; dirty = true; }
  const key = NET.name + '|' + sel + '|' + pos.join();
  const moved = key !== lastPos || !lastH;
  if (moved) {
    lastH = hess(pos, sel); lastHE = eHess(pos, sel);
    readout(lastH, lastH.l1 > 0 ? 'bowl' : lastH.l2 < 0 ? 'dome' : 'saddle', lastHE);
    lastPos = key;
  }
  const planKey = key + '|' + [vz, vx, vy, planRevision].join();
  if (planKey !== lastPlanKey) {
    c1.clearRect(0, 0, W, W); if (contImg) blit(c1, contImg, contView);
    drawSprings(); drawNetwork(); drawVectors(lastH); drawEns(lastHE); lastPlanKey = planKey;
  }
  if (moved || dirty) {
    const cls = lastH.l1 > 0 ? 'bowl' : lastH.l2 < 0 ? 'dome' : 'saddle';
    const ce = lastHE.l1 > 0 ? 'bowl' : lastHE.l2 < 0 ? 'dome' : 'saddle';
    drawSurf(sv, surf, 's', cls, true); drawSurf(sv2, surf2, 'r', ce, false); syncCam(); dirty = false;
  }
}
const visibilityObserver = new IntersectionObserver(entries => { viewerVisible = entries[0].isIntersecting; if (viewerVisible) dirty = true; }, {rootMargin: '120px'});
visibilityObserver.observe($('viz'));
narrowReadout.addEventListener('change', () => { lastTex = ''; lastTex2 = ''; lastPos = ''; dirty = true; });
reducedMotion.addEventListener('change', e => { if (e.matches) { setPlay(false); cam.spin = false; $('cam-spin').checked = false; } });
// ── interaction ──
const lay = $('layers');
const evp = e => { const c = e.target.closest('canvas'); const r = c.getBoundingClientRect(); return [(e.clientX - r.left) / r.width * W, (e.clientY - r.top) / r.height * W]; };
const zoomAt = (f, a, b) => { const nz = Math.max(1, Math.min(12, vz * f)); f = nz / vz; vx = a - W / 2 - f * (a - W / 2 - vx); vy = b - W / 2 - f * (b - W / 2 - vy); vz = nz; if (vz === 1) { vx = vy = 0; } viewChanged(); };
lay.addEventListener('wheel', e => { if (!e.target.closest('canvas')) return; e.preventDefault(); zoomAt(Math.exp(-e.deltaY * 0.0015), ...evp(e)); }, {passive: false});
let pan = null; const ptr = new Map(); let pz0 = null;
lay.addEventListener('pointerdown', e => {
  if (!e.target.closest('canvas')) return;
  ptr.set(e.pointerId, evp(e)); e.target.setPointerCapture(e.pointerId);
  if (ptr.size === 2) { const [p, q] = [...ptr.values()]; pz0 = [Math.hypot(p[0] - q[0], p[1] - q[1]), vz]; dragging = false; pan = null; return; }
  if (e.shiftKey) { pan = [...evp(e), vx, vy]; e.target.setPointerCapture(e.pointerId); return; }
  const [a, b] = evp(e), hit = (e.pointerType === 'touch' ? 22 : 12) * W / e.target.getBoundingClientRect().width; let best = -1, bd = hit;
  for (let j = 0; j < n; j++) { if (j === sel) continue; const [u, v] = toC(...X[j]); const dd = Math.hypot(u - a, v - b); if (dd < bd) { bd = dd; best = j; } }
  const [u0, v0] = toC(...pos);
  if (best >= 0 && Math.hypot(u0 - a, v0 - b) > hit) { sel = best; pos = X[sel].slice(); setPlay(false); rebuild(); return; }
  dragging = true; setPlay(false); pos = fromC(a, b); e.target.setPointerCapture(e.pointerId);
});
lay.addEventListener('pointermove', e => {
  if (!e.target.closest('canvas')) return;
  if (ptr.has(e.pointerId)) ptr.set(e.pointerId, evp(e));
  if (pz0 && ptr.size === 2) { const [p, q] = [...ptr.values()]; zoomAt(pz0[1] * Math.hypot(p[0] - q[0], p[1] - q[1]) / pz0[0] / vz, (p[0] + q[0]) / 2, (p[1] + q[1]) / 2); return; }
  if (pan) { const [a, b] = evp(e); vx = pan[2] + a - pan[0]; vy = pan[3] + b - pan[1]; viewChanged(); return; }
  if (dragging) pos = fromC(...evp(e)); });
const lift = e => { ptr.delete(e.pointerId); if (ptr.size < 2) pz0 = null; dragging = false; pan = null; };
lay.addEventListener('pointerup', lift); lay.addEventListener('pointercancel', lift);
function chooseNode() {
  const input = $('node-choice'); if (!input.reportValidity()) return;
  const next = Number(input.value); if (!Number.isInteger(next) || next < 0 || next >= n) return;
  sel = next; pos = X[sel].slice(); setPlay(false); rebuild();
}
$('node-apply').onclick = chooseNode;
$('node-choice').onchange = chooseNode;
$('node-choice').onkeydown = e => { if (e.key === 'Enter') { e.preventDefault(); chooseNode(); } };
lay.addEventListener('keydown', e => {
  const moves = {ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1]};
  if (e.key === 'Home') { e.preventDefault(); pos = X[sel].slice(); setPlay(false); dirty = true; }
  else if (moves[e.key]) { e.preventDefault(); const step = span / (e.shiftKey ? 25 : 100), v = moves[e.key]; pos = [pos[0] + v[0] * step, pos[1] + v[1] * step]; setPlay(false); dirty = true; }
});
$('pz-in').onclick = () => zoomAt(1.5, W / 2, W / 2); $('pz-out').onclick = () => zoomAt(1 / 1.5, W / 2, W / 2);
$('pz-fit').onclick = () => { vz = 1; vx = vy = 0; viewChanged(); };
$('pz-node').onclick = () => { const [a, b] = toB(...pos); vz = 4; vx = -vz * (a - W / 2); vy = -vz * (b - W / 2); viewChanged(); };
let rot = null;
const touches = new Map(); let pinch = null;
for (const cv of [sv, sv2]) {
cv.addEventListener('pointerdown', e => { cv.setPointerCapture(e.pointerId); touches.set(e.pointerId, [e.clientX, e.clientY]); if (touches.size === 2) { const [p, q] = [...touches.values()]; pinch = [Math.hypot(p[0] - q[0], p[1] - q[1]), cam.zoom]; rot = null; return; } rot = [e.clientX, e.clientY, az, el]; cv.setPointerCapture(e.pointerId); cv.style.cursor = 'grabbing'; });
cv.addEventListener('pointermove', e => { if (touches.has(e.pointerId)) touches.set(e.pointerId, [e.clientX, e.clientY]); if (pinch && touches.size === 2) { const [p, q] = [...touches.values()]; cam.zoom = Math.max(0.6, Math.min(3, pinch[1] * Math.hypot(p[0] - q[0], p[1] - q[1]) / pinch[0])); $('cam-zoom').value = cam.zoom * 100; dirty = true; return; } if (!rot) return; az = rot[2] + (e.clientX - rot[0]) * 0.01; el = Math.max(0.03, Math.min(1.45, rot[3] + (e.clientY - rot[1]) * 0.006)); elT = null; cam.spin = false; $('cam-spin').checked = false; viewBtn(null); dirty = true; });
const up = e => { touches.delete(e.pointerId); if (touches.size < 2) pinch = null; rot = null; cv.style.cursor = 'grab'; };
cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
cv.addEventListener('wheel', e => { e.preventDefault(); cam.zoom = Math.max(0.6, Math.min(3, cam.zoom * Math.exp(-e.deltaY * 0.0015))); $('cam-zoom').value = cam.zoom * 100; dirty = true; }, {passive: false});
}
// camera controls; sliders follow drags and presets
const deg2 = r => Math.round(r * 180 / Math.PI);
function syncCam() { const A = $('cam-az'), L = $('cam-el'); if (document.activeElement !== A) A.value = ((deg2(az) + 540) % 360) - 180; if (document.activeElement !== L) L.value = deg2(el); }
$('cam-az').oninput = e => { az = e.target.value * Math.PI / 180; cam.spin = false; $('cam-spin').checked = false; dirty = true; };
$('cam-el').oninput = e => { el = e.target.value * Math.PI / 180; elT = null; viewBtn(null); dirty = true; };
$('cam-zoom').oninput = e => { cam.zoom = e.target.value / 100; dirty = true; };
$('cam-h').oninput = e => { cam.h = e.target.value / 100; dirty = true; };
$('cam-spin').onchange = e => { cam.spin = e.target.checked; };
$('cam-reset').onclick = () => { az = -0.75; elT = 0.55; cam.zoom = 1; cam.h = 1; $('cam-zoom').value = 100; $('cam-h').value = 100; viewBtn('tilt'); dirty = true; };
// view presets: el = 0.03 is almost edge-on, so the profile of F_i along the viewing axis shows
let elT = null; const VIEW = {top: 1.45, tilt: 0.55, side: 0.03};
const viewBtn = v => document.querySelectorAll('[data-view]').forEach(b => b.setAttribute('aria-pressed', b.dataset.view === v));
document.querySelectorAll('[data-view]').forEach(b => b.onclick = () => { elT = VIEW[b.dataset.view]; viewBtn(b.dataset.view); });
function setPlay(v) { playing = v && sel === NET.sel; $('play').textContent = playing ? '일시정지' : 'path 재생'; }
$('play').onclick = () => { if (sel !== NET.sel) { sel = NET.sel; pos = X[sel].slice(); rebuild(); } setPlay(!playing); };
$('reset').onclick = () => { pos = X[sel].slice(); setPlay(false); };
$('gamma').oninput = e => setGamma(e.target.value / 100);
document.querySelectorAll('[data-layer]').forEach(cb => cb.onchange = () => { $(cb.dataset.layer).hidden = !cb.checked; });
const seg = split => { document.body.classList.toggle('split', split); $('stack').setAttribute('aria-pressed', !split); $('split').setAttribute('aria-pressed', split); };
$('stack').onclick = () => seg(false); $('split').onclick = () => seg(true);
$('mode').onclick = () => { const hc = document.documentElement.dataset.krdsMode !== 'high-contrast'; document.documentElement.dataset.krdsMode = hc ? 'high-contrast' : 'light'; $('mode').setAttribute('aria-pressed', hc); $('mode').textContent = hc ? '기본 화면' : '선명한 화면'; readColors(); buildContour(); buildEnsLayer(); drawLamCloud(); dirty = true; };

// ── per-network text; every math string goes through setMath (build/check_katex.cjs compiles them) ──
const MATH = {delimiters: [{left: '$$', right: '$$', display: true}, {left: '$', right: '$', display: false}], ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'option', 'canvas'], throwOnError: false};
const setMath = (id, html) => { $(id).innerHTML = html; renderMathInElement($(id), MATH); };
const sci = v => { const [m, e] = v.toExponential(1).split('e'); return `${m}\\times10^{${+e}}`; };
function describe() {
  const h0 = hess(X[NET.sel], NET.sel), eK = eig(h0.K), eS = eig(h0.S), sumw = h0.S[0] + h0.S[2], ck = NET.check, npair = n * (n - 1) / 2;
  const two = NET.B !== null, bar = two ? (NET.barrier - NET.F0).toFixed(2) : '', gain = two ? NET.F0 - NET.FB : 0, gs = Math.abs(gain).toFixed(2);
  const basinName = gain > 0 ? 'lower basin' : 'second basin', domeStep = NET.dome ? ' → dome region' : '';
  const sliceKey = two ? `<div><dt>node ${NET.sel} slice: barrier / gain</dt><dd>$+${bar}$ / $${gain > 0 ? '+' : '-'}${gs}$</dd></div>` : `<div><dt>node ${NET.sel} slice</dt><dd>single basin</dd></div>`;
  setMath('keys', `<div><dt>network</dt><dd>${NET.name}</dd></div><div><dt>size</dt><dd>$N=${n},\\ E=${E.length}$</dd></div><div><dt>constraints vs DOF</dt><dd>$${npair}$ vs $${2 * n - 3}$</dd></div><div><dt>NEI ($M=${M}$)</dt><dd>$${NET.ens.nei.toFixed(4)}$</dd></div><div><dt>compressed pairs</dt><dd>${(100 * ck.frac_comp).toFixed(1)}%</dd></div><div><dt>$\\lambda_{\\min}(\\mathbf H_\\perp)$</dt><dd>$${ck.lam_min.toFixed(2)}$</dd></div>${sliceKey}`);
  const tour = two ? `자동 경로: terminal → barrier → ${basinName}${domeStep} → terminal.` : `모든 node 의 slice 가 single basin 이라 자동 경로는 curvature tour: terminal${domeStep} → terminal.`;
  setMath('vizlead', `${NET.name} ($N=${n}$) 의 MDS terminal 에서 node ${NET.sel} (degree ${deg[NET.sel]}) 만 움직인다. ${tour}`);
  const slice = two ? `node ${NET.sel} 하나만 옮기면 barrier ($+${bar}$) 너머에 $F$ 가 terminal 보다 $${gs}$ 만큼 ${gain > 0 ? '낮은 lower basin 이 있다. terminal 은 full space 에서 local minimum 이지만 (아래 표), 한 node 의 slice 만으로도 global minimum 이 아님이 드러난다.' : '높은 second basin 이 있다. 이 slice 안에서는 terminal 쪽 basin 이 더 깊다.'}` : `이 network 의 MDS terminal 에서는 어느 node 의 slice 도 basin 이 하나뿐이다. 그래도 slice 안에는 $\lambda_2\lt0$ 인 dome region 이 있다.`;
  setMath('facts', `terminal 에서 node ${NET.sel} 의 block 은 $\\lambda_1=${h0.l1.toFixed(2)}$, $\\lambda_2=${h0.l2.toFixed(2)}$ 이다. 이 curvature 는 stiffness (eigenvalue $${eK[0].toFixed(1)},\\ ${eK[1].toFixed(1)}$, 합 $2(N-1)=${2 * (n - 1)}$) 와 prestress (eigenvalue $${eS[0].toFixed(1)},\\ ${eS[1].toFixed(1)}$) 의 합이다. 이 node 의 spring ${n - 1} 개 중 ${h0.nc} 개가 compression 이라 $\\sum_j\\omega_{ij}=${sumw.toFixed(1)}$ 이다. trace 로 보면 prestress 가 stiffness 의 약 ${Math.round(-sumw / (2 * (n - 1)) * 100)}% 를 상쇄한다. ${slice}`);
  setMath('checks', `<tr><td>stress</td><td>$F=${ck.F.toFixed(2)}$</td></tr><tr><td>full $\\mathbf H$ 의 zero mode</td><td>${ck.n_zero} 개 = translation 2 + rotation 1</td></tr><tr><td>$\\lambda_{\\min}(\\mathbf H_\\perp)$</td><td>$${ck.lam_min.toFixed(3)}\\gt0$: terminal 은 $2N$ 차원 full space 에서 local minimum</td></tr><tr><td>$\\max_i\\lVert\\nabla_iF\\rVert$</td><td>$${sci(ck.grad)}$</td></tr><tr><td>analytic $\\mathbf H_{ii}$ vs finite difference</td><td>relative difference $${sci(ck.fd)}$</td></tr><tr><td>terminal</td><td>classical MDS → SMACOF → L-BFGS</td></tr><tr><td>NEI ensemble</td><td>$M=${M}$ (census protocol), $\\mathrm{NEI}=${NET.ens.nei.toFixed(4)}$. terminal stress: min $${NET.ens.Fmin.toFixed(1)}$, median $${NET.ens.Fmed.toFixed(1)}$, max $${NET.ens.Fmax.toFixed(1)}$, $\\max\\lVert\\nabla F\\rVert=${sci(NET.ens.gmax)}$. 기준 MDS terminal 보다 stress 가 낮은 ensemble terminal: ${(100 * NET.ens.pct).toFixed(1)}%</td></tr>`);
}
function loadNet(key) {
  setNet(key); vz = 1; vx = vy = 0; sel = NET.sel; pos = X[sel].slice(); t = 0; lastTex = ''; lastTex2 = ''; lastPos = ''; lastPlanKey = '';
  document.querySelectorAll('#nets button').forEach(b => b.setAttribute('aria-pressed', b.dataset.net === key));
  describe(); rebuild(); setPlay(!matchMedia('(prefers-reduced-motion: reduce)').matches);
}
$('nets').innerHTML = Object.keys(NETS).map(k => `<button type="button" data-net="${k}" aria-pressed="false">${NETS[k].name}</button>`).join('');
$('nets').onclick = e => { const b = e.target.closest('button'); if (b) loadNet(b.dataset.net); };
$('cam-spin').checked = cam.spin;
readColors(); loadNet('__DEFAULT__');
renderMathInElement(document.body, MATH);
requestAnimationFrame(frame);
})();
</script>
</body>
</html>
"""

subs = {'__DEFAULT__': 'netscience'}
# a raw '<' inside math would open an HTML tag: escape it in the markup (not in the script)
head, sep, tail = HTML.partition('<script src=')
head = re.sub(r'\$\$[\s\S]+?\$\$|\$[^$]+?\$', lambda m: m.group(0).replace('<', '&lt;'), head)
out = head + sep + tail
for k, v in subs.items():
    out = out.replace(k, v)
out = out.replace('__NETS__', json.dumps(NETS, separators=(',', ':'), ensure_ascii=False))
open(os.path.join(HERE, '..', 'index.html'), 'w', encoding='utf-8').write(out)
print('index.html', len(out))
