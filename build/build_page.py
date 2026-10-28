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
    NETS[k] = m
    print(f"{k:10s} N={n:3d} node={sel:3d} λmin(H⊥)={m['check']['lam_min']:.3f} fd={fd:.1e} " +
          (f"barrier=+{m['barrier'] - m['F0']:.2f} gain={m['F0'] - m['FB']:+.2f}" if m['B'] else 'single basin'))

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
  --viz-ten:var(--krds-color-light-primary-50);--viz-com:var(--krds-color-light-danger-50);
  --viz-up:var(--krds-color-light-warning-40);--viz-down:var(--krds-color-light-primary-70);--viz-sad:var(--krds-color-light-success-50);
  --viz-mark:var(--krds-color-light-gray-90);
  --viz-s-lo:var(--krds-color-light-primary-40);--viz-s-mid:var(--krds-color-light-primary-10);--viz-s-hi:var(--krds-color-light-point-5);
}
[data-krds-mode="high-contrast"]{
  --viz-bg:#000;
  --viz-band-lo:var(--krds-color-high-contrast-primary-50);--viz-band-hi:var(--krds-color-high-contrast-gray-95);
  --viz-iso:var(--krds-color-high-contrast-primary-30);
  --viz-edge:var(--krds-color-high-contrast-gray-0);--viz-node:var(--krds-color-high-contrast-gray-10);--viz-node-ring:#000;
  --viz-ten:var(--krds-color-high-contrast-primary-30);--viz-com:var(--krds-color-high-contrast-danger-40);
  --viz-up:var(--krds-color-high-contrast-warning-30);--viz-down:var(--krds-color-high-contrast-primary-20);--viz-sad:var(--krds-color-high-contrast-success-40);
  --viz-mark:var(--krds-color-high-contrast-gray-0);
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
.site{font:800 2.2rem/1.2 'SUITE','Noto Sans KR',sans-serif;color:var(--gk-text);text-decoration:none}
.site small{display:block;font:500 1.3rem/1.3 'Noto Sans KR',sans-serif;color:var(--gk-text-subtle)}
.mode{min-height:4rem;padding:0 1.6rem;border:1px solid var(--gk-border);background:var(--gk-surface);color:var(--gk-text);border-radius:var(--krds-radius-medium);font:500 1.5rem 'Noto Sans KR',sans-serif;cursor:pointer}
.crumb{background:var(--gk-surface-secondary)}
.crumb ol{list-style:none;display:flex;flex-wrap:wrap;gap:.8rem;padding:1.2rem 2.4rem;margin:0 auto;max-width:124.8rem;font-size:1.5rem;color:var(--gk-text-subtle)}
.crumb li+li::before{content:'›';margin-right:.8rem}
/* title */
.title{padding:4rem 0 2.4rem;border-bottom:1px solid var(--gk-divider)}
.eyebrow{font:700 1.5rem 'Noto Sans KR',sans-serif;color:var(--gk-text-primary);margin-bottom:.8rem}
h1{font-size:var(--krds-pc-typo-heading-xlarge);font-weight:800;letter-spacing:-.02em}
.lead{font-size:var(--krds-typo-body-large);color:var(--gk-text-subtle);margin-top:1.2rem;max-width:80rem}
.lead .katex{font-size:1.05em}
/* page body: in-page nav + content */
.page{display:grid;grid-template-columns:20rem minmax(0,1fr);gap:4rem;padding:3.2rem 0 8rem}
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
/* figure: layers (left) + landscape (right) */
.viz{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:2rem;align-items:start}
.panel-t{font:700 1.5rem 'Noto Sans KR',sans-serif;margin-bottom:.8rem;display:flex;justify-content:space-between;gap:1rem}
.panel-t .hint{font-weight:400;color:var(--gk-text-subtle)}
.panel-t{align-items:center;min-height:4rem}
.seg.sm button{min-height:3.2rem;padding:0 1.2rem;font-size:1.4rem}
.layers{position:relative;aspect-ratio:1/1;border:1px solid var(--gk-border);border-radius:var(--krds-radius-large);overflow:hidden;background:var(--viz-bg);touch-action:none}
.layer{position:absolute;inset:0;margin:0}
.layer canvas{width:100%;height:100%;display:block}
.layer figcaption{display:none}
.layer[hidden]{display:none}
.layers canvas{cursor:crosshair}
.split .viz{grid-template-columns:1fr}
.split .layers{aspect-ratio:auto;display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1.2rem;border:0;overflow:visible;background:none}
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
.legend{display:flex;flex-wrap:wrap;gap:.4rem 1.6rem;font-size:1.4rem;color:var(--gk-text-subtle);margin-top:.8rem}
.legend i{display:inline-block;width:2rem;height:.4rem;vertical-align:middle;margin-right:.6rem;border-radius:.2rem}
/* live readout */
.read{margin-top:2.4rem;padding-top:1.6rem;border-top:2px solid var(--gk-text)}
#mat{overflow-x:auto;font-size:1.9rem;min-height:9rem}
.verdict{display:flex;flex-wrap:wrap;gap:.4rem 2rem;align-items:baseline;margin-top:.4rem}
#lam{font-size:1.9rem}
#verdict{font-weight:800;font-size:1.9rem}
.stat{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));margin-top:1.6rem;border-top:1px solid var(--gk-divider)}
.stat div{padding:1rem 1.2rem 0;font-size:1.4rem;color:var(--gk-text-subtle)}
.stat b{display:block;font-size:2rem;color:var(--gk-text);font-variant-numeric:tabular-nums}
.refs{padding-left:2.2rem;font-size:1.5rem}
code{font-size:.9em;background:var(--gk-surface-secondary);padding:.1rem .4rem;border-radius:.2rem}
.ft{background:var(--gk-surface-secondary);border-top:1px solid var(--gk-divider);padding:3.2rem 0;font-size:1.5rem;color:var(--gk-text-subtle)}
.ft strong{color:var(--gk-text)}
@media (max-width:1024px){.page{grid-template-columns:1fr}.toc{position:static;display:flex;flex-wrap:wrap;gap:.4rem}.toc b{width:100%}.toc a{border:1px solid var(--gk-divider);border-radius:var(--krds-radius-medium)}.viz{grid-template-columns:1fr}.split .layers{grid-template-columns:repeat(2,minmax(0,1fr))}}
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
  <p class="lead">모든 쌍이 spring 인 network 에서 node 하나를 움직이면 curvature 는 어떻게 바뀌는가. Block Hessian $\mathbf H_{ii}$ 는 언제나 positive semidefinite 인 stiffness $K$ 와, 부호가 tension 으로 정해지는 prestress $S$ 의 합이다.</p>
  <dl class="keys" id="keys" aria-label="Key numbers"></dl>
</div>

<div class="page">
<nav class="toc" aria-label="본문 메뉴"><b>Contents</b>
  <a href="#model">Model</a><a href="#viz">Slice viewer</a><a href="#decomp">$K$ and $S$</a><a href="#numbers">Numbers</a><a href="#scope">Scope</a>
</nav>
<div class="c">

<section id="model">
<h2>Model</h2>
<p>graph $G$ 의 node 를 평면에 embed 하고, 모든 쌍 $(i,j)$ 를 rest length $\delta_{ij}$ (unweighted shortest-path hop distance) 의 spring 으로 잇는다. energy 는 unweighted stress 다.</p>
$$F(X)=\sum_{i<j}\bigl(d_{ij}-\delta_{ij}\bigr)^2,\qquad d_{ij}=\lVert\mathbf x_i-\mathbf x_j\rVert,\qquad X=(\mathbf x_1,\dots,\mathbf x_N)\in\mathbb R^{2N}.$$
<p>node position 은 metric MDS 로 정한다. classical (Torgerson) MDS 를 initial condition 으로 두고 SMACOF (stress majorization) 와 L-BFGS 로 $F$ 를 줄여 멈춘 점이 <b>terminal</b>, 즉 stress 의 local minimum 이다. constraint $N(N-1)/2$ 개가 degree of freedom $2N-3$ 보다 훨씬 많은 overconstrained framework 라서, 어떤 terminal 에서도 모든 spring 이 rest length 에 있을 수 없다. 남은 stress 가 prestress 다.</p>
<p>node $i$ 하나만 움직이고 나머지를 terminal 에 고정하면 $F$ 는 2차원 slice $F_i(\mathbf x)$ 가 되고, 그 Hessian 이 full $2N\times2N$ Hessian 의 $(i,i)$ block 이다.</p>
$$F_i(\mathbf x)=\sum_{j\ne i}\bigl(\lVert\mathbf x-\mathbf x_j\rVert-\delta_{ij}\bigr)^2+\text{const},\qquad \mathbf H_{ii}=\nabla\nabla^{\mathsf T}F_i=\mathbf J(\nabla F_i).$$
<p>symmetric matrix $\mathbf H_{ii}$ 의 eigenvalue $\lambda_1\le\lambda_2$ 는 principal curvature, eigenvector 는 principal direction 이다. eigenvector $\mathbf v$ 방향으로 $\epsilon$ 만큼 움직이면 $F_i(\mathbf x+\epsilon\mathbf v)=F_i(\mathbf x)+\epsilon\,\nabla F_i\cdot\mathbf v+\tfrac12\lambda\epsilon^2+O(\epsilon^3)$ 이다.</p>
<div class="tbl"><table>
<caption>eigenvalue 부호에 따른 local curvature type. stationary point ($\nabla F_i=\mathbf 0$) 가 아닌 곳에서는 그 자리의 curvature type 일 뿐이고, stationary point 에서만 Morse index 가 된다.</caption>
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
<p><span id="vizlead"></span> 왼쪽은 네 layer 를 겹친 plan view, 오른쪽은 같은 $F_i$ 의 3D landscape 다. 다른 node 를 누르면 그 node 의 slice 로 바뀌고, 끌면 직접 움직인다.</p>

<div class="ctl" role="group" aria-label="그림 조절">
  <button type="button" class="btn pri" id="play">일시정지</button>
  <button type="button" class="btn" id="reset">terminal 로</button>
  <div class="seg" role="group" aria-label="layer 배치"><button type="button" id="stack" aria-pressed="true">overlay</button><button type="button" id="split" aria-pressed="false">split</button></div>
  <fieldset><legend>layer</legend>
    <label><input type="checkbox" data-layer="L1" checked>contour</label>
    <label><input type="checkbox" data-layer="L2" checked>spring stress</label>
    <label><input type="checkbox" data-layer="L3" checked>network</label>
    <label><input type="checkbox" data-layer="L4" checked>eigenvector</label>
  </fieldset>
  <label>landscape 강조 <input type="range" id="gamma" min="0" max="100" value="30" aria-describedby="gamma-note"></label>
</div>

<div class="viz">
  <div>
    <p class="panel-t">Plan view (layers) <span class="hint" id="selinfo"></span></p>
    <div class="layers" id="layers">
      <figure class="layer" id="L1"><canvas width="1040" height="1040" aria-label="contour layer"></canvas><figcaption><b>① contour</b>$\log F_i$ 의 contour band</figcaption></figure>
      <figure class="layer" id="L2"><canvas width="1040" height="1040" aria-label="spring stress layer"></canvas><figcaption><b>② spring stress</b>선택 node 의 $N-1$ 개 spring</figcaption></figure>
      <figure class="layer" id="L3"><canvas width="1040" height="1040" aria-label="network layer"></canvas><figcaption><b>③ network</b>graph edge, node size ∝ √degree</figcaption></figure>
      <figure class="layer" id="L4"><canvas width="1040" height="1040" aria-label="eigenvector layer"></canvas><figcaption><b>④ eigenvector</b>$\mathbf H_{ii}$ 의 principal direction, 길이 ∝ $\lvert\lambda\rvert$</figcaption></figure>
    </div>
    <div class="legend">
      <span><i style="background:var(--viz-edge)"></i>graph edge</span>
      <span><i style="background:var(--viz-ten)"></i>tension $d\gt\delta$</span>
      <span><i style="background:var(--viz-com)"></i>compression $d\lt\delta$</span>
      <span><i style="background:var(--viz-up)"></i>$\lambda\gt0$</span>
      <span><i style="background:var(--viz-down)"></i>$\lambda\lt0$</span>
    </div>
  </div>
  <div>
    <div class="panel-t"><span>Landscape $F_i(\mathbf x)$</span> <div class="seg sm" role="group" aria-label="view"><button type="button" data-view="top" aria-pressed="false">top</button><button type="button" data-view="tilt" aria-pressed="true">tilt</button><button type="button" data-view="side" aria-pressed="false">side</button></div></div>
    <canvas id="surf" width="1040" height="900" aria-label="3D landscape of F_i"></canvas>
    <div class="cam" role="group" aria-label="camera">
      <label>azimuth <input type="range" id="cam-az" min="-180" max="180" step="1"></label>
      <label>elevation <input type="range" id="cam-el" min="2" max="83" step="1"></label>
      <label>zoom <input type="range" id="cam-zoom" min="60" max="300" step="5" value="100"></label>
      <label>height scale <input type="range" id="cam-h" min="30" max="250" step="5" value="100"></label>
      <label><input type="checkbox" id="cam-spin" checked>auto-rotate</label>
      <button type="button" class="btn" id="cam-reset">reset camera</button>
    </div>
    <p class="fine">landscape 를 끌면 돌아간다 (좌우 = azimuth, 위아래 = elevation). wheel 이나 pinch 로 zoom.</p>
    <p class="fine" id="gamma-note">height 는 window 안의 $F_i$ 를 saturating linear height $1-e^{-(F_i-F_{\min})/\ell}$ 와 rank (quantile) height 사이에서 섞은 값이다 (강조 0 = linear). $\ell$ 은 preset node 에서 barrier 높이의 2배, 그 밖에는 window median 까지의 높이다. 둘 다 monotone transform 이라 contour 와 stationary point 의 위치·type 은 그대로이고, 색칠한 curvature 판정은 언제나 원래 $F_i$ 의 $\mathbf H_{ii}$ 로 한다. plan view 의 점선 사각형이 landscape window 다.</p>
  </div>
</div>

<div class="read" aria-live="polite">
  <div id="mat"></div>
  <div class="verdict"><span id="lam"></span><span id="verdict"></span></div>
  <div class="stat">
    <div>$F_i-F_i^{\rm term}$<b id="dF"></b></div>
    <div>tension pairs<b id="nT"></b></div>
    <div>compression pairs<b id="nC"></b></div>
    <div>$\lVert\nabla F_i\rVert$<b id="gN"></b></div>
  </div>
</div>
</section>

<section id="decomp">
<h2>$K$ and $S$: curvature 는 어디서 오는가</h2>
<p>쌍 방향의 unit vector 를 $\mathbf u_{ij}=(\mathbf x_i-\mathbf x_j)/d_{ij}$, projector 를 $P_{ij}=\mathbf u_{ij}\mathbf u_{ij}^{\mathsf T}$ 라 하면 block 은 정확히 두 항으로 나뉜다.</p>
$$\mathbf H_{ii}=\underbrace{2\sum_{j\ne i}P_{ij}}_{K\ \succeq\ 0\ \text{(stiffness)}}\;+\;\underbrace{\sum_{j\ne i}\omega_{ij}\bigl(\mathbf I-P_{ij}\bigr)}_{S\ \text{(prestress)}},\qquad \omega_{ij}=\frac{2\,(d_{ij}-\delta_{ij})}{d_{ij}}.$$
<p>$K$ 는 spring 을 longitudinal 방향으로 늘이는 (material) stiffness 다. 위치와 무관하게 $\operatorname{tr}K=2(N-1)$ 이므로 curvature type 을 정하는 것은 $S$ 다. $\omega_{ij}$ 는 force density (길이당 tension) 로, rigidity theory 의 stress coefficient 와 같은 양이다. $S$ 는 transverse 방향에만 작용하는 geometric (stress) stiffness 로, tension ($\omega\gt0$) 은 transverse 방향을 단단하게, compression ($\omega\lt0$) 은 무르게 만든다. 따라서</p>
$$\operatorname{tr}\mathbf H_{ii}=2(N-1)+\sum_{j\ne i}\omega_{ij},\qquad \text{DOME}\ \Rightarrow\ \sum_{j\ne i}\omega_{ij}\lt-2(N-1).$$
<p>즉 dome 은 그 자리에서 compression 의 force density 합이 고정된 stiffness 합을 넘어설 때만 생긴다. 같은 graph 라도 prestress 의 배치가 terminal 마다 다르므로 curvature map 도 terminal 마다 다르다.</p>
</section>

<section id="numbers">
<h2>Numbers</h2>
<p id="facts"></p>
<div class="tbl"><table>
<caption>terminal 에서의 check (build 때 numpy 로 계산)</caption>
<thead><tr><th scope="col">item</th><th scope="col">value</th></tr></thead>
<tbody id="checks"></tbody></table></div>
</section>

<section id="scope">
<h2>Scope</h2>
<ul>
<li><b>Slice 다.</b> barrier 와 second basin 은 다른 $N-1$ 개 node 를 고정한 conditional 값이다. 모든 node 가 함께 relax 하는 full landscape 의 minimum-energy path 와는 다른 양이며, 다른 좌표가 relax 하면 barrier 는 낮아질 수 있다.</li>
<li><b>Block 판정이다.</b> BOWL / SADDLE / DOME 은 $\mathbf H_{ii}$ 하나에 대한 것이다. terminal 전체의 stability 는 Numbers 표의 $\mathbf H_\perp$ (rigid-body mode 3개를 뺀 $2N-3$ 차원) 로 따로 확인했다.</li>
<li><b>Unweighted stress.</b> graph drawing 에서 흔한 $w_{ij}=\delta_{ij}^{-2}$ weight (Kamada–Kawai 형) 가 아니다. weight 를 넣으면 $K$ 와 $\omega_{ij}$ 에 $w_{ij}$ 가 곱해질 뿐 decomposition 의 형태는 같다.</li>
<li><b>Data.</b> 모두 largest connected component, unweighted. Zachary karate club ($N=34$), Lusseau dolphins ($N=62$), Les Misérables co-appearance ($N=77$), college football ($N=115$), ca-netscience coauthorship ($N=352$).</li>
<li><b>Terminal.</b> classical MDS → SMACOF → L-BFGS, random seed 없음. classical MDS 에서 겹치는 structurally equivalent node 는 $10^{-3}$ 크기의 golden-angle offset 으로 떼어 놓았다.</li>
<li><b>Preset node.</b> 모든 node 의 slice 에서 terminal 이 아닌 basin (grid local minimum 을 BFGS 로 다듬고, minimax path 의 barrier 가 두 minimum 보다 0.05 이상 높은 것) 을 찾아 gain $F_i^{\rm term}-F_i^{\rm basin}$ 이 가장 큰 node 를 골랐다. gain $\gt0$ 이면 slice 안에 terminal 보다 낮은 basin 이 있다. college football 은 모든 slice 가 single basin 이라, $\lambda_1$ 이 가장 작은 (가장 soft 한) node 로 dome region 까지 다녀오는 curvature tour 를 보여 준다. 현상이 잘 보이는 node 를 고른 것이지 random sample 이 아니다.</li>
<li><b>Reproduction.</b> <code>python build/make_nets.py</code>, <code>python build/build_page.py</code>, <code>node build/check_katex.cjs index.html</code> (math compile error 0).</li>
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
const cvs = ['L1', 'L2', 'L3', 'L4'].map(id => $(id).querySelector('canvas'));
const [c1, c2, c3, c4] = cvs.map(c => c.getContext('2d'));
const W = cvs[0].width;
let NET, n, X, E, Draw, span, cx, cy, S, deg;
const delta = (i, j) => Draw[i * n + j];
const toC = (x, y) => [W / 2 + (x - cx) * S, W / 2 - (y - cy) * S];
const fromC = (a, b) => [(a - W / 2) / S + cx, -(b - W / 2) / S + cy];
function setNet(key) {
  NET = NETS[key]; n = NET.N; X = NET.X; E = NET.E;
  Draw = Uint8Array.from(atob(NET.D), c => c.charCodeAt(0));
  const lo = NET.lo, hi = NET.hi; span = Math.max(hi[0] - lo[0], hi[1] - lo[1]) * 1.04;
  cx = (lo[0] + hi[0]) / 2; cy = (lo[1] + hi[1]) / 2; S = W / span;
  deg = new Array(n).fill(0); for (const [a, b] of E) { deg[a]++; deg[b]++; }
}
let C = {};
const hex = s => { s = s.trim(); if (s[0] !== '#') return [128, 128, 128]; if (s.length === 4) s = '#' + [...s.slice(1)].map(c => c + c).join(''); return [1, 3, 5].map(k => parseInt(s.slice(k, k + 2), 16)); };
const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
function readColors() { const cs = getComputedStyle(document.documentElement); for (const k of ['bg', 'band-lo', 'band-hi', 'iso', 'edge', 'node', 'node-ring', 'ten', 'com', 'up', 'down', 'sad', 'mark', 's-lo', 's-mid', 's-hi']) C[k] = hex(cs.getPropertyValue('--viz-' + k)); }
let sel, pos, t = 0, playing = true, dragging = false, dirty = true;
let gamma = 0.3, az = -0.75, el = 0.55; const cam = {zoom: 1, h: 1, spin: true};

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
let Fref = 0, win = null, surf = null;
function buildContour() {
  const G = 97, L = new Float64Array(G * G); let mn = Infinity, mx = -Infinity;
  for (let a = 0; a < G; a++) for (let b = 0; b < G; b++) { const v = Fi([cx - span / 2 + span * b / (G - 1), cy - span / 2 + span * a / (G - 1)], sel); L[a * G + b] = v; mn = Math.min(mn, v); mx = Math.max(mx, v); }
  for (let k = 0; k < G * G; k++) L[k] = Math.log(L[k] - mn + 1e-3 * (mx - mn));
  const srt = Array.from(L).sort((p, q) => p - q), NB = 22, qs = [];
  for (let k = 1; k < NB; k++) qs.push(srt[Math.floor(k / NB * (srt.length - 1))]);
  const id = c1.createImageData(W, W), px = id.data, R = 4;
  const at = (x, y) => { const bx = Math.min(G - 1.001, Math.max(0, (x - cx + span / 2) / span * (G - 1))), by = Math.min(G - 1.001, Math.max(0, (y - cy + span / 2) / span * (G - 1)));
    const i0 = Math.floor(by), j0 = Math.floor(bx), fy = by - i0, fx = bx - j0;
    return L[i0 * G + j0] * (1 - fx) * (1 - fy) + L[i0 * G + j0 + 1] * fx * (1 - fy) + L[(i0 + 1) * G + j0] * (1 - fx) * fy + L[(i0 + 1) * G + j0 + 1] * fx * fy; };
  for (let yy = 0; yy < W; yy += R) for (let xx = 0; xx < W; xx += R) {
    const v = at(...fromC(xx + R / 2, yy + R / 2)); let k = 0; while (k < qs.length && v > qs[k]) k++;
    const s = Math.pow(k / qs.length, 0.8), col = C['band-lo'].map((c, m) => c + (C['band-hi'][m] - c) * s);
    for (let dy = 0; dy < R; dy++) for (let dx = 0; dx < R; dx++) { const p4 = ((yy + dy) * W + xx + dx) * 4; px[p4] = col[0]; px[p4 + 1] = col[1]; px[p4 + 2] = col[2]; px[p4 + 3] = 255; }
  }
  c1.putImageData(id, 0, 0);
  const cell = span / (G - 1);
  qs.forEach((lev, li) => {
    c1.strokeStyle = rgba(C.iso, li < 6 ? 0.85 : 0.45); c1.lineWidth = li < 6 ? 2 : 1.2; c1.beginPath();
    for (let a = 0; a < G - 1; a++) for (let b = 0; b < G - 1; b++) {
      const v = [L[a * G + b], L[a * G + b + 1], L[(a + 1) * G + b + 1], L[(a + 1) * G + b]], P = [[b, a], [b + 1, a], [b + 1, a + 1], [b, a + 1]], pts = [];
      for (let e = 0; e < 4; e++) { const u = v[e], w = v[(e + 1) % 4]; if ((u - lev) * (w - lev) < 0) { const tt = (lev - u) / (w - u); pts.push([P[e][0] + tt * (P[(e + 1) % 4][0] - P[e][0]), P[e][1] + tt * (P[(e + 1) % 4][1] - P[e][1])]); } }
      for (let k = 0; k + 1 < pts.length; k += 2) { const A = toC(cx - span / 2 + pts[k][0] * cell, cy - span / 2 + pts[k][1] * cell), B = toC(cx - span / 2 + pts[k + 1][0] * cell, cy - span / 2 + pts[k + 1][1] * cell); c1.moveTo(A[0], A[1]); c1.lineTo(B[0], B[1]); }
    }
    c1.stroke();
  });
}
// ── landscape window and surface heights ──
function buildSurface() {
  if (sel === NET.sel) win = {x0: NET.win[0], y0: NET.win[1], s: NET.win[2]};
  else { const half = 0.2 * span; win = {x0: X[sel][0] - half, y0: X[sel][1] - half, s: 2 * half}; }
  const G = 73, F = new Float64Array(G * G);
  for (let a = 0; a < G; a++) for (let b = 0; b < G; b++) F[a * G + b] = Fi([win.x0 + win.s * b / (G - 1), win.y0 + win.s * a / (G - 1)], sel);
  // linear height saturates smoothly, 1 - exp(-(F - min)/ℓ), with ℓ = 2 × (barrier - min) for the preset node and the window median otherwise
  const srt = Array.from(F).sort((p, q) => p - q), mn = srt[0], ell = sel === NET.sel && NET.B ? 2 * (NET.barrier - mn) : srt[Math.floor(0.5 * (srt.length - 1))] - mn;
  const rank = v => { let l = 0, r = srt.length - 1; while (l < r) { const m = (l + r) >> 1; if (srt[m] < v) l = m + 1; else r = m; } return l / (srt.length - 1); };
  const lin = new Float64Array(G * G), rk = new Float64Array(G * G);
  for (let k = 0; k < G * G; k++) { lin[k] = 1 - Math.exp(-(F[k] - mn) / ell); rk[k] = rank(F[k]); }
  surf = {G, F, lin, rk, Z: new Float64Array(G * G)};
  setGamma(gamma);
}
function setGamma(gm) { gamma = gm; if (!surf) return; const {lin, rk, Z} = surf; for (let k = 0; k < Z.length; k++) Z[k] = (1 - gm) * lin[k] + gm * Math.pow(rk[k], 1.35); dirty = true; }
function zAtXY(x, y) { const {G, Z} = surf; const b = (x - win.x0) / win.s * (G - 1), a = (y - win.y0) / win.s * (G - 1);
  if (a < 0 || b < 0 || a > G - 1 || b > G - 1) return null; const i0 = Math.min(G - 2, Math.floor(a)), j0 = Math.min(G - 2, Math.floor(b)), fy = a - i0, fx = b - j0;
  return Z[i0 * G + j0] * (1 - fx) * (1 - fy) + Z[i0 * G + j0 + 1] * fx * (1 - fy) + Z[(i0 + 1) * G + j0] * (1 - fx) * fy + Z[(i0 + 1) * G + j0 + 1] * fx * fy; }

const sv = $('surf'), sx = sv.getContext('2d');
function drawSurf(cls) {
  const Wv = sv.width, Hv = sv.height, {G, Z} = surf; sx.clearRect(0, 0, Wv, Hv);
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
  const ramp = z => { const A = C['s-lo'], M = C['s-mid'], B = C['s-hi']; return z < 0.5 ? A.map((c, m) => c + (M[m] - c) * z * 2) : M.map((c, m) => c + (B[m] - c) * (z - 0.5) * 2); };
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
  sx.strokeStyle = rgba(C.iso, 0.28); sx.lineWidth = 1.2;
  for (let a = 0; a < G; a += 6) { sx.beginPath(); for (let b = 0; b < G; b++) { const p = pts[a * G + b]; b ? sx.lineTo(p[0], p[1]) : sx.moveTo(p[0], p[1]); } sx.stroke(); }
  for (let b = 0; b < G; b += 6) { sx.beginPath(); for (let a = 0; a < G; a++) { const p = pts[a * G + b]; a ? sx.lineTo(p[0], p[1]) : sx.moveTo(p[0], p[1]); } sx.stroke(); }
  const proj = p => { const z = zAtXY(p[0], p[1]); return z === null ? null : P3((p[0] - win.x0) / win.s, (p[1] - win.y0) / win.s, z); };
  // curvature patch around the moving point
  const cc = cls === 'bowl' ? C.up : cls === 'dome' ? C.down : C.sad, r = 0.06 * win.s;
  sx.strokeStyle = rgba(cc, 1); sx.fillStyle = rgba(cc, 0.45); sx.lineWidth = 4; sx.beginPath();
  for (let k = 0; k <= 40; k++) { const th = k / 40 * 2 * Math.PI, p = proj([pos[0] + r * Math.cos(th), pos[1] + r * Math.sin(th)]); if (p) k ? sx.lineTo(p[0], p[1] - 3) : sx.moveTo(p[0], p[1] - 3); }
  sx.closePath(); sx.fill(); sx.stroke();
  sx.font = '700 28px "Noto Sans KR",sans-serif'; sx.textBaseline = 'bottom';
  const label = (p, txt) => { const q = proj(p); if (!q) return; sx.strokeStyle = rgba(C.mark, 0.9); sx.lineWidth = 2; sx.beginPath(); sx.moveTo(q[0], q[1]); sx.lineTo(q[0], q[1] - 46); sx.stroke(); sx.fillStyle = rgba(C.mark, 1); sx.beginPath(); sx.arc(q[0], q[1], 5, 0, 7); sx.fill();
    const tw = sx.measureText(txt).width; sx.fillStyle = rgba(C.bg, 0.92); sx.fillRect(q[0] + 2, q[1] - 84, tw + 16, 38); sx.fillStyle = rgba(C.mark, 1); sx.fillText(txt, q[0] + 10, q[1] - 50); };
  label(X[sel], 'terminal'); if (sel === NET.sel && NET.B) { label(NET.saddle, 'barrier'); label(NET.B, NET.F0 > NET.FB ? 'lower basin' : 'second basin'); }
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
function drawVectors(h) {
  c4.clearRect(0, 0, W, W);
  const [w0, w1] = toC(win.x0, win.y0 + win.s), [w2, w3] = toC(win.x0 + win.s, win.y0);
  c4.setLineDash([10, 8]); c4.strokeStyle = rgba(C.mark, 0.7); c4.lineWidth = 2; c4.strokeRect(w0, w1, w2 - w0, w3 - w1); c4.setLineDash([]);
  const ring = (p, txt, dash) => { const [a, b] = toC(...p); c4.strokeStyle = rgba(C.mark, 1); c4.lineWidth = 2.5; c4.setLineDash(dash ? [6, 6] : []); c4.beginPath(); c4.arc(a, b, 18, 0, 7); c4.stroke(); c4.setLineDash([]);
    c4.font = '700 24px "Noto Sans KR",sans-serif'; c4.lineWidth = 6; c4.strokeStyle = rgba(C.bg, 0.9); c4.strokeText(txt, a + 22, b - 14); c4.fillStyle = rgba(C.mark, 1); c4.fillText(txt, a + 22, b - 14); };
  ring(X[sel], 'terminal', false); if (sel === NET.sel && NET.B) { ring(NET.saddle, 'barrier', true); ring(NET.B, NET.F0 > NET.FB ? 'lower basin' : 'second basin', true); }
  const [a, b] = toC(...pos), lm = Math.max(Math.abs(h.l1), Math.abs(h.l2)) || 1;
  for (const [l, v] of [[h.l1, h.v1], [h.l2, h.v2]]) { const L = 34 + 76 * Math.abs(l) / lm, c = l > 0 ? C.up : C.down; for (const sg of [1, -1]) arrow(c4, a, b, a + sg * v[0] * L, b - sg * v[1] * L, c); }
  c4.fillStyle = rgba(C.mark, 1); c4.beginPath(); c4.arc(a, b, 11, 0, 7); c4.fill(); c4.strokeStyle = rgba(C.bg, 1); c4.lineWidth = 3; c4.stroke();
}

// ── live readout (KaTeX) ──
const f2 = v => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));
const bm = M => `\\begin{bmatrix}${f2(M[0])}&${f2(M[1])}\\\\${f2(M[1])}&${f2(M[2])}\\end{bmatrix}`;
let lastTex = '';
function readout(h, cls) {
  const tex = `\\mathbf H_{ii}=\\underbrace{${bm(h.H)}}_{\\text{Hessian}}=\\underbrace{${bm(h.K)}}_{K\\,\\succeq\\,0}+\\underbrace{${bm(h.S)}}_{S}`;
  if (tex !== lastTex) { katex.render(tex, $('mat'), {displayMode: true, throwOnError: false}); lastTex = tex;
    katex.render(`\\lambda_1=${f2(h.l1)},\\quad \\lambda_2=${f2(h.l2)}`, $('lam'), {throwOnError: false}); }
  const V = $('verdict'), col = cls === 'bowl' ? 'up' : cls === 'dome' ? 'down' : 'sad';
  V.textContent = cls === 'bowl' ? 'BOWL — 모든 direction 으로 positive curvature' : cls === 'dome' ? 'DOME — compression 이 stiffness 를 이긴다' : 'SADDLE — 한 direction 은 위, 다른 direction 은 아래';
  V.style.color = `var(--viz-${col})`;
  const dF = h.f - Fref; $('dF').textContent = (dF >= 0 ? '+' : '−') + Math.abs(dF).toFixed(1);
  $('nT').textContent = h.nt; $('nC').textContent = h.nc; $('gN').textContent = Math.hypot(...h.g).toFixed(2);
}
function rebuild() { Fref = Fi(X[sel], sel); buildContour(); buildSurface(); $('selinfo').textContent = `node ${sel} · degree ${deg[sel]}` + (sel === NET.sel ? ' · preset path' : ' · drag to move'); dirty = true; }
let lastPos = '';
function frame() {
  if (playing && !dragging && sel === NET.sel) { t = (t + 1) % NET.anim.length; pos = NET.anim[t].slice(); }
  if (elT !== null) { el += (elT - el) * 0.15; if (Math.abs(elT - el) < 1e-3) { el = elT; elT = null; } dirty = true; }
  if (cam.spin && !rot) { az += 0.003; dirty = true; }
  syncCam();
  const key = pos.join() + sel;
  if (key !== lastPos || dirty) {
    const h = hess(pos, sel), cls = h.l1 > 0 ? 'bowl' : h.l2 < 0 ? 'dome' : 'saddle';
    drawSprings(); drawNetwork(); drawVectors(h); drawSurf(cls); readout(h, cls); lastPos = key; dirty = false;
  }
  requestAnimationFrame(frame);
}
// ── interaction ──
const lay = $('layers');
const evp = e => { const c = e.target.closest('canvas'); const r = c.getBoundingClientRect(); return [(e.clientX - r.left) / r.width * W, (e.clientY - r.top) / r.height * W]; };
lay.addEventListener('pointerdown', e => {
  if (!e.target.closest('canvas')) return;
  const [a, b] = evp(e); let best = -1, bd = 22;
  for (let j = 0; j < n; j++) { if (j === sel) continue; const [u, v] = toC(...X[j]); const dd = Math.hypot(u - a, v - b); if (dd < bd) { bd = dd; best = j; } }
  const [u0, v0] = toC(...pos);
  if (best >= 0 && Math.hypot(u0 - a, v0 - b) > 24) { sel = best; pos = X[sel].slice(); setPlay(false); rebuild(); return; }
  dragging = true; setPlay(false); pos = fromC(a, b); e.target.setPointerCapture(e.pointerId);
});
lay.addEventListener('pointermove', e => { if (dragging) pos = fromC(...evp(e)); });
lay.addEventListener('pointerup', () => { dragging = false; });
let rot = null;
const touches = new Map(); let pinch = null;
sv.addEventListener('pointerdown', e => { touches.set(e.pointerId, [e.clientX, e.clientY]); if (touches.size === 2) { const [p, q] = [...touches.values()]; pinch = [Math.hypot(p[0] - q[0], p[1] - q[1]), cam.zoom]; rot = null; return; } rot = [e.clientX, e.clientY, az, el]; sv.setPointerCapture(e.pointerId); sv.style.cursor = 'grabbing'; });
sv.addEventListener('pointermove', e => { if (touches.has(e.pointerId)) touches.set(e.pointerId, [e.clientX, e.clientY]); if (pinch && touches.size === 2) { const [p, q] = [...touches.values()]; cam.zoom = Math.max(0.6, Math.min(3, pinch[1] * Math.hypot(p[0] - q[0], p[1] - q[1]) / pinch[0])); $('cam-zoom').value = cam.zoom * 100; dirty = true; return; } if (!rot) return; az = rot[2] + (e.clientX - rot[0]) * 0.01; el = Math.max(0.03, Math.min(1.45, rot[3] + (e.clientY - rot[1]) * 0.006)); elT = null; cam.spin = false; $('cam-spin').checked = false; viewBtn(null); dirty = true; });
const up = e => { touches.delete(e.pointerId); if (touches.size < 2) pinch = null; rot = null; sv.style.cursor = 'grab'; };
sv.addEventListener('pointerup', up); sv.addEventListener('pointercancel', up);
// camera controls; sliders follow drags and presets
const deg2 = r => Math.round(r * 180 / Math.PI);
function syncCam() { const A = $('cam-az'), L = $('cam-el'); if (document.activeElement !== A) A.value = ((deg2(az) + 540) % 360) - 180; if (document.activeElement !== L) L.value = deg2(el); }
$('cam-az').oninput = e => { az = e.target.value * Math.PI / 180; cam.spin = false; $('cam-spin').checked = false; dirty = true; };
$('cam-el').oninput = e => { el = e.target.value * Math.PI / 180; elT = null; viewBtn(null); dirty = true; };
$('cam-zoom').oninput = e => { cam.zoom = e.target.value / 100; dirty = true; };
$('cam-h').oninput = e => { cam.h = e.target.value / 100; dirty = true; };
$('cam-spin').onchange = e => { cam.spin = e.target.checked; };
$('cam-reset').onclick = () => { az = -0.75; elT = 0.55; cam.zoom = 1; cam.h = 1; $('cam-zoom').value = 100; $('cam-h').value = 100; viewBtn('tilt'); dirty = true; };
sv.addEventListener('wheel', e => { e.preventDefault(); cam.zoom = Math.max(0.6, Math.min(3, cam.zoom * Math.exp(-e.deltaY * 0.0015))); $('cam-zoom').value = cam.zoom * 100; dirty = true; }, {passive: false});
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
$('mode').onclick = () => { const hc = document.documentElement.dataset.krdsMode !== 'high-contrast'; document.documentElement.dataset.krdsMode = hc ? 'high-contrast' : 'light'; $('mode').setAttribute('aria-pressed', hc); $('mode').textContent = hc ? '기본 화면' : '선명한 화면'; readColors(); buildContour(); dirty = true; };

// ── per-network text; every math string goes through setMath (build/check_katex.cjs compiles them) ──
const MATH = {delimiters: [{left: '$$', right: '$$', display: true}, {left: '$', right: '$', display: false}], ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'option', 'canvas'], throwOnError: false};
const setMath = (id, html) => { $(id).innerHTML = html; renderMathInElement($(id), MATH); };
const sci = v => { const [m, e] = v.toExponential(1).split('e'); return `${m}\\times10^{${+e}}`; };
function describe() {
  const h0 = hess(X[NET.sel], NET.sel), eK = eig(h0.K), eS = eig(h0.S), sumw = h0.S[0] + h0.S[2], ck = NET.check, npair = n * (n - 1) / 2;
  const two = NET.B !== null, bar = two ? (NET.barrier - NET.F0).toFixed(2) : '', gain = two ? NET.F0 - NET.FB : 0, gs = Math.abs(gain).toFixed(2);
  const basinName = gain > 0 ? 'lower basin' : 'second basin', domeStep = NET.dome ? ' → dome region' : '';
  const sliceKey = two ? `<div><dt>node ${NET.sel} slice: barrier / gain</dt><dd>$+${bar}$ / $${gain > 0 ? '+' : '-'}${gs}$</dd></div>` : `<div><dt>node ${NET.sel} slice</dt><dd>single basin</dd></div>`;
  setMath('keys', `<div><dt>network</dt><dd>${NET.name}</dd></div><div><dt>size</dt><dd>$N=${n},\\ E=${E.length}$</dd></div><div><dt>constraints vs DOF</dt><dd>$${npair}$ vs $${2 * n - 3}$</dd></div><div><dt>compressed pairs</dt><dd>${(100 * ck.frac_comp).toFixed(1)}%</dd></div><div><dt>$\\lambda_{\\min}(\\mathbf H_\\perp)$</dt><dd>$${ck.lam_min.toFixed(2)}$</dd></div>${sliceKey}`);
  const tour = two ? `자동 경로: terminal → barrier → ${basinName}${domeStep} → terminal.` : `모든 node 의 slice 가 single basin 이라 자동 경로는 curvature tour: terminal${domeStep} → terminal.`;
  setMath('vizlead', `${NET.name} ($N=${n}$) 의 MDS terminal 에서 node ${NET.sel} (degree ${deg[NET.sel]}) 만 움직인다. ${tour}`);
  const slice = two ? `node ${NET.sel} 하나만 옮기면 barrier ($+${bar}$) 너머에 $F$ 가 terminal 보다 $${gs}$ 만큼 ${gain > 0 ? '낮은 lower basin 이 있다. terminal 은 full space 에서 local minimum 이지만 (아래 표), 한 node 의 slice 만으로도 더 낮은 basin 이 보인다.' : '높은 second basin 이 있다. 이 slice 에서는 terminal 이 더 깊은 basin 이다.'}` : `이 network 의 MDS terminal 에서는 어느 node 의 slice 도 basin 이 하나뿐이다. 그래도 slice 안에는 compression 이 stiffness 를 이기는 dome region 이 있다.`;
  setMath('facts', `terminal 에서 node ${NET.sel} 의 block 은 $\\lambda_1=${h0.l1.toFixed(2)}$, $\\lambda_2=${h0.l2.toFixed(2)}$ 이다. 이 curvature 는 stiffness (eigenvalue $${eK[0].toFixed(1)},\\ ${eK[1].toFixed(1)}$, 합 $2(N-1)=${2 * (n - 1)}$) 와 prestress (eigenvalue $${eS[0].toFixed(1)},\\ ${eS[1].toFixed(1)}$) 의 합이다. 이 node 의 spring ${n - 1} 개 중 ${h0.nc} 개가 compression 이라 $\\sum_j\\omega_{ij}=${sumw.toFixed(1)}$ 이고, prestress 가 stiffness 의 약 ${Math.round(-sumw / (2 * (n - 1)) * 100)}% 를 상쇄한다. ${slice}`);
  setMath('checks', `<tr><td>stress</td><td>$F=${ck.F.toFixed(2)}$</td></tr><tr><td>full $\\mathbf H$ 의 zero mode</td><td>${ck.n_zero} 개 = translation 2 + rotation 1</td></tr><tr><td>$\\lambda_{\\min}(\\mathbf H_\\perp)$</td><td>$${ck.lam_min.toFixed(3)}\\gt0$: terminal 은 $2N$ 차원 full space 에서 local minimum</td></tr><tr><td>$\\max_i\\lVert\\nabla_iF\\rVert$</td><td>$${sci(ck.grad)}$</td></tr><tr><td>analytic $\\mathbf H_{ii}$ vs finite difference</td><td>relative difference $${sci(ck.fd)}$</td></tr><tr><td>terminal</td><td>classical MDS → SMACOF → L-BFGS</td></tr>`);
}
function loadNet(key) {
  setNet(key); sel = NET.sel; pos = X[sel].slice(); t = 0; lastTex = '';
  document.querySelectorAll('#nets button').forEach(b => b.setAttribute('aria-pressed', b.dataset.net === key));
  describe(); rebuild(); setPlay(!matchMedia('(prefers-reduced-motion: reduce)').matches);
}
$('nets').innerHTML = Object.keys(NETS).map(k => `<button type="button" data-net="${k}" aria-pressed="false">${NETS[k].name}</button>`).join('');
$('nets').onclick = e => { const b = e.target.closest('button'); if (b) loadNet(b.dataset.net); };
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
