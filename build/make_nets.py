"""make_nets.py -- stress terminals and node-slice paths for the small networks in build/data/, 2026-10-28.

For each edge list: hop distances δ, a 2-d stress terminal (classical MDS start, then SMACOF and L-BFGS
to |∇F| < 1e-6), a check that the full Hessian has exactly 3 zero modes and is otherwise positive, and the
node whose one-node slice F_i hides the deepest lower point. The preset path for that node runs
terminal → slice barrier (minimax path on the grid, refined to the saddle) → lower point → dome region → terminal.
Output build/nets/<key>.json in the same schema as build/net.json (ca-netscience, kept as given).
Usage  python build/make_nets.py
"""
import base64, json, os
import numpy as np
import networkx as nx
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, 'nets'), exist_ok=True)
NETS = {'karate': 'Zachary karate club', 'dolphins': 'Lusseau dolphins', 'lesmis': 'Les Misérables', 'football': 'College football', 'netscience': 'ca-netscience'}


def stress(x, D, iu):
    X = x.reshape(-1, 2); diff = X[:, None] - X[None]; d = np.sqrt((diff ** 2).sum(-1)); np.fill_diagonal(d, 1)
    e = d - D; np.fill_diagonal(e, 0)
    g = 2 * ((e / d)[:, :, None] * diff).sum(1)
    return (e[iu] ** 2).sum(), g.ravel()


def terminal(D):
    """classical (Torgerson) MDS start, then metric stress MDS: SMACOF and L-BFGS polish. Deterministic."""
    n = len(D); iu = np.triu_indices(n, 1)
    J = np.eye(n) - 1 / n; Bc = -0.5 * J @ (D ** 2) @ J
    lam, V = np.linalg.eigh(Bc); X = V[:, -2:] * np.sqrt(np.maximum(lam[-2:], 0))
    # structurally equivalent nodes (identical rows of δ) coincide in classical MDS; split them by 1e-3 along a golden-angle spiral
    k = np.arange(n); X = X + 1e-3 * np.c_[np.cos(2.39996 * k), np.sin(2.39996 * k)]
    for _ in range(3000):                                   # SMACOF, unit weights: X <- B(X) X / n
        d = np.linalg.norm(X[:, None] - X[None], axis=2); np.fill_diagonal(d, 1)
        B = -D / np.maximum(d, 1e-12); np.fill_diagonal(B, 0); np.fill_diagonal(B, -B.sum(1))
        Xn = B @ X / n
        if np.abs(Xn - X).max() < 1e-10: X = Xn; break
        X = Xn
    r = minimize(stress, X.ravel(), args=(D, iu), jac=True, method='L-BFGS-B', options={'gtol': 1e-11, 'ftol': 1e-15, 'maxiter': 20000})
    X = r.x.reshape(-1, 2); X -= X.mean(0)
    w, V = np.linalg.eigh(np.cov(X.T)); X = X @ V[:, ::-1]           # principal axes, wide side horizontal
    return X


def full_hessian(X, D):
    n = len(X); H = np.zeros((2 * n, 2 * n))
    for a in range(n):
        for b in range(n):
            if a == b: continue
            v = X[a] - X[b]; r = np.linalg.norm(v); u = v / r; P = np.outer(u, u)
            B = 2 * P + 2 * (1 - D[a, b] / r) * (np.eye(2) - P)
            H[2*a:2*a+2, 2*b:2*b+2] = -B; H[2*a:2*a+2, 2*a:2*a+2] += B
    return np.linalg.eigvalsh(H)


def slice_F(P, X, D, i):
    m = np.arange(len(X)) != i
    r = np.linalg.norm(P[:, None] - X[m][None], axis=2)
    return ((r - D[i][m]) ** 2).sum(1)


def block(p, X, D, i):
    m = np.arange(len(X)) != i; v = p - X[m]; r = np.linalg.norm(v, axis=1); u = v / r[:, None]
    P = np.einsum('ka,kb->kab', u, u); c = 1 - D[i][m] / r
    return np.einsum('kab->ab', 2 * P + 2 * c[:, None, None] * (np.eye(2) - P))


def grad_i(p, X, D, i):
    m = np.arange(len(X)) != i; v = p - X[m]; r = np.linalg.norm(v, axis=1)
    return 2 * (((r - D[i][m]) / r)[:, None] * v).sum(0)


def minimax_path(Fg, s, t, G):
    """bottleneck path on the grid (8-neighbour): minimise the highest point passed."""
    import heapq
    best = np.full(G * G, np.inf); prev = -np.ones(G * G, int); best[s] = Fg[s]; h = [(Fg[s], s)]
    while h:
        c, k = heapq.heappop(h)
        if k == t: break
        if c > best[k]: continue
        a, b = divmod(k, G)
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                aa, bb = a + da, b + db
                if (da or db) and 0 <= aa < G and 0 <= bb < G:
                    kk = aa * G + bb; cc = max(c, Fg[kk])
                    if cc < best[kk]: best[kk] = cc; prev[kk] = k; heapq.heappush(h, (cc, kk))
    path = [t]
    while path[-1] != s: path.append(prev[path[-1]])
    return path[::-1]


def resample(pts, m):
    pts = np.asarray(pts, float)
    for _ in range(3):                                     # Chaikin smoothing, endpoints kept
        q = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]): q += [0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b]
        q.append(pts[-1]); pts = np.array(q)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    u = np.linspace(0, s[-1], m)
    return np.c_[np.interp(u, s, pts[:, 0]), np.interp(u, s, pts[:, 1])]


def build(key):
    lines = open(os.path.join(HERE, 'data', key + '.edgelist')).read().splitlines()
    src = lines[0].lstrip('# ').split(':', 1)[1].split(';')[0].strip()
    E = [tuple(map(int, l.split())) for l in lines[1:] if l.strip()]
    G = nx.Graph(E); n = G.number_of_nodes()
    D = np.array(nx.floyd_warshall_numpy(G, nodelist=range(n)))
    iu = np.triu_indices(n, 1)
    X = terminal(D); w = full_hessian(X, D)
    nz = int((np.abs(w) < 1e-7 * w[-1]).sum())
    assert nz == 3 and w[3] > 1e-6, (key, nz, w[:5])
    F, g = stress(X.ravel(), D, iu)
    lo, hi = X.min(0) - 0.6, X.max(0) + 0.6; span = (hi - lo).max(); c = (lo + hi) / 2
    Gg = 101; xs = c[0] - span / 2 + span * np.arange(Gg) / (Gg - 1); ys = c[1] - span / 2 + span * np.arange(Gg) / (Gg - 1)
    P = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
    # for every node: other basins of its slice F_i (grid local minima refined by BFGS, > 0.3 from the terminal position),
    # each with its minimax barrier refined to a saddle. A basin counts only if its barrier stands > 0.05 above both minima.
    # Choose the node/basin with the largest gain F0 - FB (it may be negative: a higher second basin).
    cands = []
    for i in range(n):
        Fg = slice_F(P, X, D, i); F0 = slice_F(X[i][None], X, D, i)[0]; M = Fg.reshape(Gg, Gg)
        pad = np.pad(M, 1, constant_values=np.inf)
        nb = np.min([pad[1 + da:Gg + 1 + da, 1 + db:Gg + 1 + db] for da in (-1, 0, 1) for db in (-1, 0, 1) if da or db], axis=0)
        mins = np.where((M < nb).ravel())[0]
        for k in mins[np.linalg.norm(P[mins] - X[i], axis=1) > 0.3]:
            r = minimize(lambda p: slice_F(p[None], X, D, i)[0], P[k], jac=lambda p: grad_i(p, X, D, i), method='BFGS')
            if np.linalg.norm(r.x - X[i]) > 0.3: cands.append((F0 - r.fun, i, r.x, r.fun, F0, Fg))
    cands.sort(key=lambda c: -c[0])
    found = None
    for gain, i, Bpt, FB, F0, Fg in cands:
        s0 = int(np.linalg.norm(P - X[i], axis=1).argmin()); t0 = int(np.linalg.norm(P - Bpt, axis=1).argmin())
        if s0 == t0: continue
        path = minimax_path(Fg, s0, t0, Gg); top = path[int(np.argmax(Fg[path]))]
        rS = minimize(lambda p: (grad_i(p, X, D, i) ** 2).sum(), P[top], method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-14, 'maxiter': 4000})
        Spt = rS.x; ls = np.linalg.eigvalsh(block(Spt, X, D, i)); barrier = slice_F(Spt[None], X, D, i)[0]
        if not (ls[0] < 0 < ls[1]) or np.linalg.norm(Spt - P[top]) > span / 10:   # fall back to the grid bottleneck
            Spt = P[top]; barrier = Fg[top]
        if barrier - max(F0, FB) > 0.05: found = (i, Bpt, FB, F0, Fg, path, Spt, barrier, ls); break
    def dome_near(i, ref):
        l2 = np.array([np.linalg.eigvalsh(block(p, X, D, i))[1] for p in P]); dk = np.where(l2 < 0)[0]
        return P[dk[np.argmin(np.linalg.norm(P[dk] - ref, axis=1))]] if len(dk) else None
    if found:
        i, Bpt, FB, F0, Fg, path, Spt, barrier, ls = found
        dome = dome_near(i, Bpt)
        seq = [X[i]] + list(P[path[1:-1]]) + [Bpt]
        seq = [p for k, p in enumerate(seq) if k == 0 or np.linalg.norm(p - seq[k - 1]) > 1e-9]
        leg1 = resample(seq, 220); leg2 = resample([Bpt] + ([dome] if dome is not None else []) + [X[i]], 200)
        anim = np.r_[np.repeat(X[i][None], 30, 0), leg1, np.repeat(Bpt[None], 30, 0), leg2]
        info = f"node={i} barrier=+{barrier - F0:.2f} gain={F0 - FB:+.2f} saddle λ={ls.round(2)} dome={'yes' if dome is not None else 'no'}"
    else:
        # single-basin slices only: show the softest node (smallest λ1 at the terminal) and a curvature tour to its nearest dome
        i = int(np.argmin([np.linalg.eigvalsh(block(X[j], X, D, j))[0] for j in range(n)]))
        F0 = slice_F(X[i][None], X, D, i)[0]; Bpt = Spt = None; FB = barrier = None
        dome = dome_near(i, X[i])
        anim = np.r_[np.repeat(X[i][None], 30, 0), resample([X[i], dome, X[i]] if dome is not None else [X[i], X[i] + 1, X[i]], 390)]
        info = f"node={i} single basin in every slice; curvature tour, dome={'yes' if dome is not None else 'no'}"
    # window for the 3-d landscape: path box, square, padded
    a0, a1 = anim.min(0), anim.max(0); half = max((a1 - a0).max() / 2 * 1.5, span * 0.18)
    out = dict(name=NETS[key], key=key, src=src, N=n, E=[list(e) for e in sorted(G.edges())],
               X=np.round(X, 6).tolist(), D=base64.b64encode(D.astype(np.uint8).tobytes()).decode(),
               sel=int(i), anim=np.round(anim, 4).tolist(), lo=lo.tolist(), hi=hi.tolist(),
               F0=float(F0), FB=(float(FB) if found else None), barrier=(float(barrier) if found else None),
               B=(Bpt.round(4).tolist() if found else None), saddle=(np.round(Spt, 4).tolist() if found else None),
               dome=(dome.round(4).tolist() if dome is not None else None), stress_total=float(F),
               win=[float((a0[0] + a1[0]) / 2 - half), float((a0[1] + a1[1]) / 2 - half), float(2 * half)],
               check=dict(n_zero=nz, lam_min=float(w[3]), grad=float(np.abs(g).max())))
    json.dump(out, open(os.path.join(HERE, 'nets', key + '.json'), 'w'), separators=(',', ':'))
    print(f"{key:10s} N={n:3d} E={len(E):3d} zero={nz} λmin(H⊥)={w[3]:.3f} |∇F|={np.abs(g).max():.1e} " + info)


for k in NETS:
    build(k)
