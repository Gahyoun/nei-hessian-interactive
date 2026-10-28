"""ensemble.py -- the NEI ensemble: M = 500 stress terminals per network under the census protocol, 2026-10-28.

Protocol (Gahyoun/nei-architecture code/cluster/nei_gpu.py, code/model_sweep_m500.py)
  initial condition  RandomState(1000 + m).uniform(size=N*p), p = 2
  SMACOF             X <- B(X) X / N, d = 0 replaced by 1e-5, at most 500 iterations,
                     stop when (F_{k-1} - F_k) / sum_{i<j} d_ij^2 < 1e-10
  polish             L-BFGS-B (gtol 1e-12, ftol 1e-18, maxiter 1e5), analytic gradient
NEI = mean over pairs i<j of Var_m(d_ij) / Mean_m(d_ij)^2   (same as nei() in code/artifact_test.py)

Each terminal is then aligned to the reference MDS terminal of nets/<key>.json by orthogonal Procrustes
(translation + rotation/reflection, no scaling: stress is invariant under these and fixes the scale).
Output build/nets/<key>_ens.npz  (Xa: aligned terminals (M,N,2), F: terminal stress (M,), nei, gmax)
Usage  python build/ensemble.py [--M 500] [--procs 9] [keys ...]
"""
import argparse, base64, json, os, time
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
P_DIM, MAX_ITER, EPS = 2, 500, 1e-10
D_GLOBAL = None


def fg(x):
    D = D_GLOBAL; n = len(D); X = x.reshape(n, 2)
    diff = X[:, None] - X[None]; d = np.sqrt((diff ** 2).sum(-1)); np.fill_diagonal(d, 1.0)
    r = d - D; np.fill_diagonal(r, 0.0)
    g = 2 * ((r / d)[:, :, None] * diff).sum(1)
    return 0.5 * float((r * r).sum()), g.ravel()          # 0.5 * sum_{i != j} = sum_{i<j}


def run(m):
    D = D_GLOBAL; n = len(D); iu = np.triu_indices(n, 1)
    X = np.random.RandomState(1000 + m).uniform(size=n * P_DIM).reshape(n, P_DIM)
    Fold = None
    for _ in range(MAX_ITER):
        d = np.linalg.norm(X[:, None] - X[None], axis=2); d[d == 0] = 1e-5
        F = ((d - D)[iu] ** 2).sum()
        if Fold is not None and (Fold - F) / (d[iu] ** 2).sum() < EPS: break
        Fold = F
        B = -D / d; np.fill_diagonal(B, 0); np.fill_diagonal(B, -B.sum(1))
        X = B @ X / n
    r = minimize(fg, X.ravel(), jac=True, method='L-BFGS-B', options={'gtol': 1e-12, 'ftol': 1e-18, 'maxiter': 100000})
    _, g = fg(r.x)
    return r.x.reshape(n, P_DIM), r.fun, float(np.abs(g).max())


def init(D):
    global D_GLOBAL
    D_GLOBAL = D


def procrustes(Y, Xref):
    """argmin_{R orthogonal, t} ||Y R + t - Xref||."""
    Yc, Xc = Y - Y.mean(0), Xref - Xref.mean(0)
    U, _, Vt = np.linalg.svd(Yc.T @ Xc)
    return Yc @ (U @ Vt) + Xref.mean(0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--M', type=int, default=500); ap.add_argument('--procs', type=int, default=9)
    ap.add_argument('keys', nargs='*', default=['karate', 'dolphins', 'lesmis', 'football', 'netscience'])
    a = ap.parse_args()
    for key in a.keys:
        net = json.load(open(os.path.join(HERE, 'nets', key + '.json')))
        n = net['N']; Xref = np.array(net['X'])
        D = np.frombuffer(base64.b64decode(net['D']), np.uint8).reshape(n, n).astype(float)
        t0 = time.time()
        with Pool(a.procs, initializer=init, initargs=(D,)) as pool:
            res = pool.map(run, range(a.M), chunksize=1)
        Xs = np.stack([r[0] for r in res]); F = np.array([r[1] for r in res]); gmax = np.array([r[2] for r in res])
        iu = np.triu_indices(n, 1)
        P = np.linalg.norm(Xs[:, :, None] - Xs[:, None], axis=3)[:, iu[0], iu[1]]
        mu = P.mean(0); nei = float((P.var(0) / mu ** 2).mean())
        Xa = np.stack([procrustes(Y, Xref) for Y in Xs])
        np.savez_compressed(os.path.join(HERE, 'nets', key + '_ens.npz'), Xa=Xa, F=F, gmax=gmax, nei=nei)
        Fr = np.round(F, 6)
        print(f"{key:10s} N={n:3d} M={a.M} NEI={nei:.5f} F: min {F.min():.3f} median {np.median(F):.3f} max {F.max():.3f} "
              f"distinct(1e-6)={len(np.unique(Fr))} max|grad|={gmax.max():.1e} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
