"""Shirube acceptance-test harness for the proposal (Team GUEKSA, JETRO-PS-02).

All inputs (titles, factor scores, pilot counts) are ILLUSTRATIVE. Only the maths is real.
The proposal's numbers are produced by this file.   Run:  python3 shirube_sim.py
Needs: numpy, scipy
"""
import numpy as np
from scipy.stats import beta

SEED = 18102026
K = 50                                   # prior strength (pseudo-readers)
WT = np.array([.30, .20, .20, .15, .15])  # Launch Fit weights: demand, price, partner, localization, licensing
CONF = np.array([.7, .7, 1.0, .4, 1.0])   # per-factor confidence for Title A
TITLES = {                                # illustrative factor scores and pilot results
    'A': dict(F=[85, 75, 80, 75, 75], x=10, n=400),
    'B': dict(F=[71, 70, 71, 90, 88], x=24, n=400),
    'C': dict(F=[60, 62, 70, 70, 61], x=0, n=0),
}
READERS_PER_MONTH = 20000   # assumption: readers a month who finish the free chapters
UNLOCK_PRICE = 79           # assumption (INR): price of the first unlock bundle, to be tested


def post(prior, x, n):
    a, b = prior * K + x, (1 - prior) * K + n - x
    return dict(mean=a / (a + b), lo=beta.ppf(.05, a, b), hi=beta.ppf(.95, a, b), p3=beta.sf(.03, a, b))


def coverage(rng, true_sampler, prior_mean, n, R=10000, strength=K):
    p = true_sampler(R)
    x = rng.binomial(n, p)
    a, b = prior_mean * strength + x, (1 - prior_mean) * strength + n - x
    return ((p >= beta.ppf(.05, a, b)) & (p <= beta.ppf(.95, a, b))).mean() * 100


def compute():
    rng = np.random.default_rng(SEED)
    out = {}
    for k, t in TITLES.items():
        t['score'] = int(np.dot(WT, t['F']) + .5)
        t['raw'] = float(np.dot(WT, t['F']))
        t['prior'] = 0.03 + 0.01 * (t['score'] - 64) / 14
        t['post'] = post(t['prior'], t['x'], t['n'])
    A = np.array(TITLES['A']['F'], float)
    d = 20 * (1 - CONF)
    out['A_range'] = (float(WT @ (A - d)), float(WT @ (A + d)))
    out['A_conf'] = float(WT @ CONF)
    # Test 1: weight sensitivity at three jitter levels (all three titles ranked)
    T = np.array([TITLES[k]['F'] for k in 'ABC'], float)
    base_rank = tuple(np.argsort(-(T @ WT)))
    sens = {}
    for j in (.2, .5):
        same_top = same_rank = 0
        for _ in range(1000):
            w = WT * rng.uniform(1 - j, 1 + j, 5)
            w /= w.sum()
            sc = T @ w
            same_top += sc.argmax() == base_rank[0]
            same_rank += tuple(np.argsort(-sc)) == base_rank
        sens[j] = (same_top / 10, same_rank / 10)
    out['sens'] = sens
    out['gap_AB'] = float(TITLES['A']['raw'] - TITLES['B']['raw'])
    # Test 2: does the 90% range hold the true rate? Stress case: prior worth 20 pseudo-readers, cohort n = 100.
    # Baseline (the engine's own settings: prior worth 50, cohort 400) is reported alongside.
    pm = .04
    cov = {}
    for (st, n) in ((20, 100), (50, 400)):
        cov[(st, n)] = (coverage(rng, lambda R: rng.beta(pm * st, (1 - pm) * st, R), pm, n, strength=st),                  # belief right
                        coverage(rng, lambda R: rng.beta(pm * .5 * 2 * st, (1 - pm * .5) * 2 * st, R), pm, n, strength=st),  # truth = half the prior mean
                        coverage(rng, lambda R: rng.beta(pm * 2 * 2 * st, (1 - pm * 2) * 2 * st, R), pm, n, strength=st))    # truth = double the prior mean
    out['cov'] = cov
    # Revenue illustration for Title B (assumptions above)
    b = TITLES['B']['post']
    out['rev'] = [READERS_PER_MONTH * b[q] * UNLOCK_PRICE for q in ('lo', 'mean', 'hi')]
    out['share'] = 0.05 * out['rev'][1]
    out['n_sample'] = 1.96 ** 2 * .25 / .05 ** 2
    u = 900 * .98 - 500
    out['fusion'] = (u, u * 150, 100 * u * 150 / (900 * 150))
    rev = 900 * 150
    out['royalty'] = {r: (r * rev, u * 150 - r * rev, 100 * (u * 150 - r * rev) / rev) for r in (.10, .20)}  # assumptions, not benchmarks
    return out


if __name__ == '__main__':
    o = compute()
    lo, hi = o['A_range']
    print("Title A: score %.1f, range %.1f-%.1f, confidence %.2f" % (TITLES['A']['raw'], lo, hi, o['A_conf']))
    for k in 'ABC':
        t, p = TITLES[k], TITLES[k]['post']
        print("%s score %d prior %.2f%% pilot %d/%d -> mean %.2f%% 90%% range %.2f-%.2f%% P(rate>=3%%) %.1f%%" %
              (k, t['score'], t['prior'] * 100, t['x'], t['n'], p['mean'] * 100, p['lo'] * 100, p['hi'] * 100, p['p3'] * 100))
    print("Test 1 (A leads B by %.1f points)" % o['gap_AB'])
    for j, (top, rank) in o['sens'].items():
        print("  weights jittered +-%d%%: top title unchanged %.1f%%, full ranking unchanged %.1f%%" % (j * 100, top, rank))
    print("Test 2 (90% range holds true rate): [prior right, truth half, truth double]")
    for (st, n), c in o['cov'].items():
        print("  prior strength %d, n=%d: %.1f%% / %.1f%% / %.1f%%" % ((st, n) + c))
    print("Title B revenue illustration (%d readers/month x INR %d): low %.0f mean %.0f high %.0f; 5%% share %.0f" %
          ((READERS_PER_MONTH, UNLOCK_PRICE) + tuple(o['rev']) + (o['share'],)))
    print("Sample size for +-5%% at 95%%: %.1f" % o['n_sample'])
    print("Fusion Drop: unit contribution %.0f, x150 = %.0f, margin %.1f%%" % o['fusion'])
    for r, (amt, m, pct) in o['royalty'].items():
        print("  if royalty were %d%% of revenue (assumption): royalty %.0f, margin %.0f (%.1f%%)" % (r * 100, amt, m, pct))
