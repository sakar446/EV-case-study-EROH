import numpy as np, time, json, sys
from scipy.optimize import linprog

H = 24
CL = ['Delhi NCR', 'Mumbai MMR', 'Bengaluru', 'Hyderabad', 'Chennai']
C = 5
share = np.array([.28, .24, .20, .14, .14])
CAP = 1200 * share                 # MW installed (given: 1,200 MW)
DCc = .6 * CAP                     # 60% DC fast-charging (given)
SWc = .4 * CAP                     # 40% swap stations (given)
E_ST = 150 * share                 # MWh stationary (assumed 25% of 600 MWh)
P_ST = E_ST / 2                    # 0.5C
I_PK0 = 450 * share                # MWh swap-pack inventory (assumed 75% of 600 MWh)
LF = np.array([.17, .15, .15, .12, .12])        # mean utilisation (assumed)
RESH = np.array([.20, .14, .22, .24, .20])      # share of contracted RE by cluster (assumed)
ALPHA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.30
A_PART = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10
G = ALPHA * CAP                    # grid connection (MW)

RES = .95                          # 5% reserve (given)
ETA_S, SD_ST = .95, .0002
ETA_CELL, SD_P = .985, .0005
ETA_PK = .972
K_D = .025
STD_BASE, STD_OPT = .0035, .0012
P_DC, P_SW = 17.0, 15.0
HOST, VOM, DEG_ST, DEG_PK, PEN = .135, .4, 1.6, 1.0, 6.0
TAR_D, OA_IEX, PPA, OA_RE, SELL = 6.6, 1.0, 3.5, .8, .55
EPS, PHI = .4, .4


def eta_phys(l): return ETA_PK - A_PART * (1 - np.clip(l, 0, 1)) ** 2
ETA_C_OPT = eta_phys(.6)


hh = np.arange(H)
def cg(mu, sig, w): return w * np.exp(-.5 * (((hh - mu + 12) % 24) - 12) ** 2 / sig ** 2)
pdc = .35 + cg(10, 1.8, .9) + cg(14, 2, .4) + cg(19.5, 2.2, 1.9) + cg(22, 1.5, .4); pdc /= pdc.mean()
psw = .30 + cg(6.5, 1.5, 1.0) + cg(13, 2, .6) + cg(18.5, 2, 1.4) + cg(21, 2, .6); psw /= psw.mean()
sprof = np.maximum(0, np.sin(np.pi * (hh - 6) / 12)) ** 1.3; sprof = sprof / sprof.mean()
wprof = 1 + .35 * np.cos(2 * np.pi * (hh - 20) / 24)
DAMS = np.array([4.0, 3.8, 3.7, 3.6, 3.7, 4.0, 4.6, 5.2, 4.8, 4.0, 3.2, 2.8, 2.6, 2.6, 2.8, 3.4, 4.6, 6.2, 8.2, 9.0, 8.4, 7.0, 5.4, 4.5])
RE_AVG = .35 * (np.sum(LF * CAP) / .90) * RESH   # 35% of energy from RE (given)

SCEN = {
    'Normal':        dict(dT=0, solar=.85, wind=1.0, price=1.0, w=.50),
    'Heat wave':     dict(dT=5, solar=.90, wind=.90, price=1.15, w=.20),
    'Cloud+wind lull': dict(dT=-1, solar=.50, wind=.45, price=1.15, w=.20),
    'Compound stress': dict(dT=4, solar=.55, wind=.40, price=1.35, w=.10),
}


def ar1(r, n, phi, sd):
    x = np.zeros(n); x[0] = r.normal(0, sd)
    for i in range(1, n): x[i] = phi * x[i - 1] + r.normal(0, sd * np.sqrt(1 - phi ** 2))
    return x


def make_day(seed, sc, fsd=.03):
    r = np.random.default_rng(seed)
    T = sc['dT'] + r.normal(0, 1.2); wT = 1 + .022 * T
    sol = np.clip(sc['solar'] + r.normal(0, .05, C), .15, 1.0)
    wnd = np.clip(sc['wind'] * np.exp(r.normal(0, .2, C)), .1, 2)
    De_dc = .7 * (LF * CAP)[:, None] * pdc[None]; De_sw = .3 * (LF * CAP)[:, None] * psw[None]
    shock = r.normal(0, .05, C)[:, None]
    nz = lambda sd: np.array([ar1(r, H, .7, sd) for _ in range(C)])
    Ddc = De_dc * wT * (1 + shock) * (1 + nz(.06)); Dsw = De_sw * wT * (1 + shock) * (1 + nz(.06))
    seq = RE_AVG * .6 / .85; weq = RE_AVG * .4
    RE = seq[:, None] * sprof * sol[:, None] * (1 + nz(.10)) + weq[:, None] * wprof * wnd[:, None] * (1 + nz(.20))
    RE = np.maximum(RE, 0)
    lvl = sc['price'] * (1 + .02 * T) * (1 + .3 * max(0, .85 - sol.mean()) + .25 * max(0, 1 - wnd.mean()))
    dam = np.clip(DAMS * lvl * (1 + ar1(r, H, .6, .06)), 1.5, 10.0)
    # forecasts
    bias = np.array([.14, -.10, .06, -.12, .16])[:, None]
    Fb_dc = De_dc * (1 + bias) * (1 + r.normal(0, .06, (C, H))); Fb_sw = De_sw * (1 + bias) * (1 + r.normal(0, .06, (C, H)))
    wT_h = 1 + .022 * (T + r.normal(0, 1.0))
    Fm_dc = De_dc * wT_h * (1 + r.normal(0, fsd, (C, H))); Fm_sw = De_sw * wT_h * (1 + r.normal(0, fsd, (C, H)))
    REb = seq[:, None] * sprof * .85 + weq[:, None] * wprof
    REm = (seq[:, None] * sprof * (sol[:, None] + r.normal(0, .05, (C, 1))) +
           weq[:, None] * wprof * wnd[:, None] * (1 + r.normal(0, .15, (C, 1))))
    REm = np.maximum(REm, 0)
    dam_b = DAMS.copy(); dam_m = dam * (1 + r.normal(0, .06, H))
    return dict(Ddc=Ddc, Dsw=Dsw, RE=RE, dam=dam, Fb=(Fb_dc, Fb_sw, REb, dam_b), Fm=(Fm_dc, Fm_sw, REm, dam_m))


def price_resp(D, m, win=4):
    Dp = D * m ** (-EPS); out = Dp.copy(); red = PHI * np.maximum(D - Dp, 0)
    for t in range(H):
        if red[t] > 0:
            js = [(t + d) % H for d in range(-win, win + 1) if d != 0]
            w = np.array([max(0, m[t] - m[j]) for j in js])
            if w.sum() > 0: out[js] += red[t] * w / w.sum()
    return out


NV = 13
def lp_solve(c, n, e0, F0, Ddc, Dsw, RE, dam, m, I, mode, DA=None, eterm=None, Fterm=None):
    etaC = ETA_C_OPT; etaP = ETA_CELL * etaC
    Gm = RES * G[c]; Pm = RES * P_ST[c]; Ec = E_ST[c]; bstd = STD_OPT * DCc[c]
    nv = NV * n + 2
    cost = np.zeros(nv); lb = np.zeros(nv); ub = np.full(nv, np.inf)
    Aeq = np.zeros((3 * n, nv)); beq = np.zeros(3 * n)
    Aub = []; bub = []
    for j in range(n):
        o = NV * j; r = 3 * j
        if mode == 'DA': cost[o] = dam[j] + OA_IEX
        cost[o + 1] = 1.2 * dam[j] + .4 + OA_IEX; cost[o + 2] = -.75 * dam[j]
        cost[o + 3] = TAR_D; cost[o + 4] = OA_RE + SELL * dam[j]; cost[o + 6] = DEG_ST
        cost[o + 7] = -(P_DC * m[j] * (1 - HOST) - VOM) - PEN
        cost[o + 9] = -(P_SW * m[j] * (1 - HOST) - VOM - DEG_PK) - PEN
        if mode == 'DA':
            ub[o] = Gm; ub[o + 1] = 0; ub[o + 2] = 0
        else:
            lb[o] = ub[o] = DA[j]; ub[o + 2] = DA[j]
        ub[o + 3] = Gm; ub[o + 4] = RE[j]; ub[o + 5] = Pm; ub[o + 6] = Pm
        ub[o + 7] = min(Ddc[j], RES * DCc[c] * etaC); ub[o + 8] = RES * SWc[c]; ub[o + 9] = Dsw[j]
        lb[o + 10] = min(.15 * Ec, e0); ub[o + 10] = .95 * Ec
        lb[o + 11] = min(.05 * I, F0); ub[o + 11] = I
        for v, cf in ((4, 1), (0, 1), (1, 1), (2, -1), (3, 1)): Aeq[r, o + v] = cf
        Aeq[r, o + 12] = -1; Aeq[r, o + 6] = 1; Aeq[r, o + 7] = -1 / etaC; Aeq[r, o + 5] = -1; Aeq[r, o + 8] = -1
        beq[r] = bstd
        Aeq[r + 1, o + 10] = 1; Aeq[r + 1, o + 5] = -ETA_S; Aeq[r + 1, o + 6] = 1 / ETA_S
        Aeq[r + 2, o + 11] = 1; Aeq[r + 2, o + 8] = -etaP; Aeq[r + 2, o + 9] = 1
        if j == 0:
            beq[r + 1] = (1 - SD_ST) * e0; beq[r + 2] = (1 - SD_P) * F0
        else:
            Aeq[r + 1, o - NV + 10] = -(1 - SD_ST); Aeq[r + 2, o - NV + 11] = -(1 - SD_P)
        row = np.zeros(nv)
        for v, cf in ((4, 1), (0, 1), (1, 1), (2, -1), (3, 1)): row[o + v] = cf
        Aub.append(row.copy()); bub.append(Gm)
        for xf in (.25, .5, .75, 1.0):
            x0 = xf * Gm; rr = np.zeros(nv)
            for v, cf in ((4, 1), (0, 1), (1, 1), (2, -1), (3, 1)): rr[o + v] = K_D * 2 * x0 / Gm * cf
            rr[o + 12] = -1; Aub.append(rr); bub.append(K_D * x0 ** 2 / Gm)
    ol = NV * (n - 1)
    rr = np.zeros(nv); rr[ol + 10] = -1; rr[NV * n] = -1; Aub.append(rr); bub.append(-eterm)
    rr = np.zeros(nv); rr[ol + 11] = -1; rr[NV * n + 1] = -1; Aub.append(rr); bub.append(-Fterm)
    cost[NV * n] = 5.0; cost[NV * n + 1] = 5.0
    bounds = [(lb[i], None if np.isinf(ub[i]) else ub[i]) for i in range(nv)]
    res = linprog(cost, A_ub=np.array(Aub), b_ub=np.array(bub), A_eq=Aeq, b_eq=beq, bounds=bounds, method='highs')
    return res


def base_hour(c, t, e, F, Ddc, Dsw, RE, gbid, I):
    Gm = RES * G[c]; Pm = RES * P_ST[c]; Ec = E_ST[c]
    std = STD_BASE * DCc[c]
    bus_avail = max(0, Gm * (1 - K_D) - std)
    dis_cap = min(Pm, max(0, (e - .15 * Ec) * ETA_S)) if 18 <= t <= 22 else 0
    sreq = min(Ddc, RES * DCc[c])
    need = sreq / eta_phys(sreq / DCc[c]) if sreq > 0 else 0
    from_dis = min(max(0, need - bus_avail), dis_cap)
    served_bus = min(need, bus_avail + from_dis)
    sdc = sreq * served_bus / need if need > 0 else 0
    rem = bus_avail - (served_bus - from_dis)
    ch = 0
    if 1 <= t <= 5: ch = max(0, min(Pm, (.9 * Ec - e) / ETA_S, rem))
    rem -= ch
    pg = min(RES * SWc[c], max(0, rem))
    etaP = ETA_CELL * eta_phys(pg / SWc[c]) if pg > 0 else ETA_CELL * ETA_PK
    pc = max(0, min(pg, (I - F) / etaP)); rem -= pc
    sw = max(0, min(Dsw, (1 - SD_P) * F + etaP * pc - .05 * I))
    bus_used = std + (served_bus - from_dis) + ch + pc
    Gm0 = Gm
    x = Gm0 / (2 * K_D) * (1 - np.sqrt(max(0, 1 - 4 * K_D * bus_used / Gm0))) if bus_used > 0 else 0
    ru = min(RE, x); rm = x - ru
    gi = min(gbid, rm); dn = gbid - gi; gd = rm - gi
    e2 = (1 - SD_ST) * e + ETA_S * ch - from_dis / ETA_S
    F2 = (1 - SD_P) * F + etaP * pc - sw
    ld = x - bus_used + 0.0  # distribution loss (std treated as bus load)
    return dict(ru=ru, gI=gbid, up=0.0, dn=dn, gD=gd, ch=ch, dis=from_dis, sdc=sdc, pc=pc, sw=sw, e=e2, F=F2,
                x=x, ld=ld, std=std, etaP=etaP, etaC=(sdc / (served_bus) if served_bus > 0 else 1))


KEYS = ['rev', 'delv', 'dem0', 'demP', 'x', 'loss', 'l_dist', 'l_conv', 'l_std', 'l_st', 'l_pk', 'cost_e', 'cost_dev',
        'deg', 'host_vom', 'ru', 'REtot', 'dis', 'sdc', 'sw', 'unmetpk', 'mapeD', 'mapeRE']
def zero(): return {k: np.zeros(C) for k in KEYS}


def run_day(policy, day, state, Ivec, mpost_in=None, pricing=False, fsd_note=None):
    """policy: base | mlrule | opt | full ; returns per-cluster metric dict"""
    out = zero(); Fb, Fm = day['Fb'], day['Fm']
    Ddc0, Dsw0, RE, dam = day['Ddc'], day['Dsw'], day['RE'], day['dam']
    use_ml = policy != 'base'
    Fdc, Fsw, FRE, Fdam = Fm if use_ml else Fb
    mat = np.ones((C, H))
    for c in range(C):
        e, F = state['e'][c], state['F'][c]; I = Ivec[c]
        Gm = RES * G[c]
        mpost = np.ones(H)
        etaC = ETA_C_OPT
        if policy in ('opt', 'full'):
            Dp_dc, Dp_sw = Fdc[c].copy(), Fsw[c].copy()
            eterm, Fterm = .5 * E_ST[c], .8 * I
            r1 = lp_solve(c, H, e, F, Dp_dc, Dp_sw, FRE[c], Fdam, mpost, I, 'DA', eterm=eterm, Fterm=Fterm)
            if policy == 'full' and r1.status == 0:
                mu = -r1.eqlin.marginals[0::3]
                med = np.median(mu)
                mpost = np.clip(1 + .25 * (mu / med - 1), .8, 1.2)
                Dp_dc = price_resp(Fdc[c], mpost); Dp_sw = price_resp(Fsw[c], mpost)
            r2 = lp_solve(c, H, e, F, Dp_dc, Dp_sw, FRE[c], Fdam, mpost, I, 'DA', eterm=eterm, Fterm=Fterm)
            if r2.status != 0: r2 = r1
            DA = np.array([r2.x[NV * j] for j in range(H)]) if r2.status == 0 else np.zeros(H)
        else:
            # baseline / ML-rule bidding: 50% of forecast residual bus load via IEX
            xh = (Fdc[c] / .93 + Fsw[c] / .93 + STD_BASE * DCc[c]) / (1 - .02) - FRE[c]
            DA = np.minimum(.5 * np.maximum(xh, 0), .5 * Gm)
        mat[c] = mpost
        # actual demand under posted prices
        Adc = price_resp(Ddc0[c], mpost) if policy == 'full' else Ddc0[c]
        Asw = price_resp(Dsw0[c], mpost) if policy == 'full' else Dsw0[c]
        corrD = 1.0; corrR = 1.0
        for t in range(H):
            if policy in ('opt', 'full'):
                n = H - t
                dec = .8 ** np.arange(n)
                fd = (Dp_dc[t:] * (1 + (corrD - 1) * dec)); fs = (Dp_sw[t:] * (1 + (corrD - 1) * dec))
                fr = FRE[c][t:] * (1 + (corrR - 1) * .85 ** np.arange(n))
                fd = fd.copy(); fs = fs.copy(); fr = fr.copy()
                fd[0], fs[0], fr[0] = Adc[t], Asw[t], RE[c][t]
                res = lp_solve(c, n, e, F, fd, fs, fr, dam[t:], mpost[t:], I, 'RT', DA=DA[t:], eterm=.45 * E_ST[c], Fterm=.75 * I)
                if res.status == 0:
                    v = res.x[:NV]
                    gI, up, dn, gD, ru, ch, dis, sdc, pc, sw, e2, F2, ld = v
                    etaP = ETA_CELL * etaC
                    x = ru + gI + up - dn + gD
                    fl = dict(ru=ru, gI=gI, up=up, dn=dn, gD=gD, ch=ch, dis=dis, sdc=sdc, pc=pc, sw=sw, e=e2, F=F2, x=x,
                              ld=ld, std=STD_OPT * DCc[c], etaP=etaP, etaC=etaC)
                else:
                    fl = base_hour(c, t, e, F, Adc[t], Asw[t], RE[c][t], DA[t], I); fl['up'] = 0.0
            else:
                fl = base_hour(c, t, e, F, Adc[t], Asw[t], RE[c][t], DA[t], I)
            e, F = fl['e'], fl['F']
            # RT correction factors
            if Fdc[c][t] > .5: corrD = .5 + .5 * (Ddc0[c][t] + Dsw0[c][t]) / (Fdc[c][t] + Fsw[c][t])
            corrD = float(np.clip(corrD, .7, 1.3))
            if FRE[c][t] > .3: corrR = float(np.clip(.4 + .6 * RE[c][t] / FRE[c][t], .4, 1.8))
            # ---- accounting ----
            m = mpost[t]; d_ = fl['sdc'] + fl['sw']
            rev = P_DC * m * fl['sdc'] + P_SW * m * fl['sw']
            netg = fl['gI'] + fl['up'] - fl['dn']
            buy_dev = 1.2 * dam[t] + .4; sell_dev = .75 * dam[t]
            out['rev'][c] += rev; out['delv'][c] += d_
            out['dem0'][c] += Ddc0[c][t] + Dsw0[c][t]; out['demP'][c] += Adc[t] + Asw[t]
            out['x'][c] += fl['x']
            # DA part cost uses actual cleared price
            out['cost_e'][c] += DA[t] * (dam[t] + OA_IEX) + fl['gD'] * TAR_D + RE[c][t] * PPA + fl['ru'] * OA_RE - (RE[c][t] - fl['ru']) * SELL * dam[t]
            if policy in ('opt', 'full'):
                out['cost_dev'][c] += fl['up'] * (buy_dev + OA_IEX) - fl['dn'] * sell_dev
            else:
                out['cost_dev'][c] += -fl['dn'] * sell_dev
            out['deg'][c] += fl['dis'] * DEG_ST + fl['sw'] * DEG_PK
            out['host_vom'][c] += HOST * rev + VOM * d_
            out['ru'][c] += fl['ru']; out['REtot'][c] += RE[c][t]; out['dis'][c] += fl['dis']
            out['sdc'][c] += fl['sdc']; out['sw'][c] += fl['sw']
            # loss components
            etaPk = fl['etaP']
            l_dist = fl['x'] - ((fl['x']) - fl['ld']) if False else fl['ld']
            out['l_dist'][c] += l_dist
            out['l_std'][c] += fl['std']
            out['l_conv'][c] += fl['sdc'] * (1 / fl['etaC'] - 1) if fl['etaC'] > 0 else 0
            out['l_pk'][c] += fl['pc'] * (1 - etaPk)
            out['l_st'][c] += fl['ch'] * (1 - ETA_S) + fl['dis'] * (1 / ETA_S - 1)
        state['e'][c], state['F'][c] = e, F
    return out, mat


def add(a, b):
    for k in KEYS: a[k] += b[k]


def run_policy(policy, ndays=8, fsd=.03, Ivec=None, seed0=1000):
    Ivec = I_PK0 if Ivec is None else Ivec
    res = {}
    for si, (sn, sc) in enumerate(SCEN.items()):
        state = dict(e=.5 * E_ST.copy(), F=.8 * Ivec.copy())
        tot = zero(); e_start = state['e'].copy(); F_start = state['F'].copy()
        for d in range(ndays):
            day = make_day(seed0 + 100 * si + d, sc, fsd)
            o, _ = run_day(policy, day, state, Ivec)
            add(tot, o)
            # forecast MAPE (ML or base) on total demand
            Fdc, Fsw, FRE, _ = day['Fm'] if policy != 'base' else day['Fb']
            act = day['Ddc'] + day['Dsw']; fc = Fdc + Fsw
            tot['mapeD'] += np.abs(fc - act).sum(1) / act.sum(1)
            tot['mapeRE'] += np.abs(FRE - day['RE']).sum(1) / day['RE'].sum(1)
        dE = (state['e'] - e_start) ; dF = (state['F'] - F_start)
        tot['loss'] = tot['x'] - tot['delv'] - dE - dF
        for k in tot: tot[k] = tot[k] / ndays
        res[sn] = tot
    return res


def summarize(res):
    """blend scenarios into annual weights; returns dict of cluster-level and portfolio KPIs"""
    w = {k: v['w'] for k, v in SCEN.items()}
    agg = {k: sum(w[s] * res[s][k] for s in res) for k in KEYS}
    return agg


if __name__ == '__main__':
    t0 = time.time()
    r = run_policy('base', ndays=4)
    a = summarize(r)
    print('baseline reliability(cons)', a['delv'].sum() / a['dem0'].sum(), 'loss%', a['loss'].sum() / a['x'].sum())
    for s in r: print(s, r[s]['delv'].sum() / r[s]['dem0'].sum(), r[s]['loss'].sum() / r[s]['x'].sum())
    print('time', time.time() - t0)
