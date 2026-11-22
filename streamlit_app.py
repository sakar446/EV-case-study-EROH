"""Thin Streamlit front-end that reuses ev_sim.py unchanged.
Keep ev_sim.py in the same folder. run_all.py is a batch script (it runs on import and
writes res.pkl), so do NOT import it; its logic is reproduced in run_comparison() below."""
import sys
sys.argv = ['x', '0.139', '0.045']      # ev_sim reads ALPHA and A_PART from argv at import time
import numpy as np
import ev_sim as S


def run_comparison(ndays=6, fsd=0.03, gscale=1.0):
    S.G = S.ALPHA * S.CAP                                   # reset layout
    base = S.run_policy('base', ndays=ndays)
    ab = S.summarize(base)
    gross_b = (ab['rev'] - ab['cost_e'] - ab['cost_dev'] - ab['deg'] - ab['host_vom']).sum()
    fix_day = gross_b - 0.125 * ab['rev'].sum()             # fixed opex calibrated to 12.5% baseline margin
    S.G = gscale * S.ALPHA * S.CAP
    full = S.run_policy('full', ndays=ndays, fsd=fsd)
    S.G = S.ALPHA * S.CAP
    return S.summarize(base), S.summarize(full), fix_day


def kpi(a, fix_day):
    gross = a['rev'] - a['cost_e'] - a['cost_dev'] - a['deg'] - a['host_vom']
    return dict(Reliability=a['delv'].sum() / a['dem0'].sum() * 100,
                Losses=a['loss'].sum() / a['x'].sum() * 100,
                Margin=(gross.sum() - fix_day) / a['rev'].sum() * 100,
                EBITDA_cr=(gross.sum() - fix_day) * 365 / 1e4)


def main():
    import streamlit as st, pandas as pd
    st.title("EV energy allocation framework")
    ndays = st.sidebar.slider("Days per scenario", 2, 20, 6)
    fsd = st.sidebar.slider("Forecast noise", 0.01, 0.20, 0.03, 0.01)
    gscale = st.sidebar.slider("Grid capacity multiplier", 1.0, 1.4, 1.0, 0.05)
    if st.sidebar.button("Run", type="primary"):
        with st.spinner("Simulating..."):
            ab, af, fix = run_comparison(ndays, fsd, gscale)
        df = pd.DataFrame({'Baseline': kpi(ab, fix), 'Framework': kpi(af, fix)}).round(2)
        st.dataframe(df)
        st.bar_chart(pd.DataFrame({'Baseline': ab['delv'] / ab['dem0'] * 100,
                                   'Framework': af['delv'] / af['dem0'] * 100}, index=S.CL))
        st.caption("Reliability by cluster (%)")


if __name__ == '__main__':
    main()
