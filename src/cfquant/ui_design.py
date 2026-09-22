"""Shared visual language for the research workbench; no numerical logic."""
import html
import plotly.io as pio
import plotly.graph_objects as go
import streamlit as st

COLORS = ['#087f8c', '#4064b0', '#b58236', '#9c628f', '#7d899c', '#ca665e']


def install():
    pio.templates['qingxu'] = go.layout.Template(layout=dict(
        colorway=COLORS, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Microsoft YaHei, sans-serif', color='#324458', size=13),
        hovermode='x unified', hoverlabel=dict(bgcolor='white', font_size=13),
        margin=dict(l=12, r=16, t=40, b=16),
        xaxis=dict(showgrid=False, zeroline=False, linecolor='#dce3ed', title_text=''),
        yaxis=dict(gridcolor='#e6ebf2', zerolinecolor='#d3dce6', title_text=''),
        legend=dict(orientation='h', y=1.13, title_text='', font_size=12),
    ))
    pio.templates.default = 'plotly_white+qingxu'
    st.markdown('''<style>
    @import url('');
    .stApp {background:#f4f6fa;color:#24394e;}
    .block-container {max-width:1520px;padding-top:4.3rem;padding-bottom:3rem;}
    h1,h2,h3 {color:#152c43;letter-spacing:-.025em;}
    h1 {font-size:2.35rem!important;font-weight:750!important;}
    h2 {font-size:1.5rem!important;} h3 {font-size:1.18rem!important;}
    [data-testid="stSidebar"] {background:#10263b;border-right:1px solid #20394e;}
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {color:#d2e0eb;}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stRadio"] p {color:#d2e0eb!important;}
    [data-testid="stSidebar"] [role="radiogroup"] {gap:3px;}
    [data-testid="stSidebar"] [role="radiogroup"] label {padding:9px 12px;border-radius:7px;margin:0;}
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {background:#1b374f;}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background:#1d4555;box-shadow:inset 3px 0 #45c2b0;}
    [data-testid="stSidebar"] [data-baseweb="select"] * {color:#24394e;}
    [data-testid="stMetric"] {background:#fff;border:1px solid #dfe6ef;border-radius:12px;padding:18px 21px;box-shadow:0 3px 12px #182f4605;}
    [data-testid="stMetricLabel"] {font-size:13px;color:#61758a;}
    [data-testid="stMetricValue"] {font-variant-numeric:tabular-nums;font-size:1.85rem;font-weight:650;letter-spacing:-.02em;}
    [data-testid="stPlotlyChart"] {background:white;border:1px solid #e0e6ef;border-radius:12px;padding:12px 12px 4px;}
    [data-testid="stForm"], [data-testid="stExpander"] {background:#fff;border:1px solid #dfe6ef;border-radius:10px;}
    [data-testid="stTabs"] [role="tablist"] {gap:24px;border-bottom:1px solid #dce5ed;}
    [data-testid="stTabs"] [role="tab"] {padding-bottom:12px;}
    .stButton button,.stDownloadButton button {border-radius:7px;min-height:40px;transition:background-color 140ms ease-out;}
    .stButton button:focus-visible,.stDownloadButton button:focus-visible {outline:3px solid #54b4c1;outline-offset:2px;}
    .qx-eyebrow {font-size:11px;color:#08818c;letter-spacing:2.5px;font-weight:750;margin:0 0 9px;}
    .qx-intro {font-size:16px;color:#62768c;line-height:1.7;margin:-7px 0 24px;max-width:1060px;}
    .qx-brand {font-size:28px;letter-spacing:2px;font-weight:750;color:#f0f6fb;margin:0 0 4px;}
    .qx-subbrand {font-size:10px;letter-spacing:2px;color:#85b4ca;margin-bottom:24px;}
    .qx-note {border-left:3px solid #168b94;padding:11px 17px;background:#eaf3f5;color:#38576c;font-size:14px;line-height:1.7;margin:12px 0 20px;}
    .qx-step {color:#168b94;font-size:12px;letter-spacing:1px;font-weight:700;margin-top:8px;}
    @media(max-width:850px) {.block-container{padding:4rem 1rem 1.4rem;}h1{font-size:1.8rem!important;}[data-testid="stMetricValue"]{font-size:1.5rem;}}
    @media(prefers-reduced-motion:reduce) {*{transition:none!important;}}
    </style>'''.replace("    @import url('');\n", ''), unsafe_allow_html=True)


def heading(title, subtitle, eyebrow='QINGXU / RESEARCH WORKBENCH'):
    st.markdown(f'<div class="qx-eyebrow">{html.escape(eyebrow)}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="qx-intro">{html.escape(subtitle)}</div>', unsafe_allow_html=True)


def note(text):
    st.markdown(f'<div class="qx-note">{html.escape(text)}</div>', unsafe_allow_html=True)


def chart(fig, height=380, key=None):
    fig.update_layout(height=height, paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
                      legend_title_text='', font=dict(family='Microsoft YaHei, sans-serif',size=13,color='#385269'))
    fig.update_yaxes(gridcolor='#e6edf3')
    st.plotly_chart(fig, width='stretch', key=key, theme=None, config={'displaylogo':False, 'scrollZoom':False,
        'toImageButtonOptions':{'format':'png','scale':2}})
