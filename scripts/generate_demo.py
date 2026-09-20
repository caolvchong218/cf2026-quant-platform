"""Generate redistributable SYNTHETIC data. Never a substitute for empirical data."""
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    root=Path(__file__).resolve().parents[1]/"data/sample"
    root.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20260921)
    dates=pd.bdate_range("2024-01-02",periods=120)
    rows=[]
    for k in range(12):
        prices=10*np.exp(np.cumsum(rng.normal(.0002,.012,120)))
        for j,date in enumerate(dates):
            opening=prices[j]*np.exp(rng.normal(0,.002))
            rows.append(dict(date=date,asset=f"SYN{k:02d}",open=opening,high=max(opening,prices[j])*1.01,
                low=min(opening,prices[j])*.99,close=prices[j],volume=100000,
                raw_open=opening,raw_close=prices[j],adj_factor=1.,up_limit=opening*1.1,down_limit=opening*.9))
    pd.DataFrame(rows).sort_values(["date","asset"]).to_csv(root/"market.csv",index=False,float_format="%.8g")
    pd.DataFrame({"date":dates}).to_csv(root/"calendar.csv",index=False)
    (root/"README.md").write_text("固定种子 20260921 生成的合成数据，仅用于无凭据演示、测试和安装验收。不是行情数据，不能用于声称投资表现。\n",encoding="utf-8")

if __name__=="__main__":main()
