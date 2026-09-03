import polars as pl
import pandas as pd


def get_oef_tickers() -> list[str]:
    url = "https://www.ishares.com/us/products/239723/ishares-s-p-100-etf/latest-holdings.csv"

    return (
        pl.read_csv(url, skip_rows=9)
        .filter(pl.col("Asset Class") == "Equity")
        .select("Ticker")
        .drop_nulls()
        .to_series()
        .to_list()
    )


tickers = get_oef_tickers()

print(tickers)
print(len(tickers))

from vnstock import Listing

def get_vn30_tickers():
    # Thư viện sẽ tự động gọi đến API đang hoạt động tốt nhất (TCBS/SSI/VNDirect...)
    df = Listing().symbols_by_group("VN30")
    
    # Lấy cột mã chứng khoán (tùy bản cập nhật có thể là 'ticker' hoặc 'symbol')
    col_name = "ticker" if "ticker" in df.columns else "symbol"
    return df[col_name].tolist()

print(get_vn30_tickers())