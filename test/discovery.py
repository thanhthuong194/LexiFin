import requests


def get_us_company_tickers() -> list[str]:
  url = (
      "https://www.blackrock.com/varnish-api/blk-one01-product-data/product-data/api/v1/get-fund-document?"
      "appType=PRODUCT_PAGE&appSubType=ISHARES&targetSite=us-ishares&locale=en_US&"
      "portfolioId=239723&userType=individual&asOfDate=20260827&component=holdings"
  )

  data = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).json()

  # Đệ quy tìm mảng chứa danh sách cổ phiếu bên trong JSON
  def find_holdings(obj):
    if isinstance(obj, list):
      if obj and isinstance(obj[0], dict) and ("ticker" in obj[0] or "symbol" in obj[0]):
        return obj
      for item in obj:
        if (res := find_holdings(item)):
          return res
    elif isinstance(obj, dict):
      for v in obj.values():
        if (res := find_holdings(v)):
          return res
    return []

  holdings = find_holdings(data)
  tickers = [h.get("ticker") or h.get("symbol") for h in holdings]
  
  return [t.strip() for t in tickers if t and len(t.strip()) <= 6][:100]


if __name__ == "__main__":
  print(get_us_company_tickers())