import os
import yfinance as yf

project_folder = "/content/drive/MyDrive/Colab Notebooks/market_ml_project"

symbols = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "SENSEX": "^BSESN"
}

for name, symbol in symbols.items():

    df = yf.download(
        symbol,
        start="2008-01-01",
        auto_adjust=True,
        progress=False
    )

    # Flatten MultiIndex columns (optional but recommended)
    if df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)

    filepath = os.path.join(project_folder, f"{name.lower()}_daily.csv")

    df.to_csv(filepath)

    print(f"{name} saved successfully.")
    print("Latest date:", df.index.max())
    print("Saved to:", filepath)