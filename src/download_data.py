import io
import zipfile
from pathlib import Path
import requests
import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
OUT = Path("data/spam.csv")

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading UCI SMS Spam Collection...")
    r = requests.get(URL, timeout=30)
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        names = z.namelist()
        txt_name = next(n for n in names if n.endswith("SMSSpamCollection"))
        with z.open(txt_name) as f:
            df = pd.read_csv(f, sep="\t", header=None, names=["label", "message"])

    df = df.dropna().drop_duplicates().reset_index(drop=True)
    df["label"] = df["label"].str.lower().map({"ham": "ham", "spam": "spam"})
    df = df.dropna(subset=["label", "message"])
    df.to_csv(OUT, index=False)

    print(f"Saved {len(df):,} messages to {OUT}")
    print(df["label"].value_counts())

if __name__ == "__main__":
    main()
