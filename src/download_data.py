import os
import requests
import pandas as pd

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
os.makedirs(RAW_DATA_DIR, exist_ok=True)

RESULTS_URLS = [
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
    "https://raw.githubusercontent.com/martj42/international_results/main/results.csv"
]

FIFA_RANKING_URLS = [
    "https://raw.githubusercontent.com/prasertcbs/basic-dataset/master/fifa_ranking.csv",
    "https://raw.githubusercontent.com/tadhgfitzgerald/fifa_ranking/master/fifa_ranking.csv",
    "https://raw.githubusercontent.com/martj42/international_results/master/fifa_ranking.csv"
]

def download_from_sources(urls, target_name):
    target_path = os.path.join(RAW_DATA_DIR, target_name)
    for url in urls:
        print(f"Trying to download {url}...")
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(target_path, 'wb') as f:
                    f.write(response.content)
                print(f"Successfully downloaded {target_path} ({os.path.getsize(target_path)} bytes)")
                return target_path
        except Exception as e:
            print(f"Error fetching {url}: {e}")
    print(f"Failed to download {target_name} from all sources.")
    return None

def main():
    results_path = download_from_sources(RESULTS_URLS, 'results.csv')
    fifa_path = download_from_sources(FIFA_RANKING_URLS, 'fifa_ranking.csv')

    if results_path and os.path.exists(results_path):
        df_res = pd.read_csv(results_path)
        print(f"\n--- MATCHES DATASET SUMMARY ---")
        print(f"Rows: {len(df_res)}")
        print(f"Date range: {df_res['date'].min()} to {df_res['date'].max()}")
        print(f"Columns: {df_res.columns.tolist()}")
        print(df_res.head(3))

    if fifa_path and os.path.exists(fifa_path):
        df_fifa = pd.read_csv(fifa_path)
        print(f"\n--- FIFA RANKINGS DATASET SUMMARY ---")
        print(f"Rows: {len(df_fifa)}")
        print(f"Columns: {df_fifa.columns.tolist()}")
        print(df_fifa.head(3))

if __name__ == "__main__":
    main()
