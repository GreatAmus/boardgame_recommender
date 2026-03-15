import os, urllib.request

def ensure_file(url: str, local_path: str):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if not os.path.exists(local_path):
        urllib.request.urlretrieve(url, local_path)
