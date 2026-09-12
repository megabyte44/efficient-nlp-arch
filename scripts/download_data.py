"""Downloads the tiny-shakespeare text file used for fast local (CPU) iteration."""

import os
import urllib.request

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tinyshakespeare.txt")


def main():
    out_path = os.path.abspath(OUT_PATH)
    if os.path.exists(out_path):
        print(f"already present: {out_path}")
        return
    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, out_path)
    size = os.path.getsize(out_path)
    print(f"saved {size} bytes to {out_path}")


if __name__ == "__main__":
    main()
