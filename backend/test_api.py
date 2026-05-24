import urllib.request
import json
import time

def call_api(url):
    print(f"Calling: {url}")
    t0 = time.time()
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as response:
            res_body = response.read().decode("utf-8")
            print(f"Success! Status: {response.status}, Time: {time.time() - t0:.2f}s")
            data = json.loads(res_body)
            if isinstance(data, list):
                print(f"Returned list with {len(data)} items")
            else:
                print(f"Keys: {list(data.keys())}")
    except Exception as e:
        print(f"Failed! Time: {time.time() - t0:.2f}s, Error: {e}")

if __name__ == "__main__":
    codes = ["7203", "6724", "1301", "8306", "9984"]
    for code in codes:
        print(f"\n=== Code: {code} ===")
        call_api(f"http://localhost:8000/api/stock/{code}")
        call_api(f"http://localhost:8000/api/info/{code}")
        call_api(f"http://localhost:8000/api/predict/{code}")
