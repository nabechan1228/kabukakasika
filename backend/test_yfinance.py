try:
    from curl_cffi import requests as requests_cffi
    print("curl_cffi is available!")
    
    # Test session creation
    session = requests_cffi.Session(impersonate="chrome")
    print("Created Session successfully")
    
    # Try custom request with timeout override
    orig_request = session.request
    def new_request(*args, **kwargs):
        if 'timeout' not in kwargs:
            kwargs['timeout'] = 5
        return orig_request(*args, **kwargs)
    session.request = new_request
    print("Wrapped Session successfully")
    
    import yfinance as yf
    ticker = yf.Ticker("7203.T", session=session)
    print("Created yfinance Ticker with session")
    hist = ticker.history(period="1d")
    print("Successfully fetched history using session:", hist.shape)
except Exception as e:
    print("Error:", e)
