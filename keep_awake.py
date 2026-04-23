from playwright.sync_api import sync_playwright

# --- CHANGE THIS TO YOUR ACTUAL STREAMLIT APP URL ---
APP_URL = "https://your-app-name.streamlit.app"

def ping_streamlit():
    print(f"Spinning up Headless Chrome to visit {APP_URL}...")
    with sync_playwright() as p:
        # Launch Chrome invisibly
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        # 1. Go to the app
        page.goto(APP_URL)
        
        try:
            # 2. Wait for the main title to prove the JavaScript rendered
            page.wait_for_selector("text=TradersCircle Options Calculator", timeout=15000)
            print("App loaded successfully. Simulating human interaction...")
            
            # 3. Physically click the "LOAD OPTIONS" button to trigger the WebSocket
            page.click("text=LOAD OPTIONS", timeout=5000)
            print("Button clicked. Sending WebSocket signal to Streamlit server...")
            
            # 4. Wait 5 seconds for the server to process the math/data
            page.wait_for_timeout(5000) 
            
            print("Success! 'Human' activity verified. 12-hour sleep timer reset.")
        except Exception as e:
            print(f"Interaction failed, but site was visited: {e}")
            
        browser.close()

if __name__ == "__main__":
    ping_streamlit()
