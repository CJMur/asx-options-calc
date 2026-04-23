from playwright.sync_api import sync_playwright

# --- CHANGE THIS TO YOUR ACTUAL STREAMLIT APP URL ---
APP_URL = "https://asx-options-calc-myfvbqtgkfpzuwurid6jce.streamlit.app/"

def ping_streamlit():
    print(f"Spinning up Headless Chrome to visit {APP_URL}...")
    with sync_playwright() as p:
        # Launch Chrome invisibly
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        # Go to the app
        page.goto(APP_URL)
        
        # Wait 10 seconds. This is crucial! It gives Streamlit time to 
        # establish the WebSocket connection and register the "human" visit.
        page.wait_for_timeout(10000) 
        
        print("Success! App JavaScript rendered. Sleep timer reset.")
        browser.close()

if __name__ == "__main__":
    ping_streamlit()
