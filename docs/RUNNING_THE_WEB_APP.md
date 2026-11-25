# Running the classifier with Flask

The Flask server keeps your Groq API key on your computer and runs the same Python classifier you use for the evaluation. The web page never sees the key.

## Files
| File | What it does |
|---|---|
| `server.py` | The Flask server. It serves the page and answers `/api/classify`, `/api/baselines` and `/api/chat`. |
| `web/classifier_flask.html` | The web page (no API key inside). |

## One-time setup
1. Get a new key at https://console.groq.com/keys and delete the old key there. The old one was written into an older page, which is kept off GitHub for that reason.
2. Install the packages from this folder:
   ```
   py -m pip install -r requirements.txt
   ```

## Each time you run it
1. Open a terminal in the project folder.
2. Set your key. Type it here, never in a file:
   ```
   set GROQ_API_KEY=your_new_key_here
   ```
   In PowerShell, use `$env:GROQ_API_KEY="your_new_key_here"` instead.
3. Start the server:
   ```
   py server.py
   ```
4. Open **http://localhost:5000** in your browser.
5. When you're done, click in the terminal and press **Ctrl+C** to stop the server.

**Shorter address (optional):** start it with `py server.py --port 80` and open **http://127.0.0.1** (no `:5000`). To use a name such as **http://heatstrokeclassifier/** instead, add the line `127.0.0.1 heatstrokeclassifier heatstrokeclassifier.localhost` to the Windows hosts file (`C:\Windows\System32\drivers\etc\hosts`, edited as administrator).

**On a phone (same Wi-Fi):** `py server.py --port 80 --phone` prints the address to type on the phone. Anyone on that Wi-Fi can then use the page, so only do this on a network you trust.

The key only lasts while that terminal window is open. Repeat step 2 each time you open a new terminal.

## After re-running the evaluation
Rebuild the page's Results, Mistakes and comparison data from the new CSVs:
```
py build_results_into_html.py web/classifier_flask.html
```
Once the expert has filled in `rater_sheet.csv`, add `--rater rater_sheet.csv` to show the kappa score.

## At the science fair
- **Judge tour:** click "Judge tour" at the top. It turns on demo mode and Calm display, walks through 7 steps, and puts both back when it ends.
- **Demo mode** still works. Tick the box and the page replays saved answers with no internet and no key.
- **Stress test:** the AI column needs live mode (key + internet). In demo mode only the no-AI methods run on the changed versions.
- The server only accepts connections from your own computer. Run the demo on the laptop that is running the server.

## If something goes wrong
| What you see | Fix |
|---|---|
| "The server has no Groq API key" | Stop the server (Ctrl+C), do step 2, then start it again. |
| "Is server.py running?" | The terminal with the server was closed. Start it again with `py server.py`. |
| "No module named flask" | Run `py -m pip install -r requirements.txt`. |
| HTTP 401 from Groq | The key is wrong or was deleted. Make a new one. |
| HTTP 429 from Groq | You hit the rate limit. The server waits and retries on its own. Give it a minute. |
