# Signal Analyzer - web deployment

Web (Streamlit) frontend for the SIH26147 signal analyzer. Same backend as the
desktop app, zero changes to any signal-processing file.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy publicly (Streamlit Community Cloud, free)
1. Push this folder to a public GitHub repo.
2. Go to https://share.streamlit.io -> "New app".
3. Point it at the repo, branch, and `app.py`.
4. Deploy. You get a public URL like `https://<name>.streamlit.app`.

Sample .iq files to try are in `sample_signals/`.
