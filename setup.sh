mkdir -p ~/.streamlit/

echo "[theme]
base='dark'
textColor='#f1e6e6'
font='serif'
[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
