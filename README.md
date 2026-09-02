# NIFTY + BANKNIFTY Early Detector Dashboard v1.0

Separate Streamlit/Railway dashboard for the index collector.

## Tabs
- State + Conviction
- Index Snapshot
- Frozen Options
- Aggression
- OI 50%

The score mirrors the research framework used in stock v2.1:
- Option basket: 0–3
- Aggression/persistence: 0–2
- Price response: 0–2
- Futures OI positioning: 0–2
- Cumulative OI acceleration: 0–1

No stock tables are modified.

## Railway variable
`NEON_DATABASE_URL`

## Start command
`streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
