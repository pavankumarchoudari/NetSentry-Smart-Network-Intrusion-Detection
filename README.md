# GSA-DT Streamlit deployment

## Files
- `app.py`: Streamlit inference interface
- `requirements.txt`: pinned Python dependencies
- `artifacts/`: the eight trained model and preprocessing artifacts

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Input format
Upload a CSV/BINETFLOW file with `SrcAddr` and `DstAddr`. For closest compatibility with the notebook, include `Dur`, `TotPkts`, `TotBytes`, `SrcBytes`, `Sport`, `Dport`, `Proto`, and `State`.

This initial deployment supports NetFlow/BINETFLOW only. PCAP ingestion is not enabled until a packet-to-flow extractor can be validated against the notebook's feature definitions.
