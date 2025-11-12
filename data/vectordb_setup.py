import os
import logging
import uuid
from typing import Dict, Any, Optional, List

import pyarrow as pa
import lancedb
from openai import OpenAI
from pathlib import Path

# -------------------- Config --------------------
logger = logging.getLogger(__name__)
EMBED_DIM = 1536
TABLE_NAME = "documents"
ALLOWED_META_FIELDS = ["company", "ticker", "sector", "type", "index", "topic"]

# -------------------- Helpers --------------------
def _data_dir() -> Path:
    p = Path(__file__).parent.parent / "data"
    p.mkdir(parents=True, exist_ok=True)
    return p

def _connect_db() -> "lancedb.DBConnection":
    return lancedb.connect(str(_data_dir()))

def _schema() -> pa.Schema:
    return pa.schema([
        pa.field("id", pa.string(), nullable=False),
        pa.field("text", pa.large_string()),
        pa.field("embedding", pa.list_(pa.float32(), EMBED_DIM), nullable=False),
        pa.field("metadata", pa.struct([
            pa.field("company", pa.string()),
            pa.field("ticker", pa.string()),
            pa.field("sector", pa.string()),
            pa.field("type", pa.string()),
            pa.field("index", pa.string()),
            pa.field("topic", pa.string()),
        ])),
    ])

def _schemas_equal(a: pa.Schema, b: pa.Schema) -> bool:
    # strict compare names & types (ignore nullability/metadata), including nested struct fields
    if len(a) != len(b):
        return False
    for af, bf in zip(a, b):
        if af.name != bf.name:
            return False
        if not _types_equal(af.type, bf.type):
            return False
    return True

def _types_equal(a: pa.DataType, b: pa.DataType) -> bool:
    if pa.types.is_struct(a) and pa.types.is_struct(b):
        if len(a) != len(b):
            return False
        return all(
            af.name == bf.name and _types_equal(af.type, bf.type)
            for af, bf in zip(a, b)
        )
    if pa.types.is_fixed_size_list(a) and pa.types.is_fixed_size_list(b):
        return a.list_size == b.list_size and _types_equal(a.value_type, b.value_type)
    if pa.types.is_list(a) and pa.types.is_list(b):
        return _types_equal(a.value_type, b.value_type)
    return a == b

def _ensure_table(db: "lancedb.DBConnection", recreate_if_mismatch: bool = True):
    desired = _schema()
    names = set(db.table_names())
    if TABLE_NAME in names:
        table = db.open_table(TABLE_NAME)
        live = table.schema
        if not _schemas_equal(live, desired):
            msg = f"Schema mismatch. Live:\n{live}\nWanted:\n{desired}"
            if recreate_if_mismatch:
                logger.warning(msg + "\nRecreating table…")
                db.drop_table(TABLE_NAME)
                return db.create_table(TABLE_NAME, schema=desired)
            else:
                raise ValueError(msg)
        return table
    else:
        return db.create_table(TABLE_NAME, schema=desired)

def _clean_metadata(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    meta = meta or {}
    # Only keep allowed keys; set missing to None
    return {k: meta.get(k) for k in ALLOWED_META_FIELDS}

def _embed_text(text: str) -> List[float]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY env var is not set.")
    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    return resp.data[0].embedding

# -------------------- Public API --------------------
def add_document_to_lancedb(text: str, metadata: Dict[str, Any] = None) -> bool:
    """Add a document to LanceDB for RAG."""
    try:
        db = _connect_db()
        table = _ensure_table(db, recreate_if_mismatch=True)

        emb = _embed_text(text)
        if len(emb) != EMBED_DIM:
            raise ValueError(f"Embedding length {len(emb)} != {EMBED_DIM}")

        doc_data = {
            "id": str(uuid.uuid4()),
            "text": text,
            "embedding": emb,
            "metadata": _clean_metadata(metadata),
        }

        table.add([doc_data])
        logger.info(f"Added document to LanceDB table: {TABLE_NAME}")
        return True

    except Exception as e:
        logger.error(f"Error adding document to LanceDB: {e}", exc_info=True)
        return False

def populate_lancedb_automatically():
    """Populate LanceDB with sample financial data."""
    samples = [
        {
            "text": "Apple Inc. (AAPL) reported strong Q4 earnings with revenue of $119.6 billion, up 8% YoY. iPhone revenue $69.7B; services grew.",
            "metadata": {"company": "Apple", "ticker": "AAPL", "sector": "Technology", "type": "earnings"},
        },
        {
            "text": "Microsoft (MSFT) Azure revenue grew 29% YoY; Copilot driving growth in productivity and business processes.",
            "metadata": {"company": "Microsoft", "ticker": "MSFT", "sector": "Technology", "type": "business_update"},
        },
        {
            "text": "Tesla (TSLA) delivered 484,507 vehicles in Q4 2023; energy storage deployments up 125% YoY.",
            "metadata": {"company": "Tesla", "ticker": "TSLA", "sector": "Automotive", "type": "deliveries"},
        },
        {
            "text": "NVIDIA (NVDA) leads AI chips; data center revenue +279% YoY on training & inference demand.",
            "metadata": {"company": "NVIDIA", "ticker": "NVDA", "sector": "Technology", "type": "earnings"},
        },
        {
            "text": "Amazon (AMZN) holiday quarter: net sales +14% YoY; AWS +13%; ads momentum strong.",
            "metadata": {"company": "Amazon", "ticker": "AMZN", "sector": "Consumer Discretionary", "type": "earnings"},
        },
        {
            "text": "S&P 500 resilience with tech leading; ~24% YTD, driven by earnings and AI enthusiasm.",
            "metadata": {"index": "S&P 500", "type": "market_overview", "sector": "Market"},
        },
        {
            "text": "Fed policy remains key; sticky inflation suggests higher-for-longer rates impacting growth stocks.",
            "metadata": {"topic": "Federal Reserve", "type": "economic_policy", "sector": "Economics"},
        },
    ]

    ok = 0
    for s in samples:
        if add_document_to_lancedb(s["text"], s.get("metadata")):
            ok += 1
    logger.info(f"Populated LanceDB with {ok}/{len(samples)} sample documents")

def check_and_populate_lancedb():
    """Ensure table exists; populate with samples if empty."""
    db = _connect_db()
    table = _ensure_table(db, recreate_if_mismatch=True)

    # Don't use to_pandas(limit=1) — not supported. Use count_rows().
    try:
        count = table.count_rows()
    except Exception:
        # Fallback for older versions
        count = table.to_arrow().num_rows

    if count == 0:
        logger.info("LanceDB empty. Populating with sample data…")
        populate_lancedb_automatically()
    else:
        logger.info(f"LanceDB already has {count} rows.")

# -------------------- CLI --------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Setting up LanceDB with sample financial data…")
    check_and_populate_lancedb()
    print("LanceDB setup complete!")
