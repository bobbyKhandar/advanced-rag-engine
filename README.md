# Advanced RAG Engine — Motorcycle Repair Q&A Bot

A modular monolithic **Retrieval-Augmented Generation** system that answers motorcycle repair questions using a Telegram bot. Ingested PDF repair manuals are chunked hierarchically, embedded with `sentence-transformers`, stored in Qdrant, and retrieved via a **Small-to-Big** strategy with LLM-based query decomposition and bike-model-aware filtering.

## Architecture

```
Telegram User ──▶ Bot Handlers ──▶ Async Queues ──▶ RAG Pipeline ──▶ Answer
                     │                                    │
                     │                           ┌────────┴────────┐
                     │                           │  LLM (Groq)     │
                     │                           │  deconstructs   │
                     │                           │  query →        │
                     │                           │  {make, model,  │
                     │                           │   question}     │
                     │                           └────────┬────────┘
                     │                                    │
                     │                           ┌────────┴────────┐
                     │                           │ SmallToBig      │
                     │                           │ Retriever       │
                     │                           │ (child chunks → │
                     │                           │  parent docs)   │
                     │                           └────────┬────────┘
                     │                                    │
                     │                           ┌────────┴────────┐
                     │                           │ Vector Store    │
                     │                           │ (Qdrant + SBERT)│
                     │                           └────────┬────────┘
                     │                                    │
                     │                           ┌────────┴────────┐
                     │                           │ Generator (LLM) │
                     │                           │ context + Q → A │
                     │                           └─────────────────┘
                     │
               PDF Upload ──▶ IngestQueue ──▶ Ingestor
                  ┌─────────────────────────────────┐
                  │ 1. extract_bike_info() (LLM)    │
                  │ 2. load_pdf_to_markdown()       │
                  │ 3. HierarchicalChunker          │
                  │ 4. VectorStore.add_documents()  │
                  └─────────────────────────────────┘
```

## Inputs & Outputs

### Live Conversation

```
> User:
In my ktm rc 125  how do I remove the front riders seat?

> Bot:
Processing your question...

> Bot:
To remove the front rider's seat on the KTM RC 125, follow these steps:
1. Insert the ignition key in the seat lock and turn it clockwise.
2. Raise the rear of the front rider's seat, pull it toward the rear, and remove it upward.
3. Remove the ignition key from the seat lock.

> User:
What all things i should keep in mind while charging my 12v battery in ktm rc 125

> Bot:
Processing your question...

> Bot:
While charging your 12-V battery, keep the following things in mind:

1. **Select a compatible charging mode**: An incorrectly selected charging mode will damage the 12-V battery. Always choose a mode that is compatible with the type of battery.
2. **Avoid rapid recharging with high charging current**: This can shorten the service life of the battery. If the charging current, voltage, or time is exceeded, electrolyte may escape, reducing the battery's capacity.
3. **Charge the battery immediately if it's discharged by repeated starting**: If the battery is left in a discharged state for an extended period, it may become deeply discharged and sulfating can occur, destroying the battery.
4. **Be aware that the first charging process may take longer with a new 12-V battery**.
5. **Use a suitable battery charger**, such as the EU battery charger XCharge professional (00029095050), and adjust it according to the manufacturer's instructions.

> User:
My ktm duke 390 is not starting reasons?

> Bot:
Processing your question...
> Bot:
For the KTM models mentioned, one reason the motorcycle may not start is that the transport mode is still activated. According to Source 2 for the KTM 990 DUKE 2024 and Source 5 for the KTM 125 DUKE 2023, transport mode must be deactivated to be able to start the motorcycle. This procedure may apply to other KTM models, including the Duke 390, if they have a similar transport mode feature.
```

### PDF Ingestion (Upload or Bulk)

**Input:** A PDF repair manual (e.g., `KTM_RC_125_2024_manual.pdf`)
**Output:**
- Bike metadata extracted: `{"make": "KTM", "model": "RC 125", "year": 2024, "full_name": "KTM RC 125 2024"}`
- Document chunked into hierarchical sections (parents + children), embedded, and stored in Qdrant
- Telegram notification: `"Done: KTM_RC_125_2024_manual.pdf ingested (42 chunks)"`

### Bulk Ingestion

```bash
python scripts/bulk_ingest.py --input-dir data/manuals
```
Processes all PDFs in the directory through the same pipeline.

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Modular Monolith** | All code in a single deployable unit (`src/rag/`, `src/bot/`, `src/config/`), loosely coupled and independently testable |
| **Small-to-Big Retrieval** | Search small child chunks for high precision, return full parent sections for rich context — balances accuracy with completeness |
| **LLM Query Decomposition** | Extracts `{make, model, question}` from natural language at runtime, enabling bike-model-filtered retrieval with automatic fallback |
| **Hierarchical Chunking** | Markdown headers define parent sections; parents are split into fixed-size child chunks. Both stored separately in Qdrant with `parent_index` links |
| **Async Queues** | `asyncio.Queue` + `Semaphore` for both queries and ingestion, with `asyncio.Future` for clean awaitable results |
| **Local Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) runs locally — no embedding API cost; LLM inference delegated to Groq |

## Tech Stack

| Layer | Technology |
|---|---|
| Bot | `python-telegram-bot` v21+ |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2, 384d) |
| Vector Store | Qdrant (cloud, cosine distance) |
| LLM | Groq (llama-3.3-70b-versatile) via OpenAI-compatible API |
| PDF Processing | `pymupdf4llm` + `fitz` |
| Text Splitting | `langchain-text-splitters` (MarkdownHeader + RecursiveCharacter) |
| Config | `pydantic-settings` + `PyYAML` + `python-dotenv` |
| Testing | `pytest` + `pytest-asyncio` + `unittest.mock` (1194 test lines across 28 test classes) |
| CI | GitHub Actions |

## Project Structure

```
├── src/
│   ├── main.py                 # Entry point — wires bot + RAG
│   ├── config/
│   │   ├── settings.py         # pydantic-settings (reads .env)
│   │   ├── prompts.py          # RAG generation prompt templates
│   │   └── bike_prompts.py     # Bike extraction & query decomposition prompts
│   ├── rag/
│   │   ├── rag_processor.py    # Central orchestrator: decompose → retrieve → generate
│   │   ├── loader.py           # PDF → markdown conversion
│   │   ├── chunking.py         # HierarchicalChunker (header-based + recursive split)
│   │   ├── ingestor.py         # End-to-end PDF ingestion pipeline
│   │   ├── vector_store.py     # Qdrant wrapper (add, search, get parents)
│   │   ├── retriever.py        # SmallToBigRetriever (child → parent expansion)
│   │   ├── generator.py        # LLM wrapper (Groq OpenAI-compatible)
│   │   └── bike_meta.py        # LLM-based bike info extraction & query decomposition
│   ├── bot/
│   │   ├── __init__.py         # Telegram Application factory (4 handlers)
│   │   ├── base.py             # BotHandler ABC, KeyboardBuilder
│   │   ├── handlers.py         # Start, Help, Query, DocumentUpload handlers
│   │   ├── keyboards.py        # Reply + inline keyboards
│   │   ├── states.py           # ConversationState enum
│   │   ├── query_queue.py      # Async query queue (Processor interface)
│   │   └── ingest_queue.py     # Async ingestion queue
│   └── tools/                  # Stubs: confidence scoring, web search
├── scripts/
│   └── bulk_ingest.py          # CLI bulk PDF ingestion
├── data/
│   ├── manuals/                # Pre-loaded KTM repair PDFs (18 manuals)
│   └── uploads/                # User-uploaded PDFs
├── tests/
│   ├── test_rag.py             # 925 lines — chunking, loading, retrieval, generation
│   └── test_bot.py             # 333 lines — queues, handlers, ingestor
├── config.yaml                 # Central config: chunk sizes, model, Qdrant
├── Dockerfile + docker-compose.yaml
└── requirements.txt
```

## Getting Started

```bash
python -m venv venv
pip install -r requirements.txt
cp .env.example .env   # fill in TELEGRAM_TOKEN, GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY
python scripts/bulk_ingest.py --input-dir data/manuals  # (optional) pre-load manuals
python -m src.main
```

### Run Tests

```bash
pytest tests/ -v
```

## License

MIT
