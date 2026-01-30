# ISAAC: Intelligent System Architecture Advisor & Consultant

## Overview

**ISAAC** is a **Multimodal Retrieval-Augmented Generation (RAG)** system designed to assist software engineers and architects in designing scalable systems. Unlike standard text-only assistants, ISAAC retrieves and reasons over both **technical documentation** and **architectural diagrams** (images) to provide grounded, visually-supported answers.

This project was built as a "Pure RAG" application using **LangChain**, focusing on strict retrieval and generation orchestration without autonomous agents.

---

## Key Features

- **Multimodal Retrieval**: Searches across text case studies and architecture diagrams simultaneously.
- **Grounded Generation**: Answers are strictly based on retrieved context, reducing hallucinations.
- **Visual Evidence**: Displays the actual architecture diagrams used to generate the answer.
- **Interactive "Vibecoded" UI**: A modern chat interface built with **Chainlit**.
- **Automated Evaluation**: Includes a built-in evaluation pipeline for measuring retrieval recall and answer faithfulness.

## System Architecture

The system follows a standard RAG topology:

1.  **Ingestion Layer**:
    - Loads raw data from `data/raw/` (JSON format).
    - Processes text chunks and downloads/links images.
    - Generates embeddings using **Gemini Embedding** model.
    - Stores vectors and metadata in **PostgreSQL** (via `pgvector`).

2.  **Retrieval Layer**:
    - **Hybrid Search**: Combines semantic vector search with keyword filtering.
    - **Multimodal**: User queries can trigger image lookups based on semantic similarity.

3.  **Generation Layer**:
    - **LLM**: Google **Gemini 2.0 Flash**.
    - **Orchestration**: **LangChain** constructs the prompt with context (text + image captions).

4.  **Interface**:
    - **Chainlit** provides the chat UI, rendering markdown and images inline.

---

## Setup & Installation

### Prerequisites

- **Python 3.10+**
- **Docker Desktop** (for PostgreSQL with `pgvector`)
- A **Google Cloud API Key** (for Gemini)

### 1. Clone & Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the root directory:

```ini
# .env
GEMINI_API_KEY=your_google_api_key_here

# Postgres Connection (Default, assuming Docker setup below)
POSTGRES_CONNECTION_STRING=postgresql+psycopg2://admin:secret@localhost:5433/isaac_vec_db
```

### 3. Start Vector Database

Use the provided `docker-compose.yml` to start a persistent PostgreSQL instance with `pgvector` enabled:

```bash
docker-compose up -d
```

---

## Usage Guide

### Step 1: Ingest Data

Before asking questions, you must populate the vector database with the case studies and diagrams.

```bash
python -m isaac_ingestion
```

This script will:

1.  Load data from `data/raw/isaac_raw_data.json`.
2.  Download/verify images in `data/images/`.
3.  Compute embeddings.
4.  Index everything into PostgreSQL.

### Step 2: Run the UI

Start the Chat Interface:

```bash
chainlit run app.py
```

Opens automatically at `http://localhost:8000`.

### Step 3: Example Queries

Try asking:

**Text-heavy queries:**

- _"How does Gojek handle high-concurrency ride matching?"_
- _"Explain the database sharding strategy for a Twitter-like feed."_

**Visual queries (requires diagrams):**

- _"Show me the architecture diagram for the courier tracking system."_
- _"What are the components shown in the microservices data flow?"_
- _"Compare the caching strategies shown in the diagrams."_

---

## Evaluation

The project includes a comprehensive evaluation module to measure RAG performance.

**Run the full evaluation suite:**

```bash
# Run full evaluation
python -m isaac_eval

# Run retrieval-only evaluation (faster)
python -m isaac_eval --no-generation
```

Flags:

- `--create-dataset`: Regenerate the Golden Dataset/Eval Set.
- `--no-generation`: Run only retrieval metrics (faster).
- `-v`: Verbose output.

**Metrics Tracked:**

- **Retrieval**: Recall@k, MRR (Mean Reciprocal Rank).
- **Generation**: Faithfulness (Groundedness), Answer Relevance.

---

## Project Structure

```
ISAAC/
├── app.py                # Chainlit UI Entry Point
├── chainlit.md           # UI Welcome Config
├── docker-compose.yml    # Vector DB (Postgres)
├── requirements.txt      # Python Dependencies
├── data/
│   ├── images/           # Local image storage
│   └── raw/              # Raw knowledge corpus
├── isaac_core/           # Configuration & Constants
├── isaac_ingestion/      # ETL Pipeline (Load -> Chunk -> Embed -> Store)
├── isaac_generation/     # RAG Logic (Retrieval -> Generation)
├── isaac_eval/           # Evaluation Framework
└── isaac_scraper/        # (Optional) Data collection tools
```

## Implementation Report Summary

- **Domain**: Software Architecture Knowledge Base.
- **Chunking**: Semantic chunking preserving section headers.
- **Multimodal Strategy**: Images are indexed via their captions and associated text context. The "Multimodal" aspect is achieved by retrieving grounded image references and passing them to the multimodal Gemini model for analysis when needed.
- **Latency**: Optimized using `gemini-2.0-flash` for sub-5s responses on typical queries.

---

_Built with LangChain, Chainlit, and Gemini._
