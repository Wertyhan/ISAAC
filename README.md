# Product Requirements Document (POC): ISAAC

**Intelligent System Architecture Advisor & Consultant**

### [Small presentation](https://www.canva.com/design/DAG8mbRpjsg/o1aAdXqiRuFNml1KU6YWsA/edit?utm_content=DAG8mbRpjsg&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

## 1. Problem & Users

- **Target User:** "Vibe Coders," Startup Engineers, and Technical Leads.
- **Problem:** Engineers often ask LLMs for system design help but receive "hallucinated" answers or generic text without visual context. While code generation is easy, designing scalable architecture remains difficult.
- **Value Proposition:** ISAAC helps engineers design scalable systems by combining grounded architectural knowledge with real diagrams, ensuring answers are based on reality, not hallucinations.

## 2. MVP Scope

**In Scope (Core Features):**

- **Multimodal Ingestion:** Ingesting technical documents and architectural diagrams.
- **Structure-based Retrieval:** The system retrieves structured architecture including data flows, services, and responsibilities.
- **Grounded Generation:** Answers must be grounded in retrieved context to eliminate hallucinations.
- **Visual Context:** Retrieving and displaying actual diagrams relevant to the user's query.

**Out of Scope (for POC):**

- Real-time web browsing (relies on ingested corpus).
- Autonomous agentic planning.

## 3. Content & Data

- **Data Sources:** Real system architecture case studies and diagrams (e.g., Glovo, high-scale marketplaces).
- **Corpus Strategy:** A curated set of architecture descriptions and their corresponding visual diagrams.
- **Data Relationship:** Text descriptions linked to specific visual diagrams (e.g., Kafka implementation for delivery apps).

## 4. Example Queries

**Text-to-Architecture:**

1. "I want to create a food delivery app (e.g., Glovo) that will process many orders in real time. What architecture and database should I choose?".
2. "Show me the microservices diagram for a courier tracking system".
3. "How does a polyglot database strategy work for high-volume traffic?".

**Multimodal/Image Context:**

4. "Explain the data flow in this architecture diagram".
5. "What are the trade-offs of the system shown in this image?".

## 5. Success Metrics

- **Hit Rate (Recall@5):** For a set of 20 architecture questions, is the correct "gold standard" document/diagram in the top 5 results?.
- **Image Relevance Score:** For visual queries, does the top-retrieved image actually depict the requested system?.
- **Faithfulness:** The answer must not contain information unsupported by the retrieved chunks.
- **Refusal Accuracy:** The system must successfully refuse 100% of out-of-domain queries.
- **Latency:** Target < 8 seconds for the full cycle (Retrieval + Generation) on local testing.

## 6. UI Expectations

- **Vibecoded UI:** A simple, modern interface aimed at "vibe coders" and developers.
- **Visual Output:** The result is not just text; it displays structured architecture and diagrams.
- **Grounded Answers:** Clear visual indication of sources to prove "no hallucinations".

## 7. Technical Stack

- **Orchestration:** LangChain / Custom RAG Pipeline.
- **LLM:** Gemini (for generation and multimodal capabilities).
- **Embeddings:** \* **Text:** Gemini Embeddings.
  - **Image:** CLIP (e.g., `CLIP-ViT-L-14`) for features.
- **Vector Database:** Postgres with `pgvector`.
- **Backend:** FastAPI.
- **Frontend:** Chainlit.

---

## 8. Implementation Details

### Data Traceability

Every document and chunk includes full provenance tracking:

| Field        | Description                                    |
| ------------ | ---------------------------------------------- |
| `doc_id`     | Stable SHA256-based document identifier        |
| `chunk_id`   | Unique chunk ID (format: `CHK_{hash}_{index}`) |
| `source_uri` | Full URL to original GitHub source             |
| `created_at` | ISO 8601 timestamp of ingestion                |

### Image Registry

Persistent metadata storage for all architecture diagrams:

- Location: `data/images/image_registry.json`
- Tracks: image ID, project name, captions, source URLs

### Retrieval System

- **Hybrid Search:** BM25 keyword search + Vector similarity
- **Reranking:** FlashRank cross-encoder for improved precision
- **Minimum Relevance:** Score threshold (0.35) for "I don't know" responses

### Inline Citations

All generated responses include:

- Inline citations: `[Source Name]` or numbered `[1]`, `[2]`
- References section with source URLs
- Figure references for architecture diagrams

### Debug Mode (UI Settings)

Toggle debug view to display:

- Retrieval mode used (hybrid/vector/keyword)
- Chunk relevance scores
- Processing time
- Retrieved chunk previews

---

## 9. Quick Start

```bash
# 1. Start PostgreSQL with PGVector
docker-compose up -d

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
cp .env.example .env
# Edit .env with your GEMINI_API_KEY

# 4. Run data pipeline
python -m isaac_scraper    # Scrape content
python -m isaac_ingestion  # Ingest to vector DB

# 5. Start the application
chainlit run app.py
```

---

## 10. System Evaluation

Run comprehensive evaluation of retrieval and generation quality:

```bash
# Create evaluation dataset (30 queries)
python -m isaac_eval --create-dataset

# Run full evaluation
python -m isaac_eval

# Run retrieval-only evaluation (faster)
python -m isaac_eval --no-generation

# Generate markdown report
python -m isaac_eval --markdown
```

### Evaluation Metrics

**Retrieval:**
- Recall@k / Precision@k (k=1,3,5,10)
- MRR (Mean Reciprocal Rank)
- Hit Rate for text and images

**Generation:**
- Faithfulness (grounding in context)
- Citation correctness
- Answer relevance
- Refusal accuracy (off-topic)
- "I don't know" accuracy

### Evaluation Dataset

30 queries covering:
- Text-to-architecture (easy/medium/hard)
- Image similarity search
- Off-topic detection
- Edge cases

---

## 11. Data Sources & Licensing

See [DATA_SOURCES.md](DATA_SOURCES.md) for complete information about:

- Content sources and licensing (CC BY 4.0)
- Privacy constraints
- Attribution requirements
