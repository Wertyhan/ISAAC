# Product Requirements Document (POC): ISAAC
**Intelligent System Architecture Advisor & Consultant**
### [Small presentation](https://www.canva.com/design/DAG8mbRpjsg/o1aAdXqiRuFNml1KU6YWsA/edit?utm_content=DAG8mbRpjsg&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

## 1. Problem & Users
* **Target User:** "Vibe Coders," Startup Engineers, and Technical Leads.
* **Problem:** Engineers often ask LLMs for system design help but receive "hallucinated" answers or generic text without visual context. While code generation is easy, designing scalable architecture remains difficult.
* **Value Proposition:** ISAAC helps engineers design scalable systems by combining grounded architectural knowledge with real diagrams, ensuring answers are based on reality, not hallucinations.

## 2. MVP Scope
**In Scope (Core Features):**
* **Multimodal Ingestion:** Ingesting technical documents and architectural diagrams.
* **Structure-based Retrieval:** The system retrieves structured architecture including data flows, services, and responsibilities.
* **Grounded Generation:** Answers must be grounded in retrieved context to eliminate hallucinations.
* **Visual Context:** Retrieving and displaying actual diagrams relevant to the user's query.

**Out of Scope (for POC):**
* Real-time web browsing (relies on ingested corpus).
* Autonomous agentic planning.

## 3. Content & Data
* **Data Sources:** Real system architecture case studies and diagrams (e.g., Glovo, high-scale marketplaces).
* **Corpus Strategy:** A curated set of architecture descriptions and their corresponding visual diagrams.
* **Data Relationship:** Text descriptions linked to specific visual diagrams (e.g., Kafka implementation for delivery apps).

## 4. Example Queries
**Text-to-Architecture:**
1. "I want to create a food delivery app (e.g., Glovo) that will process many orders in real time. What architecture and database should I choose?".
2. "Show me the microservices diagram for a courier tracking system".
3. "How does a polyglot database strategy work for high-volume traffic?".

**Multimodal/Image Context:**

4. "Explain the data flow in this architecture diagram".
5. "What are the trade-offs of the system shown in this image?".

## 5. Success Metrics
* **Hit Rate (Recall@5):** For a set of 20 architecture questions, is the correct "gold standard" document/diagram in the top 5 results?.
* **Image Relevance Score:** For visual queries, does the top-retrieved image actually depict the requested system?.
* **Faithfulness:** The answer must not contain information unsupported by the retrieved chunks.
* **Refusal Accuracy:** The system must successfully refuse 100% of out-of-domain queries.
* **Latency:** Target < 8 seconds for the full cycle (Retrieval + Generation) on local testing.

## 6. UI Expectations
* **Vibecoded UI:** A simple, modern interface aimed at "vibe coders" and developers.
* **Visual Output:** The result is not just text; it displays structured architecture and diagrams.
* **Grounded Answers:** Clear visual indication of sources to prove "no hallucinations".

## 7. Technical Stack
* **Orchestration:** LangChain / Custom RAG Pipeline.
* **LLM:** Gemini (for generation and multimodal capabilities).
* **Embeddings:** * **Text:** Gemini Embeddings.
    * **Image:** CLIP (e.g., `CLIP-ViT-L-14`) for features.
* **Vector Database:** Postgres with `pgvector`.
* **Backend:** FastAPI.
* **Frontend:** Chainlit.
