# ISAAC - Data Sources & Licensing Documentation

## Overview

ISAAC (Intelligent System Architecture Advisor & Consultant) is a RAG-based system for system design education. This document describes the data sources, licensing, and privacy constraints for the knowledge base.

---

## Primary Data Source

### The System Design Primer

| Attribute        | Value                                                                                   |
| ---------------- | --------------------------------------------------------------------------------------- |
| **Repository**   | [donnemartin/system-design-primer](https://github.com/donnemartin/system-design-primer) |
| **Author**       | Donne Martin                                                                            |
| **License**      | Creative Commons Attribution 4.0 International (CC BY 4.0)                              |
| **License URL**  | https://creativecommons.org/licenses/by/4.0/                                            |
| **Content Type** | Educational materials on system design                                                  |
| **Last Scraped** | See `data/raw/isaac_raw_data.json` metadata                                             |

### License Terms (CC BY 4.0)

Under the CC BY 4.0 license, you are free to:

- ✅ **Share** — copy and redistribute the material in any medium or format
- ✅ **Adapt** — remix, transform, and build upon the material for any purpose, even commercially

Under the following terms:

- 📝 **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made

### Attribution Statement

This project uses content from "The System Design Primer" by Donne Martin, available at https://github.com/donnemartin/system-design-primer, licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).

---

## Data Processing Pipeline

### Scraping (`isaac_scraper/`)

1. **Source**: GitHub API + Raw content URLs
2. **Content**: Markdown files (.md) from system-design-primer repository
3. **Images**: Architecture diagrams referenced in markdown files
4. **Output**: `data/raw/isaac_raw_data.json`

### Ingestion (`isaac_ingestion/`)

1. **Input**: Raw scraped data
2. **Processing**:
   - Text chunking with markdown-aware splitting
   - Gemini embedding generation (`models/gemini-embedding-001`)
   - Image caption generation via Gemini Vision
3. **Storage**: PostgreSQL + PGVector

### Metadata Tracking

Each document and chunk includes:

| Field        | Description                                            |
| ------------ | ------------------------------------------------------ |
| `doc_id`     | Stable SHA256-based identifier                         |
| `chunk_id`   | Unique chunk identifier (format: `CHK_{hash}_{index}`) |
| `source_uri` | Full URL to original GitHub source                     |
| `created_at` | ISO 8601 timestamp of ingestion                        |
| `provenance` | Source repository name                                 |

---

## Privacy Constraints

### Data Collection

- ✅ **No user data collection**: ISAAC does not store user queries or conversations
- ✅ **No personal data**: Knowledge base contains only public educational content
- ✅ **No tracking**: No analytics or telemetry implemented

### API Usage

- **Gemini API**: Used for embeddings and generation
- **API Keys**: Stored in `.env` file (not committed to repository)
- **Rate Limiting**: Implemented to respect API quotas

### Data Storage

- **Vector Store**: Local PostgreSQL with PGVector extension
- **Images**: Local storage in `data/images/`
- **No cloud sync**: All data remains on local machine

---

## Image Sources

All architecture diagrams are sourced from the system-design-primer repository and are subject to the same CC BY 4.0 license.

### Image Registry

Images are tracked in `data/images/image_registry.json` with metadata:

```json
{
  "image_id": "IMG_abc123",
  "doc_id": "DOC_xyz789",
  "project_name": "Design a URL Shortener",
  "filename": "design_a_url_shortener_abc123.png",
  "source_url": "https://github.com/donnemartin/...",
  "caption": "AI-generated description of diagram",
  "created_at": "2024-01-15T10:30:00Z"
}
```

---

## Compliance Checklist

- [x] Source content is publicly available
- [x] License permits educational and commercial use
- [x] Attribution is provided in this document
- [x] No personal data is processed or stored
- [x] API keys are securely managed
- [x] Full provenance tracking for all content

---

## Updates

To update the knowledge base with new content:

1. Run scraper: `python -m isaac_scraper`
2. Run ingestion: `python -m isaac_ingestion`
3. Verify data in PostgreSQL

Note: Re-running ingestion will generate new `doc_id` values based on content hash.

---

## Contact

For questions about data sources or licensing, please open an issue in the project repository.
