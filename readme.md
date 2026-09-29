# Secure-Validator: Zero-Trust Clinical RAG System

A secure, privacy-preserving clinical Retrieval-Augmented Generation (RAG) system that combines:
- **Presidio** for PII redaction
- **NVIDIA NeMo Guardrails** for LLM safety and medical advice compliance
- **BioClinical ModernBERT** embeddings for semantic search
- **PostgreSQL with pgvector** for efficient vector similarity search
- **FastAPI** backend and **Streamlit** frontend

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Data Flow](#data-flow)
- [Security & Privacy](#security--privacy)
- [Development](#development)
- [Testing](#testing)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

Secure-Validator implements a zero-trust architecture for handling sensitive clinical data. The system ensures that:
1. **PII is redacted** before any LLM processing
2. **Medical advice is guarded** by NeMo Guardrails
3. **Data never leaves the trust boundary** in raw form
4. **All access is logged and controlled**

## Architecture

```mermaid
graph TD
    A[User Interface] -->|Streamlit| B(FastAPI API)
    B --> C{PII Redaction<br/>(Presidio)}
    B --> D[Embedding Service<br/>(BioClinical ModernBERT)]
    B --> E[Guardrails Service<br/>(NeMo)]
    C --> F[PostgreSQL<br/>(pgvector)]
    D --> F
    E --> G[LLM Response]
    F --> H[Clinical Context]
    H --> C
    H --> E
```

### Components

1. **Frontend** (`src/ui/app.py`)
   - Streamlit-based UI for patient selection and chat
   - Secure patient context enforcement via API
   - Automatic disclaimer on AI-generated responses

2. **Backend API** (`src/api/main.py`)
   - FastAPI application with three endpoints:
     - `/api/v1/clinical-query` - Mock DB query with PII redaction
     - `/api/v1/chat` - Production chat with pgvector search
     - `/api/v1/patients` - Fetch patients with embeddings
   - Startup loading of heavy ML models (Presidio, Guardrails, Embedder)

3. **PII Redaction Service** (`src/pii_redaction/presidio_service.py`)
   - Wraps Microsoft Presidio for clinical text de-identification
   - Customizable for healthcare-specific entities

4. **Vector Database**
   - PostgreSQL with pgvector extension
   - Stores clinical embeddings alongside structured data
   - Enables efficient similarity search

5. **Embedding Model**
   - `NeuML/bioclinical-modernbert-base-embeddings`
   - 768-dimensional BioClinical ModernBERT
   - Loaded once at startup for efficiency

6. **Guardrails**
   - NVIDIA NeMo Guardrails with RailsConfig
   - Prevents hallucinations and unsafe medical advice
   - Configured via `./src/guardrails`

## Key Features

- 🔒 **Zero-Trust Design**: PII redacted before LLM exposure
- 🏥 **Clinical Focus**: Specialized models and guardrails for healthcare
- ⚡ **High Performance**: Cached models and connection pooling
- 🔍 **Semantic Search**: Vector similarity for relevant context retrieval
- 🛡️ **Safety Guardrails**: NeMo Guardrails for medical compliance
- 📊 **Audit Ready**: Structured logging and access controls
- 🐳 **Container Ready**: Docker-compatible setup
- 📚 **Extensible**: Modular components for easy customization

## Installation

### Prerequisites
- Python 3.9+
- PostgreSQL with pgvector extension
- Git

### Setup
```bash
# Clone repository
git clone <repository-url>
cd Secure-Validator

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install spaCy model for Presidio
python -m spacy download en_core_web_lg

# Set up environment variables
cp .env.example .env  # Edit .env with your configuration
```

### Database Setup
```bash
# Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

# Run schema migrations (if any)
# See scripts/ directory for setup scripts
```

## Configuration

Environment variables are loaded from `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `DB_USER` | Database username | `postgres` |
| `DB_PASSWORD` | Database password | `password` |
| `DB_HOST` | Database host | `localhost` |
| `DB_PORT` | Database port | `5432` |
| `DB_NAME` | Database name | `clinical_db` |
| `POSTGRES_URL` | Full PostgreSQL connection string | Overrides individual DB params |
| `API_CHAT_URL` | Backend chat endpoint | `http://localhost:8000/api/v1/chat` |
| `API_PATIENTS_URL` | Backend patients endpoint | `http://localhost:8000/api/v1/patients` |

## Usage

### Development Mode
```bash
# Start backend API
uvicorn src.api.main:app --reload

# In another terminal, start Streamlit UI
streamlit run src/ui/app.py
```

### Production Deployment
See [Deployment](#deployment) section.

## API Endpoints

### Clinical Query (Mock DB)
```http
POST /api/v1/clinical-query
Content-Type: application/json

{
  "patient_id": "string",
  "prompt": "string"
}
```
Returns redacted context and LLM response using mock database.

### Production Chat
```http
POST /api/v1/chat
Content-Type: application/json

{
  "patient_id": "string",
  "messages": [
    {"role": "string", "content": "string"},
    ...
  ]
}
```
Returns LLM response with real pgvector search and guardrails.

### Get Patients
```http
GET /api/v1/patients
```
Returns list of patients with generated embeddings.

## Data Flow

1. **User selects patient** in Streamlit UI
2. **User submits question** → sent to `/api/v1/chat`
3. **Backend**:
   - Loads question and history
   - Generates embedding using BioClinical ModernBERT
   - Performs pgvector similarity search on `patient_encounters`
   - Retrieves top-k similar clinical records
   - Redacts PII from retrieved context using Presidio
   - Constructs augmented prompt with safe context
   - Processes through NeMo Guardrails for safety
   - Returns LLM response to UI
4. **UI displays response** with disclaimer

## Security & Privacy

### PII Protection
- All clinical text processed through Presidio before LLM exposure
- Entities redacted: PERSON, EMAIL, PHONE_NUMBER, MEDICAL_RECORD, etc.
- Customizable patterns for healthcare-specific identifiers

### Guardrails
- NeMo Guardrails configured to:
  - Prevent diagnostic advice
  - Ensure disclaimer presence
  - Block harmful or hallucinated content
  - Maintain clinical appropriateness

### Data Handling
- Raw PII never sent to LLM services
- Database connections use secure credentials
- All API communication can be TLS-wrapped
- Audit logging recommended for production

### Compliance
- Designed for HIPAA-compliant deployments
- Configurable for GDPR and other regulations
- Data minimization principles applied

## Development

### Code Structure
```
Secure-Validator/
├── src/
│   ├── api/              # FastAPI application
│   ├── ui/               # Streamlit frontend
│   ├── pii_redaction/    # Presidio service
│   └── guardrails/       # NeMo Guardrails config
├── scripts/              # Setup and utility scripts
├── data/                 # Sample data (if any)
├── requirements.txt      # Python dependencies
├── .env.example          # Environment template
└── README.md             # This file
```

### Adding New Features
1. **New API Endpoint**: Add to `src/api/main.py`
2. **New PII Entity**: Extend `src/pii_redaction/presidio_service.py`
3. **Guardrail Rules**: Modify files in `src/guardrails/`
4. **UI Component**: Edit `src/ui/app.py`

### Running Tests
```bash
# Install test dependencies
pip install -r requirements-test.txt  # If available

# Run tests
pytest
```

## Deployment

### Docker
```bash
# Build image
docker build -t secure-validator .

# Run containers (example)
docker-compose up -d
```

### Kubernetes
See `k8s/` directory for sample manifests (if available).

### Manual Production
1. Use process manager (systemd, supervisord, or PM2)
2. Set up reverse proxy (NGINX) for SSL termination
3. Configure database with proper backups and replication
4. Set up monitoring and logging

## Troubleshooting

### Common Issues
- **Model loading slow**: First startup loads models into memory; subsequent starts are faster
- **Database connection failed**: Verify `.env` settings and PostgreSQL accessibility
- **Embedding errors**: Ensure `sentence-transformers` model is available
- **Guardrails not working**: Check `src/guardrails/` configuration files

### Logs
- Backend logs: stdout of uvicorn process
- Frontend logs: Streamlit terminal output
- Consider implementing structured logging for production

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [Microsoft Presidio](https://github.com/microsoft/presidio) for PII redaction
- [NVIDIA NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) for LLM safety
- [Sentence Transformers](https://www.sbert.net/) for embedding models
- [pgvector](https://github.com/pgvector/pgvector) for vector similarity search
- [FastAPI](https://fastapi.tiangolo.com/) and [Streamlit](https://streamlit.io/) for web frameworks

---
*Built with 🔒 for secure, responsible AI in healthcare.*