# 🏛️ Student Republic AI Chatbot Microservice

![Build](https://img.shields.io/github/actions/workflow/status/victortsrodrigues/chatbot-republic-ai-microservice/ci-cd.yml?branch=main)  
![Docker Pulls](https://img.shields.io/docker/pulls/victortsrodrigues/chatbot-republica-ai-microservice)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An AI-powered FastAPI microservice that serves as the backend for a attendant chatbot. It uses RAG (Retrieval-Augmented Generation) with OpenAI, Pinecone vector search and MongoDB to answer student queries over WhatsApp.**

🔗 **Production URL**: [https://chatbot-republic-ai-microservice.onrender.com](https://chatbot-republic-ai-microservice.onrender.com)

---

## 🧠 Overview

This microservice exposes HTTP endpoints for:
- **Health checks** (`/health/live`, `/health/ready`)  
- **RAG-based Q&A** (`POST /rag/query`)  

It loads vector embeddings into Pinecone, rooms data in MongoDB, and invokes OpenAI’s Chat and Embedding APIs to generate context-aware responses.

---

## 🚀 Technologies

- **FastAPI** + **Uvicorn** + **Gunicorn**  
- **OpenAI** (`AsyncOpenAI`) for embeddings & chat  
- **Pinecone** (async) for vector retrieval  
- **MongoDB** with Motor (async driver)  
- **Cachetools** for TTL caching  
- **Tenacity** retry & circuit breaker patterns  
- **Docker** for containerization  
- **GitHub Actions** for CI/CD  

---

## 📦 Features

- ✅ **Retrieval-Augmented Generation** pipeline  
- ✅ **Asynchronous** OpenAI & Pinecone clients  
- ✅ **Per-user rate limiting** & global rate limiting  
- ✅ **Circuit breakers**, retries and backoff  
- ✅ **Health checks** for all dependencies  
- ✅ **Dockerized** with non-root user  
- ✅ **Unit & integration tests** with pytest marks  
- ✅ **CI/CD**: tests → build & push → deploy to Render  

---

## 🏗️ Project Structure
```
ai-microservice/ 
├── app/ 
│ ├── routers/ 
│ │ ├── health_router.py 
│ │ └── rag_router.py 
│ ├── services/ 
│ │ ├── mongo_service.py 
│ │ ├── pinecone_service.py 
│ │ ├── openai_service.py 
│ │ └── rag_service.py 
│ ├── utils/ 
│ │ └── logger.py 
│ ├── models/ 
│ │ └── schemas.py 
│ ├── config.py 
│ └── main.py 
├── tests/ 
│ ├── unit/ 
│ └── integration/ 
├── .env.example 
├── Dockerfile 
├── .dockerignore 
├── pytest.ini 
├── requirements.txt 
└── .github/workflows/ci-cd.yml
```

---

## ⚙️ Local Development

### Prerequisites

- Python 3.11+  
- MongoDB (local or Atlas)  
- Pinecone account & index  
- OpenAI API key  

### 1. Clone the repo

```bash
git clone https://github.com/your-org/ai-microservice.git
cd ai-microservice
```

### 2. Create .env
Copy `.env.example` to `.env` and fill in:
```
OPENAI_API_KEY=sk-...

PINECONE_API_KEY=pcsk-...
PINECONE_HOST=your-host
PINECONE_INDEX_NAME=your-index
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

MONGO_URI=mongodb+srv://...
MONGO_DB=chatbot-republica

LOG_LEVEL=INFO
```
⚠️ **Never commit `.env`. Add it to `.gitignore`.**

### 3. Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the app
```bash
uvicorn app.main:app \
  --host 0.0.0.0 --port 8000 \
  --reload
```
---

## 🐳 Docker

### Build:
```bash
docker build -t chatbot-republica-ai-microservice .
```

### Run:
```bash
docker run --rm -p 8000:8000 \
  --env-file .env \
  chatbot-republica-ai-microservice
```
Then access:
- Health: http://localhost:8000/health/ready
- RAG query: POST http://localhost:8000/rag/query

---

## 🧪 Automated Testing

### Run unit tests:
```bash
python -m pytest -m unit
```

### Run integration tests:
```bash
python -m pytest -m integration
```

Tests are marked with `@pytest.mark.unit` and `@pytest.mark.integration`.

---

## 🔁 CI/CD

The GitHub Actions pipeline (`.github/workflows/ci-cd.yml`) automates:
1. Unit & Integration Tests
2. Docker Build & Push to Docker Hub
3. Deploy via Render Deploy Hook

Make sure to set these GitHub Secrets:
- OPENAI_API_KEY
- PINECONE_API_KEY, PINECONE_HOST, PINECONE_INDEX_NAME
- MONGO_URI
- DOCKERHUB_USERNAME, DOCKERHUB_TOKEN
- RENDER_DEPLOY_HOOK_URL

---

## 📡 API Endpoints
### Liveness
```http
GET /health/live
```
Response: 
```json
{"status":"alive"}
```

### Readiness
```http
GET /health/ready
```
Response:
```json
{"status":"ready"}
```

### RAG Query
```http
POST /rag/query
Content-Type: application/json

{
  "query": "Show available rooms under $500",
  "user_id": "user123",
  "history": [],
  "system_message": null
}
```
Response:
```json
{
  "response": "Here are some rooms...",
  "requires_action": false
}
```

---

## 📦 Deployment
On Render (free tier):
1. Link your GitHub repo
2. Set Build Command: docker build -t service .
3. Set Start Command:
```lua
gunicorn app.main:app \
  --workers 1 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:$PORT \
  --timeout 60 \
  --max-requests 1000 \
  --max-requests-jitter 50 \
  --worker-tmp-dir /dev/shm
```
4. Add environment variables in Render dashboard.

---

## 📬 Contributing
Pull requests are welcome!  
For major changes, please open an issue first to discuss what you'd like to change.

To contribute:
1. Fork the repository  
2. Create a feature branch  
3. Commit your changes with clear messages  
4. Ensure tests are included if applicable  
5. Open a pull request 

---

## 🛡️ License
MIT © [Victor Rodrigues](https://github.com/victortsrodrigues)