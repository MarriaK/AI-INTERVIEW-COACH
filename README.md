# SMART AI Interview Coach using RAG

![Project Banner](https://via.placeholder.com/800x200?text=AI+Interview+Coach+RAG+System) <!-- Replace with actual image if available -->

An intelligent interview preparation system using Retrieval-Augmented Generation (RAG) to provide dynamic technical questions and AI-powered answer evaluation.

## Key Features
- **Dynamic Question Generation**: Retrieves/generates NLP/DL/ML interview questions
- **AI Evaluation**: Scores answers (1-10) with detailed feedback
- **Dual Interaction Modes**:
  - 🗣️ Conversational chat interface
  - 🎯 Structured interview simulation
- **RAG Architecture**: 
  - OpenAI + HuggingFace embeddings
  - ChromaDB vector store
  - GPT-4o-mini for generation/evaluation

## Tech Stack
| Component               | Technology Used          |
|-------------------------|-------------------------|
| Language Model          | GPT-4o-mini (OpenAI)    |
| Embeddings              | text-embedding-3-small  |
| Vector Database         | ChromaDB                |
| Framework               | LangChain               |
| Web Interface           | Streamlit               |
| Dataset                 | 600 Q&A pairs (JSON)    |

## Installation
```bash
git clone https://github.com/yourusername/ai-interview-coach.git
cd ai-interview-coach
