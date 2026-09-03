# AI Learning Assistant

Final project for the **Modern Data Engineering for AI Systems** course.

## Project Overview
AI Learning Assistant is a simple course knowledge assistant that applies data quality checks, text chunking, embeddings, vector storage, and semantic retrieval to answer questions from a small course knowledge base.

## Architecture

```text
Course Knowledge Base
        ↓
Data Quality Checks
        ↓
Chunking
        ↓
Embeddings
        ↓
ChromaDB Vector Database
        ↓
Semantic Retrieval
        ↓
Grounded Answer from Retrieved Context
```

## Data Quality
The project performs a few basic checks before indexing the data:
- Check that the knowledge base is not empty
- Check that it contains enough content
- Check the expected section structure
- Check for duplicate lines

## Technologies Used
- Python
- Sentence Transformers
- `all-MiniLM-L6-v2`
- ChromaDB

## Project Files
- `AI_Learning_Assistant.ipynb` — notebook version
- `app.py` — Python version
- `course_knowledge.txt` — course knowledge base
- `requirements.txt` — required packages
- `architecture.txt` — simple architecture flow

## How to Run
### Google Colab
1. Open `AI_Learning_Assistant.ipynb` in Google Colab.
2. Run the cells from top to bottom.
3. Change the sample question in the final cell to test another query.

### Local Python
```bash
pip install -r requirements.txt
python app.py
```

## Example Questions
```text
What is a vector database?
What are the main dimensions of data quality?
How does a real-time streaming pipeline work?
What is RAG?
```

## Course
This project was made for **"Modern Data Engineering for AI Systems"** course provided by [SDAIA Academy](https://github.com/SDAIAAcademy).
