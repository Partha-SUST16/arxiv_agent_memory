## Arxiv Research Agent with Memory
This Streamlit app implements a sort of research assistant that helps users search for academic papers on arXiv while maintaining a memory of user interests and past interactions. It utilizes OpenAI's GPT-4o-mini model for processing search results, Arxiv for paper browsing, and Mem0 with Qdrant for maintaining user context.

### Features

- Search interface for querying arXiv papers
- AI-powered processing of search results for improved readability
- Persistent memory of user interests and past searches
- Utilizes OpenAI's GPT-4o-mini model for intelligent processing
- Implements memory storage and retrieval using Mem0 and Qdrant

### How to get Started?

1. Clone the GitHub repository
```bash
git clone https://github.com/Partha-SUST16/arxiv_agent_memory.git
cd arxiv_agent_memory
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure Qdrant is running:
The app expects Qdrant to be running on localhost:6333. Adjust the configuration in the code if your setup is different.

```bash
docker pull qdrant/qdrant

docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

4. Run the Streamlit App
```bash
streamlit run arxiv_agent.py 