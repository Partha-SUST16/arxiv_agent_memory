# Arxiv Research Agent with Memory

This Streamlit app is a research assistant that helps you search for academic papers on arXiv. It keeps a memory of your past searches and the papers you've found, making it easy to keep track of your research.

The app uses a clean, tabbed interface for a smooth user experience. It leverages OpenAI's GPT-4o-mini model to format search results, the `arxiv` library for paper searching, and `mem0` with a Qdrant vector database for persistent memory.

## Features

- **Tabbed Interface**: A clean UI with dedicated "🔍 Search" and "🧠 Memory" tabs.
- **arXiv Paper Search**: Search for academic papers directly from the app.
- **AI-Powered Formatting**: Uses OpenAI's GPT-4o-mini to format search results into a clean, readable list.
- **Persistent Search History**: Remembers your past searches and the papers you found. Each search is saved as a distinct, formatted entry in the "Memory" tab, with the most recent searches appearing first.
- **Persistent Configuration**: Saves your OpenAI API key and username in a `config.json` file, so you don't have to enter them every time.
- **Raw Memory Storage**: Stores search history as pre-formatted markdown in a `mem0` vectordb with Qdrant, ensuring consistent display.

## How to Get Started

### 1. Clone the Repository

```bash
git clone https://github.com/Partha-SUST16/arxiv_agent_memory.git
cd arxiv_agent_memory/arxiv_agent_memory
```

### 2. Install Dependencies

Make sure you have Python 3.8+ installed. Then, install the required packages:

```bash
pip install -r requirements.txt
```

### 3. Run Qdrant with Docker

The application uses Qdrant as its vector database. Make sure you have Docker installed and running.

Pull the Qdrant image:

```bash
docker pull qdrant/qdrant
```

Run the Qdrant container. This command also mounts a local directory (`qdrant_storage`) to persist the data.

```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

_Note: The app expects Qdrant on `localhost:6333`. If your setup differs, you can adjust the settings in `config.json` after the first run._

### 4. Run the Streamlit App

Launch the app with Streamlit:

```bash
streamlit run arxiv_agent.py
```

### 5. First-time Setup

The first time you run the app, it will create a `config.json` file.

1.  Open the app in your browser.
2.  Enter your **OpenAI API Key** in the input field.
3.  Enter a **Username**.
4.  These details will be saved to `config.json` for all future sessions.

Now you're all set to start your research!
