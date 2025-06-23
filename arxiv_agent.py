import streamlit as st
import os
import json
from typing import List, Dict, Any
from mem0 import Memory
from openai import OpenAI
import arxiv


class ConfigManager:
    """Simple configuration manager for persistent settings."""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self._ensure_config_exists()
    
    def _ensure_config_exists(self):
        """Ensure configuration file exists."""
        if not os.path.exists(self.config_file):
            self.save_config({
                "openai_api_key": "",
                "default_username": "",
                "qdrant_host": "localhost",
                "qdrant_port": 6333,
                "collection_name": "arxiv_memories"
            })
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading config: {e}")
            return {}
    
    def save_config(self, config: Dict[str, Any]):
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            st.error(f"Error saving config: {e}")


class ArxivSearchEngine:
    """Handles arXiv paper searching functionality."""
    
    def __init__(self):
        self.client = arxiv.Client()
    
    def search_papers(self, search_query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search arXiv for papers based on the given query.
        
        Args:
            search_query: The search query to find papers
            max_results: Maximum number of results to return
            
        Returns:
            List of paper dictionaries with metadata
        """
        search = arxiv.Search(
            query=f"all:{search_query}",
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending
        )
        
        papers = []
        for result in self.client.results(search):
            try:
                paper = self._extract_paper_data(result)
                papers.append(paper)
            except Exception as e:
                st.warning(f"Error processing article: {e}")
        
        return papers
    
    def _extract_paper_data(self, result) -> Dict[str, Any]:
        """Extract relevant data from an arXiv result object."""
        return {
            "title": result.title,
            "id": result.get_short_id(),
            "entry_id": result.entry_id,
            "authors": [author.name for author in result.authors],
            "primary_category": result.primary_category,
            "categories": result.categories,
            "published": result.published.isoformat() if result.published else None,
            "pdf_url": result.pdf_url,
            "links": [link.href for link in result.links],
            "summary": result.summary,
            "comment": result.comment,
        }


class PaperProcessor:
    """Handles processing and formatting of paper data."""
    
    def __init__(self, openai_client: OpenAI):
        self.openai_client = openai_client
    
    def format_papers_to_markdown(self, papers: List[Dict[str, Any]]) -> str:
        """Convert paper data to structured markdown format.
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            Formatted markdown string
        """
        if not papers:
            return "No papers found for the given query."
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": self._create_formatting_prompt(papers)}],
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error formatting papers: {e}")
            return self._fallback_formatting(papers)
    
    def _create_formatting_prompt(self, papers: List[Dict[str, Any]]) -> str:
        """Create the prompt for GPT to format papers."""
        papers_text = str(papers)
        return f"""
        Based on the following arXiv search result, provide a proper structured output in markdown that is readable by the users. 
        Each paper should have a title, authors, abstract, and link.
        Search Result: {papers_text}
        Output Format: Table with the following columns: [{{"title": "Paper Title", "authors": "Author Names", "abstract": "Brief abstract", "link": "arXiv link"}}, ...]
        """
    
    def _fallback_formatting(self, papers: List[Dict[str, Any]]) -> str:
        """Fallback formatting when GPT processing fails."""
        markdown = "## Search Results\n\n"
        for i, paper in enumerate(papers, 1):
            markdown += f"### {i}. {paper.get('title', 'No title')}\n"
            markdown += f"**Authors:** {', '.join(paper.get('authors', []))}\n\n"
            markdown += f"**Abstract:** {paper.get('summary', 'No abstract available')}\n\n"
            markdown += f"**Link:** {paper.get('pdf_url', 'No link available')}\n\n"
            markdown += "---\n\n"
        return markdown



class ArxivAgentApp:
    """Main application class for the arXiv Research Agent."""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.setup_page()
        self.api_keys = self._get_api_keys()
        
        if self._validate_api_keys():
            self._initialize_components()
        else:
            self._show_api_warning()
    
    def setup_page(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="ArXiv Research Agent",
            page_icon="📚",
            layout="wide"
        )
        st.title("ArXiv Research Agent with Memory")
    
    def _get_api_keys(self) -> Dict[str, str]:
        """Get API keys from user input or persistent storage."""
        config = self.config_manager.load_config()
        
        # Load saved API key
        saved_api_key = config.get("openai_api_key", "")
        
        # API key input with saved value
        api_key = st.text_input(
            "OpenAI API Key", 
            value=saved_api_key,
            type="password", 
            help="Enter your OpenAI API key"
        )
        
        # Save API key if changed
        if api_key != saved_api_key and api_key:
            config["openai_api_key"] = api_key
            self.config_manager.save_config(config)
        
        return {'openai': api_key}
    
    def _validate_api_keys(self) -> bool:
        """Check if all required API keys are provided."""
        return all(self.api_keys.values())
    
    def _initialize_components(self):
        """Initialize all application components."""
        config = self.config_manager.load_config()
        os.environ['OPENAI_API_KEY'] = self.api_keys['openai']
        
        # Initialize services
        self.memory = self._initialize_memory(config)
        self.openai_client = OpenAI(api_key=self.api_keys['openai'])
        self.search_engine = ArxivSearchEngine()
        self.processor = PaperProcessor(self.openai_client)
        
        # Setup UI
        self._setup_sidebar(config)
        self._setup_main_content()
    
    def _initialize_memory(self, config: Dict[str, Any]) -> Memory:
        """Initialize the memory system."""
        memory_config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "host": config.get("qdrant_host", "localhost"),
                    "port": config.get("qdrant_port", 6333),
                    "collection_name": config.get("collection_name", "arxiv_memories"),
                    "on_disk": True,
                    "path": "/tmp/qdrant"
                }
            },
        }
        return Memory.from_config(memory_config)
    
    def _setup_sidebar(self, config: Dict[str, Any]):
        """Setup the sidebar components."""
        st.sidebar.header("User Settings")
        
        # Username input with saved value
        saved_username = config.get("default_username", "")
        self.user_id = st.sidebar.text_input(
            "Username",
            value=saved_username,
            help="Enter a unique username to store your research history"
        )
        
        # Save username if changed
        if self.user_id != saved_username and self.user_id:
            config["default_username"] = self.user_id
            self.config_manager.save_config(config)
    
    def _setup_main_content(self):
        """Setup the main content area with tabs."""
        # Create tabs
        tab1, tab2 = st.tabs(["🔍 Search Papers", "🧠 Memory"])
        
        with tab1:
            self._setup_search_tab()
        
        with tab2:
            self._setup_memory_tab()
    
    def _setup_search_tab(self):
        """Setup the search tab content."""
        st.header("Research Paper Search")
        
        search_query = st.text_input(
            "Research paper search query",
            placeholder="Enter your research topic or keywords...",
            help="Enter keywords or topics to search for relevant papers"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            search_button = st.button('🔍 Search for Papers', type="primary")
        
        if search_button and search_query:
            self._perform_search(search_query)
    
    def _setup_memory_tab(self):
        """Setup the memory tab content."""
        st.header("Your Research History")
        
        if not self.user_id:
            st.info("Please enter a username in the sidebar to view your memories.")
            return
        
        memories = self.get_all_memories(self.user_id)
        if "results" in memories and memories["results"]:
            # Iterate backwards to show the most recent searches first
            for mem in reversed(memories["results"]):
                st.markdown(mem['memory'])
        else:
            st.info("No memories found. Start searching for papers to build your research history!")
    
    
    def _perform_search(self, search_query: str):
        """Perform the paper search and display results."""
        if not self.user_id:
            st.error("Please enter a username in the sidebar first.")
            return
        
        with st.spinner('Searching and Processing Papers...'):
            try:
            
                # Enhance search query with memory context
                enhanced_query = self._enhance_query_with_memory(
                    search_query
                )
                
                # Search for papers
                papers = self.search_engine.search_papers(enhanced_query)
                
                if papers:
                    # Format and display results
                    formatted_results = self.processor.format_papers_to_markdown(papers)
                    st.markdown(formatted_results)
                    
                    # Store search in memory
                    self._store_search_in_memory(search_query, papers)
                else:
                    st.info("No papers found for your query. Try different keywords.")
                    
            except Exception as e:
                st.error(f"An error occurred during search: {e}")
                st.exception(e)  # This will show the full traceback for debugging
    
    def _enhance_query_with_memory(self, query: str) -> str:
        """Enhance search query with relevant memory context."""
        try:
            return query
        except Exception as e:
            st.error(f"Error enhancing query with memory: {e}")
            return query
    
    def _store_search_in_memory(self, query: str, papers: List[Dict[str, Any]]):
        """Store the search query and results in memory as a single markdown entry."""
        try:
            paper_titles = [paper.get('title', 'No Title') for paper in papers]
            
            # Create a single markdown formatted string
            markdown_memory = f"**🔍 Searched for:** {query}\n\n"
            if paper_titles:
                markdown_memory += "**📄 Found Papers:**\n"
                for title in paper_titles:
                    markdown_memory += f"- {title}\n"
            markdown_memory += "\n---\n"

            # Add the formatted markdown string as a single memory entry, without inference
            self.memory.add(
                messages=markdown_memory,
                user_id=self.user_id,
                infer=False
            )
        except Exception as e:
            st.warning(f"Could not store search in memory: {e}")
            st.exception(e) # Show full traceback for debugging

    def get_all_memories(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all memories for a user."""
        try:
            return self.memory.get_all(user_id=user_id)
        except Exception as e:
            st.warning(f"Error retrieving all memories: {e}")
            return []
    
    def _show_api_warning(self):
        """Show warning when API keys are missing."""
        st.warning("⚠️ Please enter your OpenAI API key to use this app.")
        st.info("""
        **How to get an API key:**
        1. Visit [OpenAI's website](https://platform.openai.com/api-keys)
        2. Sign up or log in to your account
        3. Create a new API key
        4. Copy and paste it above
        """)


def main():
    """Main entry point for the application."""
    try:
        app = ArxivAgentApp()
    except Exception as e:
        st.error(f"Failed to initialize application: {e}")
        st.info("Please check your configuration and try again.")


if __name__ == "__main__":
    main()