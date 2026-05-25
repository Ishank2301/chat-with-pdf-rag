"""
Updated main application file with both CLI and programmatic access.
Serves as entry point for the 2026 RAG Agent.
"""
import os
from rag_agent_orchestrator import RAGAgent
from cli import RAGAgentCLI


def main_cli():
    """Run the CLI interface."""
    cli = RAGAgentCLI()
    cli.run()


def main_programmatic():
    """Example of programmatic usage."""
    print("🚀 2026 RAG Agent - Programmatic Example\n")
    
    agent = RAGAgent()
    
    # Example: Ingest documents
    print("📂 Ingesting documents...")
    result = agent.ingest_documents("./news_articles", use_parallel=True)
    print(f"Result: {result}\n")
    
    # Example: Query documents
    if result["status"] == "success":
        print("❓ Querying documents...")
        answer = agent.query_documents("What are the main topics?")
        print(f"Answer: {answer}\n")
    
    # Example: Analyze resume
    print("📊 Analyzing resume...")
    resume_path = "./sample_resume.txt"
    if os.path.exists(resume_path):
        analysis = agent.analyze_resume(resume_path)
        print(f"ATS Score: {analysis.get('ats_score', 'N/A')}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--programmatic":
        main_programmatic()
    else:
        main_cli()
