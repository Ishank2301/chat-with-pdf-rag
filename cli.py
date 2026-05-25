"""Interactive command-line interface for the RAG agent."""

import logging
import os
import sys

from rag_agent_orchestrator import RAGAgent

logger = logging.getLogger(__name__)


class RAGAgentCLI:
    """Interactive command-line interface for RAG Agent."""

    def __init__(self):
        """Initialize CLI with RAG Agent."""
        self.agent = RAGAgent()
        self.current_doc_path = None

    def display_banner(self) -> None:
        """Display welcome banner."""
        banner = """
============================================================
                       RAG AGENT CLI
============================================================
        """
        print(banner)

    def display_menu(self) -> None:
        """Display main menu options."""
        menu = """
Main Menu
1. Ingest Documents
2. Query Documents
3. Summarize Document
4. Analyze Resume
5. Database Stats
6. Clear Database
7. Help
0. Exit
        """
        print(menu)

    def ingest_documents(self) -> None:
        """Handle document ingestion."""
        print("\nDOCUMENT INGESTION")
        print("-" * 50)

        directory = input("Enter directory path: ").strip()

        if not os.path.exists(directory):
            print("Directory not found.")
            return

        try:
            use_parallel = input("Use parallel loading? (y/n, default: y): ").strip().lower() != "n"
            num_workers = 4

            if use_parallel:
                workers_input = input("Number of workers (default: 4): ").strip()
                if workers_input.isdigit():
                    num_workers = int(workers_input)

            print("\nIngesting documents...")
            result = self.agent.ingest_documents(directory, use_parallel, num_workers)

            if result["status"] == "success":
                print("\nSuccess.")
                print(f"Documents loaded: {result['documents_loaded']}")
                print(f"Chunks created: {result['chunks_created']}")
                print(f"Total in database: {result['total_in_database']}")
            else:
                print(f"\nError: {result['message']}")

        except Exception as e:
            print(f"\nError: {e}")

    def query_documents(self) -> None:
        """Handle document querying."""
        print("\nQUERY DOCUMENTS")
        print("-" * 50)

        stats = self.agent.get_database_stats()
        if stats.get("document_count", 0) == 0:
            print("No documents in database. Please ingest documents first.")
            return

        question = input("Enter your question: ").strip()
        if not question:
            print("Question cannot be empty.")
            return

        try:
            n_results = input("Number of relevant chunks to retrieve (default: 3): ").strip()
            if n_results.isdigit():
                n_results = int(n_results)
            else:
                n_results = 3

            print("\nProcessing query...")
            result = self.agent.query_documents(
                question,
                n_results=n_results,
                return_relevant_chunks=True,
            )

            if result["status"] == "success":
                print("\nANSWER:")
                print("-" * 50)
                print(result["answer"])
                print("-" * 50)
                print(f"Sources used: {result['num_sources']}")

                show_chunks = input("\nShow relevant chunks? (y/n): ").strip().lower() == "y"
                if show_chunks and result.get("relevant_chunks"):
                    print("\nRELEVANT CHUNKS:")
                    for i, chunk in enumerate(result["relevant_chunks"], 1):
                        print(f"\n[Chunk {i}]")
                        print(chunk[:500] + "..." if len(chunk) > 500 else chunk)
            else:
                print(f"\nError: {result['message']}")

        except Exception as e:
            print(f"\nError: {e}")

    def summarize_document(self) -> None:
        """Handle document summarization."""
        print("\nSUMMARIZE DOCUMENT")
        print("-" * 50)

        file_path = input("Enter document path: ").strip()

        if not os.path.exists(file_path):
            print("File not found.")
            return

        try:
            print("\nSummary length:")
            print("  1. Short (1-2 sentences)")
            print("  2. Detailed (1-2 paragraphs)")
            choice = input("Choose (1 or 2, default: 1): ").strip()

            summary_length = "detailed" if choice == "2" else "short"

            print("\nGenerating summary...")
            result = self.agent.summarize_document(file_path, summary_length)

            if result["status"] == "success":
                print("\nSUMMARY:")
                print("-" * 50)
                print(result["summary"])
                print("-" * 50)
            else:
                print(f"\nError: {result['message']}")

        except Exception as e:
            print(f"\nError: {e}")

    def analyze_resume(self) -> None:
        """Handle resume ATS analysis."""
        print("\nRESUME ATS ANALYSIS")
        print("-" * 50)

        resume_path = input("Enter resume file path: ").strip()

        if not os.path.exists(resume_path):
            print("File not found.")
            return

        try:
            print("\nAnalyzing resume for ATS compatibility...")
            result = self.agent.analyze_resume(resume_path)

            if result["status"] == "success":
                print("\nATS ANALYSIS RESULTS:")
                print("=" * 50)
                print(f"ATS Score: {result['ats_score']}/100")
                print("=" * 50)

                print("\nSTRENGTHS:")
                for strength in result.get("strengths", []):
                    print(f"  - {strength}")

                print("\nWEAKNESSES:")
                for weakness in result.get("weaknesses", []):
                    print(f"  - {weakness}")

                print("\nSUGGESTIONS TO IMPROVE:")
                for suggestion in result.get("suggestions", []):
                    print(f"  - {suggestion}")

                print("\nKEY MISSING ELEMENTS:")
                for missing in result.get("missing_elements", []):
                    print(f"  - {missing}")

                print("\n" + "=" * 50)
            else:
                print(f"\nError: {result['message']}")

        except Exception as e:
            print(f"\nError: {e}")

    def database_stats(self) -> None:
        """Display database statistics."""
        print("\nDATABASE STATISTICS")
        print("-" * 50)

        try:
            stats = self.agent.get_database_stats()

            if stats.get("status") == "healthy":
                print(f"Collection: {stats['collection_name']}")
                print(f"Total documents: {stats['document_count']}")
                print(f"Status: {stats['status'].upper()}")
            else:
                print(f"Error: {stats.get('message', 'Unknown error')}")

        except Exception as e:
            print(f"\nError: {e}")

    def clear_database(self) -> None:
        """Clear database with confirmation."""
        print("\nCLEAR DATABASE")
        print("-" * 50)

        confirm = input("This will delete ALL documents. Confirm? (yes/no): ").strip().lower()

        if confirm == "yes":
            try:
                result = self.agent.clear_database()
                if result["status"] == "success":
                    print(result["message"])
                else:
                    print(f"Error: {result['message']}")
            except Exception as e:
                print(f"\nError: {e}")
        else:
            print("Operation cancelled.")

    def show_help(self) -> None:
        """Show help information."""
        help_text = """
HELP GUIDE

1. INGEST DOCUMENTS
    - Load text documents from a directory
    - Supports: TXT, PDF, DOCX, Markdown
    - Processes documents in parallel for speed
    - Creates semantic chunks for better retrieval

2. QUERY DOCUMENTS
    - Ask questions about your documents
    - Returns relevant answers with sources
    - Can display supporting document chunks

3. SUMMARIZE DOCUMENT
    - Get quick summary of any document
    - Choose between short or detailed summaries
    - Great for quick document overviews

4. ANALYZE RESUME
    - Check resume ATS (Applicant Tracking System) score
    - Get strengths and weaknesses analysis
    - Receive actionable suggestions to improve score
    - Identify missing key elements

5. DATABASE STATS
    - View current database statistics
    - Check number of indexed documents

6. CLEAR DATABASE
    - Remove all documents from database
    - Requires confirmation
    - Clears embedding cache as well

TIPS:
  - Start by ingesting documents
  - Then query them or get summaries
  - Use ATS analysis for resume optimization
  - All operations are logged to logs/rag_agent.log
        """
        print(help_text)

    def run(self) -> None:
        """Run the CLI application."""
        self.display_banner()

        while True:
            print()
            self.display_menu()

            choice = input("\nSelect option (0-7): ").strip()

            if choice == "1":
                self.ingest_documents()
            elif choice == "2":
                self.query_documents()
            elif choice == "3":
                self.summarize_document()
            elif choice == "4":
                self.analyze_resume()
            elif choice == "5":
                self.database_stats()
            elif choice == "6":
                self.clear_database()
            elif choice == "7":
                self.show_help()
            elif choice == "0":
                print("\nThank you for using RAG Agent.")
                print("Repository: https://github.com/Ishank2301/chat-with-pdf-rag")
                break
            else:
                print("Invalid option. Please try again.")


def main():
    """Main entry point."""
    try:
        cli = RAGAgentCLI()
        cli.run()
    except KeyboardInterrupt:
        print("\n\nApplication interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal error: {e}")
        logger.error("Fatal error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
