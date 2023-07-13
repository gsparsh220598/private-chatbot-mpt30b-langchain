# Python script to scrape arXiv.org using the arxiv python package

# Import the necessary libraries
import arxiv
import os

anthropic_research = [
    "Towards Measuring the Representation of Subjective Global Opinions in Language Models",
    "The Capacity for Moral Self-Correction in Large Language Models",
    "Discovering Language Model Behaviors with Model-Written Evaluations",
    "Constitutional AI: Harmlessness from AI Feedback",
    "Measuring Progress on Scalable Oversight for Large Language Models",
    "Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned",
    "Language Models (Mostly) Know What They Know",
    "Scaling Laws and Interpretability of Learning from Repeated Data",
    "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback",
    "Predictability and Surprise in Large Generative Models",
    "A General Language Assistant as a Laboratory for Alignment",
]


# Define your search query parameters
def query_arxiv(search_query, start=0, max_results=1):
    search = arxiv.Search(
        query=search_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    return search


download_dir = "./source_documents/anthropic/alignment/"
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

# Perform the search
for query in anthropic_research:
    results = query_arxiv(query, max_results=1).results()
    for result in results:
        query = query.replace(" ", "_")
        result.download_pdf(dirpath=download_dir, filename=f"{query}.pdf")
