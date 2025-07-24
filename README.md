# PubMed Research Assistant

A powerful, extensible assistant for advanced PubMed research and biomedical literature analysis, with optional AI-powered insights via Google Gemini.

---

## Overview

**PubMed Research Assistant** enables researchers, students, and professionals to:
- Search PubMed for biomedical articles using natural language queries.
- Analyze research trends across topics.
- Perform comparative analysis between research domains.
- Generate visualizations and word clouds from PubMed results.
- (Optional) Use Google Gemini AI for advanced, natural language research questions and summarization.

The assistant is built using Python, LangChain, pandas, matplotlib, seaborn, and integrates with Google Generative AI.

---

## Features

- **PubMed Search**: Query PubMed and retrieve structured article data.
- **Trend Analysis**: Analyze and visualize research trends for multiple topics.
- **Comparative Analysis**: Compare two research topics side-by-side.
- **Word Clouds & Plots**: Visualize keyword and topic distributions.
- **AI Integration (Optional)**: Use Gemini for intelligent Q&A and summarization (requires API key).
- **Caching**: Results are cached for efficiency during interactive sessions.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd pubmed-insight-langchain
   ```
2. **(Recommended) Create a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

You can run the main demonstration script directly:

```bash
python main.py
```

This will:
- Demonstrate basic PubMed search
- Analyze research trends for sample topics
- Perform a comparative analysis
- Show how to enable AI-powered features

### Enabling AI Features (Optional)
To use Google Gemini for advanced queries, get a free API key at [makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey).

Uncomment and set your key in `main.py`:
```python
# researcher = AdvancedPubMedResearcher(gemini_api_key="your-gemini-api-key")
```

---

## Notes
- For AI features, you must provide a Gemini API key.
- The project is for educational and research purposes. For large-scale or production use, review PubMed and Google API terms.
