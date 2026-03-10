# Multinational Healthcare Perception Analysis

NLP pipeline for analyzing qualitative interview data on specialized healthcare perceptions across five global cities: **Vancouver, London, Seoul, Bangalore, and Yazd**.

> **Associated publication:**  
> Moayedifar, S., Akhavansadr, S., Taheri, P., Tayebi, S., & Sharifi, A. (2026). *Mapping Local Perspectives on Specialized Healthcare: Insights from Five Global Cities.* Health & Place, 98, 103637.

---

## Overview

This project applies a multi-stage NLP pipeline to semi-structured interview data collected across five cities spanning four languages. The pipeline moves from raw text through preprocessing, sentiment scoring, topic discovery, and cross-city comparison, producing both quantitative outputs and publication-quality visualizations.

**Pipeline stages:**

1. **Text Preprocessing** — Tokenization, lemmatization, custom stopword filtering, and normalization of domain-specific multi-word expressions (e.g., `health care system` → `healthcare`)
2. **Sentiment Analysis** — VADER compound scoring with category-level distribution plots and cross-category comparison
3. **Topic Modeling** — BERTopic with `all-MiniLM-L6-v2` sentence embeddings, UMAP dimensionality reduction, and HDBSCAN clustering
4. **Co-occurrence Network** — Term co-occurrence graph with weighted edges and force-directed layout
5. **Correlation Analysis** — Spearman rank correlations across interview categories with significance annotation
6. **Cross-City Comparison** — Mean sentiment heatmap, grouped bar chart, radar chart, violin distributions, per-city profiles, and bump chart of city rankings

---

## Repository Structure

```
├── healthcare_nlp_analysis.py   # Full pipeline (single-file)
├── data/
│   └── Interview Answers.csv    # Input data (not included — see Data section)
├── analysis_results/            # Auto-generated output directory
│   ├── processed_data.csv
│   ├── combined_wordclouds.png
│   ├── topic_visualization.png
│   ├── correlation_heatmap.png
│   ├── heatmap.png
│   ├── radar_chart.png
│   ├── bump_chart.png
│   └── ...
├── logs/
└── requirements.txt
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/healthcare-nlp-analysis.git
cd healthcare-nlp-analysis

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Download required NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('vader_lexicon'); nltk.download('averaged_perceptron_tagger')"
```

---

## Usage

1. Place your interview CSV at `data/Interview Answers.csv`  
   Expected columns: `Interviewee`, `Location`, and one column per interview category.

2. Run the pipeline:
```bash
python healthcare_nlp_analysis.py
```

Results are saved to a timestamped folder under `analysis_results/`.

---

## Data

The raw interview data is not included in this repository to protect participant confidentiality. The CSV should follow this structure:

| Interviewee | Location  | General Experience with the Health System | Accessibility and Justice | ... |
|-------------|-----------|-------------------------------------------|---------------------------|-----|
| P01         | Vancouver | Response text...                          | Response text...          | ... |

---

## Dependencies

| Package              | Purpose                              |
|----------------------|--------------------------------------|
| `bertopic`           | Topic modeling                       |
| `sentence-transformers` | Sentence embeddings (MiniLM)      |
| `umap-learn`         | Dimensionality reduction             |
| `hdbscan`            | Density-based clustering             |
| `nltk`               | Tokenization, lemmatization, VADER   |
| `scikit-learn`       | Vectorization, preprocessing         |
| `pandas`, `numpy`    | Data manipulation                    |
| `matplotlib`, `seaborn`, `plotly` | Visualization          |
| `wordcloud`          | Word cloud generation                |
| `networkx`           | Co-occurrence network                |
| `scipy`              | Statistical tests (Spearman)         |

See `requirements.txt` for pinned versions.

---

## Key Design Decisions

- **VADER over transformer-based sentiment** — Interview responses are short and opinion-dense; VADER's lexicon-based approach performs competitively on this register and is fully interpretable.
- **BERTopic over LDA** — Sentence-transformer embeddings capture semantic similarity that bag-of-words models miss, especially important for cross-lingual content where surface-form overlap is limited.
- **Spearman over Pearson correlation** — Sentiment scores are not assumed normally distributed; Spearman is robust to the small per-city sample sizes in this dataset.
- **Custom stopword list** — Standard English stopwords were supplemented with domain-specific terms (city names, interview scaffolding verbs, ubiquitous healthcare nouns) that carry no discriminative signal across categories.

---

## Citation

If you use this code, please cite:

```bibtex
@article{moayedifar2026mapping,
  title   = {Mapping Local Perspectives on Specialized Healthcare: Insights from Five Global Cities},
  author  = {Moayedifar, S. and Akhavansadr, S. and Taheri, P. and Tayebi, S. and Sharifi, A.},
  journal = {Health \& Place},
  volume  = {98},
  pages   = {103637},
  year    = {2026}
}
```
