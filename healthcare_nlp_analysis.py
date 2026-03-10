"""
Multinational Healthcare Perception Analysis
=============================================
NLP pipeline for analyzing qualitative interview data on specialized healthcare
perceptions across five cities: Vancouver, London, Seoul, Bangalore, and Yazd.

Methods: Text preprocessing, VADER sentiment analysis, BERTopic topic modeling,
co-occurrence network analysis, and cross-city comparative visualization.

Published: Moayedifar, S., Akhavansadr, S., et al. (2026).
"Mapping Local Perspectives on Specialized Healthcare: Insights from Five Global Cities."
Health & Place, 98, 103637.

Usage:
    Place your interview CSV at the path specified in main() and run:
        python healthcare_nlp_analysis.py
"""

# =============================================================================
# Imports
# =============================================================================

import os
import logging
import statistics
from collections import Counter
from datetime import datetime
from itertools import combinations
from math import pi
from pathlib import Path

import hdbscan
import matplotlib.pyplot as plt
import networkx as nx
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import seaborn as sns
import spacy
import textwrap

from bertopic import BERTopic
from docx import Document
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from scipy import stats
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from umap import UMAP
from wordcloud import WordCloud


# =============================================================================
# Step 1: Text Preprocessing
# =============================================================================

class TextPreprocessor:
    """
    Preprocesses raw interview text for NLP analysis.

    Handles tokenization, lemmatization, stopword removal, and normalization
    of domain-specific multi-word expressions (e.g., 'health care' -> 'healthcare').
    """

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.custom_stops = self._get_custom_stopwords()
        self.stop_words.update(self.custom_stops)
        self._add_lemmatized_forms()

    # -------------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------------

    def preprocess_text(self, text):
        """Tokenize, normalize, lemmatize, and filter a raw text string."""
        if pd.isna(text):
            return ""

        text = str(text).lower()

        # Join multi-word healthcare phrases before tokenization
        for phrase in ['health care system', 'health-care system', 'healthcare system', 'local resident']:
            text = text.replace(phrase, phrase.replace(' ', '-'))

        tokens = word_tokenize(text)
        processed_tokens = []

        for token in tokens:
            cleaned = self._clean_word(token)
            if cleaned.isalnum():
                normalized = self._normalize_word(cleaned)
                lemma = self.lemmatizer.lemmatize(normalized)
                if lemma not in self.stop_words:
                    processed_tokens.append(lemma)

        return ' '.join(processed_tokens)

    def verify_filtered_words(self, text):
        """
        Audit which words were removed during preprocessing and why.
        Returns a DataFrame categorizing filtered words.
        """
        original_words = set(word_tokenize(text.lower()))
        processed_words = set(word_tokenize(self.preprocess_text(text).lower()))
        filtered_words = original_words - processed_words

        filter_results = []
        for word in filtered_words:
            filter_results.append({
                'word': word,
                'is_stop': word in self.stop_words,
                'is_custom_stop': word in self.custom_stops,
                'is_short': len(word) <= 3,
                'is_non_alpha': not word.isalnum(),
                'length': len(word)
            })

        df = pd.DataFrame(filter_results)
        df.to_csv('filtered_words_audit.csv', index=False)
        return df

    def identify_potential_stopwords(self, texts, min_freq=0.8, max_meaningful_length=3):
        """
        Identify candidate stopwords from a corpus based on document frequency,
        word length, and POS tag heuristics.

        Returns a set of candidate stopwords and a frequency DataFrame.
        """
        all_words = []
        doc_word_sets = []

        for text in texts:
            if pd.isna(text):
                continue
            words = [w for w in word_tokenize(str(text).lower()) if w.isalnum()]
            all_words.extend(words)
            doc_word_sets.append(set(words))

        word_freq = Counter(all_words)
        total_docs = len(doc_word_sets)

        doc_freq = {
            word: sum(1 for doc in doc_word_sets if word in doc) / total_docs
            for word in set(all_words)
        }

        pos_dict = dict(nltk.pos_tag(list(set(all_words))))

        potential_stopwords = set()
        for word, freq in word_freq.items():
            tag = pos_dict.get(word, '')
            if (doc_freq[word] >= min_freq
                    or len(word) <= max_meaningful_length
                    or tag.startswith(('DT', 'IN', 'CC', 'PRP', 'TO'))):
                for form in [word,
                             self.lemmatizer.lemmatize(word),
                             self.lemmatizer.lemmatize(word, pos='v')]:
                    potential_stopwords.add(form)

        word_stats = pd.DataFrame({
            'word': list(word_freq.keys()),
            'frequency': [word_freq[w] for w in word_freq],
            'doc_frequency': [doc_freq[w] for w in word_freq],
            'length': [len(w) for w in word_freq],
            'pos_tag': [pos_dict.get(w, '') for w in word_freq],
            'is_potential_stop': [w in potential_stopwords for w in word_freq]
        }).sort_values('frequency', ascending=False)

        return potential_stopwords, word_stats

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _clean_word(self, word):
        """Strip punctuation and lowercase."""
        return re.sub(r'[^\w\s]', '', word.lower())

    def _normalize_word(self, word):
        """
        Collapse surface-form variants into canonical hyphenated forms.
        E.g., 'nonlocal', 'non local', 'non-locals' -> 'non-local'.
        """
        variations = {
            'non-local': ['nonlocal', 'non-locals', 'nonlocals', 'non local'],
            'non-resident': ['nonresident', 'non-residents', 'nonresidents', 'non resident'],
            'local-resident': ['local resident', 'local residents', 'local-residents'],
            'co-creation': ['cocreation', 'co creation'],
            'healthcare': ['health-care', 'health care', 'health care system',
                          'health-care system', 'healthcare system'],
            'follow-up': ['followup', 'follow up'],
            'real-time': ['realtime', 'real time'],
            'day-to-day': ['daytoday', 'day to day'],
        }
        for standard, variants in variations.items():
            if word in variants or word == standard:
                return standard
            if ' ' in word and any(v in word for v in variants):
                return standard
        return word

    def _add_lemmatized_forms(self):
        """Expand stop word list with lemmatized verb and noun forms."""
        extras = set()
        for word in self.stop_words:
            extras.add(self.lemmatizer.lemmatize(word))
            extras.add(self.lemmatizer.lemmatize(word, pos='v'))
        self.stop_words.update(extras)

    def _get_custom_stopwords(self):
        """
        Domain-specific stopwords for healthcare interview data.
        Includes city names, generic interview scaffolding words, and
        high-frequency terms that carry little discriminative signal.
        """
        return {
            # City/country names (spatial identifiers, not substantive content)
            'yazd', 'london', 'vancouver', 'bangalore', 'iran',
            'canada', 'uk', 'india', 'seoul', 'korea',

            # Interview scaffolding
            'thank', 'interview', 'interviewee', 'study', 'respondent',
            'research', 'questions', 'share', 'like', 'feel', 'believe',
            'think', 'opinion', 'noted', 'yes', 'no', 'certainly',
            'definitely', 'challenge', 'challenges', 'highlight', 'ensure',
            'face', 'similar', 'across', 'might', 'relatively', 'general',
            'system', 'systems', 'systemic', 'including', 'local-resident',

            # Generic evaluative terms
            'significant', 'significantly', 'considerable', 'considerably',
            'adequate', 'inadequate', 'positive', 'negative', 'noticeable',
            'noticeably',

            # Generic descriptors
            'specific', 'specialized', 'specialization', 'various', 'different',
            'certain', 'particularly', 'especially', 'related', 'regarding',
            'existing', 'current', 'present',

            # Time references
            'time', 'future', 'during', 'over', 'currently', 'often',
            'sometimes', 'yet', 'within', 'overall',

            # Location/institution terms
            'city', 'cities', 'area', 'region', 'public', 'private',
            'center', 'centers', 'ministry', 'hospital', 'hospitals',
            'clinic', 'clinics',

            # Qualifiers and connectors
            'very', 'much', 'many', 'some', 'most', 'further', 'more',
            'due', 'despite', 'also', 'though', 'although', 'furthermore',
            'moreover', 'nevertheless', 'however', 'therefore', 'thus', 'hence',

            # Healthcare domain terms (too ubiquitous to discriminate)
            'service', 'services', 'patient', 'patients', 'healthcare',
            'health', 'medical', 'medicine', 'doctor', 'doctors', 'care',
            'caring', 'treatment', 'treatments', 'facility', 'facilities',

            # Common verbs
            'provide', 'provides', 'provided', 'providing', 'need', 'needs',
            'needed', 'needing', 'use', 'uses', 'used', 'using', 'seem',
            'seems', 'work', 'works', 'worked', 'working', 'considered',
            'considers', 'access',

            # Reporting/speech verbs
            'say', 'said', 'saying', 'tells', 'told', 'telling', 'explain',
            'explains', 'explained', 'explaining', 'describe', 'describes',
            'described', 'describing', 'mention', 'mentions', 'mentioned',
            'remain', 'answer', 'answers', 'answered', 'ask', 'asks',
            'asked', 'asking', 'response', 'responses', 'responded',
            'responding',

            # Cognition/belief verbs (interview artifacts)
            'believes', 'believed', 'believing', 'thinks', 'thought',
            'thinking', 'feels', 'felt', 'feeling', 'shares', 'shared',
            'sharing', 'ensures', 'ensured', 'ensuring', 'faces', 'faced',
            'facing', 'highlights', 'highlighted', 'highlighting',

            # Meta-interview terms
            'interviews', 'interviewed', 'interviewing', 'respondents',
            'local', 'resident', 'locals', 'residents', 'participant',
            'participants', 'participated', 'participating', 'healthcaresystem',
        }


# =============================================================================
# Step 2: Sentiment Analysis
# =============================================================================

class SentimentAnalyzer:
    """
    Sentiment analysis using VADER (Valence Aware Dictionary and sEntiment Reasoner).

    VADER is well-suited to short, opinion-dense texts such as interview
    excerpts. Compound scores range from -1 (most negative) to +1 (most positive).
    """

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, texts):
        """
        Compute VADER sentiment scores for a collection of texts.

        Returns a dict with compound scores, full score DataFrame, and summary stats.
        """
        sentiments = [self.sia.polarity_scores(str(t)) for t in texts]
        df = pd.DataFrame(sentiments)
        df['category'] = df['compound'].apply(self._interpret_sentiment)

        return {
            'compound_scores': df['compound'].tolist(),
            'detailed_scores': df,
            'summary': self._generate_summary(df),
        }

    def plot_sentiment_distribution(self, scores, title, save_path=None):
        """Histogram + boxplot of compound sentiment scores for a single category."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        sns.histplot(scores, bins=20, ax=ax1)
        ax1.set_title(f'Sentiment Distribution — {title}', fontsize=18, fontweight='bold')
        ax1.set_xlabel('Sentiment Score', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Count', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='both', labelsize=14)
        ax1.axvline(x=-0.05, color='r', linestyle='--', alpha=0.5)
        ax1.axvline(x=0.05, color='r', linestyle='--', alpha=0.5)
        ax1.text(-0.5, ax1.get_ylim()[1] * 0.9, 'Negative', ha='center',
                 fontsize=14, fontweight='bold')
        ax1.text(0,   ax1.get_ylim()[1] * 0.9, 'Neutral',  ha='center',
                 fontsize=14, fontweight='bold')
        ax1.text(0.5, ax1.get_ylim()[1] * 0.9, 'Positive', ha='center',
                 fontsize=14, fontweight='bold')

        sns.boxplot(y=scores, ax=ax2)
        ax2.set_title('Score Distribution', fontsize=14)
        ax2.set_ylabel('Sentiment Score', fontsize=16, fontweight='bold')

        # Category breakdown annotation
        categories = [self._interpret_sentiment(s) for s in scores]
        counts = pd.Series(categories).value_counts()
        note = "Sentiment breakdown:\n" + "\n".join(
            f"{cat}: {cnt} ({cnt/len(categories)*100:.1f}%)"
            for cat, cnt in counts.items()
        )
        plt.figtext(0.15, 0.02, note,
                    bbox=dict(facecolor='white', alpha=0.8),
                    fontsize=12, fontweight='bold')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()

    def plot_sentiment_comparison(self, sentiment_data, categories,
                                  title="Sentiment Comparison Across Categories",
                                  save_path=None):
        """Boxplot comparing sentiment distributions across interview categories."""
        # Exclude metadata column
        filtered = {k: v for k, v in sentiment_data.items() if k != 'Location'}
        filtered_cats = [c for c in categories if c != 'Location']

        plt.figure(figsize=(16, 10))
        sns.boxplot(data=[filtered[c] for c in filtered_cats])

        plt.ylabel('Sentiment Score', fontsize=18, fontweight='bold')

        y_min, y_max = plt.ylim()
        pad = (y_max - y_min) * 0.25
        plt.ylim(y_min - pad / 2, y_max + pad)

        multiline_labels = [
            'General\nExperience', 'Accessibility\nand Justice',
            'Impact of\nSpecialization', 'Cultural and\nSocial Impacts',
            'Economic\nImpacts', 'Infrastructure\nImpacts',
            'Institutional\nImpacts',
        ]
        plt.xticks(range(len(filtered_cats)), multiline_labels,
                   rotation=0, fontsize=14, fontweight='bold', ha='center')
        plt.yticks(fontsize=14, fontweight='bold')

        means = [np.mean(filtered[c]) for c in filtered_cats]
        for i, mean in enumerate(means):
            plt.text(i, plt.ylim()[0] + 0.04, f'Mean: {mean:.2f}',
                     fontsize=14, fontweight='bold', ha='center', va='bottom')

        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.subplots_adjust(bottom=0.2)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _interpret_sentiment(self, score):
        if score >= 0.05:
            return 'Positive'
        elif score <= -0.05:
            return 'Negative'
        return 'Neutral'

    def _generate_summary(self, df):
        return {
            'mean_compound':  df['compound'].mean(),
            'std_compound':   df['compound'].std(),
            'positive_count': (df['compound'] >= 0.05).sum(),
            'neutral_count':  (df['compound'].abs() < 0.05).sum(),
            'negative_count': (df['compound'] <= -0.05).sum(),
            'mean_pos': df['pos'].mean(),
            'mean_neg': df['neg'].mean(),
            'mean_neu': df['neu'].mean(),
        }


# =============================================================================
# Step 3: Visualization Tools
# =============================================================================

class VisualizationTools:
    """
    Word cloud, co-occurrence network, and multi-panel visualization utilities.
    """

    CATEGORY_COLORS = {
        'General Experience':        '#4B0082',
        'Accessibility and Justice': '#1E90FF',
        'Impact of Specialization':  '#228B22',
        'Cultural and Social Impacts': '#8B008B',
        'Economic Impacts':          '#FF4500',
        'Infrastructure Impacts':    '#20B2AA',
        'Institutional Impacts':     '#4682B4',
    }

    def __init__(self, stop_words):
        self.stop_words = stop_words

    def create_wordcloud(self, text, title, save_path=None):
        """
        Word cloud + top-20 frequency bar chart for a single text corpus.
        Returns a word-frequency DataFrame.
        """
        try:
            stop_words = {w.lower() for w in self.stop_words}
            text = str(text).lower()
            filtered_text = ' '.join(
                w for w in text.split() if w not in stop_words
            )

            wc = WordCloud(
                background_color='white', width=800, height=400,
                max_words=100, stopwords=stop_words,
                collocations=False, min_word_length=4,
            ).generate(filtered_text)

            freq_df = (
                pd.DataFrame(wc.process_text(filtered_text).items(),
                             columns=['Word', 'Frequency'])
                .sort_values('Frequency', ascending=False)
            )

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
            plt.subplots_adjust(hspace=1)

            ax1.imshow(wc, interpolation='bilinear')
            ax1.axis('off')
            ax1.set_title('(A)', pad=10, fontsize=20, fontweight='bold')

            top = freq_df.head(20)
            bars = ax2.bar(range(len(top)), top['Frequency'])
            ax2.set_xticks(range(len(top)))
            ax2.set_xticklabels(top['Word'], rotation=45, ha='right',
                                fontsize=14, fontweight='bold')
            ax2.set_title('(B)', pad=10, fontsize=20, fontweight='bold')
            ax2.set_xlabel('Words', fontsize=18, fontweight='bold')
            ax2.set_ylabel('Frequency', fontsize=18, fontweight='bold')
            ax2.grid(axis='y', linestyle='--', alpha=0.7, color='gray')
            ax2.tick_params(axis='y', labelsize=14)
            for label in ax2.get_yticklabels():
                label.set_fontweight('bold')
            for bar in bars:
                h = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2, h,
                         str(int(h)), ha='center', va='bottom',
                         fontsize=12, fontweight='bold')

            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.show()
            return freq_df

        except Exception as e:
            print(f"Error creating word cloud for '{title}': {e}")
            return pd.DataFrame()

    def create_combined_wordclouds(self, texts_dict, save_path=None):
        """
        2-row grid of word clouds (one per interview category),
        with the last cloud spanning both rows.
        """
        fig = plt.figure(figsize=(24, 12))
        gs = plt.GridSpec(2, 4, figure=fig,
                          wspace=0.001, hspace=0.001,
                          left=0.001, right=0.999,
                          top=0.95, bottom=0.08)

        for idx, (category, text) in enumerate(texts_dict.items()):
            if idx < 3:
                ax = fig.add_subplot(gs[0, idx])
                w, h = 1200, 1000
            elif idx < 6:
                ax = fig.add_subplot(gs[1, idx - 3])
                w, h = 1200, 1000
            else:
                ax = plt.subplot(gs[:, -1])
                w, h = 1200, 2000

            color = self.CATEGORY_COLORS.get(category, '#333333')
            filtered = ' '.join(
                word for word in str(text).lower().split()
                if word not in self.stop_words
            )
            wc = WordCloud(
                background_color='white', width=w, height=h,
                max_words=100, stopwords=self.stop_words,
                collocations=False, min_word_length=4,
                color_func=lambda *a, **kw: color,
                prefer_horizontal=0.6,
                min_font_size=8, max_font_size=150, random_state=42,
            ).generate(filtered)

            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')

        self._draw_legend_boxes(fig, self.CATEGORY_COLORS, y_position=0.02)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300,
                        facecolor='white', pad_inches=0)
        plt.show()

    def create_cooccurrence_network(self, text, min_edge_weight=3, max_nodes=30):
        """
        Build and visualize a term co-occurrence network.

        Nodes = terms; edges = co-occurrence within a sliding window.
        Edge weight = co-occurrence count. Returns the NetworkX graph.
        """
        pairs = self._extract_word_pairs(text, window_size=5)
        pair_counts = Counter(pairs)
        significant = {p: c for p, c in pair_counts.items()
                       if c >= min_edge_weight}

        if not significant:
            print(f"No term pairs meet min_edge_weight={min_edge_weight}.")
            return None

        G = nx.Graph()
        for (u, v), weight in significant.items():
            G.add_edge(u, v, weight=weight)

        # Keep only the top-N most connected nodes
        strengths = dict(G.degree(weight='weight'))
        top_nodes = sorted(strengths, key=strengths.get, reverse=True)[:max_nodes]
        G = G.subgraph(top_nodes).copy()

        if not G.nodes():
            print("No nodes remain after filtering.")
            return None

        self._draw_network(G)
        return G

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _extract_word_pairs(self, text, window_size=5):
        words = [
            w for w in text.lower().split()
            if w not in self.stop_words and len(w) > 3
            and w.isalpha()
        ]
        pairs = []
        for i in range(len(words)):
            for j in range(i + 1, min(i + window_size, len(words))):
                if words[i] != words[j]:
                    pairs.append(tuple(sorted([words[i], words[j]])))
        return pairs

    def _draw_network(self, G):
        degrees = dict(G.degree(weight='weight'))
        max_deg = max(degrees.values())
        node_sizes = [3000 * (degrees[n] / max_deg) for n in G.nodes()]
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        max_w = max(edge_weights)

        plt.figure(figsize=(15, 15), facecolor='white')
        pos = nx.spring_layout(G, k=3.5, iterations=100, seed=42)

        for u, v, w in G.edges(data='weight'):
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                                   alpha=min(0.1 + (w / max_w) * 0.5, 0.7),
                                   width=2 * (w / max_w),
                                   edge_color='#1f77b4')

        nodes = nx.draw_networkx_nodes(G, pos,
                                       node_size=node_sizes,
                                       node_color=list(degrees.values()),
                                       cmap=plt.cm.Blues, alpha=0.7)

        for node in G.nodes():
            size = degrees[node]
            label_size = min(12 + (size / max_deg) * 12, 18)
            plt.text(pos[node][0], pos[node][1], node,
                     fontsize=label_size, ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.3', fc='white',
                               ec='gray', alpha=0.8))

        cbar = plt.colorbar(nodes, label='Node Strength (Weighted Degree)')
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label('Node Strength (Weighted Degree)',
                       fontsize=16, fontweight='bold')

        plt.title(
            f"Term Co-occurrence Network\n"
            f"(Top {G.number_of_nodes()} terms, "
            f"min co-occurrences: {min(d['weight'] for *_, d in G.edges(data=True))})",
            fontsize=16, pad=20,
        )
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        print(f"\nNodes: {G.number_of_nodes()}  |  "
              f"Edges: {G.number_of_edges()}  |  "
              f"Density: {nx.density(G):.3f}")
        print("Top 10 terms by weighted degree:")
        for term, w in sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {term}: {w}")

    def _draw_legend_boxes(self, fig, categories_colors, y_position=0.02):
        n = len(categories_colors)
        box_w, box_h = 0.14, 0.04
        spacing = (1.0 - n * box_w) / (n + 1)

        for idx, (cat, color) in enumerate(categories_colors.items()):
            x = spacing + idx * (box_w + spacing)
            fig.add_artist(
                plt.Rectangle((x, y_position), box_w, box_h,
                               facecolor=color, transform=fig.transFigure,
                               alpha=0.8, edgecolor='black', linewidth=1)
            )
            label = cat if len(cat) <= 20 else '\n'.join(
                cat.split()[:len(cat.split()) // 2] +
                [' '.join(cat.split()[len(cat.split()) // 2:])]
            )
            fig.text(x + box_w / 2, y_position - 0.02, label,
                     ha='center', va='top', fontsize=20, fontweight='bold',
                     transform=fig.transFigure,
                     bbox=dict(facecolor='white', alpha=0.7,
                               edgecolor='none', pad=2))


# =============================================================================
# Step 4: Topic Modeling (BERTopic)
# =============================================================================

class TopicModeler:
    """
    Topic modeling using BERTopic with UMAP dimensionality reduction
    and HDBSCAN clustering.

    BERTopic leverages sentence-transformer embeddings to capture semantic
    similarity before clustering, making it more context-aware than LDA.
    """

    def __init__(self, stop_words):
        self.stop_words = stop_words
        self.model = None
        self.topics = None
        self.topic_info = None
        self.filtered_topics = {}

    def perform_topic_modeling(self, texts, min_topic_size=5, n_neighbors=5):
        """
        Fit a BERTopic model on preprocessed interview texts.

        Returns a dict with topics, probabilities, topic_info, model, and
        filtered_topics (excluding the outlier cluster -1).
        """
        try:
            processed = [str(t).strip() for t in texts if str(t).strip()]
            if len(processed) < min_topic_size:
                print(f"Insufficient documents ({len(processed)} < {min_topic_size}).")
                return None

            print(f"Fitting BERTopic on {len(processed)} documents...")

            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

            umap_model = UMAP(
                n_neighbors=3, n_components=10,
                min_dist=0.1, metric='cosine', random_state=42,
            )

            hdbscan_model = hdbscan.HDBSCAN(
                min_cluster_size=2, min_samples=1,
                metric='euclidean', cluster_selection_epsilon=0.1,
                prediction_data=True,
            )

            vectorizer = CountVectorizer(
                stop_words=list(self.stop_words),
                min_df=1, max_df=0.90,
                token_pattern=r'(?u)\b[A-Za-z]{4,}\b',
            )

            self.model = BERTopic(
                embedding_model=embedding_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer,
                min_topic_size=3,
                nr_topics=5,
                calculate_probabilities=True,
                verbose=True,
            )

            self.topics, probs = self.model.fit_transform(processed)
            self.topic_info = self.model.get_topic_info()

            self.filtered_topics = {
                tid: words[:10]
                for tid, words in self.model.get_topics().items()
                if tid != -1 and words
            }

            print("\nDiscovered topics:")
            for tid, words in self.filtered_topics.items():
                print(f"  Topic {tid}: {', '.join(w for w, _ in words[:5])}")

            return {
                'topics': self.topics,
                'probabilities': probs,
                'topic_info': self.topic_info,
                'model': self.model,
                'filtered_topics': self.filtered_topics,
            }

        except Exception as e:
            print(f"Topic modeling error: {e}")
            return None

    def visualize_topics(self, save_path=None):
        """Bar chart of topic sizes + document distribution across topics."""
        if self.model is None:
            print("Run perform_topic_modeling() first.")
            return

        try:
            rows = [
                {'Topic': tid,
                 'Words': ', '.join(w for w, _ in words[:5]),
                 'Size': sum(s for _, s in words)}
                for tid, words in self.filtered_topics.items()
            ]
            topics_df = pd.DataFrame(rows)

            if topics_df.empty:
                print("No topics to visualize.")
                return

            plt.figure(figsize=(20, 10))

            ax1 = plt.subplot(1, 2, 1)
            ax1.barh(topics_df['Topic'].astype(str), topics_df['Size'])
            ax1.grid(axis='x', linestyle='--', alpha=0.4)
            ax1.set_title('(A)', fontsize=24, fontweight='bold')
            ax1.set_xlabel('Size', fontsize=20, fontweight='bold')
            ax1.set_ylabel('Topic', fontsize=20, fontweight='bold')
            ax1.tick_params(labelsize=16)
            for label in ax1.get_xticklabels() + ax1.get_yticklabels():
                label.set_fontweight('bold')

            ax2 = plt.subplot(1, 2, 2)
            if self.topics:
                topic_dist = pd.Series(
                    [t for t in self.topics if t != -1]
                ).value_counts()
                topic_dist.plot(kind='bar', ax=ax2)
                ax2.grid(axis='y', linestyle='--', alpha=0.4)
                ax2.set_title('(B)', fontsize=24, fontweight='bold')
                ax2.set_xlabel('Topic', fontsize=20, fontweight='bold')
                ax2.set_ylabel('Documents', fontsize=20, fontweight='bold')
                ax2.tick_params(labelsize=16)
                for label in ax2.get_xticklabels() + ax2.get_yticklabels():
                    label.set_fontweight('bold')

            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.show()

            print("\nTopic words:")
            for _, row in topics_df.iterrows():
                print(f"  Topic {row['Topic']}: {row['Words']}")

        except Exception as e:
            print(f"Topic visualization error: {e}")


# =============================================================================
# Step 5: Correlation Analysis
# =============================================================================

class CorrelationAnalyzer:
    """
    Spearman rank correlation analysis across interview response categories.

    Spearman is used (over Pearson) because sentiment scores are not
    assumed to be normally distributed and the sample sizes per city
    are small.
    """

    def __init__(self):
        self.correlation_matrix = None
        self.pvalues_matrix = None

    def calculate_correlations(self, sentiment_data):
        """
        Compute pairwise Spearman correlations and p-values across categories.

        Parameters
        ----------
        sentiment_data : dict
            {category_name: [compound_score, ...]}

        Returns
        -------
        correlation_matrix, pvalues_matrix : pd.DataFrame
        """
        df = pd.DataFrame(sentiment_data)
        self.correlation_matrix = df.corr(method='spearman')
        self.pvalues_matrix = pd.DataFrame(
            np.zeros_like(self.correlation_matrix),
            index=self.correlation_matrix.index,
            columns=self.correlation_matrix.columns,
        )
        for i in self.correlation_matrix.index:
            for j in self.correlation_matrix.columns:
                _, p = stats.spearmanr(df[i].dropna(), df[j].dropna())
                self.pvalues_matrix.loc[i, j] = p

        return self.correlation_matrix, self.pvalues_matrix

    def plot_correlation_heatmap(self, save_path=None):
        """
        Lower-triangle heatmap with significance stars annotated on each cell.
        * p<0.05  ** p<0.01  *** p<0.001
        """
        if self.correlation_matrix is None:
            raise ValueError("Run calculate_correlations() first.")

        # Shorten verbose category name for display
        matrix = self.correlation_matrix.rename(
            index={'General Experience with the Health System': 'General Experience'},
            columns={'General Experience with the Health System': 'General Experience'},
        )

        # Build annotation matrix (value + significance stars)
        annot = matrix.round(2).astype(str)
        pval = self.pvalues_matrix.rename(
            index={'General Experience with the Health System': 'General Experience'},
            columns={'General Experience with the Health System': 'General Experience'},
        )
        for i in matrix.index:
            for j in matrix.columns:
                p = pval.loc[i, j]
                stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                annot.loc[i, j] += stars

        mask = np.triu(np.ones_like(matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(
            matrix, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
            square=True, annot=annot, fmt='', linewidths=0.5,
            cbar_kws={'shrink': 0.5},
            annot_kws={'fontweight': 'bold', 'fontsize': 12},
        )

        # Wrap axis tick labels
        def wrap_label(label):
            if 'Impacts' in label:
                return label.split(' Impacts')[0] + '\nImpacts'
            if ' and ' in label:
                return label.replace(' and ', '\nand ')
            if ' of ' in label:
                return label.replace(' of ', '\nof ')
            return label

        ax.set_xticklabels(
            [wrap_label(t.get_text()) for t in ax.get_xticklabels()],
            rotation=0, fontsize=10, ha='center', fontweight='bold',
        )
        ax.set_yticklabels(
            [wrap_label(t.get_text()) for t in ax.get_yticklabels()],
            fontsize=10, va='center', fontweight='bold',
        )

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=12)
        for label in cbar.ax.get_yticklabels():
            label.set_fontweight('bold')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()

    def generate_correlation_report(self):
        """
        Return a DataFrame of all statistically significant (p<0.05)
        pairwise correlations with strength and direction labels.
        """
        if self.correlation_matrix is None:
            raise ValueError("Run calculate_correlations() first.")

        rows = []
        for i in self.correlation_matrix.index:
            for j in self.correlation_matrix.columns:
                if i == j:
                    continue
                r = self.correlation_matrix.loc[i, j]
                p = self.pvalues_matrix.loc[i, j]
                if p < 0.05 and r != 0:
                    rows.append({
                        'aspect_1':  i,
                        'aspect_2':  j,
                        'spearman_r': r,
                        'p_value':   p,
                        'strength':  ('strong' if abs(r) > 0.7
                                      else 'moderate' if abs(r) > 0.4
                                      else 'weak'),
                        'direction': 'positive' if r > 0 else 'negative',
                    })
        return pd.DataFrame(rows)


# =============================================================================
# Step 6: Cross-City Comparative Analysis
# =============================================================================

def analyze_cities_comparison(df):
    """
    Compute mean VADER sentiment scores per city × category,
    plus pairwise Spearman correlations within each city.

    Returns a dict with:
        sentiment_means    : pd.DataFrame (cities × categories)
        sentiment_detailed : dict of raw score lists
        correlations       : dict of per-city correlation matrices
        strongest_correlations : dict of top-3 pairs per city
        cities, categories : lists
    """
    if 'Location' not in df.columns:
        print("No 'Location' column found — skipping city comparison.")
        return None

    sia = SentimentIntensityAnalyzer()
    df = df.copy()
    df['Location'] = df['Location'].str.strip()
    cities = df['Location'].unique()
    categories = [c for c in df.columns
                  if c not in ('Interviewee', 'Location',
                               'combined_text', 'processed_text')]

    city_means, city_detailed = {}, {}

    for city in cities:
        subset = df[df['Location'] == city]
        city_means[city], city_detailed[city] = {}, {}

        for cat in categories:
            responses = subset[cat].dropna()
            if len(responses):
                scores = [sia.polarity_scores(str(r))['compound']
                          for r in responses]
                city_means[city][cat] = np.mean(scores)
                city_detailed[city][cat] = scores
            else:
                city_means[city][cat] = np.nan
                city_detailed[city][cat] = []

    sentiment_df = pd.DataFrame(city_means).T

    # Per-city correlations
    city_corr, strongest = {}, {}
    for city in cities:
        valid = {c: s for c, s in city_detailed[city].items() if s}
        if len(valid) > 1:
            min_len = min(len(s) for s in valid.values())
            corr_df = pd.DataFrame(
                {c: s[:min_len] for c, s in valid.items()}
            )
            city_corr[city] = corr_df.corr(method='spearman')
            upper = city_corr[city].where(
                np.triu(np.ones_like(city_corr[city]), k=1).astype(bool)
            )
            strongest[city] = (
                upper.unstack().dropna().sort_values(ascending=False).head(3)
            )

    return {
        'sentiment_means': sentiment_df,
        'sentiment_detailed': city_detailed,
        'correlations': city_corr,
        'strongest_correlations': strongest,
        'cities': cities,
        'categories': categories,
    }


def create_enhanced_visualizations(comparison_results, results_dir=None):
    """
    Generate six comparative visualizations from cross-city sentiment data:
      (A) Heatmap of mean sentiment scores
      (B) Grouped bar chart
      (C) Radar chart
      (D) Violin/strip distribution plot
      (E) Per-city horizontal bar profiles
      (F) Bump chart of city rankings per category
    """
    if comparison_results is None:
        return

    results_dir = Path(results_dir or Path.cwd() / 'results')
    results_dir.mkdir(parents=True, exist_ok=True)

    sentiment_df = comparison_results['sentiment_means']
    categories = comparison_results['categories']

    RENAME = {
        'General Experience with the Health System': 'General Experience',
        'Accessibility and Justice':  'Accessibility\nand Justice',
        'Impact of Specialization':   'Impact of\nSpecialization',
        'Cultural and Social Impacts': 'Cultural and\nSocial Impacts',
        'Economic Impacts':           'Economic\nImpacts',
        'Infrastructure Impacts':     'Infrastructure\nImpacts',
        'Institutional Impacts':      'Institutional\nImpacts',
    }
    sentiment_df = sentiment_df.rename(columns=RENAME)
    categories = [RENAME.get(c, c) for c in categories]

    def wrap(text, width=15):
        return textwrap.fill(str(text), width)

    # ── (A) Heatmap ──────────────────────────────────────────────────────────
    plt.figure(figsize=(14, 10))
    ax = sns.heatmap(sentiment_df, annot=True, cmap='RdYlBu_r', center=0,
                     fmt='.2f', linewidths=0.5,
                     mask=np.isnan(sentiment_df.values),
                     annot_kws={'fontsize': 12, 'fontweight': 'bold'})
    plt.ylabel('City', fontsize=16, fontweight='bold')
    plt.xlabel('Category', fontsize=16, fontweight='bold')
    plt.xticks(np.arange(len(categories)) + 0.5,
               [wrap(c) for c in categories],
               rotation=0, fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    cbar = ax.collections[0].colorbar
    cbar.set_label('Mean Sentiment Score', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_fontweight('bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ── (B) Grouped bar chart ─────────────────────────────────────────────────
    sentiment_long = (
        sentiment_df.reset_index()
        .melt(id_vars='index', value_vars=categories,
              var_name='Category', value_name='Sentiment')
        .rename(columns={'index': 'City'})
    )

    plt.figure(figsize=(16, 12))
    ax = sns.barplot(x='City', y='Sentiment', hue='Category',
                     data=sentiment_long, palette='Pastel2')
    plt.axhline(0, color='black', linestyle='-', alpha=0.3)
    plt.ylabel('Mean Sentiment Score', fontsize=16, fontweight='bold')
    plt.xlabel('City', fontsize=16, fontweight='bold', labelpad=20)
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=9, fontweight='bold')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend().remove()
    plt.legend(handles, [wrap(l, 20) for l in labels],
               title='Category', bbox_to_anchor=(0, -0.2, 1, 0.1),
               loc='upper center', ncol=len(labels),
               fontsize=12, title_fontsize=14,
               frameon=True, fancybox=True, shadow=True)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(results_dir / 'grouped_bar.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ── (C) Radar chart ───────────────────────────────────────────────────────
    CITY_COLORS = {
        'Vancouver': '#E74C3C', 'London': '#3498DB',
        'Korea':     '#2ECC71', 'Yazd':   '#F39C12',
        'India':     '#9B59B6',
    }
    N = len(categories)
    angles = [n / N * 2 * pi for n in range(N)] + [0]

    fig = plt.figure(figsize=(14, 12))
    ax = plt.subplot(111, polar=True)
    ax.set_facecolor('#fafafa')

    vals = sentiment_df.values.flatten()
    vals = vals[~np.isnan(vals)]
    y_min = max(np.floor(vals.min() * 10) / 10 - 0.1, -1.0)
    y_max = min(np.ceil(vals.max() * 10) / 10 + 0.1,  1.0)
    ax.set_ylim(y_min, y_max)

    SHORT = {
        'General Experience':         'General\nExperience',
        'Accessibility\nand Justice':  'Accessibility\n& Justice',
        'Impact of\nSpecialization':   'Impact of\nSpecialization',
        'Cultural and\nSocial Impacts': 'Cultural &\nSocial',
        'Economic\nImpacts':           'Economic\nImpacts',
        'Infrastructure\nImpacts':     'Infrastructure\nImpacts',
        'Institutional\nImpacts':      'Institutional\nImpacts',
    }
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([SHORT.get(c, c) for c in categories],
                       fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', pad=35)

    for i, city in enumerate(sentiment_df.index):
        vals_city = np.nan_to_num(sentiment_df.loc[city].values).tolist()
        vals_city += vals_city[:1]
        color = CITY_COLORS.get(city, plt.cm.Set1(i))
        ax.plot(angles, vals_city, linewidth=3, linestyle='-',
                label=city, color=color, marker='o', markersize=8,
                markerfacecolor=color, markeredgecolor='white',
                markeredgewidth=2)
        ax.fill(angles, vals_city, color=color, alpha=0.1)

    ax.grid(True, linestyle='-', alpha=0.3, color='gray', linewidth=0.5)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(1)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5,
               frameon=True, fancybox=True, shadow=True,
               prop={'weight': 'bold', 'size': 14},
               handlelength=3, columnspacing=2.0, borderpad=1.5)
    plt.subplots_adjust(top=0.85, bottom=0.2, left=0.05, right=0.95)
    plt.savefig(results_dir / 'radar_chart.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close()

    # ── (D) Violin + strip distribution ──────────────────────────────────────
    plt.figure(figsize=(15, 10))
    ax = sns.violinplot(x='Category', y='Sentiment', data=sentiment_long,
                        palette='viridis', inner='box')
    sns.stripplot(x='Category', y='Sentiment', data=sentiment_long,
                  color='black', size=4, alpha=0.3)
    plt.ylabel('Sentiment Score', fontsize=14)
    plt.xlabel('Category', fontsize=14)
    ax.set_xticklabels([wrap(t.get_text()) for t in ax.get_xticklabels()],
                       rotation=0, fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'violin_distribution.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # ── (E) Per-city horizontal bar profiles ──────────────────────────────────
    n_cities = len(sentiment_df.index)
    n_cols = min(3, n_cities)
    n_rows = int(np.ceil(n_cities / n_cols))

    plt.figure(figsize=(18, 12))
    for i, city in enumerate(sentiment_df.index):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        city_data = sentiment_df.loc[city].sort_values(ascending=False)
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(city_data)))
        bars = ax.barh(range(len(city_data)), city_data, color=colors)

        ax.set_yticks(range(len(city_data)))
        ax.set_yticklabels([wrap(c) for c in city_data.index],
                           fontsize=10, fontweight='bold')
        for j, bar in enumerate(bars):
            w = bar.get_width()
            ax.text(w + 0.02 if w >= 0 else w - 0.08,
                    bar.get_y() + bar.get_height() / 2,
                    f'{w:.2f}', va='center', fontsize=10)
        ax.set_title(f'{city}', fontsize=14)
        ax.set_xlim(-1, 1)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.7)
        ax.grid(axis='x', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / 'city_profiles.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # ── (F) Bump chart (city rankings per category) ───────────────────────────
    rank_df = pd.DataFrame({
        cat: sentiment_df[cat].rank(ascending=False)
        for cat in categories
    }, index=sentiment_df.index)

    rank_long = (
        rank_df.reset_index()
        .melt(id_vars='index', value_vars=categories,
              var_name='Category', value_name='Rank')
        .rename(columns={'index': 'City'})
    )

    x_pos = {cat: i for i, cat in enumerate(categories)}
    final_ranks = {
        city: float(rank_long.loc[
            (rank_long['City'] == city) &
            (rank_long['Category'] == categories[-1]),
            'Rank'
        ].values[0])
        for city in rank_df.index
    }

    rank_counts = Counter(final_ranks.values())
    offsets = {}
    for rank_val, count in rank_counts.items():
        same = [c for c, r in final_ranks.items() if r == rank_val]
        for pos, city in enumerate(same):
            offsets[city] = (pos - (count - 1) / 2) * 0.25

    plt.figure(figsize=(16, 9))
    for i, city in enumerate(rank_df.index):
        city_data = rank_long[rank_long['City'] == city]
        xs = [x_pos[c] for c in city_data['Category']]
        ys = city_data['Rank'].values
        color = CITY_COLORS.get(city, plt.cm.tab10.colors[i % 10])

        plt.plot(xs, ys, marker='o', linewidth=3, markersize=10,
                 label=city, color=color,
                 markeredgecolor='white', markeredgewidth=1)
        plt.text(len(categories) - 0.35,
                 final_ranks[city] + offsets[city],
                 f' {city}', fontsize=14, fontweight='bold',
                 va='center', ha='left', color=color)

    plt.gca().invert_yaxis()
    plt.ylabel('Rank', fontsize=16, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.yticks(range(1, len(rank_df.index) + 1),
               [str(i) for i in range(1, len(rank_df.index) + 1)],
               fontsize=16, fontweight='bold')
    plt.xticks(list(x_pos.values()),
               ['\n'.join(c.split('\n')) for c in categories],
               fontsize=12, fontweight='bold')
    for x in x_pos.values():
        plt.axvline(x, color='gray', linestyle='-', alpha=0.1)
    plt.xlim(-0.3, len(categories) + 0.45)
    plt.legend().remove()
    plt.tight_layout(rect=[0, 0.1, 0.97, 0.96])
    plt.savefig(results_dir / 'bump_chart.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"All visualizations saved to: {results_dir}")


# =============================================================================
# Step 7: Interview Analyzer (Orchestration)
# =============================================================================

class InterviewAnalyzer:
    """
    End-to-end orchestration class for the interview analysis pipeline.

    Loads data, runs preprocessing, sentiment analysis, topic modeling,
    co-occurrence networks, correlation analysis, and cross-city comparison.
    """

    def __init__(self, csv_path):
        self._setup_logging()
        self.logger.info("Initializing InterviewAnalyzer...")
        try:
            self.df = self._load_data(csv_path)
            self.preprocessor = TextPreprocessor()
            self.sentiment_analyzer = SentimentAnalyzer()

            all_stops = self.preprocessor.stop_words | self.preprocessor.custom_stops
            self.visualizer = VisualizationTools(all_stops)
            self.topic_modeler = TopicModeler(all_stops)

            self.results_dir = (
                Path('analysis_results') /
                datetime.now().strftime('%Y%m%d_%H%M%S')
            )
            self.results_dir.mkdir(parents=True, exist_ok=True)

        except Exception as e:
            self.logger.error(f"Initialization error: {e}")
            raise

    # -------------------------------------------------------------------------
    # Public pipeline methods
    # -------------------------------------------------------------------------

    def prepare_data(self):
        """Drop empty rows, create combined text column, run preprocessing."""
        self.logger.info("Preparing data...")
        initial = len(self.df)
        self.df = self.df.dropna(how='all')
        self.logger.info(f"Dropped {initial - len(self.df)} empty rows.")

        self.df['combined_text'] = self.df.iloc[:, 1:].apply(
            lambda row: ' '.join(str(v) for v in row if pd.notna(v)), axis=1
        )
        self.df['processed_text'] = self.df['combined_text'].apply(
            self.preprocessor.preprocess_text
        )
        self.df.to_csv(self.results_dir / 'processed_data.csv', index=False)
        self.logger.info("Data preparation complete.")

    def analyze_by_category(self):
        """Run sentiment analysis, word clouds, and word frequency per category."""
        self.logger.info("Running category-level analysis...")
        categories = self._get_analysis_columns()
        results = {}
        all_sentiment = {}

        for cat in categories:
            responses = self.df[cat].dropna()
            if responses.empty:
                self.logger.warning(f"No responses for '{cat}' — skipping.")
                continue

            sentiment = self.sentiment_analyzer.analyze_sentiment(responses)
            all_sentiment[cat] = sentiment['compound_scores']

            self.sentiment_analyzer.plot_sentiment_distribution(
                sentiment['compound_scores'], cat,
                save_path=self.results_dir / f'sentiment_{cat}.png',
            )

            combined = ' '.join(responses)
            results[cat] = {
                'sentiment': sentiment,
                'wordcloud': self.visualizer.create_wordcloud(
                    combined, cat,
                    save_path=self.results_dir / f'wordcloud_{cat}.png',
                ),
                'word_freq': self._word_frequencies(responses, cat),
            }
            self._save_category_results(cat, results[cat])

            s = sentiment['summary']
            print(f"\n[{cat}]  mean={s['mean_compound']:.3f}  "
                  f"+{s['positive_count']} ={s['neutral_count']} -{s['negative_count']}")

        if len(all_sentiment) > 1:
            self.sentiment_analyzer.plot_sentiment_comparison(
                all_sentiment, list(all_sentiment.keys())
            )
            plt.savefig(self.results_dir / 'sentiment_comparison.png',
                        bbox_inches='tight', dpi=300)

        return results

    def analyze_correlations(self):
        """Spearman correlation analysis across sentiment scores per category."""
        self.logger.info("Running correlation analysis...")
        categories = self._get_analysis_columns()
        analyzer = CorrelationAnalyzer()

        sentiment_data = {}
        for cat in categories:
            responses = self.df[cat].dropna()
            if len(responses):
                res = self.sentiment_analyzer.analyze_sentiment(responses)
                sentiment_data[cat] = res['compound_scores']

        if len(sentiment_data) < 2:
            self.logger.warning("Not enough categories for correlation analysis.")
            return None

        corr_matrix, pvalues = analyzer.calculate_correlations(sentiment_data)
        analyzer.plot_correlation_heatmap(
            save_path=self.results_dir / 'correlation_heatmap.png'
        )
        report = analyzer.generate_correlation_report()
        report.to_csv(self.results_dir / 'correlation_report.csv', index=False)
        self.logger.info(f"Found {(report['p_value'] < 0.05).sum()} "
                         f"significant correlations.")
        return {'matrix': corr_matrix, 'pvalues': pvalues, 'report': report}

    def run_complete_analysis(self):
        """
        Execute the full pipeline:
        1. Preprocess data
        2. Overall word cloud
        3. Combined category word clouds
        4. BERTopic topic modeling
        5. Category-level sentiment & word clouds
        6. Spearman correlation analysis
        7. Co-occurrence network
        8. Cross-city comparative visualizations
        """
        print("Starting full analysis pipeline...")

        self.prepare_data()

        # Additional custom-stop filtering on processed text
        self.df['processed_text'] = self.df['processed_text'].apply(
            lambda t: ' '.join(
                w for w in t.split()
                if w not in self.preprocessor.custom_stops
            )
        )

        # Overall word cloud
        all_text = ' '.join(self.df['processed_text'].dropna())
        self.visualizer.create_wordcloud(
            all_text, "All Responses",
            save_path=self.results_dir / 'wordcloud_overall.png',
        )

        # Combined category word clouds
        RENAME = {'General Experience with the Health System': 'General Experience'}
        cats = self._get_analysis_columns()
        texts_by_cat = {
            RENAME.get(c, c): ' '.join(self.df[c].dropna())
            for c in cats
        }
        self.visualizer.create_combined_wordclouds(
            texts_by_cat,
            save_path=self.results_dir / 'combined_wordclouds.png',
        )

        # Topic modeling
        topic_results = self.topic_modeler.perform_topic_modeling(
            texts=self.df['processed_text'].dropna().tolist(),
            min_topic_size=3, n_neighbors=5,
        )
        if topic_results:
            self.topic_modeler.visualize_topics(
                save_path=self.results_dir / 'topic_visualization.png'
            )
            self._save_topic_words(topic_results)

        # Category-level analysis
        category_results = self.analyze_by_category()

        # Correlation analysis
        correlation_results = self.analyze_correlations()

        # Co-occurrence network
        network = self.visualizer.create_cooccurrence_network(
            text=' '.join(self.df['processed_text']),
            min_edge_weight=6, max_nodes=20,
        )

        # Cross-city comparison
        comparison = analyze_cities_comparison(self.df)
        if comparison:
            create_enhanced_visualizations(comparison, self.results_dir)

        print(f"\nAnalysis complete. Outputs saved to: {self.results_dir}")
        return {
            'by_category': category_results,
            'topic_modeling': topic_results,
            'correlations': correlation_results,
            'network': network,
            'city_comparison': comparison,
        }

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _setup_logging(self):
        self.logger = logging.getLogger('InterviewAnalyzer')
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        for handler in (logging.StreamHandler(),
                        logging.FileHandler('interview_analysis.log')):
            handler.setFormatter(fmt)
            self.logger.addHandler(handler)

    def _load_data(self, csv_path):
        for enc in ('utf-8', 'latin1', 'cp1252'):
            try:
                df = pd.read_csv(csv_path, encoding=enc)
                if df.empty:
                    raise ValueError("CSV is empty.")
                self.logger.info(f"Loaded data with {enc} encoding.")
                return df
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not read CSV with any supported encoding.")

    def _get_analysis_columns(self):
        return [c for c in self.df.columns
                if c not in ('Interviewee', 'Location',
                             'combined_text', 'processed_text')]

    def _word_frequencies(self, texts, category):
        words = ' '.join(str(t) for t in texts).split()
        freq_df = pd.Series(words).value_counts().reset_index()
        freq_df.columns = ['word', 'frequency']
        freq_df.to_csv(
            self.results_dir / f'word_freq_{category}.csv', index=False
        )
        return freq_df

    def _save_category_results(self, category, results):
        cat_dir = self.results_dir / category
        cat_dir.mkdir(exist_ok=True)

        s = results['sentiment']
        pd.DataFrame({
            'compound': s['compound_scores'],
            'positive': s['detailed_scores']['pos'],
            'neutral':  s['detailed_scores']['neu'],
            'negative': s['detailed_scores']['neg'],
            'label':    s['detailed_scores']['category'],
        }).to_csv(cat_dir / 'sentiment_scores.csv', index=False)

        pd.DataFrame([s['summary']]).to_csv(
            cat_dir / 'sentiment_summary.csv', index=False
        )
        results['word_freq'].to_csv(cat_dir / 'word_frequencies.csv')

    def _save_topic_words(self, topic_results):
        rows = [
            {'topic': tid, 'word': word, 'weight': weight}
            for tid, words in topic_results['filtered_topics'].items()
            for word, weight in words
        ]
        pd.DataFrame(rows).to_csv(
            self.results_dir / 'topic_words.csv', index=False
        )


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """
    Run the full analysis pipeline.

    Update `csv_path` to point to your interview data CSV.
    Expected columns: 'Interviewee', 'Location', and one column per
    interview category (e.g., 'General Experience with the Health System').
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(
                Path('logs') / f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            ),
            logging.StreamHandler(),
        ],
    )
    Path('logs').mkdir(exist_ok=True)

    csv_path = Path("data/Interview Answers.csv")   # <-- update this path

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Interview data not found at '{csv_path}'.\n"
            "Please update the csv_path in main() to point to your CSV file."
        )

    analyzer = InterviewAnalyzer(csv_path)
    results = analyzer.run_complete_analysis()
    return results


if __name__ == "__main__":
    main()
