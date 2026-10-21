# NewsBot Intelligence System: End-to-End NLP Pipeline

## Overview
The NewsBot Intelligence System is an automated NLP pipeline that ingests, cleans, classifies, and extracts structured intelligence from raw news articles. Built on the BBC News dataset (1,490 articles across 5 balanced categories: Business, Entertainment, Politics, Sport, Tech), the system extracts actionable signals including semantic topics, grammatical distributions, Subject-Verb-Object (SVO) relationships, sentiment tone, and key named entities.

## System Architecture & Pipeline Components
1. **Module 1 & 2: Text Preprocessing Pipeline**
   - URL/HTML removal, punctuation/digit filtering, tokenization, stop-word elimination, and WordNet lemmatization.
2. **Module 3: TF-IDF Feature Extraction**
   - Extracted 2,500 unigram/bigram features to identify class-distinctive vocabulary.
3. **Module 4: Part-of-Speech (POS) Analysis**
   - Profiled grammatical composition; identified high noun density in Tech/Business (0.275 / 0.262) and elevated verb usage in Politics (0.137).
4. **Module 5: Syntax Parsing & SVO Action Extraction**
   - Parsed dependency trees with spaCy to extract relational triplets (*Subject -> Action -> Object*).
5. **Module 6: Sentiment & Emotion Polarity Analysis**
   - Evaluated article tone using NLTK VADER; captured positive skews in Entertainment (mean +0.599) versus balanced, contentious neutrality in Politics (+0.058).
6. **Module 7: Multi-Class Text Classification**
   - Benchmarked Multinomial Naive Bayes (96.98%), Logistic Regression (96.64%), and Linear Support Vector Classifier (97.65%).
   - Deployed **LinearSVC** as the production classifier (F1-score: 0.98 across all classes).
7. **Module 8: Named Entity Recognition (NER)**
   - Extracted and tracked high-frequency organizations, public figures, and geopolitical entities across sectors.

## Performance Summary
| Model | Accuracy | Macro F1 |
| :--- | :---: | :---: |
| **Linear Support Vector Classifier** | **97.65%** | **0.98** |
| Multinomial Naive Bayes | 96.98% | 0.97 |
| Logistic Regression | 96.64% | 0.97 |

## Project Structure
ITAI2373-NewsBot-Midterm/
├── NewsBot_Pipeline_ITAI2373.ipynb   # Executed notebook with all outputs
├── README.md                          # Architecture and benchmark overview
└── assets/                            # Visualizations
├── tfidf_keywords.png
├── pos_distribution_heatmap.png
├── sentiment_distribution.png
├── confusion_matrix.png
└── top_entities.png
## How to Run
1. Open `NewsBot_Pipeline_ITAI2373.ipynb` in Google Colab.
2. Run cells sequentially to configure Kaggle API credentials, ingest the BBC News dataset, and initialize pipeline modules.
3. Pass any unseen news text into `newsbot_analyze(article_text)` for instant classification, sentiment scoring, and entity/action extraction.
