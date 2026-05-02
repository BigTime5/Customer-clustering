# Customer Segmentation and Archetype Intelligence

A full-stack machine learning application that segments an e-commerce customer base into behavioural archetypes, predicts the lifetime segment of a new customer from their very first basket, and surfaces actionable business intelligence through an interactive dashboard.

**Live application:** [https://customer-clustering-project.vercel.app/](https://customer-clustering-project.vercel.app/)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Machine Learning Pipeline](#3-machine-learning-pipeline)
   - 3.1 [Data Cleaning and Feature Engineering](#31-data-cleaning-and-feature-engineering)
   - 3.2 [Product Taxonomy via NLP and K-Means](#32-product-taxonomy-via-nlp-and-k-means)
   - 3.3 [Customer Segmentation](#33-customer-segmentation)
   - 3.4 [Archetype Classifier](#34-archetype-classifier)
4. [Segment Profiles](#4-segment-profiles)
5. [Application Architecture](#5-application-architecture)
6. [Backend API](#6-backend-api)
7. [Frontend](#7-frontend)
8. [Repository Structure](#8-repository-structure)
9. [Getting Started](#9-getting-started)
   - 9.1 [Prerequisites](#91-prerequisites)
   - 9.2 [Running the Backend](#92-running-the-backend)
   - 9.3 [Running the Frontend](#93-running-the-frontend)
10. [Model Artefacts](#10-model-artefacts)
11. [Technology Stack](#11-technology-stack)

---

## 1. Project Overview

This project answers a core question every e-commerce business faces: **who are my customers, and how should I treat them differently?**

Using one year of transactional data from a UK-based online retailer, the pipeline performs a two-stage unsupervised segmentation: products are first grouped into a five-category taxonomy using natural language processing, and customers are then segmented into eleven behavioural archetypes using K-Means clustering on basket-level behavioural features. A supervised ensemble classifier is trained on top of the segmentation, enabling the system to assign any new customer to an archetype from a single basket — in real time.

The results are surfaced through a six-page React application with live API connectivity, interactive charts, a first-purchase predictor, and a natural-language Business Intelligence assistant.

---

## 2. Dataset

| Attribute | Value |
|---|---|
| Source | UCI Online Retail Dataset |
| Coverage | 1 December 2010 — 9 December 2011 |
| Total customers | 4,334 |
| Total transactions | 18,405 |
| Total revenue | £8,743,914 |
| Average basket value | £475.08 |
| UK customer share | 90.4% |
| Cancellation rate | 2.2% |
| Product segments | 5 |
| Customer segments | 11 |

The dataset records invoices from a UK-based and registered non-store online retailer. The retailer primarily sells unique all-occasion gifts, and a significant portion of its customers are wholesale buyers.

---

## 3. Machine Learning Pipeline

The pipeline is fully reproducible via `backend/train_and_export.py`, which reads the raw `data.csv` and produces all artefacts consumed by the API.

### 3.1 Data Cleaning and Feature Engineering

Raw data undergoes the following transformations before any modelling takes place.

- Rows with missing `CustomerID` values are removed, as they cannot be attributed to a customer profile.
- Duplicate invoice lines are dropped.
- Cancelled orders (invoice numbers prefixed with `C`) are separated and excluded from behavioural features but are retained for KPI reporting.
- Non-product stock codes (`POST`, `D`, `M`, `BANK CHARGES`, `PADS`, `DOT`) are removed to prevent administrative line items from contaminating product or revenue features.
- Rows with zero or negative unit price or quantity are removed.
- A `TotalPrice` column is derived as `Quantity * UnitPrice` for each line item.
- The dataset is split chronologically: the first ten months (December 2010 to September 2011) form the training window, and the final two months (October to December 2011) serve as the hold-out test set.

### 3.2 Product Taxonomy via NLP and K-Means

Before customers can be characterised by their spending patterns across product categories, those categories must be constructed. There is no predefined taxonomy in the raw data, so one is derived entirely from product descriptions using the following approach.

**Noun extraction.** Each product description is tokenised using NLTK's word tokeniser, and part-of-speech tagging is applied to isolate nouns. Nouns are stemmed using the Snowball stemmer to map morphological variants to a common root (for example, "hearts", "heart", and "hearted" resolve to the same stem).

**Keyword matrix construction.** The top 100 most frequent noun stems are selected. For each product, a binary indicator vector is constructed whose dimensions correspond to whether each of the top 100 keywords appears in that product's description.

**K-Means clustering.** A K-Means model with five clusters, 100 initialisations, and 600 maximum iterations is fitted to the product keyword matrix. Each product is assigned to one of five latent product categories, producing a `StockCode → cluster` mapping used in all downstream steps.

Five clusters were selected based on the separation observed in the notebook analysis and the interpretability of the resulting groupings, which loosely correspond to household, seasonal, giftware, decorative, and miscellaneous product families.

### 3.3 Customer Segmentation

With a product taxonomy in place, each customer can be described by a compact behavioural feature vector.

**Per-basket aggregation.** Transactions are first aggregated to basket (invoice) level, computing total basket spend per visit.

**Per-customer feature vector.** Nine features are derived for each customer over the training window.

| Feature | Description |
|---|---|
| `count` | Number of purchases (visits) |
| `min` | Minimum single-basket value |
| `max` | Maximum single-basket value |
| `mean` | Mean basket value |
| `categ_0` through `categ_4` | Share of total spend directed at each product category |

Note that `sum` (total spend) is used for profiling but excluded from the clustering feature set to prevent it from dominating the distance metric.

**Standardisation and clustering.** Features are standardised using `StandardScaler` before K-Means is applied. An eleven-cluster solution was selected, with 100 random initialisations and 600 iterations, to ensure stable convergence and to capture the meaningful heterogeneity that exists across the customer base — particularly the distinction between retail customers, frequent mid-tier buyers, and wholesale accounts.

### 3.4 Archetype Classifier

K-Means assigns labels to customers seen during training. To extend the system to new customers — specifically, to classify a customer from their very first basket before repeat behaviour is observed — a supervised ensemble classifier is trained on the K-Means segment assignments.

**Features.** The classifier uses six features available at first purchase: mean basket value and the five product category spend proportions.

**Ensemble architecture.** A `VotingClassifier` with hard voting is constructed from three base learners.

| Base Learner | Role |
|---|---|
| Logistic Regression (`max_iter=1000`) | Captures linear decision boundaries between archetypes |
| K-Nearest Neighbours (`k=5`) | Captures local, non-linear patterns in feature space |
| Gradient Boosting (`100 estimators`) | Captures complex interactions and provides robustness |

**Evaluation.** The ensemble is evaluated on the hold-out test set, where ground-truth labels are generated by applying the trained K-Means model to the scaled test features. The ensemble achieves **80.85% accuracy** on the test set, meaning the predicted archetype from a single basket matches the segment assignment based on the full purchase history in more than four out of five cases.

At inference time, the API also returns individual model votes and confidence scores for each base learner, giving the business user visibility into prediction agreement across the ensemble.

---

## 4. Segment Profiles

The eleven segments span a wide range of behavioural patterns. The table below summarises the profile of each segment derived from the training data.

| Segment | Customers | Avg Basket (£) | Avg Total Spend (£) | Archetype |
|---|---|---|---|---|
| 0 | 883 | 301 | 666 | Standard Retail Customer |
| 1 | 465 | 279 | 724 | Standard Retail Customer |
| 2 | 1 | 77,184 | 77,184 | Extreme Wholesale Outlier |
| 3 | 1,273 | 416 | 1,266 | Standard Retail Customer |
| 4 | 7 | 382 | 35,722 | Wholesale / Bulk Buyer |
| 5 | 41 | 311 | 461 | One-Time / At-Risk Customer |
| 6 | 170 | 328 | 982 | Standard Retail Customer |
| 7 | 149 | 486 | 9,379 | High-Value Occasional Buyer |
| 8 | 464 | 441 | 1,328 | Standard Retail Customer |
| 9 | 10 | 6,542 | 92,846 | Wholesale / Bulk Buyer |
| 10 | 150 | 325 | 660 | Standard Retail Customer |

Each segment is enriched at API serve-time with a heuristic persona label, a churn-risk classification, a plain-English description of the segment's behaviour, and a concrete marketing recommendation — derived from the visit frequency and basket value thresholds surfaced by the clustering analysis.

---

## 5. Application Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│  Browser (React 19 / Vite)                                        │
│  Vercel CDN — customer-clustering-project.vercel.app              │
│                                                                   │
│  Dashboard  |  Customer Lookup  |  Predictor  |  Segment Explorer │
│  Category Explorer  |  Business Intelligence                      │
└──────────────────────────────┬────────────────────────────────────┘
                               │ HTTPS / REST
┌──────────────────────────────▼────────────────────────────────────┐
│  FastAPI Backend (Uvicorn)                                        │
│  Render — customer-clustering-e031.onrender.com                   │
│                                                                   │
│  /api/kpis        /api/segments     /api/customers                │
│  /api/categories  /api/predict      /api/bi/query                 │
└──────────────────────────────┬────────────────────────────────────┘
                               │
┌──────────────────────────────▼────────────────────────────────────┐
│  Model Artefacts  (./backend/models/)                             │
│                                                                   │
│  scaler_customer.pkl          kmeans_customer.pkl                 │
│  kmeans_products.pkl          classifier_archetype.pkl            │
│  taxonomy_data.pkl            customer_lookup.parquet             │
│  archetype_profiles.json      category_profiles.json             │
│  kpis.json                    stockcode_cluster_map.json          │
└───────────────────────────────────────────────────────────────────┘
```

---

## 6. Backend API

The API is built with FastAPI and served by Uvicorn. All endpoints return JSON. CORS is enabled for all origins to support the separately hosted frontend.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Liveness check; confirms all artefacts loaded |
| GET | `/api/kpis` | Dashboard KPIs: revenue, customer count, basket value, cancellation rate |
| GET | `/api/segments` | All eleven segment profiles with generated persona descriptions |
| GET | `/api/segments/{id}` | Single segment profile including category preferences |
| GET | `/api/segments/revenue` | Per-segment estimated revenue for chart rendering |
| GET | `/api/categories` | All five product category profiles with top keywords and revenue share |
| GET | `/api/categories/{id}` | Single category profile with sample product descriptions |
| GET | `/api/customers` | Paginated customer list (limit / offset query parameters) |
| GET | `/api/customers/{id}` | Full profile for a single customer ID |
| POST | `/api/predict/simple` | Predict segment from basket value and category mix |
| POST | `/api/predict` | Predict segment from a list of purchase items with descriptions |
| POST | `/api/bi/query` | Natural-language BI query; returns a structured answer with suggested follow-up questions |

The `/api/predict/simple` endpoint accepts a `basket_value` and five `categ_N` floats representing the share of spend per product category, feeds them through the scaler and ensemble classifier, and returns the predicted segment ID along with the individual vote and confidence from each base learner.

---

## 7. Frontend

The frontend is a single-page application built with React 19 and Vite 7, deployed to Vercel. Navigation is handled by React Router 7. Charts are rendered with Recharts 3 and UI transitions are powered by Framer Motion 12.

**Dashboard.** Presents headline KPIs (total customers, total revenue, average basket value, cancellation rate) alongside a bar chart of customer distribution across segments and a revenue donut chart broken down by segment.

**Customer Lookup.** Allows the user to browse all 4,334 customers with pagination, or search by CustomerID directly. Each customer card shows their segment assignment, visit count, average and total basket values, and category preference breakdown.

**First-Purchase Predictor.** Provides interactive sliders for basket value and the percentage of spend directed at each product category. Submitting the form calls `/api/predict/simple` and displays the predicted archetype persona, churn risk, business recommendation, and a confidence breakdown by individual model.

**Segment Explorer.** Lists all eleven archetypes. Selecting a segment opens a detail card with a radar chart of category preferences and the full set of profile metrics.

**Category Explorer.** Presents the five NLP-derived product categories, including revenue share, product count, top keywords, and sample product descriptions.

**Business Intelligence.** A conversational interface backed by the `/api/bi/query` endpoint. The assistant has direct access to all KPI and segment data and can answer questions such as "Which segment has the highest churn risk?" or "What is the average basket of segment 7?" Suggested follow-up questions are surfaced after each answer.

---

## 8. Repository Structure

```
.
├── backend/
│   ├── main.py                  # FastAPI application with all route handlers
│   ├── train_and_export.py      # Full ML pipeline; run once to regenerate artefacts
│   ├── requirements.txt         # Python dependencies
│   └── models/
│       ├── scaler_customer.pkl          # StandardScaler fitted on training features
│       ├── kmeans_customer.pkl          # 11-cluster K-Means for customer segmentation
│       ├── kmeans_products.pkl          # 5-cluster K-Means for product taxonomy
│       ├── classifier_archetype.pkl     # VotingClassifier ensemble
│       ├── taxonomy_data.pkl            # Keyword list and stemming maps for inference
│       ├── customer_lookup.parquet      # Pre-computed features for all training customers
│       ├── archetype_profiles.json      # Aggregated metrics per segment
│       ├── category_profiles.json       # Aggregated metrics per product category
│       ├── kpis.json                    # Global KPI summary
│       └── stockcode_cluster_map.json   # StockCode to product cluster mapping
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx                      # Root component; routing and API health polling
│   │   ├── api/client.js                # Axios instance and all API call wrappers
│   │   ├── components/
│   │   │   ├── Sidebar.jsx              # Navigation sidebar with segment colour palette
│   │   │   └── RadarChart.jsx           # Reusable radar chart for category preferences
│   │   └── pages/
│   │       ├── Dashboard.jsx            # KPI overview and distribution charts
│   │       ├── CustomerLookup.jsx       # Customer search and profile cards
│   │       ├── Predictor.jsx            # First-purchase archetype predictor
│   │       ├── SegmentExplorer.jsx      # Segment cards with radar charts
│   │       ├── CategoryExplorer.jsx     # Product taxonomy explorer
│   │       └── BusinessIntelligence.jsx # NLP-powered BI chat interface
│   ├── package.json
│   └── vite.config.js
│
├── customer-segmentation-storytelling.ipynb   # Exploratory analysis and methodology notebook
└── README.md
```

---

## 9. Getting Started

### 9.1 Prerequisites

- Python 3.10 or later
- Node.js 20 or later
- The raw dataset file `data.csv` (UCI Online Retail dataset) placed in the repository root

### 9.2 Running the Backend

```bash
cd backend
pip install -r requirements.txt

# Regenerate all model artefacts from the raw data.
# Skip this step if the pre-built artefacts in ./models/ are already present.
python train_and_export.py

# Start the API server
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Interactive documentation is auto-generated at `http://localhost:8000/docs`.

If you regenerate the artefacts, update the `baseURL` in `frontend/src/api/client.js` to point to your local server before starting the frontend.

### 9.3 Running the Frontend

```bash
cd frontend
npm install
npm run dev
```

The application will be served at `http://localhost:5173` by default.

---

## 10. Model Artefacts

All artefacts are serialised with `joblib` (pickled models) or stored as Parquet and JSON files. They are loaded once at API startup and held in memory for the lifetime of the server process.

| Artefact | Format | Description |
|---|---|---|
| `scaler_customer.pkl` | joblib | `StandardScaler` fitted on nine customer features |
| `kmeans_customer.pkl` | joblib | 11-cluster `KMeans` trained on scaled customer features |
| `kmeans_products.pkl` | joblib | 5-cluster `KMeans` trained on product keyword matrix |
| `classifier_archetype.pkl` | joblib | `VotingClassifier` (LR + KNN + GB) |
| `taxonomy_data.pkl` | joblib | Top-100 keyword list and stemming maps required for inference |
| `customer_lookup.parquet` | Parquet | Feature vectors and segment IDs for all training customers |
| `archetype_profiles.json` | JSON | Aggregated profile metrics per segment |
| `category_profiles.json` | JSON | Revenue share and top keywords per product category |
| `kpis.json` | JSON | Global KPI summary used by the Dashboard |
| `stockcode_cluster_map.json` | JSON | Lookup map from `StockCode` string to product cluster integer |

---

## 11. Technology Stack

**Machine Learning and Data**

| Library | Version | Purpose |
|---|---|---|
| scikit-learn | 1.6.1 | K-Means, StandardScaler, VotingClassifier, evaluation metrics |
| pandas | 2.2.3 | Data ingestion, cleaning, and feature engineering |
| numpy | 2.4.1 | Numerical array operations |
| nltk | 3.9.1 | Tokenisation, POS tagging, and Snowball stemming |
| joblib | 1.4.2 | Model serialisation and deserialisation |
| fastparquet | 2024.2.0 | Parquet file I/O for customer lookup table |

**Backend**

| Library | Version | Purpose |
|---|---|---|
| FastAPI | 0.115.0 | REST API framework |
| Uvicorn | 0.30.0 | ASGI server |
| Pydantic | (FastAPI dependency) | Request and response schema validation |

**Frontend**

| Library | Version | Purpose |
|---|---|---|
| React | 19.2.0 | UI component framework |
| Vite | 7.3.1 | Build tool and development server |
| React Router | 7.13.0 | Client-side routing |
| Recharts | 3.7.0 | Bar, pie, and radar chart rendering |
| Framer Motion | 12.34.2 | Page transition and element animations |
| Axios | 1.13.5 | HTTP client for API communication |
| Lucide React | 0.575.0 | Icon library |

**Infrastructure**

| Service | Role |
|---|---|
| Vercel | Frontend hosting and CDN |
| Render | Backend hosting |
