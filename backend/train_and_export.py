"""
train_and_export.py
===================
Reads data.csv, replicates the notebook's full pipeline, and saves all model
artifacts to ./models/.

Run once from the /backend directory:
    python train_and_export.py
"""

import os
import re
import sys
import json
import string
import joblib
import nltk
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn import metrics

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
os.makedirs("models", exist_ok=True)
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data.csv")

# Download NLTK resources (run once)
for resource in ["punkt", "punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# 1. Load the raw data
# ---------------------------------------------------------------------------
print("Loading data …")
df = pd.read_csv(DATA_PATH, encoding="ISO-8859-1", dtype={"CustomerID": str})

# ---------------------------------------------------------------------------
# 2. Data Cleaning (mirrors notebook)
# ---------------------------------------------------------------------------
print("Cleaning data …")

# Remove missing CustomerIDs
df.dropna(subset=["CustomerID"], inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Engineer basket price per line
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

# Parse date
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Separate cancellations (InvoiceNo starts with 'C')
cancellations = df[df["InvoiceNo"].astype(str).str.startswith("C")].copy()
orders = df[~df["InvoiceNo"].astype(str).str.startswith("C")].copy()

# Remove special / non-product stock codes
special_codes = {"POST", "D", "M", "BANK CHARGES", "PADS", "DOT"}
orders = orders[~orders["StockCode"].astype(str).str.upper().isin(special_codes)]
orders = orders[orders["UnitPrice"] > 0]
orders = orders[orders["Quantity"] > 0]

print(f"  Cleaned dataset: {len(orders):,} order lines, {orders['CustomerID'].nunique():,} customers")

# ---------------------------------------------------------------------------
# 3. Product Taxonomy via NLP + K-Means (mirrors notebook)
# ---------------------------------------------------------------------------
print("Building product taxonomy …")

stemmer = nltk.stem.SnowballStemmer("english")

def is_noun(pos_tag: str) -> bool:
    return pos_tag.startswith("NN")


def keywords_inventory(description_series: pd.Series):
    """Extract canonical noun-stems from product descriptions."""
    keywords_roots: dict[str, set] = {}
    keywords_select: dict[str, str] = {}
    category_keys: list[str] = []
    count_keywords: dict[str, int] = {}

    for s in description_series:
        if pd.isnull(s):
            continue
        tokens = nltk.word_tokenize(s.lower())
        nouns = [w for w, pos in nltk.pos_tag(tokens) if is_noun(pos)]
        for t in nouns:
            root = stemmer.stem(t)
            if root in keywords_roots:
                keywords_roots[root].add(t)
                count_keywords[root] += 1
            else:
                keywords_roots[root] = {t}
                count_keywords[root] = 1

    for root, variants in keywords_roots.items():
        canonical = min(variants, key=len)
        category_keys.append(canonical)
        keywords_select[root] = canonical

    return category_keys, keywords_roots, keywords_select, count_keywords


# Build product list (unique descriptions)
products = orders[["StockCode", "Description"]].drop_duplicates()
# Deduplicate by StockCode – keep first description
products = products.groupby("StockCode").first().reset_index()

print(f"  Extracting keywords from {len(products):,} products …")
category_keys, keywords_roots, keywords_select, count_keywords = keywords_inventory(
    products["Description"]
)
print(f"  Extracted {len(category_keys)} keywords")

# Build keyword matrix for products
def product_category_vec(description: str, keyword_list: list[str]) -> list[int]:
    if pd.isnull(description):
        return [0] * len(keyword_list)
    desc_lower = description.lower()
    return [1 if kw in desc_lower else 0 for kw in keyword_list]


# Limit to top-N keywords by frequency to keep matrix manageable
TOP_N_KEYWORDS = 100
sorted_keys = sorted(count_keywords, key=count_keywords.get, reverse=True)[:TOP_N_KEYWORDS]
canonical_top = [keywords_select.get(k, k) for k in sorted_keys]

print(f"  Building product keyword matrix (top {TOP_N_KEYWORDS}) …")
X_products = np.array([product_category_vec(d, canonical_top) for d in products["Description"]])

# Cluster products into 5 categories
kmeans_products = KMeans(n_clusters=5, n_init=100, max_iter=600, random_state=42)
kmeans_products.fit(X_products)
products["product_cluster"] = kmeans_products.labels_
print("  Product taxonomy done.")

# Build a mapping: StockCode → product_cluster
stockcode_to_cluster = dict(zip(products["StockCode"], products["product_cluster"]))

# ---------------------------------------------------------------------------
# 4. Build per-customer features (10-month training window)
# ---------------------------------------------------------------------------
print("Building customer features …")

# Split by date – notebook uses 10 months train, 2 months test
# Date range in data: Dec 2010 – Dec 2011
train_cutoff = pd.Timestamp("2011-10-01")

train = orders[orders["InvoiceDate"] < train_cutoff].copy()
test = orders[orders["InvoiceDate"] >= train_cutoff].copy()

print(f"  Train period: {train['InvoiceDate'].min()} -> {train['InvoiceDate'].max()}")
print(f"  Test period : {test['InvoiceDate'].min()}  -> {test['InvoiceDate'].max()}")

# Assign product cluster to every transaction
train["product_cluster"] = train["StockCode"].map(stockcode_to_cluster)
test["product_cluster"] = test["StockCode"].map(stockcode_to_cluster)

# Per-basket aggregation
def build_basket_features(df_: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to basket (InvoiceNo) level."""
    basket = (
        df_.groupby(["CustomerID", "InvoiceNo"])
        .agg(basket_price=("TotalPrice", "sum"))
        .reset_index()
    )
    return basket


train_baskets = build_basket_features(train)
test_baskets = build_basket_features(test)


def build_customer_features(df_: pd.DataFrame, baskets: pd.DataFrame) -> pd.DataFrame:
    """Build per-customer behavioral summary."""
    stats = (
        baskets.groupby("CustomerID")["basket_price"]
        .agg(["count", "min", "max", "mean", "sum"])
        .reset_index()
    )
    # Category percentages
    cat_spend = (
        df_.groupby(["CustomerID", "product_cluster"])["TotalPrice"]
        .sum()
        .unstack(fill_value=0)
    )
    cat_spend.columns = [f"categ_{int(c)}" for c in cat_spend.columns]
    for c in range(5):
        if f"categ_{c}" not in cat_spend.columns:
            cat_spend[f"categ_{c}"] = 0.0
    cat_spend = cat_spend[[f"categ_{i}" for i in range(5)]]

    customer_spend_total = df_.groupby("CustomerID")["TotalPrice"].sum()
    for col in cat_spend.columns:
        cat_spend[col] = cat_spend[col] / customer_spend_total.reindex(cat_spend.index)

    result = stats.merge(cat_spend.reset_index(), on="CustomerID", how="left")
    result[cat_spend.columns] = result[cat_spend.columns].fillna(0)
    return result


train_features = build_customer_features(train, train_baskets)
test_features = build_customer_features(test, test_baskets)

print(f"  Train customers: {len(train_features):,}")
print(f"  Test customers : {len(test_features):,}")

# ---------------------------------------------------------------------------
# 5. Customer Segmentation – 11-cluster K-Means
# ---------------------------------------------------------------------------
print("Training customer K-Means (11 clusters) …")

list_cols = ["count", "min", "max", "mean", "categ_0", "categ_1", "categ_2", "categ_3", "categ_4"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_features[list_cols])

kmeans_cust = KMeans(n_clusters=11, n_init=100, max_iter=600, random_state=42)
kmeans_cust.fit(X_train_scaled)
train_features["segment"] = kmeans_cust.labels_
print("  Customer segmentation done.")

# ---------------------------------------------------------------------------
# 6. Archetype Classifier
# ---------------------------------------------------------------------------
print("Training archetype classifier …")

# Training labels from K-Means assignments
Y_train = train_features["segment"].values
# Features: first-purchase observables (mean basket + category mix)
clf_cols = ["mean", "categ_0", "categ_1", "categ_2", "categ_3", "categ_4"]
X_clf_train = train_features[clf_cols].values

# Classifiers
lr = LogisticRegression(max_iter=1000, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

voting_clf = VotingClassifier(
    estimators=[("lr", lr), ("knn", knn), ("gb", gb)],
    voting="hard",
)
voting_clf.fit(X_clf_train, Y_train)

# Quick accuracy on test set
X_test_scaled = scaler.transform(test_features[list_cols])
Y_test = kmeans_cust.predict(X_test_scaled)
X_clf_test = test_features[clf_cols].values
preds = voting_clf.predict(X_clf_test)
accuracy = metrics.accuracy_score(Y_test, preds)
print(f"  Ensemble accuracy on test set: {accuracy * 100:.2f}%")

# ---------------------------------------------------------------------------
# 7. Build Archetype Profiles
# ---------------------------------------------------------------------------
print("Building archetype profiles …")

profile_cols = ["count", "min", "max", "mean", "sum", "categ_0", "categ_1", "categ_2", "categ_3", "categ_4", "segment"]
profiles_df = train_features[profile_cols].copy()

archetype_profiles = {}
for seg_id in sorted(profiles_df["segment"].unique()):
    subset = profiles_df[profiles_df["segment"] == seg_id]
    profile = {
        "id": int(seg_id),
        "customer_count": int(len(subset)),
        "avg_basket_value": float(subset["mean"].mean()),
        "min_basket_value": float(subset["min"].mean()),
        "max_basket_value": float(subset["max"].mean()),
        "avg_total_spend": float(subset["sum"].mean()),
        "avg_visit_count": float(subset["count"].mean()),
        "category_preferences": {
            f"categ_{i}": float(subset[f"categ_{i}"].mean()) for i in range(5)
        },
    }
    archetype_profiles[str(seg_id)] = profile

# ---------------------------------------------------------------------------
# 8. Build Category Profiles (product taxonomy)
# ---------------------------------------------------------------------------
print("Building category profiles …")

category_profiles = {}
for cat_id in range(5):
    subset_prods = products[products["product_cluster"] == cat_id]
    # Top keywords: descriptions of products in this cluster
    all_words = []
    for desc in subset_prods["Description"].dropna():
        tokens = nltk.word_tokenize(desc.lower())
        nouns = [w for w, pos in nltk.pos_tag(tokens) if is_noun(pos)]
        all_words.extend(nouns)

    word_freq: dict[str, int] = {}
    for w in all_words:
        word_freq[w] = word_freq.get(w, 0) + 1
    top_words = sorted(word_freq, key=word_freq.get, reverse=True)[:20]

    # Revenue share from training set
    cat_revenue = train[train["product_cluster"] == cat_id]["TotalPrice"].sum()
    total_revenue = train["TotalPrice"].sum()
    revenue_share = float(cat_revenue / total_revenue) if total_revenue > 0 else 0.0

    category_profiles[str(cat_id)] = {
        "id": cat_id,
        "product_count": int(len(subset_prods)),
        "top_keywords": top_words,
        "revenue_share": revenue_share,
        "sample_products": subset_prods["Description"].dropna().head(10).tolist(),
    }

# ---------------------------------------------------------------------------
# 9. Build KPI summary
# ---------------------------------------------------------------------------
print("Aggregating KPIs …")

kpis = {
    "total_customers": int(orders["CustomerID"].nunique()),
    "total_transactions": int(orders["InvoiceNo"].nunique()),
    "total_revenue": float(orders["TotalPrice"].sum()),
    "avg_basket_value": float(orders.groupby(["CustomerID", "InvoiceNo"])["TotalPrice"].sum().mean()),
    "num_segments": 11,
    "num_product_categories": 5,
    "classifier_accuracy": float(accuracy),
    "uk_customer_pct": float(
        orders[orders["Country"] == "United Kingdom"]["CustomerID"].nunique()
        / orders["CustomerID"].nunique()
    ),
    "cancellation_rate": float(
        len(cancellations) / (len(cancellations) + len(orders))
    ),
    "date_range": {
        "start": str(orders["InvoiceDate"].min().date()),
        "end": str(orders["InvoiceDate"].max().date()),
    },
}

# Build per-segment customer count list for charts
kpis["segment_distribution"] = [
    {"segment_id": int(sid), "count": int(p["customer_count"])}
    for sid, p in archetype_profiles.items()
]

# ---------------------------------------------------------------------------
# 10. Persist all artefacts
# ---------------------------------------------------------------------------
print("Saving artefacts …")

joblib.dump(scaler, "models/scaler_customer.pkl")
joblib.dump(kmeans_products, "models/kmeans_products.pkl")
joblib.dump(kmeans_cust, "models/kmeans_customer.pkl")
joblib.dump(voting_clf, "models/classifier_archetype.pkl")

# Save taxonomy data for inference
taxonomy_data = {
    "canonical_top": canonical_top,
    "keywords_select": {k: v for k, v in keywords_select.items()},
    "count_keywords": {k: int(v) for k, v in count_keywords.items()},
}
joblib.dump(taxonomy_data, "models/taxonomy_data.pkl")

# Save customer lookup table
customer_lookup = (
    train_features[["CustomerID", "count", "min", "max", "mean", "sum",
                     "categ_0", "categ_1", "categ_2", "categ_3", "categ_4", "segment"]]
    .copy()
)
customer_lookup.to_parquet("models/customer_lookup.parquet", index=False)

# Save JSON profiles
with open("models/archetype_profiles.json", "w") as f:
    json.dump(archetype_profiles, f, indent=2)

with open("models/category_profiles.json", "w") as f:
    json.dump(category_profiles, f, indent=2)

with open("models/kpis.json", "w") as f:
    json.dump(kpis, f, indent=2)

# Save stockcode → cluster mapping
stockcode_map = {str(k): int(v) for k, v in stockcode_to_cluster.items()}
with open("models/stockcode_cluster_map.json", "w") as f:
    json.dump(stockcode_map, f)

print("\n[SUCCESS] All artefacts saved to ./models/")
print(f"   Classifier accuracy: {accuracy * 100:.2f}%")
print(f"   Total customers (full dataset): {kpis['total_customers']:,}")
