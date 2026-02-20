"""
main.py — FastAPI Backend for Customer Segmentation App
"""

import json
import os
import re
import joblib
import numpy as np
import pandas as pd
import nltk

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# ─────────────────────────────── Setup ────────────────────────────────────
app = FastAPI(title="Customer Segmentation API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# ──────────────────────────── Load Artefacts ──────────────────────────────

def load(filename):
    return joblib.load(os.path.join(MODELS_DIR, filename))


def load_json(filename):
    with open(os.path.join(MODELS_DIR, filename)) as f:
        return json.load(f)


try:
    scaler = load("scaler_customer.pkl")
    kmeans_products = load("kmeans_products.pkl")
    kmeans_customer = load("kmeans_customer.pkl")
    classifier = load("classifier_archetype.pkl")
    taxonomy_data = load("taxonomy_data.pkl")
    customer_df = pd.read_parquet(os.path.join(MODELS_DIR, "customer_lookup.parquet"))
    archetype_profiles = load_json("archetype_profiles.json")
    category_profiles = load_json("category_profiles.json")
    kpis_data = load_json("kpis.json")
    stockcode_map = load_json("stockcode_cluster_map.json")
    ARTEFACTS_LOADED = True
    LOAD_ERROR = ""
except Exception as exc:
    ARTEFACTS_LOADED = False
    LOAD_ERROR = str(exc)

# ──────────────────────────── NLTK helpers ────────────────────────────────
stemmer = nltk.stem.SnowballStemmer("english")


def is_noun(pos_tag: str) -> bool:
    return pos_tag.startswith("NN")


def description_to_category_vec(description: str, canonical_top: list) -> list:
    if not description:
        return [0] * len(canonical_top)
    desc_lower = description.lower()
    return [1 if kw in desc_lower else 0 for kw in canonical_top]


def predict_product_cluster(description: str) -> int:
    canonical_top = taxonomy_data["canonical_top"]
    vec = description_to_category_vec(description, canonical_top)
    return int(kmeans_products.predict([vec])[0])


# ──────────────────────── Archetype Descriptions ─────────────────────────

def _generate_archetype_description(profile: dict) -> dict:
    """Generate human-readable description + business recommendation from profile metrics."""
    visits = profile.get("avg_visit_count", 0)
    basket = profile.get("avg_basket_value", 0)
    spend = profile.get("avg_total_spend", 0)
    cats = profile.get("category_preferences", {})
    top_cat = max(cats, key=cats.get) if cats else "categ_0"
    top_cat_pct = round(cats.get(top_cat, 0) * 100, 1)

    # Heuristic persona
    if visits >= 10 and basket >= 200:
        persona = "High-Frequency Big Spender"
        description = f"This is a premium, loyal customer who visits frequently ({visits:.0f} visits) and spends heavily (avg £{basket:.0f}/basket). They are your most valuable segment."
        recommendation = "Prioritise with personalised loyalty rewards and early access to new products. Churn risk: LOW."
        churn_risk = "Low"
    elif visits >= 8 and basket < 100:
        persona = "Frequent Bargain Shopper"
        description = f"Visits often ({visits:.0f} times) but keeps basket values low (avg £{basket:.0f}). Sensitive to price and promotions."
        recommendation = "Run targeted bundle promotions and volume discounts to increase basket size. Churn risk: MEDIUM."
        churn_risk = "Medium"
    elif visits <= 3 and basket >= 300:
        persona = "High-Value Occasional Buyer"
        description = f"Rarely visits ({visits:.0f} times) but trans acts large amounts (avg £{basket:.0f}/basket). Likely a B2B or wholesale buyer."
        recommendation = "Send personalised re-engagement campaigns and exclusive offers to increase visit frequency. Churn risk: HIGH."
        churn_risk = "High"
    elif visits <= 2 and basket < 100:
        persona = "One-Time / At-Risk Customer"
        description = f"Very few visits ({visits:.0f}) and low basket value (£{basket:.0f}). May have only purchased once."
        recommendation = "Activate win-back campaign with a discount code within 30 days of last purchase. Churn risk: VERY HIGH."
        churn_risk = "Very High"
    elif basket >= 500:
        persona = "Wholesale / Bulk Buyer"
        description = f"Average basket of £{basket:.0f} indicates bulk purchasing behaviour. Likely a business customer."
        recommendation = "Offer a trade account with volume pricing. Upsell complementary product lines. Churn risk: LOW."
        churn_risk = "Low"
    else:
        persona = "Standard Retail Customer"
        description = f"An average customer with {visits:.0f} visits and £{basket:.0f} avg basket. Broadly representative of the mainstream base."
        recommendation = "Focus on cross-sell recommendations and seasonal campaigns. Churn risk: MEDIUM."
        churn_risk = "Medium"

    return {
        "persona": persona,
        "description": description,
        "recommendation": recommendation,
        "churn_risk": churn_risk,
        "top_category": top_cat,
        "top_category_pct": top_cat_pct,
    }


# ─────────────────────────────── Schemas ──────────────────────────────────

class PurchaseItem(BaseModel):
    description: str
    quantity: int = 1
    unit_price: float


class FirstPurchaseRequest(BaseModel):
    items: List[PurchaseItem]
    stock_codes: Optional[List[str]] = None


class SimplePredictRequest(BaseModel):
    basket_value: float
    categ_0: float = 0.0
    categ_1: float = 0.0
    categ_2: float = 0.0
    categ_3: float = 0.0
    categ_4: float = 0.0


class BIQueryRequest(BaseModel):
    question: str


# ─────────────────────────────── Routes ───────────────────────────────────

@app.get("/api/health")
def health():
    if not ARTEFACTS_LOADED:
        raise HTTPException(status_code=503, detail=f"Models not loaded: {LOAD_ERROR}")
    return {"status": "ok", "customers": len(customer_df)}


@app.get("/api/kpis")
def get_kpis():
    """Dashboard KPIs."""
    if not ARTEFACTS_LOADED:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return kpis_data


@app.get("/api/segments")
def get_all_segments():
    """Return a summary of all 11 archetypes with generated descriptions."""
    if not ARTEFACTS_LOADED:
        raise HTTPException(status_code=503, detail="Models not loaded")
    enriched = []
    for key, profile in archetype_profiles.items():
        p = dict(profile)
        p["archetype_info"] = _generate_archetype_description(p)
        enriched.append(p)
    return {"segments": enriched}


@app.get("/api/segments/revenue")
def get_segments_revenue():
    """Per-segment estimated revenue for donut chart."""
    if not ARTEFACTS_LOADED:
        raise HTTPException(status_code=503, detail="Models not loaded")
    revenue_data = []
    for key, profile in archetype_profiles.items():
        est_revenue = profile["avg_total_spend"] * profile["customer_count"]
        revenue_data.append({
            "segment_id": profile["id"],
            "estimated_revenue": round(est_revenue, 2),
            "customer_count": profile["customer_count"],
        })
    return {"revenue": revenue_data}


@app.get("/api/segments/{segment_id}")
def get_segment(segment_id: int):
    """Detailed profile for a specific archetype."""
    if not ARTEFACTS_LOADED:
        raise HTTPException(status_code=503, detail="Models not loaded")
    key = str(segment_id)
    if key not in archetype_profiles:
        raise HTTPException(status_code=404, detail=f"Segment {segment_id} not found")
    profile = dict(archetype_profiles[key])
    profile["archetype_info"] = _generate_archetype_description(profile)
    return profile


@app.get("/api/categories")
def get_categories():
    """Return all 5 product category profiles."""
    if not ARTEFACTS_LOADED:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {"categories": list(category_profiles.values())}


@app.get("/api/categories/{category_id}")
def get_category(category_id: int):
    """Detailed profile for a product category."""
    if not ARTEFACTS_LOADED:
        raise HTTPException(status_code=503, detail="Models not loaded")
    key = str(category_id)
    if key not in category_profiles:
        raise HTTPException(status_code=404, detail=f"Category {category_id} not found")
    return category_profiles[key]


@app.get("/api/customers/{customer_id}")
def get_customer(customer_id: str):
    """Look up a customer by ID and return their behavioral profile and segment."""
    if not ARTEFACTS_LOADED:
        raise HTTPException(status_code=503, detail="Models not loaded")
    row = customer_df[customer_df["CustomerID"] == customer_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    record = row.iloc[0].to_dict()
    segment_id = int(record["segment"])
    profile = archetype_profiles.get(str(segment_id), {})
    archetype_info = _generate_archetype_description(profile)
    return {
        "customer_id": customer_id,
        "stats": {
            "visit_count": int(record["count"]),
            "min_basket": float(record["min"]),
            "max_basket": float(record["max"]),
            "avg_basket": float(record["mean"]),
            "total_spend": float(record["sum"]),
            "category_split": {f"categ_{i}": float(record[f"categ_{i}"]) for i in range(5)},
        },
        "segment_id": segment_id,
        "segment_profile": profile,
        "archetype_info": archetype_info,
    }


@app.get("/api/customers")
def list_customers(limit: int = 50, offset: int = 0):
    """Paginated list of customers with their segments."""
    if not ARTEFACTS_LOADED:
        raise HTTPException(status_code=503, detail="Models not loaded")
    total = len(customer_df)
    page = customer_df.iloc[offset: offset + limit].copy()
    records = []
    for _, row in page.iterrows():
        records.append({
            "customer_id": str(row["CustomerID"]),
            "avg_basket": round(float(row["mean"]), 2),
            "visit_count": int(row["count"]),
            "total_spend": round(float(row["sum"]), 2),
            "segment_id": int(row["segment"]),
        })
    return {"total": total, "customers": records}


@app.post("/api/predict")
def predict_archetype(request: FirstPurchaseRequest):
    """
    Given items from a customer's FIRST purchase, predict their lifetime archetype.
    Input: list of { description, quantity, unit_price }
    """
    if not ARTEFACTS_LOADED:
        raise HTTPException(status_code=503, detail="Models not loaded")

    items = request.items
    if not items:
        raise HTTPException(status_code=422, detail="At least one item is required")

    total_spend = sum(it.unit_price * it.quantity for it in items)
    if total_spend <= 0:
        raise HTTPException(status_code=422, detail="Basket total must be > 0")

    mean_basket = total_spend

    category_spend = {i: 0.0 for i in range(5)}
    item_details = []
    for it in items:
        cluster = predict_product_cluster(it.description)
        spend = it.unit_price * it.quantity
        category_spend[cluster] += spend
        item_details.append({
            "description": it.description,
            "unit_price": it.unit_price,
            "quantity": it.quantity,
            "product_category": cluster,
            "line_total": round(spend, 2),
        })

    category_pcts = {i: category_spend[i] / total_spend for i in range(5)}

    feature_vec = np.array([[
        mean_basket,
        category_pcts[0], category_pcts[1], category_pcts[2],
        category_pcts[3], category_pcts[4]
    ]])

    segment_id = int(classifier.predict(feature_vec)[0])
    profile = archetype_profiles.get(str(segment_id), {})
    archetype_info = _generate_archetype_description(profile)

    # Per-classifier confidence from individual estimators
    classifier_votes = {}
    for name, est in classifier.estimators:
        try:
            pred = int(est.predict(feature_vec)[0])
            classifier_votes[name] = {"predicted_segment": pred}
            if hasattr(est, "predict_proba"):
                proba = est.predict_proba(feature_vec)[0]
                classes = list(est.classes_)
                classifier_votes[name]["confidence"] = round(float(proba[classes.index(pred)]) * 100, 1)
            else:
                classifier_votes[name]["confidence"] = 100.0 if pred == segment_id else 0.0
        except Exception:
            pass

    return {
        "predicted_segment_id": segment_id,
        "segment_profile": profile,
        "archetype_info": archetype_info,
        "classifier_votes": classifier_votes,
        "basket": {
            "total": round(total_spend, 2),
            "items": item_details,
            "category_pcts": {f"categ_{i}": round(v * 100, 1) for i, v in category_pcts.items()},
        },
    }


@app.post("/api/predict/simple")
def predict_simple(req: SimplePredictRequest):
    """
    Simple slider-based prediction. Accepts basket_value + category percentages (0-1 each).
    Returns predicted segment, archetype info, and per-classifier votes.
    """
    if not ARTEFACTS_LOADED:
        raise HTTPException(status_code=503, detail="Models not loaded")

    total_pct = req.categ_0 + req.categ_1 + req.categ_2 + req.categ_3 + req.categ_4
    if total_pct == 0:
        c0, c1, c2, c3, c4 = 0.2, 0.2, 0.2, 0.2, 0.2
    else:
        # Normalize to sum = 1
        c0 = req.categ_0 / total_pct
        c1 = req.categ_1 / total_pct
        c2 = req.categ_2 / total_pct
        c3 = req.categ_3 / total_pct
        c4 = req.categ_4 / total_pct

    feature_vec = np.array([[req.basket_value, c0, c1, c2, c3, c4]])

    segment_id = int(classifier.predict(feature_vec)[0])
    profile = archetype_profiles.get(str(segment_id), {})
    archetype_info = _generate_archetype_description(profile)

    classifier_votes = {}
    for name, est in classifier.estimators:
        try:
            pred = int(est.predict(feature_vec)[0])
            classifier_votes[name] = {"predicted_segment": pred}
            if hasattr(est, "predict_proba"):
                proba = est.predict_proba(feature_vec)[0]
                classes = list(est.classes_)
                classifier_votes[name]["confidence"] = round(float(proba[classes.index(pred)]) * 100, 1)
            else:
                classifier_votes[name]["confidence"] = 100.0 if pred == segment_id else 0.0
        except Exception:
            pass

    return {
        "predicted_segment_id": segment_id,
        "segment_profile": profile,
        "archetype_info": archetype_info,
        "classifier_votes": classifier_votes,
        "normalized_features": {
            "basket_value": req.basket_value,
            "categ_0": round(c0, 3), "categ_1": round(c1, 3),
            "categ_2": round(c2, 3), "categ_3": round(c3, 3),
            "categ_4": round(c4, 3),
        },
    }


@app.post("/api/bi/query")
def bi_query(req: BIQueryRequest):
    """
    Rule-based Business Intelligence query endpoint.
    Answers natural-language questions about the customer segments and KPIs.
    """
    if not ARTEFACTS_LOADED:
        raise HTTPException(status_code=503, detail="Models not loaded")

    q = req.question.lower().strip()

    # ── Compute derived metrics once ──
    profiles_list = list(archetype_profiles.values())
    seg_by_ltv = sorted(profiles_list, key=lambda p: p["avg_total_spend"], reverse=True)
    seg_by_basket = sorted(profiles_list, key=lambda p: p["avg_basket_value"], reverse=True)
    seg_by_visits = sorted(profiles_list, key=lambda p: p["avg_visit_count"], reverse=True)
    seg_by_size = sorted(profiles_list, key=lambda p: p["customer_count"], reverse=True)

    # Churn risk: low visits = high risk
    seg_by_churn = sorted(profiles_list, key=lambda p: p["avg_visit_count"])

    # Revenue estimate per segment
    seg_revenue = {str(p["id"]): p["avg_total_spend"] * p["customer_count"] for p in profiles_list}
    highest_rev_seg = max(seg_revenue, key=seg_revenue.get)
    cat_revenue = {k: v["revenue_share"] for k, v in category_profiles.items()}
    top_cat = max(cat_revenue, key=cat_revenue.get)

    def seg_info(p):
        i = _generate_archetype_description(p)
        return f"Segment {p['id']} ({i['persona']})"

    # ── Pattern matching ──

    # Highest LTV / lifetime value
    if re.search(r"highest.*(ltv|lifetime value|spend|revenue)", q) or re.search(r"most valuable segment", q):
        top = seg_by_ltv[0]
        info = _generate_archetype_description(top)
        return {
            "question": req.question,
            "answer": f"**{seg_info(top)}** has the highest average lifetime value at **£{top['avg_total_spend']:,.0f}** per customer, with {top['customer_count']} customers in this segment.",
            "data": {"segment_id": top["id"], "avg_total_spend": top["avg_total_spend"]},
        }

    # Churn risk
    if re.search(r"churn|at.risk|risk", q):
        risky = seg_by_churn[0]
        info = _generate_archetype_description(risky)
        return {
            "question": req.question,
            "answer": f"**{seg_info(risky)}** has the highest churn risk. They average only **{risky['avg_visit_count']:.1f} visits** — indicating low engagement and risk of lapsing. {info['recommendation']}",
            "data": {"segment_id": risky["id"], "avg_visit_count": risky["avg_visit_count"], "churn_risk": info["churn_risk"]},
        }

    # Average basket of a specific segment
    m = re.search(r"(average|avg|mean)?\s*basket\s*(of|for)?\s*segment\s*(\d+)", q)
    if m:
        sid = str(m.group(3))
        if sid in archetype_profiles:
            p = archetype_profiles[sid]
            return {
                "question": req.question,
                "answer": f"The average basket value for **Segment {sid}** is **£{p['avg_basket_value']:,.2f}**. This segment contains {p['customer_count']} customers with an average of {p['avg_visit_count']:.1f} visits each.",
                "data": {"segment_id": int(sid), "avg_basket": p["avg_basket_value"]},
            }

    # Biggest segment
    if re.search(r"biggest|largest|most customers|highest count", q):
        top = seg_by_size[0]
        return {
            "question": req.question,
            "answer": f"**{seg_info(top)}** is the largest segment with **{top['customer_count']:,} customers** — {round(top['customer_count']/kpis_data['total_customers']*100,1)}% of the total customer base.",
            "data": {"segment_id": top["id"], "customer_count": top["customer_count"]},
        }

    # Highest basket
    if re.search(r"highest.*(basket|spend per visit|order value)", q) or re.search(r"big.*(spender|basket)", q):
        top = seg_by_basket[0]
        return {
            "question": req.question,
            "answer": f"**{seg_info(top)}** has the highest average basket value at **£{top['avg_basket_value']:,.2f}** per visit.",
            "data": {"segment_id": top["id"], "avg_basket": top["avg_basket_value"]},
        }

    # Most frequent visitors
    if re.search(r"(most|highest).*(visit|frequent|loyal)", q) or re.search(r"visit.*(most|frequently)", q):
        top = seg_by_visits[0]
        return {
            "question": req.question,
            "answer": f"**{seg_info(top)}** visits most frequently, averaging **{top['avg_visit_count']:.1f} visits** per customer.",
            "data": {"segment_id": top["id"], "avg_visits": top["avg_visit_count"]},
        }

    # Revenue by category
    if re.search(r"(category|product).*(revenue|most|drive)", q) or re.search(r"revenue.*(category|product)", q):
        cat = category_profiles[top_cat]
        return {
            "question": req.question,
            "answer": f"**Product Category {top_cat}** drives the most revenue, accounting for **{cat['revenue_share']*100:.1f}%** of total training-period revenue. Its top keywords include: {', '.join(cat['top_keywords'][:5])}.",
            "data": {"category_id": int(top_cat), "revenue_share": cat["revenue_share"], "top_keywords": cat["top_keywords"][:5]},
        }

    # Total customers
    if re.search(r"(how many|total|number of)\s+customers", q):
        return {
            "question": req.question,
            "answer": f"The dataset contains **{kpis_data['total_customers']:,} unique customers** across {kpis_data['num_segments']} behavioural segments, spanning {kpis_data['date_range']['start']} to {kpis_data['date_range']['end']}.",
            "data": {"total_customers": kpis_data["total_customers"]},
        }

    # Total revenue
    if re.search(r"(total|overall)\s+revenue", q):
        return {
            "question": req.question,
            "answer": f"Total revenue across the full dataset (Dec 2010 – Dec 2011) is **£{kpis_data['total_revenue']:,.0f}**, with an average basket value of **£{kpis_data['avg_basket_value']:,.2f}**.",
            "data": {"total_revenue": kpis_data["total_revenue"], "avg_basket": kpis_data["avg_basket_value"]},
        }

    # Cancellation rate
    if re.search(r"cancel|cancellation|refund|return", q):
        rate = kpis_data.get("cancellation_rate", 0)
        return {
            "question": req.question,
            "answer": f"The cancellation rate across all transactions is **{rate*100:.1f}%**. This includes all orders beginning with 'C' (cancellations/reversals).",
            "data": {"cancellation_rate": rate},
        }

    # Model accuracy
    if re.search(r"accuracy|classifier|model.*(perform|score|result)", q):
        acc = kpis_data.get("classifier_accuracy", 0)
        return {
            "question": req.question,
            "answer": f"The ensemble classifier (Logistic Regression + kNN + Gradient Boosting voting) achieves **{acc*100:.1f}% accuracy** on held-out test customers — predicting a customer's lifetime segment from their very first purchase.",
            "data": {"classifier_accuracy": acc},
        }

    # Segment revenue
    if re.search(r"revenue.*(segment|group)\s*(\d+)", q) or re.search(r"(segment|group)\s*(\d+).*(revenue|worth)", q):
        m2 = re.search(r"(\d+)", q)
        if m2:
            sid = str(m2.group(1))
            if sid in archetype_profiles:
                rev = seg_revenue[sid]
                p = archetype_profiles[sid]
                return {
                    "question": req.question,
                    "answer": f"**Segment {sid}** has an estimated lifetime revenue contribution of **£{rev:,.0f}** ({p['customer_count']} customers × £{p['avg_total_spend']:,.0f} avg LTV).",
                    "data": {"segment_id": int(sid), "estimated_revenue": rev},
                }

    # Number of segments
    if re.search(r"(how many|number of)\s+segment", q):
        return {
            "question": req.question,
            "answer": f"The model identified **{kpis_data['num_segments']} distinct customer segments** (archetypes) using K-Means clustering on 10 months of behavioural data.",
            "data": {"num_segments": kpis_data["num_segments"]},
        }

    # UK customers
    if re.search(r"uk|united kingdom|british", q):
        pct = kpis_data.get("uk_customer_pct", 0)
        return {
            "question": req.question,
            "answer": f"**{pct*100:.1f}%** of all customers are from the United Kingdom, reflecting this retailer's primary market.",
            "data": {"uk_customer_pct": pct},
        }

    # Fallback
    return {
        "question": req.question,
        "answer": "I couldn't find a direct answer to that question. Try asking about: segment churn risk, lifetime value, average basket by segment, total revenue, cancellation rate, model accuracy, or product category revenue.",
        "data": {},
        "suggested_questions": [
            "Which segment has the highest churn risk?",
            "What is the average basket of segment 5?",
            "Which product category drives the most revenue?",
            "What is the total revenue?",
            "What is the cancellation rate?",
            "Which segment has the highest lifetime value?",
        ],
    }
