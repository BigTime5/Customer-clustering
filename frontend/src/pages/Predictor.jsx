import { useState, useEffect } from 'react';
import { predictSimple } from '../api/client';
import { SEGMENT_COLORS } from '../components/Sidebar';
import { RadarChart } from '../components/RadarChart';
import { Loader2 } from 'lucide-react';

export default function Predictor() {
  const [basket, setBasket] = useState(150);
  const [categories, setCategories] = useState({ c0: 20, c1: 20, c2: 20, c3: 20, c4: 20 });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const runPrediction = async () => {
    setLoading(true);
    try {
      const res = await predictSimple({
        basket_value: parseFloat(basket),
        categ_0: categories.c0 / 100,
        categ_1: categories.c1 / 100,
        categ_2: categories.c2 / 100,
        categ_3: categories.c3 / 100,
        categ_4: categories.c4 / 100,
      });
      setResult(res);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  // Auto-run on mount
  useEffect(() => { runPrediction(); }, []);

  const handleCatChange = (name, val) => {
    setCategories(prev => ({ ...prev, [name]: parseInt(val) }));
  };

  return (
    <div className="page-enter">
      <div className="page-header">
        <h1 className="page-title">First-Purchase Predictor</h1>
        <p className="page-subtitle">Predict a customer's lifetime archetype from the composition of their very first basket.</p>
      </div>

      <div className="customer-profile">
        {/* Input Form */}
        <div className="card">
          <div className="card-title">Simulate Basket</div>

          <div className="form-group" style={{ marginBottom: 24, marginTop: 16 }}>
            <label className="form-label">Total Basket Value (£)</label>
            <div className="range-wrapper">
              <input
                type="range" min="10" max="1500" step="10"
                value={basket} onChange={(e) => setBasket(e.target.value)}
                className="range-input"
              />
              <span className="range-value">£{basket}</span>
            </div>
            <input
              type="number" className="form-input" style={{ width: 100, marginTop: 8 }}
              value={basket} onChange={(e) => setBasket(e.target.value)}
            />
          </div>

          <div className="card-title">Category % Spend Mix</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16, marginTop: 16 }}>
            {[0, 1, 2, 3, 4].map(i => (
              <div className="form-group" key={i}>
                <label className="form-label" style={{ fontSize: '0.7rem' }}>Category {i} (%)</label>
                <div className="range-wrapper">
                  <input
                    type="range" min="0" max="100" step="1"
                    value={categories[`c${i}`]} onChange={(e) => handleCatChange(`c${i}`, e.target.value)}
                    className="range-input"
                    style={{ background: `linear-gradient(90deg, ${SEGMENT_COLORS[(i + 4) % 11]} ${(categories[`c${i}`] / 100) * 100}%, var(--border) ${(categories[`c${i}`] / 100) * 100}%)` }}
                  />
                  <span className="range-value" style={{ color: SEGMENT_COLORS[(i + 4) % 11] }}>{categories[`c${i}`]}%</span>
                </div>
              </div>
            ))}
          </div>

          <button
            className="btn btn-primary"
            style={{ width: '100%', marginTop: 32, justifyContent: 'center' }}
            onClick={runPrediction}
            disabled={loading}
          >
            {loading ? <Loader2 size={16} className="spinner" style={{ margin: 0, width: 16, height: 16 }} /> : 'Predict Archetype'}
          </button>
        </div>

        {/* Results Panel */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {result && (
            <>
              <div className="predict-result page-enter">
                <div className="card-title" style={{ color: 'var(--indigo-light)' }}>Predicted Segment</div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 8, marginBottom: 16 }}>
                  <div className="segment-badge" style={{ background: SEGMENT_COLORS[result.predicted_segment_id % 11], width: 36, height: 36, fontSize: '1.1rem' }}>
                    {result.predicted_segment_id}
                  </div>
                  <div className="predict-result-headline">{result.archetype_info.persona}</div>
                </div>
                <p style={{ fontSize: '0.9rem', lineHeight: 1.5, marginBottom: 16, color: 'rgba(255,255,255,0.9)' }}>
                  {result.archetype_info.description}
                </p>
                <div className={`churn-badge churn-${result.archetype_info.churn_risk.toLowerCase().replace(' ', '-')}`}>
                  Churn Risk: {result.archetype_info.churn_risk}
                </div>
              </div>

              <div className="card page-enter">
                <div className="card-title">Ensemble Model Confidence</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 16, marginTop: 16 }}>
                  {Object.entries(result.classifier_votes).map(([clf, data]) => (
                    <div key={clf}>
                      <div className="confidence-bar-wrap">
                        <span className="confidence-bar-label">{clf.toUpperCase()}</span>
                        <div className="confidence-bar-bg">
                          <div className="confidence-bar-fill" style={{ width: `${data.confidence}%` }} />
                        </div>
                        <span className="confidence-pct">{data.confidence.toFixed(1)}%</span>
                      </div>
                      <div style={{ fontSize: '0.65rem', color: '#94a3b8', paddingLeft: 112 }}>
                        Voted for Segment {data.predicted_segment}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}

          {!result && loading && (
            <div className="empty-state">
              <Loader2 size={32} className="spinner" />
              Evaluating input vector...
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
