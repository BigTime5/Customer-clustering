import { useState, useEffect } from 'react';
import { getSegments } from '../api/client';
import { RadarChart } from '../components/RadarChart';
import { SEGMENT_COLORS } from '../components/Sidebar';
import { Loader2 } from 'lucide-react';

export default function SegmentExplorer() {
  const [segments, setSegments] = useState([]);
  const [selected, setSelected] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getSegments()
      .then(res => {
        // Sort by ID
        const sorted = res.segments.sort((a, b) => a.id - b.id);
        setSegments(sorted);
        setLoading(false);
      })
      .catch(console.error);
  }, []);

  if (loading) return <div className="loading-state"><Loader2 className="spinner" />Loading Segments...</div>;

  return (
    <div className="page-enter">
      <div className="page-header">
        <h1 className="page-title">Segment Explorer</h1>
        <p className="page-subtitle">Deep dive into the 11 behavioural archetypes identified by the KMeans model.</p>
      </div>

      <div className="segments-grid">
        {segments.map((seg) => (
          <div
            key={seg.id}
            className="segment-card"
            style={{ '--seg-color': SEGMENT_COLORS[seg.id % 11] }}
            onClick={() => setSelected(seg)}
          >
            <div className="segment-card-header">
              <span className="segment-badge" style={{ background: SEGMENT_COLORS[seg.id % 11] }}>{seg.id}</span>
              <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>{seg.customer_count} customers</span>
            </div>
            <div className="segment-persona">{seg.archetype_info.persona}</div>

            <div className="segment-stats">
              <div className="segment-stat">
                <div className="segment-stat-val">£{seg.avg_basket_value.toFixed(0)}</div>
                <div className="segment-stat-lbl">Avg Basket</div>
              </div>
              <div className="segment-stat">
                <div className="segment-stat-val">{seg.avg_visit_count.toFixed(1)}</div>
                <div className="segment-stat-lbl">Visits</div>
              </div>
            </div>

            <div style={{ marginTop: 12, fontSize: '0.7rem', color: '#38bdf8' }}>
              Prefers: Category {seg.archetype_info.top_category.replace('categ_', '')} ({(seg.archetype_info.top_category_pct).toFixed(1)}%)
            </div>
          </div>
        ))}
      </div>

      {selected && (
        <div className="segment-detail page-enter">
          <div className="card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <h2 style={{ fontSize: '1.2rem', fontWeight: 800, marginBottom: 8, color: SEGMENT_COLORS[selected.id % 11] }}>
              Segment {selected.id}: {selected.archetype_info.persona}
            </h2>
            <p style={{ color: '#94a3b8', fontSize: '0.85rem', marginBottom: 24, padding: '0 20px', textAlign: 'center' }}>
              {selected.archetype_info.description}
            </p>

            <RadarChart
              data={{
                visits: selected.avg_visit_count / 50,
                avg_basket: selected.avg_basket_value / 1000,
                max_basket: selected.max_basket_value / 1500,
                categ_0: selected.category_preferences.categ_0,
                categ_1: selected.category_preferences.categ_1,
                categ_2: selected.category_preferences.categ_2,
                categ_3: selected.category_preferences.categ_3,
                categ_4: selected.category_preferences.categ_4,
              }}
              color={SEGMENT_COLORS[selected.id % 11]}
              size={320}
            />
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            <div className="card">
              <div className="card-title" style={{ color: 'var(--indigo-light)' }}>Recommended Action</div>
              <p style={{ fontSize: '0.875rem', lineHeight: 1.5, marginBottom: 12 }}>{selected.archetype_info.recommendation}</p>
              <div className={`churn-badge churn-${selected.archetype_info.churn_risk.toLowerCase().replace(' ', '-')}`}>
                Risk: {selected.archetype_info.churn_risk}
              </div>
            </div>

            <div className="card">
              <div className="card-title">Category Spend Averages</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginTop: 12 }}>
                {[0, 1, 2, 3, 4].map((i) => {
                  const pct = selected.category_preferences[`categ_${i}`] * 100;
                  return (
                    <div key={i}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: 6 }}>
                        <span style={{ color: '#94a3b8' }}>Category {i}</span>
                        <span style={{ fontWeight: 600 }}>{pct.toFixed(1)}%</span>
                      </div>
                      <div className="progress-bar">
                        <div
                          className="progress-bar-fill"
                          style={{ width: `${pct}%`, background: SEGMENT_COLORS[(i + 4) % 11] }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
