import { useState } from 'react';
import { getCustomer } from '../api/client';
import { Search, Loader2, AlertCircle } from 'lucide-react';
import { RadarChart } from '../components/RadarChart';
import { SEGMENT_COLORS } from '../components/Sidebar';

export default function CustomerLookup() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [data, setData] = useState(null);

  const handleSearch = async (e) => {
    e?.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await getCustomer(query.trim());
      setData(res);
    } catch (err) {
      setError(err.response?.data?.detail || 'Customer not found. Please try a valid CustomerID (e.g. 12346 - 18287).');
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-enter">
      <div className="page-header">
        <h1 className="page-title">Customer Lookup</h1>
        <p className="page-subtitle">Instantly view a customer's behavioural profile and predicted archetype.</p>
      </div>

      <form className="search-bar" onSubmit={handleSearch}>
        <input
          type="text"
          className="search-input"
          placeholder="Enter CustomerID (e.g. 17850, 12346)"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button type="submit" className="btn btn-primary" disabled={loading}>
          {loading ? <Loader2 size={16} className="spinner" style={{ margin: 0, width: 16, height: 16 }} /> : <Search size={16} />}
          Search
        </button>
      </form>

      {error && (
        <div className="error-state">
          <AlertCircle size={20} style={{ marginBottom: 8 }} />
          <div>{error}</div>
        </div>
      )}

      {!data && !error && !loading && (
        <div className="empty-state">
          <Search size={40} opacity={0.3} />
          <div className="empty-state-title">No customer selected</div>
          <div className="empty-state-desc">Enter a CustomerID above to retrieve their full archetype and behaviour fingerprint.</div>
        </div>
      )}

      {data && (
        <div className="customer-profile page-enter">
          {/* Left Col: Persona & Radar */}
          <div className="card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center' }}>
            <div className="segment-badge" style={{ background: SEGMENT_COLORS[data.segment_id % 11], marginBottom: 16 }}>
              {data.segment_id}
            </div>
            <h2 style={{ fontSize: '1.4rem', fontWeight: 800, marginBottom: 4 }}>
              {data.archetype_info.persona}
            </h2>
            <p style={{ color: '#94a3b8', fontSize: '0.85rem', marginBottom: 24, padding: '0 20px' }}>
              {data.archetype_info.description}
            </p>

            <RadarChart
              data={{
                visits: data.stats.visit_count / 50, // rough normalisation for visual
                avg_basket: data.stats.avg_basket / 1000,
                max_basket: data.stats.max_basket / 1500,
                categ_0: data.stats.category_split.categ_0,
                categ_1: data.stats.category_split.categ_1,
                categ_2: data.stats.category_split.categ_2,
                categ_3: data.stats.category_split.categ_3,
                categ_4: data.stats.category_split.categ_4,
              }}
              compareData={{
                visits: data.segment_profile.avg_visit_count / 50,
                avg_basket: data.segment_profile.avg_basket_value / 1000,
                max_basket: data.segment_profile.max_basket_value / 1500,
                categ_0: data.segment_profile.category_preferences.categ_0,
                categ_1: data.segment_profile.category_preferences.categ_1,
                categ_2: data.segment_profile.category_preferences.categ_2,
                categ_3: data.segment_profile.category_preferences.categ_3,
                categ_4: data.segment_profile.category_preferences.categ_4,
              }}
              color={SEGMENT_COLORS[data.segment_id % 11]}
              size={300}
            />
            <div style={{ display: 'flex', gap: 16, marginTop: 16, fontSize: '0.75rem', color: '#94a3b8' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', background: SEGMENT_COLORS[data.segment_id % 11] }} />
                Customer
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', background: 'transparent', border: '1px dashed #10b981' }} />
                Segment Avg
              </div>
            </div>
          </div>

          {/* Right Col: Stats & Recommendation */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            <div className="card">
              <div className="card-title">Customer Statistics</div>
              <div className="stat-row">
                <span className="stat-label">Total Visits</span>
                <span className="stat-value">{data.stats.visit_count}</span>
              </div>
              <div className="stat-row">
                <span className="stat-label">Lifetime Value</span>
                <span className="stat-value">£{data.stats.total_spend.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
              </div>
              <div className="stat-row">
                <span className="stat-label">Average Basket</span>
                <span className="stat-value">£{data.stats.avg_basket.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
              </div>
              <div className="stat-row">
                <span className="stat-label">Maximum Basket</span>
                <span className="stat-value">£{data.stats.max_basket.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
              </div>
            </div>

            <div className="card">
              <div className="card-title">Category Spend Breakdown</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginTop: 12 }}>
                {[0, 1, 2, 3, 4].map((i) => {
                  const pct = data.stats.category_split[`categ_${i}`] * 100;
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

            <div className="card" style={{ borderColor: 'var(--border-accent)', background: 'rgba(99,102,241,0.03)' }}>
              <div className="card-title" style={{ color: 'var(--indigo-light)' }}>Recommended Action</div>
              <p style={{ fontSize: '0.875rem', lineHeight: 1.5, marginBottom: 12 }}>{data.archetype_info.recommendation}</p>
              <div className={`churn-badge churn-${data.archetype_info.churn_risk.toLowerCase().replace(' ', '-')}`}>
                Risk: {data.archetype_info.churn_risk}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
