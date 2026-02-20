import { useState, useEffect } from 'react';
import { getCategories } from '../api/client';
import { SEGMENT_COLORS } from '../components/Sidebar';
import { Loader2, Package } from 'lucide-react';

export default function CategoryExplorer() {
  const [categories, setCategories] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getCategories()
      .then(res => {
        setCategories(res.categories);
        setLoading(false);
      })
      .catch(console.error);
  }, []);

  if (loading) return <div className="loading-state"><Loader2 className="spinner" />Loading Categories...</div>;

  return (
    <div className="page-enter">
      <div className="page-header">
        <h1 className="page-title">Product Category Explorer</h1>
        <p className="page-subtitle">NLP keyword extraction grouped {categories.reduce((acc, c) => acc + c.product_count, 0).toLocaleString()} products into 5 categories.</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: 20 }}>
        {categories.map((cat, i) => {
          const color = SEGMENT_COLORS[(i + 4) % 11];
          return (
            <div key={cat.id} className="card" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <div style={{ width: 40, height: 40, background: `${color}20`, color, borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <Package size={20} />
                  </div>
                  <div>
                    <div style={{ fontWeight: 800, fontSize: '1.1rem' }}>Category {cat.id}</div>
                    <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>{cat.product_count.toLocaleString()} products</div>
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontWeight: 700, color, fontSize: '1.2rem' }}>{(cat.revenue_share * 100).toFixed(1)}%</div>
                  <div style={{ fontSize: '0.65rem', color: '#94a3b8', textTransform: 'uppercase' }}>Rev Share</div>
                </div>
              </div>

              <div>
                <div className="card-title">Top NLP Keywords</div>
                <div className="wordcloud">
                  {cat.top_keywords.map((kw, idx) => (
                    <span
                      key={kw}
                      className="wordcloud-word"
                      style={{ fontSize: Math.max(0.75, 1.3 - (idx * 0.04)) + 'rem', opacity: Math.max(0.4, 1 - (idx * 0.05)) }}
                    >
                      {kw}
                    </span>
                  ))}
                </div>
              </div>

              <div>
                <div className="card-title">Sample Products</div>
                <ul style={{ listStyleType: 'none', padding: 0, margin: 0, fontSize: '0.8rem', color: '#94a3b8', display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {cat.sample_products.slice(0, 5).map(prod => (
                    <li key={prod} style={{ background: 'var(--bg-main)', padding: '6px 10px', borderRadius: 4 }}>
                      {prod}
                    </li>
                  ))}
                </ul>
              </div>

            </div>
          );
        })}
      </div>
    </div>
  );
}
