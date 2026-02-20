import { useState, useEffect } from 'react';
import { getKPIs, getSegmentsRevenue, getCategories } from '../api/client';
import { Users, CreditCard, RefreshCw, BarChart2, PieChart as PieIcon } from 'lucide-react';
import { BarChart, Bar, XAxis, Tooltip, ResponsiveContainer, Cell, PieChart, Pie, Legend } from 'recharts';
import { SEGMENT_COLORS } from '../components/Sidebar';

export default function Dashboard() {
  const [kpis, setKpis] = useState(null);
  const [revenueData, setRevenueData] = useState([]);
  const [categories, setCategories] = useState([]);

  useEffect(() => {
    Promise.all([getKPIs(), getSegmentsRevenue(), getCategories()])
      .then(([k, rev, cat]) => {
        // Sort segments for bar chart
        k.segment_distribution.sort((a, b) => a.segment_id - b.segment_id);
        setKpis(k);
        setRevenueData(rev.revenue);
        setCategories(cat.categories);
      })
      .catch(console.error);
  }, []);

  if (!kpis) return <div className="loading-state"><div className="spinner" />Loading Dashboard...</div>;

  return (
    <div className="page-enter">
      <div className="page-header">
        <h1 className="page-title">Dashboard Overview</h1>
        <p className="page-subtitle">Real-time summary of your customer base and segments.</p>
      </div>

      <div className="kpi-grid">
        <div className="kpi-card">
          <div className="kpi-icon"><Users size={20} /></div>
          <div className="kpi-value">{kpis.total_customers.toLocaleString()}</div>
          <div className="kpi-label">Total Customers</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-icon"><CreditCard size={20} /></div>
          <div className="kpi-value">£{(kpis.total_revenue / 1000).toFixed(0)}k</div>
          <div className="kpi-label">Total Revenue</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-icon"><RefreshCw size={20} /></div>
          <div className="kpi-value">{(kpis.cancellation_rate * 100).toFixed(1)}%</div>
          <div className="kpi-label">Cancellation Rate</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-icon"><BarChart2 size={20} /></div>
          <div className="kpi-value">£{kpis.avg_basket_value.toFixed(0)}</div>
          <div className="kpi-label">Avg Basket Value</div>
        </div>
      </div>

      <div className="charts-grid">
        {/* Segment Size Bar Chart */}
        <div className="card">
          <div className="card-title">Customer Distribution by Segment</div>
          <div style={{ height: 280, marginTop: 20 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={kpis.segment_distribution} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <XAxis dataKey="segment_id" tick={{ fill: '#94a3b8', fontSize: 12 }} axisLine={false} tickLine={false} />
                <Tooltip
                  cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                  contentStyle={{ background: '#181b28', borderColor: 'rgba(255,255,255,0.1)', borderRadius: 8 }}
                  formatter={(val, name, props) => [val, `Segment ${props.payload.segment_id}`]}
                />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {kpis.segment_distribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={SEGMENT_COLORS[entry.segment_id % SEGMENT_COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Revenue Donut */}
        <div className="card">
          <div className="card-title">Lifetime Revenue by Segment</div>
          <div style={{ height: 280, marginTop: 20 }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={revenueData}
                  cx="50%" cy="50%"
                  innerRadius={70} outerRadius={110}
                  paddingAngle={2}
                  dataKey="estimated_revenue"
                  nameKey="segment_id"
                  stroke="none"
                >
                  {revenueData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={SEGMENT_COLORS[entry.segment_id % SEGMENT_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ background: '#181b28', borderColor: 'rgba(255,255,255,0.1)', borderRadius: 8 }}
                  formatter={(val, name) => [`£${val.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, `Segment ${name}`]}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Categories Share */}
        <div className="card">
          <div className="card-title">Revenue by Product Category</div>
          <div style={{ height: 280, marginTop: 20 }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={categories}
                  cx="50%" cy="50%"
                  outerRadius={100}
                  dataKey="revenue_share"
                  nameKey="id"
                  stroke="none"
                >
                  {categories.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={SEGMENT_COLORS[(index + 4) % SEGMENT_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ background: '#181b28', borderColor: 'rgba(255,255,255,0.1)', borderRadius: 8 }}
                  formatter={(val, name) => [`${(val * 100).toFixed(1)}%`, `Category ${name}`]}
                />
                <Legend iconType="circle" wrapperStyle={{ fontSize: 12, color: '#94a3b8' }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
          <div className="kpi-icon" style={{ background: 'rgba(16, 185, 129, 0.15)', color: '#10b981' }}>
            <PieIcon size={20} />
          </div>
          <div>
            <div className="page-title" style={{ fontSize: '1.2rem', marginBottom: 8 }}>Voting Classifier</div>
            <p style={{ color: '#94a3b8', fontSize: '0.85rem', lineHeight: 1.5 }}>
              The First-Purchase Predictor is running an ensemble of Logistic Regression, k-Nearest Neighbors, and Gradient Boosting.
              <br /><br />
              <strong>Test Set Accuracy:</strong> {(kpis.classifier_accuracy * 100).toFixed(1)}%
              <br />
              <strong>Training Window:</strong> {kpis.date_range.start} to {kpis.date_range.end}
            </p>
          </div>
        </div>

      </div>
    </div>
  );
}
