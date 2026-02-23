import { NavLink, useLocation } from 'react-router-dom';
import {
  LayoutDashboard, Users, Zap, PieChart, Package, MessageSquare, X
} from 'lucide-react';

const SEGMENT_COLORS = [
  '#6366f1', '#10b981', '#f59e0b', '#f43f5e', '#38bdf8',
  '#a78bfa', '#34d399', '#fb923c', '#e879f9', '#22d3ee', '#84cc16'
];

const NAV_ITEMS = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/customers', icon: Users, label: 'Customer Lookup' },
  { to: '/predict', icon: Zap, label: 'First-Purchase Predictor' },
  { to: '/segments', icon: PieChart, label: 'Segment Explorer' },
  { to: '/categories', icon: Package, label: 'Category Explorer' },
  { to: '/bi', icon: MessageSquare, label: 'Business Intelligence' },
];

export function Sidebar({ apiStatus, isOpen, onClose }) {
  return (
    <aside className={`sidebar ${isOpen ? 'open' : ''}`}>
      <div className="sidebar-logo">
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div className="sidebar-logo-icon">🧠</div>
          <div>
            <div className="sidebar-logo-text">CustomerIQ</div>
            <div className="sidebar-logo-sub">Segmentation Intelligence</div>
          </div>
        </div>
        <button className="sidebar-close btn-ghost" onClick={onClose}>
          <X size={20} />
        </button>
      </div>

      <nav className="sidebar-nav">
        <div className="nav-section-label">Navigation</div>
        {NAV_ITEMS.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            onClick={onClose}
            className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}
          >
            <Icon size={16} />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="sidebar-status">
          <div className={`status-dot${apiStatus === 'offline' ? ' offline' : ''}`} />
          {apiStatus === 'online' ? 'API Connected' : apiStatus === 'offline' ? 'API Offline' : 'Connecting…'}
        </div>
      </div>
    </aside>
  );
}

export { SEGMENT_COLORS };
