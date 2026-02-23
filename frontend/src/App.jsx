import { useState, useEffect } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Sidebar } from './components/Sidebar'
import Dashboard from './pages/Dashboard'
import CustomerLookup from './pages/CustomerLookup'
import Predictor from './pages/Predictor'
import SegmentExplorer from './pages/SegmentExplorer'
import CategoryExplorer from './pages/CategoryExplorer'
import BusinessIntelligence from './pages/BusinessIntelligence'
import { healthCheck } from './api/client'
import { Menu } from 'lucide-react'

function App() {
  const [apiStatus, setApiStatus] = useState('connecting'); // 'connecting' | 'online' | 'offline'
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => {
    // Initial check
    healthCheck()
      .then(() => setApiStatus('online'))
      .catch(() => setApiStatus('offline'));

    // Periodic check
    const interval = setInterval(() => {
      healthCheck()
        .then(() => setApiStatus('online'))
        .catch(() => setApiStatus('offline'));
    }, 15000);

    return () => clearInterval(interval);
  }, []);

  return (
    <BrowserRouter>
      <div className="app-layout">
        <button className="mobile-menu-btn btn-ghost" onClick={() => setSidebarOpen(true)}>
          <Menu size={20} />
        </button>
        {sidebarOpen && <div className="sidebar-overlay" onClick={() => setSidebarOpen(false)} />}
        <Sidebar apiStatus={apiStatus} isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/customers" element={<CustomerLookup />} />
            <Route path="/predict" element={<Predictor />} />
            <Route path="/segments" element={<SegmentExplorer />} />
            <Route path="/categories" element={<CategoryExplorer />} />
            <Route path="/bi" element={<BusinessIntelligence />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}

export default App
