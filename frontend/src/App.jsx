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

function App() {
  const [apiStatus, setApiStatus] = useState('connecting'); // 'connecting' | 'online' | 'offline'

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
        <Sidebar apiStatus={apiStatus} />
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
