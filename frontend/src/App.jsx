import React, { useState } from 'react';
import axios from 'axios';
import FileUpload from './components/FileUpload';
import ResultsDashboard from './components/ResultsDashboard';

function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleUpload = async (selectedFile) => {
    setFile(selectedFile);
    setLoading(true);
    setError(null);
    setResults(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post('http://127.0.0.1:8000/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResults(response.data);
    } catch (err) {
      console.error("Upload error:", err);
      setError(err.response?.data?.detail || "An error occurred during inference.");
    } finally {
      setLoading(false);
    }
  };

  const resetApp = () => {
    setFile(null);
    setResults(null);
    setError(null);
  };

  return (
    <>
      <div className="noise-overlay"></div>
      <div className="scanline"></div>
      
      <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '2rem' }}>
        
        <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: 'var(--border-thick)', paddingBottom: '1.5rem', marginBottom: '3rem' }}>
          <div>
            <h1 className="heading-gradient" style={{ fontSize: '2.5rem', margin: 0, letterSpacing: '-0.05em' }}>
              NEUROSLEEP
            </h1>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '0.25rem', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
              DEEP LEARNING PSG CLASSIFICATION
            </p>
          </div>
          
          <div style={{ display: 'flex', gap: '2rem', alignItems: 'center' }}>
            <a href="https://github.com/SaliqBashir/NeuroSleep" target="_blank" rel="noreferrer" style={{ background: 'var(--text-primary)', color: 'var(--text-light)', padding: '0.75rem 1.5rem', fontWeight: 700, fontSize: '0.85rem', textTransform: 'uppercase', transition: 'var(--transition-fast)', boxShadow: 'var(--shadow-brutal)' }} onMouseOver={(e) => { e.target.style.boxShadow = 'none'; e.target.style.transform = 'translate(4px, 4px)'; }} onMouseOut={(e) => { e.target.style.boxShadow = 'var(--shadow-brutal)'; e.target.style.transform = 'translate(0, 0)'; }}>
              VIEW SRC <span style={{ color: 'var(--accent-cyan)' }}>↗</span>
            </a>
          </div>
        </header>

        <main style={{ position: 'relative', zIndex: 1 }}>
          <div style={{ marginBottom: '2rem', fontSize: '0.7rem', color: 'var(--text-muted)', letterSpacing: '0.15em', textTransform: 'uppercase' }}>
            <span style={{ color: 'var(--accent-red)', fontWeight: 700 }}>01</span> WORKSPACE
          </div>

          {error && (
            <div className="glass-panel" style={{ padding: '1.5rem', marginBottom: '2rem', borderLeft: '8px solid var(--stage-rem)' }}>
              <p style={{ color: 'var(--text-primary)', fontWeight: 700, margin: 0, fontFamily: 'var(--font-sans)' }}>[ERR] {error}</p>
            </div>
          )}

          {!results && !loading && (
            <FileUpload onUpload={handleUpload} />
          )}

          {loading && (
            <div className="glass-panel" style={{ padding: '5rem 2rem', textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '2rem', background: 'var(--accent-cyan)' }}>
              <div style={{ width: '48px', height: '48px', border: '4px solid var(--text-primary)', borderTopColor: 'transparent' }} className="spinner"></div>
              <div>
                <h3 style={{ fontSize: '2rem', marginBottom: '0.5rem', color: 'var(--text-primary)' }}>RENDERING...</h3>
                <p style={{ color: 'var(--text-secondary)', fontWeight: 700 }}>Dual-Branch CNN & LSTM inference on {file?.name}</p>
              </div>
            </div>
          )}

          {results && (
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
                <h2 style={{ fontSize: '1.5rem', margin: 0, color: 'var(--text-primary)', background: 'var(--accent-cyan)', padding: '0.5rem 1rem', border: 'var(--border-thick)', boxShadow: '4px 4px 0 #000' }}>
                  TARGET_ <span style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>{results.filename}</span>
                </h2>
                <button onClick={resetApp} style={{ background: 'var(--bg-primary)', border: 'var(--border-thick)', color: 'var(--text-primary)', padding: '0.75rem 1.5rem', cursor: 'pointer', fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: '0.9rem', boxShadow: 'var(--shadow-brutal)', transition: 'var(--transition-fast)' }} onMouseOver={(e) => { e.target.style.background = 'var(--text-primary)'; e.target.style.color = 'var(--text-light)'; }} onMouseOut={(e) => { e.target.style.background = 'var(--bg-primary)'; e.target.style.color = 'var(--text-primary)'; }}>
                  [ NEW_ANALYSIS ]
                </button>
              </div>
              <ResultsDashboard data={results} />
            </div>
          )}
        </main>
      </div>
    </>
  );
}

export default App;
