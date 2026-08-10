import React, { useCallback, useState } from 'react';

const FileUpload = ({ onUpload }) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file) => {
    if (file.name.endsWith('.edf')) {
      setSelectedFile(file);
    } else {
      alert("Please upload an .edf file.");
    }
  };

  const submitFile = () => {
    if (selectedFile) {
      onUpload(selectedFile);
    }
  };

  return (
    <div className="glass-panel" style={{ padding: '4rem 2rem', textAlign: 'center', transition: 'var(--transition-fast)', border: dragActive ? '4px dashed var(--accent-red)' : 'var(--border-thick)', background: dragActive ? 'var(--accent-cyan)' : 'var(--bg-secondary)', boxShadow: 'var(--shadow-brutal)' }} onDragEnter={handleDrag} onDragLeave={handleDrag} onDragOver={handleDrag} onDrop={handleDrop}>
      
      {!selectedFile ? (
        <>

          <h3 style={{ fontSize: '2rem', marginBottom: '0.5rem', color: 'var(--text-primary)', fontFamily: 'var(--font-sans)', textTransform: 'uppercase' }}>INITIALIZE UPLOAD</h3>
          <p style={{ color: 'var(--text-secondary)', marginBottom: '2.5rem', maxWidth: '400px', margin: '0 auto 2.5rem', fontFamily: 'var(--font-mono)' }}>Drop raw PSG recording or click below to browse system files.</p>
          
          <input type="file" id="file-upload" accept=".edf" style={{ display: 'none' }} onChange={handleChange} />
          <label htmlFor="file-upload" style={{ background: 'var(--bg-primary)', border: 'var(--border-thick)', color: 'var(--text-primary)', padding: '1rem 2rem', cursor: 'pointer', fontFamily: 'var(--font-mono)', fontWeight: 700, display: 'inline-block', transition: 'var(--transition-fast)', boxShadow: 'var(--shadow-brutal)' }} onMouseOver={(e) => { e.target.style.boxShadow = 'none'; e.target.style.transform = 'translate(4px, 4px)'; }} onMouseOut={(e) => { e.target.style.boxShadow = 'var(--shadow-brutal)'; e.target.style.transform = 'translate(0, 0)'; }}>
            BROWSE SYSTEM
          </label>
        </>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '2rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', background: 'var(--bg-primary)', padding: '1rem 1.5rem', border: 'var(--border-thick)', boxShadow: '4px 4px 0 #000' }}>
            <span style={{ fontWeight: 700, color: 'var(--accent-red)' }}>DATA_STREAM:</span>
            <span style={{ fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>{selectedFile.name}</span>
            <button onClick={() => setSelectedFile(null)} style={{ background: 'var(--text-primary)', border: 'none', color: 'var(--accent-cyan)', cursor: 'pointer', padding: '0.2rem 0.5rem', fontFamily: 'var(--font-mono)', fontSize: '0.75rem', fontWeight: 700 }} title="Remove file">
              X
            </button>
          </div>
          <button onClick={submitFile} style={{ background: 'var(--accent-red)', color: '#fff', border: 'var(--border-thick)', padding: '1rem 3rem', cursor: 'pointer', fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: '1.05rem', transition: 'var(--transition-fast)', boxShadow: 'var(--shadow-brutal)' }} onMouseOver={(e) => { e.target.style.boxShadow = 'none'; e.target.style.transform = 'translate(4px, 4px)'; }} onMouseOut={(e) => { e.target.style.boxShadow = 'var(--shadow-brutal)'; e.target.style.transform = 'translate(0, 0)'; }}>
            [ EXECUTE_ANALYSIS ]
          </button>
        </div>
      )}
    </div>
  );
};

export default FileUpload;
