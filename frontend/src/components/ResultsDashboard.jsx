import React, { useMemo } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer 
} from 'recharts';

const valueMap = {
  "Wake": 0,
  "REM": -1,
  "N1": -2,
  "N2": -3,
  "N3": -4
};

const colorMap = {
  "Wake": "var(--stage-wake)",
  "REM": "var(--stage-rem)",
  "N1": "var(--stage-n1)",
  "N2": "var(--stage-n2)",
  "N3": "var(--stage-n3)"
};

const CustomTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="glass-panel" style={{ padding: '1rem', background: 'var(--bg-primary)', border: 'var(--border-thick)', boxShadow: '4px 4px 0 #000' }}>
        <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '0.85rem', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>T_ {data.time}</p>
        <p style={{ margin: '0.25rem 0 0 0', fontWeight: 700, color: 'var(--text-primary)', fontSize: '1.05rem', fontFamily: 'var(--font-sans)', textTransform: 'uppercase' }}>
          [{data.label === "N3" ? "N3_DEEP" : data.label}]
        </p>
      </div>
    );
  }
  return null;
};

const StatItem = ({ title, value, color, percent }) => (
  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', background: 'var(--bg-primary)', padding: '1.5rem', border: 'var(--border-thick)', boxShadow: '4px 4px 0 #000', borderLeft: `8px solid ${color}` }}>
    <span style={{ color: 'var(--text-secondary)', fontWeight: 700, fontSize: '0.8rem', fontFamily: 'var(--font-mono)', letterSpacing: '0.05em' }}>{title}</span>
    <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem' }}>
      <h3 style={{ fontSize: '2rem', fontWeight: 700, margin: 0, color: 'var(--text-primary)', lineHeight: 1, fontFamily: 'var(--font-mono)' }}>{value}</h3>
      {percent && <span style={{ color: 'var(--text-light)', fontWeight: 700, fontSize: '0.9rem', background: color, padding: '0 4px' }}>{percent}%</span>}
    </div>
  </div>
);

const ResultsDashboard = ({ data }) => {
  const { summary, stages } = data;
  
  const chartData = useMemo(() => {
    return stages.map((stage, index) => {
      // Each epoch is 30s
      const totalSeconds = index * 30;
      const hours = Math.floor(totalSeconds / 3600);
      const minutes = Math.floor((totalSeconds % 3600) / 60);
      
      return {
        epoch: index,
        time: `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}`,
        value: valueMap[stage],
        label: stage
      };
    });
  }, [stages]);

  const getPercent = (count) => ((count / summary.total_epochs) * 100).toFixed(1);
  const getHours = (count) => ((count * 30) / 3600).toFixed(1);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '3rem' }}>
      
      {/* Unified Overview Stats Panel */}
      <div className="glass-panel" style={{ padding: '2.5rem', background: 'var(--bg-secondary)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '2.5rem' }}>
          <div>
            <h3 style={{ fontWeight: 800, fontSize: '1.25rem', color: 'var(--text-primary)', margin: 0, fontFamily: 'var(--font-sans)' }}>DATA_SUMMARY</h3>
            <span style={{ color: 'var(--accent-red)', fontSize: '0.8rem', fontFamily: 'var(--font-mono)', fontWeight: 700 }}>INSIGHTS EXTRACTED</span>
          </div>
          <button className="no-print" onClick={() => window.print()} style={{ background: 'var(--text-primary)', border: 'none', color: 'var(--accent-cyan)', padding: '0.75rem 1.25rem', cursor: 'pointer', fontWeight: 700, fontSize: '0.85rem', fontFamily: 'var(--font-mono)', boxShadow: 'var(--shadow-brutal)', transition: 'var(--transition-fast)' }} onMouseOver={(e) => { e.target.style.boxShadow = 'none'; e.target.style.transform = 'translate(4px, 4px)'; }} onMouseOut={(e) => { e.target.style.boxShadow = 'var(--shadow-brutal)'; e.target.style.transform = 'translate(0, 0)'; }}>
            [ PRINT ]
          </button>
        </div>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '2rem' }}>
          <StatItem 
            title="TOTAL_SLEEP" 
            value={`${(summary.total_minutes / 60).toFixed(1)}h`} 
            color="var(--accent-red)" 
          />
          <StatItem 
            title="EFFICIENCY" 
            value={`${((1 - (summary.Wake / summary.total_epochs)) * 100).toFixed(1)}%`} 
            color="var(--accent-cyan)" 
          />
          <StatItem 
            title="DEEP(N3)" 
            value={`${getHours(summary.N3)}h`} 
            percent={getPercent(summary.N3)}
            color="var(--text-primary)" 
          />
          <StatItem 
            title="REM_STAGE" 
            value={`${getHours(summary.REM)}h`} 
            percent={getPercent(summary.REM)}
            color="var(--accent-red)" 
          />
          <StatItem 
            title="LIGHT(N1+N2)" 
            value={`${getHours(summary.N1 + summary.N2)}h`} 
            percent={getPercent(summary.N1 + summary.N2)}
            color="var(--text-secondary)" 
          />
          <StatItem 
            title="AWAKE_TIME" 
            value={`${getHours(summary.Wake)}h`} 
            percent={getPercent(summary.Wake)}
            color="var(--accent-cyan)" 
          />
        </div>
      </div>

      {/* Hypnogram Chart */}
      <div className="glass-panel no-print" style={{ padding: '2.5rem', height: '500px', position: 'relative' }}>
        <div style={{ position: 'absolute', top: '-14px', left: '24px', background: 'var(--accent-red)', color: 'var(--text-light)', padding: '4px 12px', fontSize: '0.75rem', fontWeight: 700, letterSpacing: '0.1em' }}>
          HYPNOGRAM_RENDER
        </div>
        
        <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="2 2" stroke="var(--border-color)" vertical={false} opacity={0.2} />
              <XAxis 
                dataKey="time" 
                stroke="var(--text-primary)" 
                tick={{ fill: 'var(--text-primary)', fontSize: 12, fontWeight: 700, fontFamily: 'var(--font-mono)' }}
                minTickGap={50}
                tickLine={false}
                axisLine={{ strokeWidth: 2 }}
              dy={15}
            />
            <YAxis 
              domain={[-4, 0]} 
              ticks={[0, -1, -2, -3, -4]}
              tickFormatter={(val) => Object.keys(valueMap).find(k => valueMap[k] === val)}
              stroke="var(--text-primary)"
              tick={{ fill: 'var(--text-primary)', fontSize: 12, fontWeight: 700, fontFamily: 'var(--font-mono)' }}
              axisLine={{ strokeWidth: 2 }}
              tickLine={false}
              dx={-15}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ stroke: 'rgba(0,0,0,0.1)', strokeWidth: 2 }} />
            <Line 
              type="stepAfter" 
              dataKey="value" 
              stroke="var(--accent-red)" 
              strokeWidth={4} 
              dot={false}
              activeDot={{ r: 0 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      
    </div>
  );
};

export default ResultsDashboard;
