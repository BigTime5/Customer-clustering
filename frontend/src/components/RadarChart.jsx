/**
 * RadarChart — custom SVG radar for customer segment fingerprints.
 * Axes: Visit Count, Avg Basket, Max Basket, Min Basket, Categ0-4 (configurable)
 */

const DEFAULT_AXES = [
  { key: 'visits', label: 'Visits' },
  { key: 'avg_basket', label: 'Avg Basket' },
  { key: 'max_basket', label: 'Max Basket' },
  { key: 'categ_0', label: 'Category 0' },
  { key: 'categ_1', label: 'Category 1' },
  { key: 'categ_2', label: 'Category 2' },
  { key: 'categ_3', label: 'Category 3' },
  { key: 'categ_4', label: 'Category 4' },
];

function polarToCartesian(cx, cy, r, angleDeg) {
  const rad = (angleDeg - 90) * (Math.PI / 180);
  return {
    x: cx + r * Math.cos(rad),
    y: cy + r * Math.sin(rad),
  };
}

export function RadarChart({ data, axes = DEFAULT_AXES, color = '#6366f1', compareData = null, compareColor = '#10b981', size = 260 }) {
  const cx = size / 2;
  const cy = size / 2;
  const R = size * 0.38;
  const n = axes.length;
  const levels = [0.25, 0.5, 0.75, 1.0];

  // Normalise 0–1
  const norm = (val) => Math.min(1, Math.max(0, val));

  const points = (values) =>
    axes.map((ax, i) => {
      const angle = (360 / n) * i;
      const r = R * norm(values[ax.key] ?? 0);
      return polarToCartesian(cx, cy, r, angle);
    });

  const toPath = (pts) =>
    pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ') + ' Z';

  const mainPts = points(data);
  const comparePts = compareData ? points(compareData) : null;

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} style={{ overflow: 'visible' }}>
      {/* Grid rings */}
      {levels.map((level) => {
        const ringPts = axes.map((_, i) => {
          const angle = (360 / n) * i;
          return polarToCartesian(cx, cy, R * level, angle);
        });
        return (
          <polygon
            key={level}
            points={ringPts.map(p => `${p.x},${p.y}`).join(' ')}
            fill="none"
            stroke="rgba(255,255,255,0.07)"
            strokeWidth="1"
          />
        );
      })}

      {/* Axis lines */}
      {axes.map((ax, i) => {
        const angle = (360 / n) * i;
        const end = polarToCartesian(cx, cy, R, angle);
        return <line key={i} x1={cx} y1={cy} x2={end.x} y2={end.y} stroke="rgba(255,255,255,0.1)" strokeWidth="1" />;
      })}

      {/* Compare shape */}
      {comparePts && (
        <path
          d={toPath(comparePts)}
          fill={`${compareColor}20`}
          stroke={compareColor}
          strokeWidth="1.5"
          strokeDasharray="4 3"
        />
      )}

      {/* Main shape */}
      <path
        d={toPath(mainPts)}
        fill={`${color}28`}
        stroke={color}
        strokeWidth="2"
      />

      {/* Data points */}
      {mainPts.map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r="3.5" fill={color} />
      ))}

      {/* Labels */}
      {axes.map((ax, i) => {
        const angle = (360 / n) * i;
        const labelR = R + 22;
        const pos = polarToCartesian(cx, cy, labelR, angle);
        const textAnchor = pos.x < cx - 5 ? 'end' : pos.x > cx + 5 ? 'start' : 'middle';
        return (
          <text
            key={i}
            x={pos.x}
            y={pos.y}
            textAnchor={textAnchor}
            dominantBaseline="middle"
            fontSize="10"
            fill="rgba(148,163,184,0.9)"
            fontFamily="Inter, sans-serif"
          >
            {ax.label}
          </text>
        );
      })}
    </svg>
  );
}
