const PALETTE = ["#EDE7FB", "#DEEEFB", "#E3F3E9", "#FBEDE3", "#FBE7ED"]

function PredictionList({ predictions, loading, topK }) {
  if (loading) {
    return (
      <ol className="results-grid">
        {Array.from({ length: topK }).map((_, i) => (
          <li key={i} className="result-card skeleton" style={{ backgroundColor: PALETTE[i % PALETTE.length] }} />
        ))}
      </ol>
    )
  }

  if (!predictions || predictions.length === 0) {
    return null
  }

  return (
    <ol className="results-grid">
      {predictions.map((p, i) => (
        <li key={i} className="result-card" style={{ backgroundColor: PALETTE[i % PALETTE.length] }}>
          <span className="result-rank">{String(i + 1).padStart(2, "0")}</span>
          <span className="result-word">{p.word}</span>
          <span className="result-prob">{(p.probability * 100).toFixed(1)}%</span>
        </li>
      ))}
    </ol>
  )
}

export default PredictionList