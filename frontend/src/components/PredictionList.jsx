function PredictionList({ predictions }) {
  if (!predictions || predictions.length === 0) {
    return null
  }

  return (
    <ul>
      {predictions.map((p, i) => (
        <li key={i}>
          <strong>{p.word}</strong> — {(p.probability * 100).toFixed(1)}%
        </li>
      ))}
    </ul>
  )
}

export default PredictionList