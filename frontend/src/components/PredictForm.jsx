import { useState } from "react"
import { getPrediction } from "../api/client"
import PredictionList from "./PredictionList"

function PredictForm() {
  const [text, setText] = useState("")
  const [topK, setTopK] = useState(5)
  const [predictions, setPredictions] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleSubmit(e) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const result = await getPrediction(text, topK)
      setPredictions(result.predictions)
    } catch (err) {
      setError(err.message)
      setPredictions([])
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="card">
      <form onSubmit={handleSubmit}>
        <textarea
          className="seed-input"
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={4}
          placeholder="Begin a sentence…"
        />

        <div className="controls">
          <label className="stepper">
            <span>Candidates</span>
            <input
              type="number"
              min={1}
              max={20}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
            />
          </label>

          <button id='pred-but' type="submit" disabled={loading || text.trim().length === 0}>
            {loading ? "Predicting…" : "Predict next word"}
          </button>
        </div>
      </form>

      {error && <p className="error" role="alert">{error}</p>}
      <PredictionList predictions={predictions} loading={loading} topK={topK} />
    </section>
  )
}

export default PredictForm