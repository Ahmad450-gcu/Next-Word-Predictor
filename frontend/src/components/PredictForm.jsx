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
    <div>
      <form onSubmit={handleSubmit}>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={4}
          placeholder="Type some text..."
        />
        <div>
          <label>
            Number of predictions:
            <input
              type="number"
              min={1}
              max={20}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
            />
          </label>
        </div>
        <button type="submit" disabled={loading || text.trim().length === 0}>
          {loading ? "Predicting..." : "Predict"}
        </button>
      </form>

      {error && <p style={{ color: "red" }}>{error}</p>}
      <PredictionList predictions={predictions} />
    </div>
  )
}

export default PredictForm