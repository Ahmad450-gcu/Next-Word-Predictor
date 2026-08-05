import PredictForm from "./components/PredictForm"

function App() {
  return (
    <div className="page">
      <header className="hero">
        <h1 className="hero-title">
          Predict. <span>The next word, before you write it.</span>
        </h1>
        <p className="hero-sub">
          A tied-weight LSTM trained on WikiText-2. Try out by writing some Wikipedia style text.
        </p>
      </header>
      <PredictForm />
    </div>
  )
}

export default App