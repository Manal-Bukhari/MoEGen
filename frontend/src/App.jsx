import { useState, useEffect } from 'react'
import './styles/App.css'
import TextGenerator from './components/TextGenerator'
import ExpertSelector from './components/ExpertSelector'
import OutputDisplay from './components/OutputDisplay'
import Header from './components/Header'
import Footer from './components/Footer'
import { getExperts } from './services/api'

function App() {
  const [experts, setExperts] = useState([])
  const [selectedExpert, setSelectedExpert] = useState('auto')
  const [generatedResult, setGeneratedResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    // Fetch available experts on mount
    const fetchExperts = async () => {
      try {
        const data = await getExperts()
        setExperts(data.experts)
      } catch (err) {
        console.error('Error fetching experts:', err)
      }
    }
    fetchExperts()
  }, [])

  const handleGenerate = (result) => {
    setGeneratedResult(result)
    setError(null)
  }

  const handleError = (err) => {
    setError(err)
    setGeneratedResult(null)
  }

  const handleClear = () => {
    setGeneratedResult(null)
    setError(null)
  }

  return (
    <div className="app">
      <Header />
      
      <main className="main-content">
        <div className="container">
          <div className="intro-section">
            <h2>🎯 Mixture-of-Experts Text Generation</h2>
            <p>
              Enter your prompt below and our AI will automatically select the best expert
              (Story, Poem, or Email) to generate your content. Or choose an expert manually!
            </p>
          </div>

          <div className="generator-section">
            <ExpertSelector 
              experts={experts}
              selectedExpert={selectedExpert}
              onSelectExpert={setSelectedExpert}
            />

            <TextGenerator 
              selectedExpert={selectedExpert}
              onGenerate={handleGenerate}
              onError={handleError}
              setLoading={setLoading}
            />
          </div>

          {loading && (
            <div className="loading-indicator">
              <div className="spinner"></div>
              <p>Generating your content...</p>
            </div>
          )}

          {error && (
            <div className="error-message">
              <h3>❌ Error</h3>
              <p>{error}</p>
            </div>
          )}

          {generatedResult && (
            <OutputDisplay 
              result={generatedResult}
              onClear={handleClear}
            />
          )}
        </div>
      </main>

      <Footer />
    </div>
  )
}

export default App
