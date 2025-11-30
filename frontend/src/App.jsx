import { useState, useEffect, useRef } from 'react'
import './styles/App.css'
import TextGenerator from './components/TextGenerator'
import ExpertSelector from './components/ExpertSelector'
import OutputDisplay from './components/OutputDisplay'
import Header from './components/Header'
import Footer from './components/Footer'
import { getExperts } from './services/api'
import { MessageSquare, Sparkles } from 'lucide-react'

function App() {
  const [experts, setExperts] = useState([])
  const [selectedExpert, setSelectedExpert] = useState('auto')
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const messagesEndRef = useRef(null)

  useEffect(() => {
    const fetchExperts = async () => {
      try {
        const data = await getExperts()
        
        if (Array.isArray(data)) {
          setExperts(data)
        } else if (data.experts && Array.isArray(data.experts)) {
          setExperts(data.experts)
        } else {
          console.error('Unexpected experts format:', data)
          setExperts([])
        }
      } catch (err) {
        console.error('Error fetching experts:', err)
        setError('Failed to load experts. Please ensure the backend is running.')
        setExperts([])
      }
    }
    fetchExperts()
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleUserMessage = (userPrompt) => {
    // Add user message immediately
    const userMessage = {
      type: 'user',
      content: userPrompt,
      timestamp: new Date()
    }
    setMessages(prev => [...prev, userMessage])
    setError(null)
    setLoading(true)
  }

  const handleGenerate = (result) => {
    // Add AI response when it arrives
    const aiMessage = {
      type: 'ai',
      content: result.generated_text,
      result: result,
      timestamp: new Date()
    }

    setMessages(prev => [...prev, aiMessage])
    setError(null)
    setLoading(false)
  }

  const handleError = (err) => {
    setError(err)
    setLoading(false)
    const errorMessage = {
      type: 'error',
      content: err,
      timestamp: new Date()
    }
    setMessages(prev => [...prev, errorMessage])
  }

  const handleClear = () => {
    setMessages([])
    setError(null)
  }

  return (
    <div className="app">
      <Header sidebarOpen={sidebarOpen} setSidebarOpen={setSidebarOpen} />
      
      <div className="app-container">
        <ExpertSelector 
          experts={experts}
          selectedExpert={selectedExpert}
          onSelectExpert={setSelectedExpert}
          sidebarOpen={sidebarOpen}
          setSidebarOpen={setSidebarOpen}
        />

        <main className="chat-main">
          <div className="chat-container">
            {messages.length === 0 ? (
              <div className="welcome-screen">
                <div className="welcome-content">
                  <div className="welcome-icon">
                    <img src="/logo.png" alt="MoE Generator Logo" className="welcome-logo" />
                  </div>
                  <h1>Mixture-of-Experts AI</h1>
                  <p>Select an expert mode and start a conversation to generate content</p>
                  <div className="expert-badges">
                    <span className="expert-badge-item">Story</span>
                    <span className="expert-badge-item">Poem</span>
                    <span className="expert-badge-item">Email</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="messages-container">
                {messages.map((message, index) => (
                  <OutputDisplay
                    key={index}
                    message={message}
                    onClear={handleClear}
                  />
                ))}
                {loading && (
                  <div className="message message-ai">
                    <div className="message-content">
                      <div className="typing-indicator">
                        <span></span>
                        <span></span>
                        <span></span>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
            )}

            {error && messages.length === 0 && (
              <div className="error-banner">
                <p>{error}</p>
              </div>
            )}

            <TextGenerator 
              selectedExpert={selectedExpert}
              onUserMessage={handleUserMessage}
              onGenerate={handleGenerate}
              onError={handleError}
              loading={loading}
            />
          </div>
        </main>
      </div>

      <Footer />
    </div>
  )
}

export default App
