import React, { useState, useRef, useEffect } from 'react'
import { generateText } from '../services/api'
import { Send, Sparkles, BookOpen, PenTool, Mail } from 'lucide-react'

const expertIcons = {
  auto: Sparkles,
  story: BookOpen,
  poem: PenTool,
  email: Mail
}

function TextGenerator({ selectedExpert, onUserMessage, onGenerate, onError, loading }) {
  const [prompt, setPrompt] = useState('')
  const [showExamples, setShowExamples] = useState(false)
  const textareaRef = useRef(null)

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`
    }
  }, [prompt])

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    if (!prompt.trim() || loading) {
      return
    }

    const userPrompt = prompt.trim()
    
    // Clear input immediately
    setPrompt('')
    
    // Notify parent to add user message immediately
    onUserMessage(userPrompt)
    
    // Make API call
    try {
      const result = await generateText(
        userPrompt,
        selectedExpert
      )
      onGenerate(result)
    } catch (err) {
      onError(err.toString())
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  const examplePrompts = {
    auto: [
      "Write a short story about a robot learning to paint",
      "Write a poem about the ocean at sunset",
      "Write an email requesting a meeting with a client"
    ],
    story: [
      "Once upon a time in a magical forest...",
      "The detective entered the abandoned mansion...",
      "In a distant galaxy, a young pilot discovered..."
    ],
    poem: [
      "The moon hangs low in the midnight sky...",
      "Whispers of autumn leaves...",
      "Love is like a river..."
    ],
    email: [
      "Write a professional email requesting time off",
      "Draft an email following up on a job application",
      "Compose an email thanking a colleague"
    ]
  }

  const getCurrentExamples = () => {
    return examplePrompts[selectedExpert] || examplePrompts.auto
  }

  const Icon = expertIcons[selectedExpert] || Sparkles

  return (
    <div className="chat-input-container">
      {showExamples && (
        <div className="example-prompts-panel">
          <div className="example-prompts-header">
            <span>Example Prompts</span>
            <button 
              className="close-examples"
              onClick={() => setShowExamples(false)}
            >
              ×
            </button>
          </div>
          <div className="example-prompts-list">
            {getCurrentExamples().map((example, index) => (
              <button
                key={index}
                type="button"
                className="example-prompt-btn"
                onClick={() => {
                  setPrompt(example)
                  setShowExamples(false)
                }}
              >
                {example}
              </button>
            ))}
          </div>
        </div>
      )}

      <form className="chat-input-form" onSubmit={handleSubmit}>
        <div className="chat-input-wrapper">
          <div className="expert-indicator">
            <Icon size={18} />
          </div>
          <textarea
            ref={textareaRef}
            className="chat-input"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your message here... (Press Enter to send, Shift+Enter for new line)"
            rows={1}
            disabled={loading}
          />
          <div className="chat-input-actions">
            <button
              type="button"
              className="example-toggle-btn"
              onClick={() => setShowExamples(!showExamples)}
              title="Show examples"
            >
              <Sparkles size={18} />
            </button>
            <button 
              type="submit" 
              className="send-button"
              disabled={!prompt.trim() || loading}
              title="Send message"
            >
              <Send size={20} />
            </button>
          </div>
        </div>
        <div className="input-footer">
          <span className="input-hint">AI can make mistakes. Check important info.</span>
        </div>
      </form>
    </div>
  )
}

export default TextGenerator
