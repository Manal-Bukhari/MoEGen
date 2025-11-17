import React, { useState } from 'react'
import { generateText } from '../services/api'

function TextGenerator({ selectedExpert, onGenerate, onError, setLoading }) {
  const [prompt, setPrompt] = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    if (!prompt.trim()) {
      onError('Please enter a prompt')
      return
    }

    setLoading(true)
    try {
      const result = await generateText(
        prompt,
        selectedExpert
      )
      onGenerate(result)
    } catch (err) {
      onError(err.toString())
    } finally {
      setLoading(false)
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

  return (
    <div className="text-generator">
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="prompt">Enter Your Prompt:</label>
          <textarea
            id="prompt"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Type your prompt here... (e.g., 'Write a story about a brave knight' or 'Compose a poem about nature')"
            rows="5"
            required
          />
        </div>

        <div className="example-prompts">
          <p className="example-label">Try these examples:</p>
          {getCurrentExamples().map((example, index) => (
            <button
              key={index}
              type="button"
              className="example-btn"
              onClick={() => setPrompt(example)}
            >
              {example}
            </button>
          ))}
        </div>

        <button type="submit" className="generate-btn">
          ✨ Generate Text
        </button>
      </form>
    </div>
  )
}

export default TextGenerator
