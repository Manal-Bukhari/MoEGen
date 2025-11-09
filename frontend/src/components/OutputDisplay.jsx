import React from 'react'

const expertInfo = {
  story: {
    icon: '📚',
    color: '#4A90E2',
    name: 'Story Expert'
  },
  poem: {
    icon: '✍️',
    color: '#9B59B6',
    name: 'Poem Expert'
  },
  email: {
    icon: '📧',
    color: '#27AE60',
    name: 'Email Expert'
  }
}

function OutputDisplay({ result, onClear }) {
  const expert = expertInfo[result.expert_used] || { icon: '🤖', color: '#95A5A6', name: 'Unknown Expert' }
  
  const handleCopy = () => {
    navigator.clipboard.writeText(result.generated_text)
    alert('Text copied to clipboard!')
  }

  const confidencePercentage = (result.confidence * 100).toFixed(1)
  const getConfidenceColor = (confidence) => {
    if (confidence > 0.7) return '#27AE60'
    if (confidence > 0.4) return '#F39C12'
    return '#E74C3C'
  }

  return (
    <div className="output-display">
      <div className="output-header">
        <div className="expert-badge" style={{ backgroundColor: expert.color }}>
          <span className="expert-icon">{expert.icon}</span>
          <span className="expert-name">{expert.name}</span>
        </div>
        
        <div className="confidence-indicator">
          <span className="confidence-label">Confidence:</span>
          <div className="confidence-bar-container">
            <div 
              className="confidence-bar" 
              style={{ 
                width: `${confidencePercentage}%`,
                backgroundColor: getConfidenceColor(result.confidence)
              }}
            ></div>
          </div>
          <span 
            className="confidence-value"
            style={{ color: getConfidenceColor(result.confidence) }}
          >
            {confidencePercentage}%
          </span>
        </div>
      </div>

      <div className="original-prompt">
        <h4>Your Prompt:</h4>
        <p>{result.prompt}</p>
      </div>

      <div className="generated-content">
        <h4>Generated Content:</h4>
        <div className="generated-text">
          {result.generated_text}
        </div>
      </div>

      <div className="output-actions">
        <button onClick={handleCopy} className="action-btn copy-btn">
          📋 Copy Text
        </button>
        <button onClick={onClear} className="action-btn clear-btn">
          🗑️ Clear Output
        </button>
      </div>
    </div>
  )
}

export default OutputDisplay
