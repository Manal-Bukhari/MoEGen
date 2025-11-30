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

  const [showEnhancedQuery, setShowEnhancedQuery] = React.useState(false)

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

      {result.enhanced_query && (
        <div className="enhanced-query-section">
          <button 
            onClick={() => setShowEnhancedQuery(!showEnhancedQuery)}
            className="toggle-enhanced-btn"
            style={{ 
              marginBottom: '10px',
              padding: '8px 16px',
              backgroundColor: '#3498db',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '14px'
            }}
          >
            {showEnhancedQuery ? '🔽 Hide' : '🔍 Show'} Enhanced Query Details
          </button>
          
          {showEnhancedQuery && (
            <div className="enhanced-query-details" style={{
              backgroundColor: '#f8f9fa',
              padding: '15px',
              borderRadius: '8px',
              marginBottom: '20px',
              border: '1px solid #dee2e6'
            }}>
              <h4 style={{ marginTop: 0, color: '#2c3e50' }}>🔍 Enhanced Query (from Gemini):</h4>
              <div style={{ fontSize: '14px', lineHeight: '1.6' }}>
                {result.expert_used === 'email' && (
                  <>
                    <p><strong>Email Type:</strong> {result.enhanced_query.email_type || 'N/A'}</p>
                    <p><strong>Tone:</strong> {result.enhanced_query.tone || 'N/A'}</p>
                    <p><strong>Recipient Type:</strong> {result.enhanced_query.recipient_type || 'N/A'}</p>
                    <p><strong>Key Points:</strong> {result.enhanced_query.key_points?.join(', ') || 'None'}</p>
                    <p><strong>Special Requirements:</strong> {result.enhanced_query.special_requirements?.join(', ') || 'None'}</p>
                    <p><strong>Enhanced Instruction:</strong></p>
                    <p style={{ fontStyle: 'italic', color: '#555' }}>{result.enhanced_query.enhanced_instruction || 'N/A'}</p>
                  </>
                )}
                {result.expert_used === 'story' && (
                  <>
                    <p><strong>Genre:</strong> {result.enhanced_query.genre || 'N/A'}</p>
                    <p><strong>Tone:</strong> {result.enhanced_query.tone || 'N/A'}</p>
                    <p><strong>Key Elements:</strong> {result.enhanced_query.key_elements?.join(', ') || 'None'}</p>
                    <p><strong>Length Preference:</strong> {result.enhanced_query.length_preference || 'N/A'}</p>
                    <p><strong>Special Requirements:</strong> {result.enhanced_query.special_requirements?.join(', ') || 'None'}</p>
                    <p><strong>Enhanced Instruction:</strong></p>
                    <p style={{ fontStyle: 'italic', color: '#555' }}>{result.enhanced_query.enhanced_instruction || 'N/A'}</p>
                  </>
                )}
                {result.expert_used === 'poem' && (
                  <>
                    <p><strong>Poem Type:</strong> {result.enhanced_query.poem_type || 'N/A'}</p>
                    <p><strong>Tone:</strong> {result.enhanced_query.tone || 'N/A'}</p>
                    <p><strong>Theme:</strong> {result.enhanced_query.theme || 'N/A'}</p>
                    <p><strong>Rhyme Scheme:</strong> {result.enhanced_query.rhyme_scheme || 'N/A'}</p>
                    <p><strong>Special Requirements:</strong> {result.enhanced_query.special_requirements?.join(', ') || 'None'}</p>
                    <p><strong>Enhanced Instruction:</strong></p>
                    <p style={{ fontStyle: 'italic', color: '#555' }}>{result.enhanced_query.enhanced_instruction || 'N/A'}</p>
                  </>
                )}
                <details style={{ marginTop: '10px' }}>
                  <summary style={{ cursor: 'pointer', fontWeight: 'bold', color: '#7f8c8d' }}>View Raw JSON</summary>
                  <pre style={{ 
                    backgroundColor: '#2c3e50', 
                    color: '#ecf0f1', 
                    padding: '10px', 
                    borderRadius: '4px',
                    overflow: 'auto',
                    fontSize: '12px',
                    marginTop: '10px'
                  }}>
                    {JSON.stringify(result.enhanced_query, null, 2)}
                  </pre>
                </details>
              </div>
            </div>
          )}
        </div>
      )}

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
