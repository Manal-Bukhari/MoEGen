import React from 'react'
import { BookOpen, PenTool, Mail, Bot, User, Copy, Trash2, ChevronDown, ChevronUp, Code, AlertCircle } from 'lucide-react'

const expertInfo = {
  story: {
    Icon: BookOpen,
    color: '#4A90E2',
    name: 'Story Expert'
  },
  poem: {
    Icon: PenTool,
    color: '#9B59B6',
    name: 'Poem Expert'
  },
  email: {
    Icon: Mail,
    color: '#27AE60',
    name: 'Email Expert'
  }
}

function OutputDisplay({ message, onClear }) {
  const [showEnhancedQuery, setShowEnhancedQuery] = React.useState(false)
  const [copied, setCopied] = React.useState(false)

  if (message.type === 'user') {
    return (
      <div className="message message-user">
        <div className="message-avatar user-avatar">
          <User size={20} />
        </div>
        <div className="message-content">
          <div className="message-text">{message.content}</div>
        </div>
      </div>
    )
  }

  if (message.type === 'error') {
    return (
      <div className="message message-error">
        <div className="message-avatar error-avatar">
          <AlertCircle size={20} />
        </div>
        <div className="message-content">
          <div className="message-text error-text">{message.content}</div>
        </div>
      </div>
    )
  }

  if (message.type === 'ai' && message.result) {
    const result = message.result
    const expert = expertInfo[result.expert_used] || { 
      Icon: Bot, 
      color: '#95A5A6', 
      name: 'Unknown Expert' 
    }
    const { Icon: ExpertIcon } = expert

    const handleCopy = async () => {
      try {
        await navigator.clipboard.writeText(result.generated_text)
        setCopied(true)
        setTimeout(() => setCopied(false), 2000)
      } catch (err) {
        console.error('Failed to copy:', err)
      }
    }

    const confidencePercentage = (result.confidence * 100).toFixed(0)

    return (
      <div className="message message-ai">
        <div className="message-avatar ai-avatar" style={{ backgroundColor: `${expert.color}20` }}>
          <ExpertIcon size={20} style={{ color: expert.color }} />
        </div>
        <div className="message-content">
          <div className="message-header">
            <div className="expert-badge" style={{ color: expert.color }}>
              <ExpertIcon size={16} />
              <span>{expert.name}</span>
            </div>
            <div className="confidence-badge">
              <span>{confidencePercentage}%</span>
            </div>
          </div>

          <div className="message-text ai-text">
            {result.generated_text}
          </div>

          {result.enhanced_query && (
            <div className="enhanced-query-section">
              <button 
                onClick={() => setShowEnhancedQuery(!showEnhancedQuery)}
                className="enhanced-query-toggle"
              >
                {showEnhancedQuery ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                <span>Enhanced Query Details</span>
              </button>
              
              {showEnhancedQuery && (
                <div className="enhanced-query-details">
                  <div className="enhanced-query-content">
                    {result.expert_used === 'email' && (
                      <div className="query-detail-grid">
                        <div className="query-detail-item">
                          <strong>Email Type:</strong> {result.enhanced_query.email_type || 'N/A'}
                        </div>
                        <div className="query-detail-item">
                          <strong>Tone:</strong> {result.enhanced_query.tone || 'N/A'}
                        </div>
                        <div className="query-detail-item">
                          <strong>Recipient:</strong> {result.enhanced_query.recipient_type || 'N/A'}
                        </div>
                        {result.enhanced_query.key_points?.length > 0 && (
                          <div className="query-detail-item full-width">
                            <strong>Key Points:</strong> {result.enhanced_query.key_points.join(', ')}
                          </div>
                        )}
                        {result.enhanced_query.enhanced_instruction && (
                          <div className="query-detail-item full-width">
                            <strong>Enhanced Instruction:</strong>
                            <p className="enhanced-instruction">{result.enhanced_query.enhanced_instruction}</p>
                          </div>
                        )}
                      </div>
                    )}
                    {result.expert_used === 'story' && (
                      <div className="query-detail-grid">
                        <div className="query-detail-item">
                          <strong>Genre:</strong> {result.enhanced_query.genre || 'N/A'}
                        </div>
                        <div className="query-detail-item">
                          <strong>Tone:</strong> {result.enhanced_query.tone || 'N/A'}
                        </div>
                        <div className="query-detail-item">
                          <strong>Length:</strong> {result.enhanced_query.length_preference || 'N/A'}
                        </div>
                        {result.enhanced_query.key_elements?.length > 0 && (
                          <div className="query-detail-item full-width">
                            <strong>Key Elements:</strong> {result.enhanced_query.key_elements.join(', ')}
                          </div>
                        )}
                        {result.enhanced_query.enhanced_instruction && (
                          <div className="query-detail-item full-width">
                            <strong>Enhanced Instruction:</strong>
                            <p className="enhanced-instruction">{result.enhanced_query.enhanced_instruction}</p>
                          </div>
                        )}
                      </div>
                    )}
                    {result.expert_used === 'poem' && (
                      <div className="query-detail-grid">
                        <div className="query-detail-item">
                          <strong>Poem Type:</strong> {result.enhanced_query.poem_type || 'N/A'}
                        </div>
                        <div className="query-detail-item">
                          <strong>Tone:</strong> {result.enhanced_query.tone || 'N/A'}
                        </div>
                        <div className="query-detail-item">
                          <strong>Theme:</strong> {result.enhanced_query.theme || 'N/A'}
                        </div>
                        {result.enhanced_query.enhanced_instruction && (
                          <div className="query-detail-item full-width">
                            <strong>Enhanced Instruction:</strong>
                            <p className="enhanced-instruction">{result.enhanced_query.enhanced_instruction}</p>
                          </div>
                        )}
                      </div>
                    )}
                    <details className="json-viewer">
                      <summary>
                        <Code size={14} />
                        View Raw JSON
                      </summary>
                      <pre>{JSON.stringify(result.enhanced_query, null, 2)}</pre>
                    </details>
                  </div>
                </div>
              )}
            </div>
          )}

          <div className="message-actions">
            <button 
              onClick={handleCopy}
              className="action-button copy-button"
              title="Copy text"
            >
              <Copy size={16} />
              {copied ? 'Copied!' : 'Copy'}
            </button>
          </div>
        </div>
      </div>
    )
  }

  return null
}

export default OutputDisplay
