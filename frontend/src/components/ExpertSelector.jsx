import React from 'react'

const expertIcons = {
  auto: '🎯',
  story: '📚',
  poem: '✍️',
  email: '📧'
}

function ExpertSelector({ experts = [], selectedExpert, onSelectExpert }) {  // ✅ Added default value
  // ✅ Safety check: Return loading state if experts not loaded yet
  if (!experts || experts.length === 0) {
    return (
      <div className="expert-selector">
        <h3>Select Expert Mode:</h3>
        <div className="expert-buttons">
          <button
            className={`expert-btn ${selectedExpert === 'auto' ? 'active' : ''}`}
            onClick={() => onSelectExpert('auto')}
          >
            {expertIcons.auto} Auto (Router Decides)
          </button>
        </div>
        <p className="expert-hint">Loading experts...</p>
      </div>
    )
  }

  return (
    <div className="expert-selector">
      <h3>Select Expert Mode:</h3>
      <div className="expert-buttons">
        <button
          className={`expert-btn ${selectedExpert === 'auto' ? 'active' : ''}`}
          onClick={() => onSelectExpert('auto')}
        >
          {expertIcons.auto} Auto (Router Decides)
        </button>
        
        {experts.map((expert) => (
          <button
            key={expert.name}
            className={`expert-btn ${selectedExpert === expert.name ? 'active' : ''} ${!expert.available ? 'disabled' : ''}`}
            onClick={() => onSelectExpert(expert.name)}
            title={expert.description}
            disabled={!expert.available}
          >
            {expertIcons[expert.name] || '🤖'} {expert.name.charAt(0).toUpperCase() + expert.name.slice(1)}
            {!expert.available && ' (Coming Soon)'}
          </button>
        ))}
      </div>
      
      {selectedExpert === 'auto' && (
        <p className="expert-hint">
          The router will analyze your prompt and automatically select the best expert!
        </p>
      )}
      
      {selectedExpert !== 'auto' && (
        <p className="expert-hint">
          {experts.find(e => e.name === selectedExpert)?.description || 'Expert selected'}
        </p>
      )}
    </div>
  )
}

export default ExpertSelector