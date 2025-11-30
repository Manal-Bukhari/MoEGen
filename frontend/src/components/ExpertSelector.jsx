import React from 'react'
import { Sparkles, BookOpen, PenTool, Mail, X } from 'lucide-react'

const expertIcons = {
  auto: Sparkles,
  story: BookOpen,
  poem: PenTool,
  email: Mail
}

const expertColors = {
  auto: '#6366f1',
  story: '#4A90E2',
  poem: '#9B59B6',
  email: '#27AE60'
}

function ExpertSelector({ experts = [], selectedExpert, onSelectExpert, sidebarOpen, setSidebarOpen }) {
  const expertOptions = [
    { name: 'auto', label: 'Auto Select', description: 'Let AI choose the best expert' },
    ...experts.filter(e => e.available).map(e => ({
      name: e.name,
      label: e.name.charAt(0).toUpperCase() + e.name.slice(1),
      description: e.description || `${e.name} expert`
    }))
  ]

  const Icon = expertIcons[selectedExpert] || Sparkles
  const color = expertColors[selectedExpert] || '#6366f1'

  return (
    <>
      {sidebarOpen && <div className="sidebar-overlay" onClick={() => setSidebarOpen(false)} />}
      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <h2>Expert Mode</h2>
          <button 
            className="sidebar-close"
            onClick={() => setSidebarOpen(false)}
            aria-label="Close sidebar"
          >
            <X size={20} />
          </button>
        </div>

        <div className="sidebar-content">
          <div className="expert-list">
            {expertOptions.map((expert) => {
              const ExpertIcon = expertIcons[expert.name] || Sparkles
              const isActive = selectedExpert === expert.name
              
              return (
                <button
                  key={expert.name}
                  className={`expert-option ${isActive ? 'active' : ''}`}
                  onClick={() => onSelectExpert(expert.name)}
                  style={isActive ? { 
                    borderLeftColor: expertColors[expert.name] || '#6366f1',
                    backgroundColor: `${expertColors[expert.name] || '#6366f1'}15`
                  } : {}}
                >
                  <ExpertIcon 
                    size={20} 
                    style={{ color: isActive ? (expertColors[expert.name] || '#6366f1') : 'inherit' }}
                  />
                  <div className="expert-option-content">
                    <span className="expert-option-label">{expert.label}</span>
                    <span className="expert-option-desc">{expert.description}</span>
                  </div>
                  {isActive && (
                    <div 
                      className="expert-option-indicator"
                      style={{ backgroundColor: expertColors[expert.name] || '#6366f1' }}
                    />
                  )}
                </button>
              )
            })}
          </div>

          <div className="sidebar-footer">
            <div className="selected-expert-info">
              <Icon size={18} style={{ color }} />
              <span>Selected: <strong>{expertOptions.find(e => e.name === selectedExpert)?.label}</strong></span>
            </div>
          </div>
        </div>
      </aside>
    </>
  )
}

export default ExpertSelector
