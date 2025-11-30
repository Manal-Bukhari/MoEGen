import React from 'react'
import { Bot, Menu, X } from 'lucide-react'

function Header({ sidebarOpen, setSidebarOpen }) {
  return (
    <header className="header">
      <div className="header-content">
        <button 
          className="sidebar-toggle"
          onClick={() => setSidebarOpen(!sidebarOpen)}
          aria-label="Toggle sidebar"
        >
          {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
        </button>
        
        <div className="header-title">
          <Bot size={24} className="header-icon" />
          <div>
            <h1 className="logo">MoE Generator</h1>
            <p className="tagline">Mixture-of-Experts AI System</p>
          </div>
        </div>
      </div>
    </header>
  )
}

export default Header
