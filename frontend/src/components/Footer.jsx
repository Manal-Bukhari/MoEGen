import React from 'react'
import { Code, Heart, Github } from 'lucide-react'

function Footer() {
  return (
    <footer className="footer">
      <div className="footer-content">
        <p className="footer-text">
          <Code size={16} className="footer-icon" />
          Built with React, FastAPI, and Gemini AI
        </p>
      </div>
    </footer>
  )
}

export default Footer
