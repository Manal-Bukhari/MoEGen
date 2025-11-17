import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

/**
 * Generate text using the MoE system
 * Parameters (max_length, temperature) are now configured in backend .env file
 */
export const generateText = async (prompt, expert = null) => {
  try {
    const response = await api.post('/generate', {
      prompt,
      expert: expert === 'auto' ? null : expert,
    })
    return response.data
  } catch (error) {
    console.error('Error generating text:', error)
    throw error.response?.data?.detail || 'Failed to generate text'
  }
}

/**
 * Get list of available experts
 */
export const getExperts = async () => {
  try {
    const response = await api.get('/experts')
    return response.data
  } catch (error) {
    console.error('Error fetching experts:', error)
    throw error.response?.data?.detail || 'Failed to fetch experts'
  }
}

/**
 * Health check
 */
export const checkHealth = async () => {
  try {
    const response = await api.get('/health')
    return response.data
  } catch (error) {
    console.error('Health check failed:', error)
    throw error
  }
}

/**
 * Generate text with a specific expert
 */
export const generateWithExpert = async (expertName, prompt, maxLength = 150, temperature = 0.7) => {
  try {
    const response = await api.post(`/generate/${expertName}`, {
      prompt,
      max_length: maxLength,
      temperature,
    })
    return response.data
  } catch (error) {
    console.error(`Error generating with ${expertName}:`, error)
    throw error.response?.data?.detail || `Failed to generate with ${expertName}`
  }
}

export default api
