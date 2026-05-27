import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

process.env.BROWSER = 'chrome'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    open: true,
  }
})
