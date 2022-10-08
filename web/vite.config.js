import { defineConfig } from 'vite';

import path from 'path';
import react from '@vitejs/plugin-react';

export default defineConfig({
  resolve: {
    alias: {
      '@': path.resolve(__dirname, '/src'),
      '@images': path.resolve(__dirname, '/src/images'),
      '@constants': path.resolve(__dirname, '/src/constants'),
    },
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
      },
    },
  },
  plugins: [react()],
});
