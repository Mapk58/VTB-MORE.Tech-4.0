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
    host: '0.0.0.0',
    proxy: {
      '/api': {
        target: 'http://0.0.0.0:8000',
      },
    },
  },
  plugins: [react()],
});
