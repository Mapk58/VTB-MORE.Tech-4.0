import React from 'react';
import { createRoot } from 'react-dom/client';
import { Provider } from 'react-redux';
import { BrowserRouter, Routes, Route } from 'react-router-dom';

import { store } from './store/config';

import { LoginPage } from './pages';
import { MainPage } from './pages';

import './index.sass';
import "react-responsive-carousel/lib/styles/carousel.min.css"; // requires a loader

const container = document.getElementById('root');
const root = createRoot(container);
root.render(
  <Provider store={store}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<LoginPage />} />
          <Route path="/main/" element={<MainPage />} />
        </Routes>
      </BrowserRouter>
  </Provider>
);
