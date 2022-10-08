import { configureStore } from '@reduxjs/toolkit';
import rootState from './reducers/root';

const store = configureStore({
  reducer: {
    rootReducer: rootState,
  },
});

export { store };
