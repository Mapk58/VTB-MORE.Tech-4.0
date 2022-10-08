import { createSlice } from '@reduxjs/toolkit';

const initialState = {
  rootState: {
    status: false,
    userID: null,
  },
};

const rootSlice = createSlice({
  name: 'root',
  initialState,
  reducers: {
    loginStatus({ rootState }) {
      rootState.status = true;
    },
    setUserID({ rootState }, action) {
      rootState.userID = action.payload;
    },
  },
});

export const {
  actions: { loginStatus, setUserID },
  reducer,
} = rootSlice;

export default reducer;
