import { useSelector } from 'react-redux';

export const getRootState = () =>
  useSelector(({ rootReducer: { rootState } }) => rootState);
