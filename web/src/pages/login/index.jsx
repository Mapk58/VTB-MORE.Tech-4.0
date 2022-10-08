import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import Logo from '@images/logo.svg';

import styles from './styles.module.sass';
import axios from 'axios';
import { useDispatch } from 'react-redux';
import { getRootState } from '../../store/selectors';
import { loginStatus, setUserID } from '../../store/actions';

// НАСТРОЙКА ЗАПРОСА

const url = '/api/auth';

const LoginPage = () => {
  const [login, setLogin] = useState('');
  const [password, setPassword] = useState('');

  const [error, setError] = useState(false);

  const dispatch = useDispatch();
  const { status, userID } = getRootState();

  const navigate = useNavigate();

  const buttonHandler = () => {
    axios
      .post(url, { login: login, password: password })
      .then((response) => {
        const {
          data: { status, userID },
        } = response;

        if (status) {
          dispatch(loginStatus());
          dispatch(setUserID(userID));
          navigate('/main');
        } else {
          setError(true);
        }
      })
      .catch(() => {
        setError(true);
      });
  };

  return (
    <div className={styles.page}>
      <div className={styles.container}>
        <img className={styles.image} src={Logo} />
        <input
          className={`${styles.input} ${error && styles.error}`}
          placeholder="Логин / Email"
          type="text"
          value={login}
          onChange={(e) => setLogin(e.target.value)}
        />
        <input
          className={`${styles.input} ${error && styles.error}`}
          placeholder="Пароль"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        <Link className={styles.recovery}>Восстановить</Link>
        <button className={styles.button} onClick={() => buttonHandler()}>
          Войти
        </button>
        <Link className={styles.problems}>Не удается войти?</Link>
      </div>
    </div>
  );
};

export default LoginPage;
