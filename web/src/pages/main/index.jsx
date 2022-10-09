import React, { useEffect, useState } from 'react';
import Carousel from 'react-multi-carousel';
import styles from './styles.module.sass';
import 'react-multi-carousel/lib/styles.css';
import axios from 'axios';

const rooturl = './api/sendData';

const responsive = {
  desktop: {
    breakpoint: { max: 3000, min: 1024 },
    items: 4,
    slidesToSlide: 1, // optional, default to 1.
  },
};

const mockData = [
  {
    title: 'Тренды: Сегодня',
    items: [
      {
        subtitle:
          'Keanu Reeves Departs Martin Scorsese, Leonardo DiCaprio’s ‘Devil in the White City’ at Hulu',
        author: '',
      },
      {
        subtitle:
          'Keanu Reeves Departs Martin Scorsese, Leonardo DiCaprio’s ‘Devil in the White City’ at Hulu',
        author: '',
      },
      {
        subtitle:
          'Keanu Reeves Departs Martin Scorsese, Leonardo DiCaprio’s ‘Devil in the White City’ at Hulu',
        author: '',
      },
    ],
  },
  {
    title: 'Тренды: Завтра',
    items: [
      {
        subtitle:
          'Keanu Reeves Departs Martin Scorsese, Leonardo DiCaprio’s ‘Devil in the White City’ at Hulu',
        author: '',
      },
      {
        subtitle:
          'Keanu Reeves Departs Martin Scorsese, Leonardo DiCaprio’s ‘Devil in the White City’ at Hulu',
        author: '',
      },
      {
        subtitle:
          'Keanu Reeves Departs Martin Scorsese, Leonardo DiCaprio’s ‘Devil in the White City’ at Hulu',
        author: '',
      },
    ],
  },
  {
    title: 'Тренды: Кто я...',
    items: [
      {
        subtitle:
          'Keanu Reeves Departs Martin Scorsese, Leonardo DiCaprio’s ‘Devil in the White City’ at Hulu',
        author: '',
      },
      {
        subtitle:
          'Keanu Reeves Departs Martin Scorsese, Leonardo DiCaprio’s ‘Devil in the White City’ at Hulu',
        author: '',
      },
      {
        subtitle:
          'Keanu Reeves Departs Martin Scorsese, Leonardo DiCaprio’s ‘Devil in the White City’ at Hulu',
        author: '',
      },
    ],
  },
  {
    title: 'Тренды: Кто я...',
    items: [
      {
        subtitle:
          'Keanu Reeves Departs Martin Scorsese, Leonardo DiCaprio’s ‘Devil in the White City’ at Hulu',
        author: '',
      },
      {
        subtitle:
          'Keanu Reeves Departs Martin Scorsese, Leonardo DiCaprio’s ‘Devil in the White City’ at Hulu',
        author: '',
      },
      {
        subtitle:
          'Keanu Reeves Departs Martin Scorsese, Leonardo DiCaprio’s ‘Devil in the White City’ at Hulu',
        author: '',
      },
    ],
  },
  ,
  {
    title: 'Тренды: Кто я...',
    items: [
      {
        subtitle:
          'Keanu Reeves Departs Martin Scorsese, Leonardo DiCaprio’s ‘Devil in the White City’ at Hulu',
        author: 'Hollywood Reporter',
      },
      {
        subtitle:
          'Keanu Reeves Departs Martin Scorsese, Leonardo DiCaprio’s ‘Devil in the White City’ at Hulu',
        author: 'Hollywood Reporter',
      },
      {
        subtitle:
          'Keanu Reeves Departs Martin Scorsese, Leonardo DiCaprio’s ‘Devil in the White City’ at Hulu',
        author: 'Hollywood Reporter',
      },
    ],
  },
  {
    title: 'Тренды: Кто я...',
    items: [
      {
        subtitle:
          'Keanu Reeves Departs Martin Scorsese, Leonardo DiCaprio’s ‘Devil in the White City’ at Hulu',
        author: 'Hollywood Reporter',
      },
      {
        subtitle:
          'Keanu Reeves Departs Martin Scorsese, Leonardo DiCaprio’s ‘Devil in the White City’ at Hulu',
        author: 'Hollywood Reporter',
      },
      {
        subtitle:
          'Keanu Reeves Departs Martin Scorsese, Leonardo DiCaprio’s ‘Devil in the White City’ at Hulu',
        author: 'Hollywood Reporter',
      },
    ],
  },
];

const ItemComponent = (props) => {
  const { data } = props;

  return (
    <Carousel responsive={responsive} slidesToSlide={1} itemClass={styles.item}>
      {data &&
        data.map((el, i1) => {
          const { title, items } = el;

          return (
            <div key={`item-${i1}`} className={styles.item__content}>
              <span className={styles.item__title}>{title}</span>
              <div className={styles.items__container}>
                {items &&
                  items.map((el, i2) => {
                    const { subtitle, author } = el;

                    return (
                      <div
                        className={styles.item__subitem}
                        key={`item-${i1}-subitem-${i2}`}
                      >
                        <span className={styles.subitem__subtitle}>
                          {subtitle}
                        </span>
                        <span className={styles.subitem__author}>{author}</span>
                      </div>
                    );
                  })}
              </div>
            </div>
          );
        })}
    </Carousel>
  );
};

const CheckboxRender = (props) => {
  const { checkboxParams, setCheckboxParams } = props;

  for (const prop in checkboxParams) {
    const key = prop;
    const value = checkboxParams[key];

    return (
      <CheckboxComponent key={key} value={value} setState={setCheckboxParams} />
    );
  }
};

const CheckboxComponent = (props) => {
  const {
    itemkey,
    value: { status, role },
    setState,
  } = props;

  return (
    <label className={styles.checkbox__container}>
      <input
        className={`${styles.checkbox} ${status && styles.checkbox__checked}`}
        type="checkbox"
        onChange={() => {
          console.log([itemkey]);
          setState((params) => ({
            ...params,
            [itemkey]: { status: !status, role: role },
          }));
        }}
        checked={status}
      />
      <span className={styles.checkbox__text}>{role}</span>
    </label>
  );
};

const MainPage = (props) => {
  const [checkboxStatus, setCheckboxStatus] = useState(false);
  const [checkboxParams, setCheckboxParams] = useState({
    0: { status: false, role: 'Генеральный директор' },
    1: { status: false, role: 'Бухгалтер' },
    2: { status: false, role: 'Общее' },
  });

  const [url, setUrl] = useState('');

  const name = 'Иван';

  const clickHandler = () => {
    let roles = [];

    Object.entries(checkboxParams).forEach(
      ([key, { status, role }]) => status && roles.push(role)
    );

    const data = {
      link: url,
      roles: roles,
    };

    axios.post(rooturl, data).then((r) => {
      console.log(r.data);
    });

    console.log(data);
  };

  return (
    <div className={styles.container}>
      <div className={styles.background}>
        <span className={styles.title}>{`Здравствуйте, ${name}!`}</span>
        <div className={styles.content}>
          <ItemComponent data={mockData} />
        </div>
        <span className={styles.subtitle}>
          Добавить новый источник новостей
        </span>
        <div className={styles.bottom__container}>
          <input
            className={styles.input}
            type="text"
            placeholder="Вставьте ссылку"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
          />
          <div>
            <button
              className={styles.categories__button}
              onClick={() => setCheckboxStatus(!checkboxStatus)}
            >
              Выберите основные категории
            </button>
            {checkboxStatus && (
              <div className={styles.input_group__container}>
                {Object.keys(checkboxParams).map((key) => {
                  const value = checkboxParams[key];

                  return (
                    <CheckboxComponent
                      itemkey={key}
                      value={value}
                      setState={setCheckboxParams}
                    />
                  );
                })}
                <div className={styles.button_container}>
                  <button className={styles.clear__button}>Очистить</button>
                  <button className={styles.accept__button}>Применить</button>
                </div>
              </div>
            )}
          </div>
          <button
            className={styles.push__button}
            onClick={() => clickHandler()}
          >
            Добавить
          </button>
        </div>
      </div>
    </div>
  );
};

export default MainPage;
