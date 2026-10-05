"""Русскоязычный интерфейс; ML-логика находится в независимых модулях."""

import logging
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

from car_price.config import load_config
from car_price.predict import load_model, predict_prices

LOGGER = logging.getLogger(__name__)


@st.cache_resource
def cached_model(path: str, modified_ns: int) -> Pipeline:
    """Время изменения файла обновляет кеш после переобучения."""
    return load_model(Path(path))


def main() -> None:
    st.set_page_config(page_title="Оценка стоимости автомобиля", page_icon="🚗")
    st.title("🚗 Оценка стоимости автомобиля")
    st.write("Укажите характеристики подержанного автомобиля и получите прогноз цены.")
    try:
        config = load_config()
    except (OSError, ValueError, KeyError, TypeError):
        LOGGER.exception("Ошибка конфигурации")
        st.error("Не удалось прочитать конфигурацию проекта.")
        return
    st.info(
        "Учебная модель на данных индийского рынка. Цены указаны в лакхах "
        "индийских рупий: 1 лакх = 100 000 ₹. Возраст автомобиля рассчитывается "
        f"на {config.reference_year} год."
    )
    with st.form("car_price"):
        price = st.number_input(
            "Цена аналогичного нового автомобиля, лакхи ₹",
            min_value=0.01,
            max_value=100.0,
            value=5.0,
            step=0.5,
            help="Например, 5 лакхов — это 500 000 индийских рупий.",
        )
        owner = st.radio("Количество предыдущих владельцев", (0, 1, 2, 3))
        year = st.number_input(
            "Год покупки автомобиля",
            min_value=1900,
            max_value=config.reference_year,
            value=min(2015, config.reference_year),
            step=1,
        )
        fuel = st.selectbox("Тип топлива", ("Бензин", "Дизель", "Газ (CNG)"))
        seller = st.selectbox("Продавец", ("Дилер", "Частное лицо"))
        transmission = st.selectbox("Коробка передач", ("Механическая", "Автоматическая"))
        submitted = st.form_submit_button("Рассчитать стоимость")
    if submitted:
        row = pd.DataFrame(
            [
                {
                    "Present_Price": price,
                    "Owner": owner,
                    "Year": year,
                    "Fuel_Type": {"Бензин": "Petrol", "Дизель": "Diesel", "Газ (CNG)": "CNG"}[fuel],
                    "Seller_Type": "Individual" if seller == "Частное лицо" else "Dealer",
                    "Transmission": "Manual" if transmission == "Механическая" else "Automatic",
                }
            ]
        )
        try:
            model = cached_model(str(config.model_path), config.model_path.stat().st_mtime_ns)
            output = float(predict_prices(model, row)[0])
        except FileNotFoundError:
            st.error("Модель не найдена. Сначала выполните: poetry run car-train")
            return
        except (OSError, ValueError, TypeError, AttributeError, ImportError):
            LOGGER.exception("Ошибка расчёта стоимости")
            st.error("Не удалось получить прогноз. Выполните: poetry run car-train")
            return
        amount = f"{output:.2f}".replace(".", ",")
        rupees = f"{output * 100_000:,.0f}".replace(",", " ")
        st.success(f"Ориентировочная стоимость: {amount} лакхов ₹")
        st.caption(f"Примерно {rupees} индийских рупий. Это прогноз учебной модели.")
