"""
Telegram-бот для анализа депрессивных и суицидальных текстов.

Гибридная модель:
- sentence-transformers для семантических эмбеддингов
- CatBoost для классификации
- Лингвистические признаки из feature_extractor.py

Команды:
/start - начало работы
/help - справка
/stats - статистика анализов (анонимная)
/about - о модели
"""

import asyncio
import json
import logging
import os
import csv
import hashlib
from datetime import datetime

import numpy as np
from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message
from sentence_transformers import SentenceTransformer
from catboost import CatBoostClassifier

from feature_extractor import extract_features, FEATURE_NAMES
from lexicons import (
    HOPELESSNESS, SUICIDAL_IDEATION, DEPRESSION_EMOTIONAL,
    COGNITIVE_DISTORTIONS, SOCIAL_ISOLATION, PHYSICAL_SYMPTOMS,
)

# ==========================================
# 1. НАСТРОЙКИ
# ==========================================

# Замените на свой токен от @BotFather
BOT_TOKEN = "ВАШ_ТОКЕН_ЗДЕСЬ"

LOG_FILE = "bot_logs.csv"
STATS_FILE = "bot_stats.json"
MODEL_PATH = "cb_suicide_model.cbm"
CONFIG_PATH = "model_config.json"

logging.basicConfig(level=logging.INFO)

# ==========================================
# 2. ЗАГРУЗКА МОДЕЛИ И КОНФИГА
# ==========================================

print("=" * 60)
print("ЗАГРУЗКА МОДЕЛИ")
print("=" * 60)

print("[1/3] Загружаем NLP-энкодер (sentence-transformers)...")
encoder = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device="cpu"
)
print(f"       OK | Модель: paraphrase-multilingual-MiniLM-L12-v2")
print(f"       OK | Размерность эмбеддингов: {encoder.get_sentence_embedding_dimension()}")

print("\n[2/3] Загружаем ML-модель (CatBoost)...")
cb_model = CatBoostClassifier()
if os.path.exists(MODEL_PATH):
    cb_model.load_model(MODEL_PATH)
    print(f"       OK | Модель загружена: {MODEL_PATH}")
else:
    print(f"       ERR | Модель {MODEL_PATH} не найдена!")
    print(f"       Запустите: python train_demo.py")
    raise FileNotFoundError(f"Модель {MODEL_PATH} не найдена. Сначала обучите модель.")

print("\n[3/3] Загружаем конфигурацию...")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    THRESHOLD = config.get("threshold", 0.5)
    EMBEDDING_DIM = config.get("embedding_dim", 384)
    META_COUNT = config.get("meta_feature_count", len(FEATURE_NAMES))
    print(f"       OK | Порог: {THRESHOLD:.3f}")
    print(f"       OK | Эмбеддинги: {EMBEDDING_DIM}")
    print(f"       OK | Лингв. признаки: {META_COUNT}")
else:
    print(f"       WARN | Конфиг {CONFIG_PATH} не найден, используем defaults")
    THRESHOLD = 0.5
    EMBEDDING_DIM = 384
    META_COUNT = len(FEATURE_NAMES)

print("\n" + "=" * 60)
print("МОДЕЛЬ ГОТОВА К РАБОТЕ")
print("=" * 60)

# ==========================================
# 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================

def anonymize_id(user_id: int) -> str:
    """Хеширует ID пользователя для анонимности."""
    return hashlib.sha256(str(user_id).encode()).hexdigest()[:12]


def load_stats() -> dict:
    """Загружает статистику из файла."""
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"total_analyses": 0, "risk_detected": 0, "daily": {}}


def save_stats(stats: dict):
    """Сохраняет статистику."""
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def log_analysis(user_id: int, text: str, prob: float, label: str, details: dict):
    """Сохраняет результат анализа в CSV (анонимно)."""
    file_exists = os.path.isfile(LOG_FILE)
    anon_id = anonymize_id(user_id)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(LOG_FILE, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "datetime", "anon_user_id", "text_length", "probability",
                "prediction_label", "depression_index", "suicide_risk_index",
                "emotional_balance",
            ])
        writer.writerow([
            current_time, anon_id, len(text), round(prob, 4), label,
            round(details.get("depression_index", 0), 2),
            round(details.get("suicide_risk_index", 0), 2),
            round(details.get("emotional_balance", 0), 2),
        ])
    
    # Обновляем статистику
    stats = load_stats()
    stats["total_analyses"] += 1
    if label == "Тревога":
        stats["risk_detected"] += 1
    today = datetime.now().strftime("%Y-%m-%d")
    stats["daily"][today] = stats["daily"].get(today, 0) + 1
    save_stats(stats)


def analyze_text(text: str) -> tuple:
    """
    Полный анализ текста.
    
    Returns:
        (probability, features_dict, details_dict)
    """
    # 1. Эмбеддинги
    vec = encoder.encode([text], convert_to_numpy=True)
    
    # 2. Лингвистические признаки
    features = extract_features(text)
    meta_values = np.array([features[name] for name in FEATURE_NAMES]).reshape(1, -1)
    
    # 3. Объединение
    X = np.hstack((vec, meta_values))
    
    # 4. Предсказание
    prob = float(cb_model.predict_proba(X)[0][1])
    
    # 5. Детали по категориям
    details = {
        "depression_index": features["depression_index"],
        "suicide_risk_index": features["suicide_risk_index"],
        "emotional_balance": features["emotional_balance"],
        "negative_index": features["negative_emotion_index"],
        "positive_index": features["positive_emotion_index"],
        "risk_ratio": features["risk_protective_ratio"],
        "word_count": features["word_count"],
        "all_depressive_freq": features["all_depressive_freq"],
        "hopelessness_freq": features["hopelessness_freq"],
        "suicidal_freq": features["suicidal_freq"],
        "social_isolation_freq": features["social_isolation_freq"],
        "cognitive_distortions_freq": features["cognitive_distortions_freq"],
        "lexical_diversity": features["lexical_diversity"],
        "intensification_index": features["intensification_index"],
        "i_pronoun_freq": features["i_pronoun_freq"],
    }
    
    return prob, features, details


def format_result(prob: float, details: dict) -> str:
    """Форматирует результат анализа для Telegram."""
    
    if prob >= THRESHOLD:
        risk_level = "?? ВЫСОКИЙ РИСК"
        emoji = "??"
        color_emoji = "??"
    elif prob >= THRESHOLD * 0.6:
        risk_level = "?? ПОВЫШЕННЫЙ РИСК"
        emoji = "??"
        color_emoji = "??"
    else:
        risk_level = "?? НОРМА"
        emoji = "?"
        color_emoji = "??"
    
    # Шкала вероятности
    bar_length = 20
    filled = int(prob * bar_length)
    bar = "-" * filled + "-" * (bar_length - filled)
    
    message = f"""
{emoji} *Результат анализа*

{color_emoji} *Уровень риска:* {risk_level}
*Уверенность модели:* {prob:.1%}
`{bar}`

*?? Лингвистические маркеры:*
• Индекс депрессии: `{details["depression_index"]:.1f}`
• Суицидальный риск: `{details["suicide_risk_index"]:.1f}`
• Эмоциональный баланс: `{details["emotional_balance"]:+.1f}`
  (отрицательный = преобладает негатив)

*?? Текстовые метрики:*
• Длина текста: `{int(details["word_count"])} слов`
• Лексическое разнообразие: `{details["lexical_diversity"]:.2f}`
• Частота местоимений "я/меня": `{details["i_pronoun_freq"]:.1f}%`
• Частота депрессивных маркеров: `{details["all_depressive_freq"]:.1f}%`
"""
    
    # Детальные маркеры для высокого риска
    if prob >= THRESHOLD * 0.5:
        message += "\n*?? Детализация маркеров:*\n"
        if details["hopelessness_freq"] > 0:
            message += f"• Безысходность: `{details['hopelessness_freq']:.1f}%`\n"
        if details["suicidal_freq"] > 0:
            message += f"• Суицидальные маркеры: `{details['suicidal_freq']:.1f}%` ??\n"
        if details["social_isolation_freq"] > 0:
            message += f"• Социальная изоляция: `{details['social_isolation_freq']:.1f}%`\n"
        if details["cognitive_distortions_freq"] > 0:
            message += f"• Когнитивные искажения: `{details['cognitive_distortions_freq']:.1f}%`\n"
        if details["intensification_index"] > 0.5:
            message += f"• Эмоциональная интенсификация: `{details['intensification_index']:.1f}`\n"
    
    # Рекомендации
    if prob >= THRESHOLD:
        message += """
?? *Рекомендация:*
Текст содержит значительное количество маркеров депрессивного состояния. Если это ваш текст, рекомендуется обратиться к специалисту.

*Телефоны доверия:*
?? 8-800-2000-122 (бесплатно, круглосуточно)
?? 8-800-333-44-34 (Телефон доверия)
"""
    
    return message


# ==========================================
# 4. TELEGRAM БОТ
# ==========================================

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


@dp.message(Command("start"))
async def cmd_start(message: Message):
    welcome_text = """
?? *Привет! Я — NLP-ассистент для анализа эмоционального состояния.*

Я использую гибридную модель машинного обучения:
?? *Трансформер* (384-мерные эмбеддинги) -- понимаю смысл текста
?? *CatBoost* -- классифицирую риск
?? *30+ лингвистических признаков* -- анализирую паттерны речи

*Как пользоваться:*
Просто напишите мне о своих мыслях, переживаниях или чувствах.
Минимум ~10 слов для точного анализа контекста.

*Команды:*
/help -- подробная справка
/stats -- анонимная статистика
/about -- о модели

_Все данные анонимны. ID хешируется._
"""
    await message.answer(welcome_text, parse_mode="Markdown")


@dp.message(Command("help"))
async def cmd_help(message: Message):
    help_text = f"""
*?? Справка*

*Как работает анализ:*
1. Ваш текст преобразуется в 384-мерный эмбеддинг через нейросеть
2. Извлекаются 30+ лингвистических признаков:
   • Частота депрессивных/суицидальных маркеров
   • Эмоциональные индексы
   • Когнитивные искажения
   • Социальная изоляция
   • Безысходность, интенсификация
3. CatBoost-классификатор оценивает риск

*Интерпретация результатов:*
?? *Норма* (< {THRESHOLD * 0.6:.0%}) -- текст без тревожных маркеров
?? *Повышенный* ({THRESHOLD * 0.6:.0%}--{THRESHOLD:.0%}) -- есть отдельные маркеры
?? *Высокий* (? {THRESHOLD:.0%}) -- значительное количество маркеров

_Важно: бот не ставит диагноз. При высоком риске рекомендуется обратиться к психологу или психиатру._
"""
    await message.answer(help_text, parse_mode="Markdown")


@dp.message(Command("about"))
async def cmd_about(message: Message):
    about_text = f"""
*?? О модели*

Это гибридная модель для курсовой работы:
"Разработка гибридной модели машинного обучения для выявления признаков депрессивных и суицидальных состояний на основе анализа текстового цифрового следа"

*Архитектура:*
• Эмбеддинги: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
• Классификатор: `CatBoostClassifier`
• Лингв. признаки: `30+` (лексиконы, синтаксис, стиль)
• Порог: `{THRESHOLD:.3f}`

*Маркеры:*
• Безысходность ({len(HOPELESSNESS)} слов)
• Суицидальная идеация ({len(SUICIDAL_IDEATION)} слов)
• Эмоциональная депрессия ({len(DEPRESSION_EMOTIONAL)} слов)
• Когнитивные искажения ({len(COGNITIVE_DISTORTIONS)} слов)
• Социальная изоляция ({len(SOCIAL_ISOLATION)} слов)
• Физические симптомы ({len(PHYSICAL_SYMPTOMS)} слов)

_Бот создан в научных целях. Не заменяет профессиональную помощь._
"""
    await message.answer(about_text, parse_mode="Markdown")


@dp.message(Command("stats"))
async def cmd_stats(message: Message):
    stats = load_stats()
    total = stats["total_analyses"]
    risk = stats["risk_detected"]
    normal = total - risk
    
    today = datetime.now().strftime("%Y-%m-%d")
    today_count = stats["daily"].get(today, 0)
    
    stats_text = f"""
*?? Анонимная статистика*

Всего анализов: `{total}`
?? Норма: `{normal}`
?? Риск выявлен: `{risk}`

*Сегодня:* `{today_count}` анализов

_Данные анонимны. User ID хешируется SHA-256._
"""
    await message.answer(stats_text, parse_mode="Markdown")


@dp.message()
async def analyze_message(message: Message):
    text = message.text or message.caption or ""
    
    # Игнорируем команды
    if text.startswith("/"):
        return
    
    # Проверка длины
    words = text.split()
    if len(words) < 5:
        await message.answer(
            "?? Текст слишком короткий для точного анализа контекста. "
            "Напишите хотя бы 5-10 слов о своих мыслях или переживаниях."
        )
        return
    
    # Индикатор "печатает"
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    
    try:
        # Анализ
        prob, features, details = analyze_text(text)
        
        # Формируем ответ
        response = format_result(prob, details)
        
        # Логируем (анонимно)
        label = "Тревога" if prob >= THRESHOLD else "Норма"
        log_analysis(message.from_user.id, text, prob, label, details)
        
        await message.answer(response, parse_mode="Markdown")
        
    except Exception as e:
        logging.error(f"Ошибка анализа: {e}")
        await message.answer(
            "? Произошла ошибка при анализе. Попробуйте ещё раз или обратитесь к администратору."
        )


# ==========================================
# 5. ЗАПУСК
# ==========================================

async def main():
    print("\n?? Бот запущен и готов к работе!")
    print(f"   Порог: {THRESHOLD:.3f}")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
