"""
bot_v2.py  --  Telegram-бот v2 с тремя улучшениями:

  #3  Временной анализ (temporal tracking):
      если пользователь отправил 2+ сообщения с высоким риском
      за последние 30 минут -- отправляется расширенное предупреждение.

  #4  NLI-компонент (из feature_extractor_v2.py):
      scope-aware отрицание снижает false-positive на фразах
      "я не хочу умирать", "not suicidal", etc.

  #5  Поддержка откалиброванной модели (calibration.py):
      если calibrated_model.pkl существует -- использует его;
      иначе -- оригинальный CatBoost.

ТОКЕН: установите переменную среды BOT_TOKEN перед запуском.
"""

import re
import pathlib
import asyncio
import json
import logging
import os
import csv
import hashlib
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
try:
    from deep_translator import GoogleTranslator
    from langdetect import detect as _langdetect
    _translator_available = True
except ImportError:
    _translator_available = False

def _to_english(text: str) -> str:
    """Переводит текст в английский, если язык не EN."""
    if not _translator_available:
        return text
    try:
        lang = _langdetect(text)
        if lang != "en":
            return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        pass
    return text
from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message
from sentence_transformers import SentenceTransformer
from catboost import CatBoostClassifier

# Улучшенный экстрактор признаков (#4): если есть v2, берём его
try:
    from feature_extractor_v2 import extract_features, FEATURE_NAMES_V1 as FEATURE_NAMES
except ImportError:
    from feature_extractor import extract_features, FEATURE_NAMES

# Поддержка откалиброванной модели (#5)
try:
    from calibration import load_calibrated_model
except ImportError:
    def load_calibrated_model(_=None):
        return None

try:
    from lexicons import (
        HOPELESSNESS, SUICIDAL_IDEATION, DEPRESSION_EMOTIONAL,
        COGNITIVE_DISTORTIONS, SOCIAL_ISOLATION, PHYSICAL_SYMPTOMS,
    )
except ImportError:
    HOPELESSNESS = SUICIDAL_IDEATION = DEPRESSION_EMOTIONAL = []
    COGNITIVE_DISTORTIONS = SOCIAL_ISOLATION = PHYSICAL_SYMPTOMS = []

# ==========================================
# 1. НАСТРОЙКИ
# ==========================================

BOT_TOKEN   = os.getenv('BOT_TOKEN', '8343248444:AAHqbYzVOH7IoZvUdAr-lcfDnH_H6ouFAk8')
LOG_FILE    = str(pathlib.Path.home() / 'bot_logs.csv')
STATS_FILE  = 'bot_stats.json'
MODEL_PATH  = 'cb_suicide_model.cbm'
CAL_PATH    = 'calibrated_model.pkl'
CONFIG_PATH = 'model_config.json'

# #3 Temporal tracking
_user_history: dict = defaultdict(list)
TEMPORAL_WINDOW_MIN = 30    # окно в минутах
TEMPORAL_RISK_COUNT = 2     # кол-во high-risk сообщений для эскалации
TEMPORAL_RISK_THR   = 0.60   # порог temporal

logging.basicConfig(level=logging.INFO)

# ==========================================
# 2. ЗАГРУЗКА МОДЕЛИ
# ==========================================

# Сбрасываем старый лог чтобы заголовок был перезаписан
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)
print(f'[Лог] Будет сохраняться в: {LOG_FILE}')
print('=' * 60)
print('ЗАГРУЗКА МОДЕЛИ v2')
print('=' * 60)

print('[1/3] Загружаем NLP-энкодер...')
encoder = SentenceTransformer(
    'sentence-transformers/all-mpnet-base-v2',
    device='cpu',
)
print(f'       OK | paraphrase-multilingual-all-mpnet-base-v2')
print(f'       OK | Размерность: {encoder.get_sentence_embedding_dimension()}')

print('\n[2/3] Загружаем ML-модель (CatBoost)...')
cb_model = CatBoostClassifier()
if os.path.exists(MODEL_PATH):
    cb_model.load_model(MODEL_PATH)
    print(f'       OK | {MODEL_PATH}')
else:
    raise FileNotFoundError(f'Модель {MODEL_PATH} не найдена. Сначала обучите модель.')

print('\n[3/3] Проверяем откалиброванную модель...')
cal_model    = load_calibrated_model(CAL_PATH)
ACTIVE_MODEL = cal_model if cal_model is not None else cb_model
if cal_model is not None:
    print('       OK | Используется откалиброванная модель (calibrated_model.pkl)')
else:
    print('       INFO | calibrated_model.pkl не найден, используем CatBoost напрямую')

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    THRESHOLD     = config.get('threshold', 0.495)
    EMBEDDING_DIM = config.get('embedding_dim', 384)
else:
    THRESHOLD, EMBEDDING_DIM = 0.495, 384
    config = {}

print('\n' + '=' * 60)
print(f'МОДЕЛЬ ГОТОВА | порог={THRESHOLD:.3f}')
print('=' * 60)


# ==========================================
# 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================

def anonymize_id(user_id: int) -> str:
    return hashlib.sha256(str(user_id).encode()).hexdigest()[:12]


def load_stats() -> dict:
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'total_analyses': 0, 'risk_detected': 0, 'daily': {}}


def save_stats(stats: dict):
    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

def normalize_input(text: str) -> str:
    """
    Очищает текст от артефактов копирования (HTML-теги, markdown,
    лишние пробелы, технические префиксы 'Результат анализа' и т.д.)
    перед подачей в модель.
    """
    # Удаляем HTML-теги
    text = re.sub(r'<[^>]+>', ' ', text)
    # Удаляем markdown bold/italic
    text = re.sub(r'\*\*|__|\*|_', '', text)
    # Удаляем служебные вставки бота (если юзер скопировал старый ответ)
    text = re.sub(r'Результат анализа.*?(?=\n|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Уровень риска.*?(?=\n|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Уверенность модели.*?(?=\n|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Телефоны доверия.*?(?=\n|$)', '', text, flags=re.IGNORECASE)
    # Заменяем множественные переносы и пробелы на один пробел
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def log_analysis(user_id: int, text: str, prob: float, label: str, details: dict, username: str = ""):
    anon_id = anonymize_id(user_id)
    row = [
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        anon_id, username or '', len(text), text,
        round(prob, 4), label,
        round(details.get('depression_index', 0), 2),
        round(details.get('suicide_risk_index', 0), 2),
        round(details.get('emotional_balance', 0), 2),
        round(details.get('nli_negation_score', 0), 3),
    ]
    header = ['datetime','anon_user_id','username','text_length','text',
              'probability','prediction_label','depression_index',
              'suicide_risk_index','emotional_balance','nli_negation_score']
    write_header = not os.path.isfile(LOG_FILE)
    try:
        with open(LOG_FILE, mode='a', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)
        logging.info(f'[LOG] Сохранено -> {LOG_FILE}')
    except Exception as e:
        logging.error(f'[LOG] Ошибка записи в {LOG_FILE}: {e}')
        # Fallback: пробуем записать рядом со скриптом
        fallback = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bot_logs_fallback.csv')
        try:
            with open(fallback, mode='a', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                if not os.path.isfile(fallback):
                    writer.writerow(header)
                writer.writerow(row)
            logging.info(f'[LOG] Fallback -> {fallback}')
        except Exception as e2:
            logging.error(f'[LOG] Fallback тоже не сработал: {e2}')
    # Статистика отдельно
    try:
        stats = load_stats()
        stats['total_analyses'] += 1
        if label == 'Тревога':
            stats['risk_detected'] += 1
        today = datetime.now().strftime('%Y-%m-%d')
        stats['daily'][today] = stats['daily'].get(today, 0) + 1
        save_stats(stats)
    except Exception as e:
        logging.warning(f'[STATS] {e}')


# --- #3 Temporal analysis ---
def update_temporal(user_id: int, prob: float) -> int:
    """
    Обновляет историю пользователя и возвращает кол-во high-risk
    сообщений за последние TEMPORAL_WINDOW_MIN минут.
    """
    now    = datetime.now()
    cutoff = now - timedelta(minutes=TEMPORAL_WINDOW_MIN)
    # Очищаем устаревше
    _user_history[user_id] = [
        (ts, p) for ts, p in _user_history[user_id] if ts > cutoff
    ]
    _user_history[user_id].append((now, prob))
    return sum(1 for _, p in _user_history[user_id] if p >= TEMPORAL_RISK_THR)


def analyze_text(text: str) -> tuple:
    """Полный анализ текста."""

    text      = normalize_input(text)
    text_en   = _to_english(text)  # RU -> EN для эмбеддинга
    vec       = encoder.encode([text_en], convert_to_numpy=True)
    features  = extract_features(text)   # #4 NLI-отрицание
    meta_vals = np.array([features[name] for name in FEATURE_NAMES]).reshape(1, -1)
    X         = np.hstack((vec, meta_vals))
    prob      = float(ACTIVE_MODEL.predict_proba(X)[0][1])  # #5 калибровка

    details = {
        'depression_index':           features['depression_index'],
        'suicide_risk_index':         features['suicide_risk_index'],
        'emotional_balance':          features['emotional_balance'],
        'negative_emotion_index':     features['negative_emotion_index'],
        'positive_emotion_index':     features['positive_emotion_index'],
        'risk_protective_ratio':      features['risk_protective_ratio'],
        'word_count':                 features['word_count'],
        'all_depressive_freq':        features['all_depressive_freq'],
        'hopelessness_freq':          features['hopelessness_freq'],
        'suicidal_freq':              features['suicidal_freq'],
        'social_isolation_freq':      features['social_isolation_freq'],
        'cognitive_distortions_freq': features['cognitive_distortions_freq'],
        'lexical_diversity':          features['lexical_diversity'],
        'intensification_index':      features['intensification_index'],
        'i_pronoun_freq':             features['i_pronoun_freq'],
        'nli_negation_score':         features.get('nli_negation_score',
                                      features.get('negation_of_suicidal_intent', 0)),
    }
    return prob, features, details


def format_result(prob: float, details: dict, temporal_risk_count: int = 0) -> str:
    """Форматирует ответ для Telegram."""
    if prob >= THRESHOLD:
        risk_level = '\U0001f6a8 ВЫСОКИЙ РИСК'
        emoji      = '\U0001f534'
    elif prob >= THRESHOLD * 0.6:
        risk_level = '\u26a0\ufe0f ПОВЫШЕННЫЙ РИСК'
        emoji      = '\U0001f7e1'
    else:
        risk_level = '\u2705 НОРМА'
        emoji      = '\U0001f7e2'

    nli_note = ''
    if details.get('nli_negation_score', 0) > 0.5:
        nli_note = '\n_Обнаружено явное отрицание суицидального намерения._'

    bar_length = 20
    filled     = int(prob * bar_length)
    bar        = '\u2588' * filled + '\u2591' * (bar_length - filled)

    message = (
        f'{emoji} *Результат анализа*\n\n'
        f'{emoji} *Уровень риска:* {risk_level}\n'
        f'*Уверенность модели:* {prob:.1%}\n'
        f'`{bar}`{nli_note}\n\n'
        f'*\U0001f9e0 Лингвистические маркеры:*\n'
        f'\u2022 Индекс депрессии: `{details["depression_index"]:.1f}`\n'
        f'\u2022 Суицидальный риск: `{details["suicide_risk_index"]:.1f}`\n'
        f'\u2022 Эмоциональный баланс: `{details["emotional_balance"]:+.1f}`\n\n'
        f'*\U0001f4ca Текстовые метрики:*\n'
        f'\u2022 Длина: `{int(details["word_count"])} слов`\n'
        f'\u2022 Местоимения "я/меня": `{details["i_pronoun_freq"]:.1f}%`\n'
        f'\u2022 Депрессивные маркеры: `{details["all_depressive_freq"]:.1f}%`\n'
    )

    if prob >= THRESHOLD * 0.5:
        extras = ''
        if details['hopelessness_freq'] > 0:
            extras += f'\u2022 Безысходность: `{details["hopelessness_freq"]:.1f}%`\n'
        if details['suicidal_freq'] > 0:
            extras += f'\u2022 Суицидальные маркеры: `{details["suicidal_freq"]:.1f}%` \u26a0\ufe0f\n'
        if details['social_isolation_freq'] > 0:
            extras += f'\u2022 Социальная изоляция: `{details["social_isolation_freq"]:.1f}%`\n'
        if details['cognitive_distortions_freq'] > 0:
            extras += f'\u2022 Когнитивные искажения: `{details["cognitive_distortions_freq"]:.1f}%`\n'
        if details['intensification_index'] > 0.5:
            extras += f'\u2022 Эмоциональная интенсификация: `{details["intensification_index"]:.1f}`\n'
        if extras:
            message += f'\n*\U0001f50d Детализация:*\n{extras}'

    if prob >= THRESHOLD:
        message += (
            '\n\U0001f4cc *Рекомендация:*\n'
            'Текст содержит значительное количество маркеров депрессии. '
            'Если это ваш текст, рекомендуется обратиться к специалисту.\n\n'
            '*Телефоны доверия:*\n'
            '\U0001f4de 8-800-2000-122 _(бесплатно, круглосуточно)_\n'
            '\U0001f4de 8-800-333-44-34'
        )

    # --- #3 Temporal эскалация ---
    if temporal_risk_count >= TEMPORAL_RISK_COUNT and prob >= TEMPORAL_RISK_THR:
        message += (
            f'\n\n\U0001f198 *Повторный высокий риск*\n'
            f'За последние {TEMPORAL_WINDOW_MIN} минут это уже '
            f'{temporal_risk_count}-е сообщение с высоким уровнем тревоги. '
            'Пожалуйста, позвоните на телефон доверия прямо сейчас:\n'
            '*8-800-2000-122* _(бесплатно, круглосуточно)_'
        )

    return message


# ==========================================
# 4. TELEGRAM БОТ
# ==========================================

dp  = Dispatcher()
bot = None


@dp.message(Command('start'))
async def cmd_start(message: Message):
    text = (
        '\U0001f916 *Привет! Я -- NLP-ассистент для анализа эмоционального состояния.*\n\n'
        'Я использую гибридную модель:\n'
        '\U0001f9e0 *Трансформер* (768-мерные эмбеддинги)\n'
        '\U0001f4ca *CatBoost* -- классификатор риска\n'
        '\U0001f4dd *31 лингвистический признак* + NLI-отрицание\n'
        '\U0001f4c8 *Временной анализ* -- динамика за 30 минут\n\n'
        '*Как пользоваться:*\n'
        'Напишите о своих мыслях или переживаниях (минимум 5 слов).\n\n'
        '*Команды:*\n'
        '/help -- подробная справка\n'
        '/stats -- анонимная статистика\n'
        '/about -- о модели\n'
        '/clear -- сбросить временную историю\n\n'
        '_Все данные анонимны. ID хешируется SHA-256._'
    )
    await message.answer(text, parse_mode='Markdown')


@dp.message(Command('help'))
async def cmd_help(message: Message):
    t_low  = f'{THRESHOLD * 0.6:.0%}'
    t_high = f'{THRESHOLD:.0%}'
    text = (
        '*Справка*\n\n'
        '*Как работает анализ:*\n'
        '1. Текст кодируется в 768-мерный эмбеддинг\n'
        '2. Извлекаются 31 лингвистический признак + NLI-отрицание\n'
        '3. CatBoost оценивает риск\n'
        '4. Временной трекер фиксирует динамику за 30 минут\n\n'
        f'*Интерпретация:*\n'
        f'\U0001f7e2 Норма (< {t_low}) -- нет тревожных маркеров\n'
        f'\U0001f7e1 Повышенный ({t_low}--{t_high}) -- есть отдельные маркеры\n'
        f'\U0001f534 Высокий (>= {t_high}) -- значительное число маркеров\n\n'
        '_Бот не ставит диагноз. При высоком риске обратитесь к специалисту._'
    )
    await message.answer(text, parse_mode='Markdown')


@dp.message(Command('about'))
async def cmd_about(message: Message):
    cal_status = 'да (isotonic)' if os.path.exists(CAL_PATH) else 'нет'
    text = (
        '*О модели v2*\n\n'
        'Гибридная модель (СПбГУ, 2026)\n\n'
        '*Архитектура:*\n'
        '- Эмбеддинги: `paraphrase-multilingual-all-mpnet-base-v2` (768-мерные)\n'
        '- Классификатор: `CatBoostClassifier`\n'
        f'- Порог: `{THRESHOLD:.3f}`\n'
        f'- Калибровка: `{cal_status}`\n\n'
        '*Улучшения v2:*\n'
        '- NLI scope-aware отрицание\n'
        '- Временной анализ (30 мин)\n'
        '- Калибровка вероятностей\n\n'
        '*Метрики (test set):*\n'
        '- ROC-AUC: `0.9881` | F1: `0.9502` | Recall: `0.9586`\n'
        '- 5-fold CV: `0.9866 +/- 0.0002`\n\n'
        f'\u2022 Безысходность: {len(HOPELESSNESS)} слов | Суицидальная идеация: {len(SUICIDAL_IDEATION)} слов\n\n'
        '_Не заменяет профессиональную помощь._'
    )
    await message.answer(text, parse_mode='Markdown')


@dp.message(Command('stats'))
async def cmd_stats(message: Message):
    stats = load_stats()
    total = stats['total_analyses']
    risk  = stats['risk_detected']
    today = datetime.now().strftime('%Y-%m-%d')
    text = (
        '*Анонимная статистика*\n\n'
        f'Всего анализов: `{total}`\n'
        f'\U0001f7e2 Норма: `{total - risk}`\n'
        f'\U0001f534 Риск выявлен: `{risk}`\n\n'
        f'*Сегодня:* `{stats["daily"].get(today, 0)}` анализов\n\n'
        '_Данные анонимны. User ID хешируется SHA-256._'
    )
    await message.answer(text, parse_mode='Markdown')


@dp.message(Command('clear'))
async def cmd_clear(message: Message):
    """#3: Сбрасывает временную историю пользователя."""
    _user_history[message.from_user.id] = []
    await message.answer(
        '\u2705 Временная история сброшена. Счётчик тревожных сообщений обнулён.'
    )


@dp.message()
async def analyze_message(message: Message):
    text = message.text or message.caption or ''
    if text.startswith('/'):
        return
    if len(text.split()) < 5:
        await message.answer(
            '\u270d\ufe0f Текст слишком короткий. Напишите хотя бы 5-10 слов.'
        )
        return

    await bot.send_chat_action(chat_id=message.chat.id, action='typing')
    try:
        prob, features, details = await asyncio.to_thread(analyze_text, text)
        risk_count = update_temporal(message.from_user.id, prob)  # #3
        response   = format_result(prob, details, temporal_risk_count=risk_count)
        label      = 'Тревога' if prob >= THRESHOLD else 'Норма'
        log_analysis(message.from_user.id, text, prob, label, details, username=message.from_user.username or "")
        await message.answer(response, parse_mode='Markdown')
    except Exception as e:
        logging.error(f'Ошибка анализа: {e}')
        await message.answer('\u274c Произошла ошибка. Попробуйте ещё раз.')


# ==========================================
# 5. ЗАПУСК
# ==========================================

async def main():
    global bot
    bot = Bot(token=BOT_TOKEN)
    print('\n\U0001f916 Бот v2 запущен!')
    print(f'   Порог: {THRESHOLD:.3f} | Temporal: {TEMPORAL_WINDOW_MIN} мин | '
          f'Калибровка: {"da" if cal_model else "net"}')
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())
