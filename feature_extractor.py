"""
Продвинутый экстрактор лингвистических признаков для анализа
депрессивных и суицидальных текстов.

Включает:
- Лексиконный анализ (по словарям маркеров)
- Синтаксические признаки (длина предложений, знаки препинания)
- Стилистические признаки (языковые паттерны)
- Эмоциональные индексы
"""

import re
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple
import nltk

from lexicons import (
    HOPELESSNESS, SUICIDAL_IDEATION, DEPRESSION_EMOTIONAL,
    COGNITIVE_DISTORTIONS, SOCIAL_ISOLATION, PHYSICAL_SYMPTOMS,
    POSITIVE_MARKERS, INTENSIFIERS, ALL_DEPRESSIVE,
)


def lemmatize_text(text: str) -> List[str]:
    """
    Простая лемматизация для русского языка.
    В реальном проекте используй pymorphy2 или natasha.
    """
    # Очистка и нормализация
    text = text.lower()
    text = re.sub(r"[^а-яёa-z0-9\s]", " ", text)
    words = text.split()
    
    # Простые суффиксные правила русского языка
    lemmas = []
    for word in words:
        if len(word) < 2:
            continue
        # Простая стемминг-подобная нормализация
        if word.endswith("ться"):
            word = word[:-4]
        elif word.endswith("тся"):
            word = word[:-3]
        elif word.endswith("ость"):
            word = word[:-4]
        elif word.endswith("ости"):
            word = word[:-4]
        elif word.endswith("ован"):
            word = word[:-2]
        elif word.endswith("ована"):
            word = word[:-3]
        elif word.endswith("овано"):
            word = word[:-3]
        elif word.endswith("ировать"):
            word = word[:-5]
        elif word.endswith("ировать"):
            word = word[:-5]
        elif word.endswith("аю"):
            word = word[:-2]
        elif word.endswith("яю"):
            word = word[:-2]
        elif word.endswith("аешь"):
            word = word[:-4]
        elif word.endswith("яешь"):
            word = word[:-4]
        elif word.endswith("ает"):
            word = word[:-3]
        elif word.endswith("яет"):
            word = word[:-3]
        elif word.endswith("аем"):
            word = word[:-3]
        elif word.endswith("яем"):
            word = word[:-3]
        elif word.endswith("ают"):
            word = word[:-3]
        elif word.endswith("яют"):
            word = word[:-3]
        elif word.endswith("ала"):
            word = word[:-3]
        elif word.endswith("яла"):
            word = word[:-3]
        elif word.endswith("ало"):
            word = word[:-3]
        elif word.endswith("яло"):
            word = word[:-3]
        elif word.endswith("али"):
            word = word[:-3]
        elif word.endswith("яли"):
            word = word[:-3]
        elif word.endswith("ал"):
            word = word[:-2]
        elif word.endswith("ял"):
            word = word[:-2]
        elif word.endswith("ан"):
            word = word[:-2]
        elif word.endswith("ян"):
            word = word[:-2]
        elif word.endswith("ена"):
            word = word[:-3]
        elif word.endswith("ено"):
            word = word[:-3]
        elif word.endswith("ены"):
            word = word[:-3]
        elif word.endswith("ен"):
            word = word[:-2]
        elif word.endswith("ен"):
            word = word[:-2]
        elif word.endswith("ован"):
            word = word[:-2]
        elif word.endswith("ована"):
            word = word[:-3]
        elif word.endswith("овано"):
            word = word[:-3]
        elif word.endswith("ованы"):
            word = word[:-3]
        elif word.endswith("ти"):
            word = word[:-2]
        elif word.endswith("ть"):
            word = word[:-2]
        elif word.endswith("ил"):
            word = word[:-2]
        elif word.endswith("ила"):
            word = word[:-3]
        elif word.endswith("ило"):
            word = word[:-3]
        elif word.endswith("или"):
            word = word[:-3]
        elif word.endswith("ить"):
            word = word[:-3]
        elif word.endswith("ить"):
            word = word[:-3]
        elif word.endswith("ю"):
            word = word[:-1]
        elif word.endswith("ешь"):
            word = word[:-3]
        elif word.endswith("ет"):
            word = word[:-2]
        elif word.endswith("ем"):
            word = word[:-2]
        elif word.endswith("ете"):
            word = word[:-3]
        elif word.endswith("ут"):
            word = word[:-2]
        elif word.endswith("ют"):
            word = word[:-2]
        elif word.endswith("л"):
            word = word[:-1]
        elif word.endswith("ла"):
            word = word[:-2]
        elif word.endswith("ло"):
            word = word[:-2]
        elif word.endswith("ли"):
            word = word[:-2]
        elif word.endswith("ный"):
            word = word[:-3]
        elif word.endswith("ная"):
            word = word[:-3]
        elif word.endswith("ное"):
            word = word[:-3]
        elif word.endswith("ные"):
            word = word[:-3]
        elif word.endswith("ный"):
            word = word[:-3]
        elif word.endswith("ная"):
            word = word[:-3]
        elif word.endswith("ное"):
            word = word[:-3]
        elif word.endswith("ные"):
            word = word[:-3]
        elif word.endswith("ного"):
            word = word[:-4]
        elif word.endswith("ному"):
            word = word[:-4]
        elif word.endswith("ным"):
            word = word[:-3]
        elif word.endswith("ной"):
            word = word[:-3]
        elif word.endswith("ном"):
            word = word[:-3]
        elif word.endswith("ных"):
            word = word[:-3]
        elif word.endswith("ым"):
            word = word[:-2]
        elif word.endswith("ой"):
            word = word[:-2]
        elif word.endswith("ом"):
            word = word[:-2]
        elif word.endswith("ых"):
            word = word[:-2]
        elif word.endswith("ую"):
            word = word[:-2]
        elif word.endswith("ым"):
            word = word[:-2]
        elif word.endswith("ему"):
            word = word[:-3]
        elif word.endswith("его"):
            word = word[:-3]
        elif word.endswith("ем"):
            word = word[:-2]
        elif word.endswith("ом"):
            word = word[:-2]
        elif word.endswith("ая"):
            word = word[:-2]
        elif word.endswith("яя"):
            word = word[:-2]
        elif word.endswith("о"):
            word = word[:-1]
        elif word.endswith("е"):
            word = word[:-1]
        elif word.endswith("и"):
            word = word[:-1]
        elif word.endswith("ы"):
            word = word[:-1]
        elif word.endswith("а"):
            word = word[:-1]
        elif word.endswith("я"):
            word = word[:-1]
        elif word.endswith("у"):
            word = word[:-1]
        elif word.endswith("ю"):
            word = word[:-1]
        elif word.endswith("с"):
            word = word[:-1]
        elif word.endswith("ш"):
            word = word[:-1]
        elif word.endswith("ь"):
            word = word[:-1]
        
        lemmas.append(word)
    return lemmas


def find_lexicon_matches(text: str, lexicon: set) -> Tuple[int, List[str]]:
    """
    Находит количество и список совпадений с лексиконом.
    Учитывает многословные выражения.
    """
    text_lower = text.lower()
    count = 0
    matches = []
    
    for phrase in lexicon:
        # Проверяем точные совпадения
        if " " in phrase:
            # Многословное выражение
            if phrase in text_lower:
                count += 1
                matches.append(phrase)
        else:
            # Однословное — проверяем как отдельное слово
            pattern = r'\b' + re.escape(phrase) + r'\b'
            found = re.findall(pattern, text_lower)
            count += len(found)
            matches.extend(found)
    
    return count, matches


def extract_features(text: str) -> Dict[str, float]:
    """
    Извлекает полный набор лингвистических признаков из текста.
    
    Returns:
        dict: словарь признаков -> значения (float)
    """
    if not text or len(text.strip()) < 3:
        return {name: 0.0 for name in FEATURE_NAMES}
    
    # Базовые текстовые метрики
    text_lower = text.lower()
    words = text_lower.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    char_count = len(text)
    word_count = len(words)
    sentence_count = max(len(sentences), 1)
    
    # Средние длины
    avg_word_len = np.mean([len(w) for w in words]) if words else 0.0
    avg_sentence_len = word_count / sentence_count if sentence_count > 0 else 0.0
    
    # Лемматизированные слова
    lemmas = lemmatize_text(text)
    lemma_count = len(lemmas)
    unique_lemmas = len(set(lemmas))
    lexical_diversity = unique_lemmas / lemma_count if lemma_count > 0 else 0.0
    
    # --- ЛЕКСИКОННЫЙ АНАЛИЗ ---
    
    # Безысходность
    hope_count, _ = find_lexicon_matches(text, HOPELESSNESS)
    hope_freq = hope_count / word_count * 100 if word_count > 0 else 0.0
    
    # Суицидальные маркеры
    suicide_count, _ = find_lexicon_matches(text, SUICIDAL_IDEATION)
    suicide_freq = suicide_count / word_count * 100 if word_count > 0 else 0.0
    
    # Эмоциональная депрессия
    depr_count, _ = find_lexicon_matches(text, DEPRESSION_EMOTIONAL)
    depr_freq = depr_count / word_count * 100 if word_count > 0 else 0.0
    
    # Когнитивные искажения
    cogn_count, _ = find_lexicon_matches(text, COGNITIVE_DISTORTIONS)
    cogn_freq = cogn_count / word_count * 100 if word_count > 0 else 0.0
    
    # Социальная изоляция
    isol_count, _ = find_lexicon_matches(text, SOCIAL_ISOLATION)
    isol_freq = isol_count / word_count * 100 if word_count > 0 else 0.0
    
    # Физические симптомы
    phys_count, _ = find_lexicon_matches(text, PHYSICAL_SYMPTOMS)
    phys_freq = phys_count / word_count * 100 if word_count > 0 else 0.0
    
    # Позитивные маркеры
    pos_count, _ = find_lexicon_matches(text, POSITIVE_MARKERS)
    pos_freq = pos_count / word_count * 100 if word_count > 0 else 0.0
    
    # Интенсификаторы
    intens_count, _ = find_lexicon_matches(text, INTENSIFIERS)
    intens_freq = intens_count / word_count * 100 if word_count > 0 else 0.0
    
    # Все депрессивные маркеры (комбинированные)
    all_depr_count = hope_count + suicide_count + depr_count + cogn_count + isol_count + phys_count
    all_depr_freq = all_depr_count / word_count * 100 if word_count > 0 else 0.0
    
    # --- СТИЛИСТИЧЕСКИЕ ПРИЗНАКИ ---
    
    # Местоимения "я/меня/мне"
    i_pattern = r'\b(я|меня|мне|мной|мною|мо[ейё]|мои|моих|моему|моем|моим)\b'
    i_count = len(re.findall(i_pattern, text_lower))
    i_freq = i_count / word_count * 100 if word_count > 0 else 0.0
    
    # Отрицания
    neg_pattern = r'\b(не|нет|никогда|ничего|никто|никак|нигде|ни за что|ни о чем|ни к чему|ни с чем|ни на что|ни перед чем|ни после чего|ни от чего|ни к кому|ни с кем|ни о ком|ни за кого|ни перед кем|ни после кого|ни от кого|ни откуда|ни куда|ни от кого|ни с чего|ни на чем|ни в чем|ни о чем|ни при чем|ни при каких|ни в каких|ни на каких|ни в каком|ни в какой|ни в какие|ни в каких|ни о каком|ни о какой|ни о каких|ни о каких|ни для какого|ни для какой|ни для каких|ни из какого|ни из какой|ни из каких|ни по какому|ни по какой|ни по каким|ни с каким|ни с какой|ни с какими|ни у какого|ни у какой|ни у каких|ни за каким|ни за какой|ни за какими|ни над каким|ни над какой|ни над какими|ни под каким|ни под какой|ни под какими|ни перед каким|ни перед какой|ни перед какими|ни при каком|ни при какой|ни при каких|ни после какого|ни после какой|ни после каких|ни между каким|ни между какой|ни между какими|ни внутри какого|ни внутри какой|ни внутри каких|ни вне какого|ни вне какой|ни вне каких|ни против какого|ни против какой|ни против каких|ни вопреки какому|ни вопреки какой|ни вопреки каким|ни благодаря какому|ни благодаря какой|ни благодаря каким|ни согласно какому|ни согласно какой|ни согласно каким|ни ввиду какого|ни ввиду какой|ни ввиду каких|ни вследствие какого|ни вследствие какой|ни вследствие каких|ни вместо какого|ни вместо какой|ни вместо каких|ни ради какого|ни ради какой|ни ради каких|ни несмотря на какое|ни несмотря на какую|ни несмотря на какие|ни несмотря на каких|ни вопреки какому|ни вопреки какой|ни вопреки каким|ни невзирая на какое|ни невзирая на какую|ни невзирая на какие|ни невзирая на каких)\b'
    neg_count = len(re.findall(neg_pattern, text_lower))
    neg_freq = neg_count / word_count * 100 if word_count > 0 else 0.0
    
    # Вопросительные предложения
    question_count = text.count('?')
    question_ratio = question_count / sentence_count if sentence_count > 0 else 0.0
    
    # Восклицательные предложения
    excl_count = text.count('!')
    excl_ratio = excl_count / sentence_count if sentence_count > 0 else 0.0
    
    # Многоточия (паузы, незаконченные мысли)
    ellipsis_count = text.count('...') + text.count('..') + text.count('…')
    ellipsis_ratio = ellipsis_count / sentence_count if sentence_count > 0 else 0.0
    
    # Заглавные буквы (крик, аффект)
    caps_words = [w for w in words if w.isupper() and len(w) > 1]
    caps_freq = len(caps_words) / word_count * 100 if word_count > 0 else 0.0
    
    # --- ЭМОЦИОНАЛЬНЫЕ ИНДЕКСЫ ---
    
    # Негативный эмоциональный индекс (взвешенная сумма депрессивных маркеров)
    # Суицидальные маркеры весят больше
    negative_index = (
        hope_freq * 1.0 +
        suicide_freq * 3.0 +
        depr_freq * 1.5 +
        cogn_freq * 1.2 +
        isol_freq * 1.3 +
        phys_freq * 0.8
    )
    
    # Позитивный эмоциональный индекс
    positive_index = pos_freq * 2.0
    
    # Нетто-эмоциональный баланс
    emotional_balance = negative_index - positive_index
    
    # Индекс интенсификации (усиление эмоций)
    intensification_index = intens_freq * 1.5
    
    # Индекс самообращения (фокус на себе)
    self_reference_index = i_freq * 1.0
    
    # --- КОМПОЗИТНЫЕ МЕТРИКИ ---
    
    # Общий депрессивный индекс
    depression_index = negative_index + intensification_index * 0.5
    
    # Суицидальный риск-индекс
    suicide_risk_index = suicide_freq * 3.0 + hope_freq * 1.5 + cogn_freq * 0.5
    
    # Риск/защитный фактор (ratio)
    risk_protective_ratio = negative_index / (positive_index + 0.01)
    
    # Текстовая сложность
    text_complexity = avg_word_len * avg_sentence_len
    
    # --- СТРУКТУРНЫЕ ПРИЗНАКИ ---
    
    features = {
        # Базовые метрики
        "text_length": float(char_count),
        "word_count": float(word_count),
        "sentence_count": float(sentence_count),
        "avg_word_length": float(avg_word_len),
        "avg_sentence_length": float(avg_sentence_len),
        "lexical_diversity": float(lexical_diversity),
        
        # Лексиконные частоты
        "hopelessness_freq": float(hope_freq),
        "suicidal_freq": float(suicide_freq),
        "depression_emotional_freq": float(depr_freq),
        "cognitive_distortions_freq": float(cogn_freq),
        "social_isolation_freq": float(isol_freq),
        "physical_symptoms_freq": float(phys_freq),
        "positive_markers_freq": float(pos_freq),
        "intensifiers_freq": float(intens_freq),
        "all_depressive_freq": float(all_depr_freq),
        
        # Структурные признаки
        "i_pronoun_freq": float(i_freq),
        "negation_freq": float(neg_freq),
        "question_ratio": float(question_ratio),
        "exclamation_ratio": float(excl_ratio),
        "ellipsis_ratio": float(ellipsis_ratio),
        "caps_freq": float(caps_freq),
        
        # Эмоциональные индексы
        "negative_emotion_index": float(negative_index),
        "positive_emotion_index": float(positive_index),
        "emotional_balance": float(emotional_balance),
        "intensification_index": float(intensification_index),
        "self_reference_index": float(self_reference_index),
        
        # Композитные метрики
        "depression_index": float(depression_index),
        "suicide_risk_index": float(suicide_risk_index),
        "risk_protective_ratio": float(risk_protective_ratio),
        "text_complexity": float(text_complexity),
    }
    
    return features


# Список имён признаков (для создания DataFrame)
FEATURE_NAMES = [
    "text_length",
    "word_count",
    "sentence_count",
    "avg_word_length",
    "avg_sentence_length",
    "lexical_diversity",
    "hopelessness_freq",
    "suicidal_freq",
    "depression_emotional_freq",
    "cognitive_distortions_freq",
    "social_isolation_freq",
    "physical_symptoms_freq",
    "positive_markers_freq",
    "intensifiers_freq",
    "all_depressive_freq",
    "i_pronoun_freq",
    "negation_freq",
    "question_ratio",
    "exclamation_ratio",
    "ellipsis_ratio",
    "caps_freq",
    "negative_emotion_index",
    "positive_emotion_index",
    "emotional_balance",
    "intensification_index",
    "self_reference_index",
    "depression_index",
    "suicide_risk_index",
    "risk_protective_ratio",
    "text_complexity",
]
