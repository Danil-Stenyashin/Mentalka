"""
feature_extractor_v2.py

Улучшения:
- Улучшение #4: NLI-компонент для обработки отрицательных конструкций
  Вместо простого pattern matching используется scope-aware negation:
  отслеживается «область действия» отрицания (до 5 токенов после NOT/НЕ)
  и подавляются суицидальные/депрессивные маркеры внутри этой области.
"""

import re
from typing import Dict

try:
    from lexicons import (
        HOPELESSNESS, SUICIDAL_IDEATION, DEPRESSION_EMOTIONAL,
        COGNITIVE_DISTORTIONS, SOCIAL_ISOLATION, PHYSICAL_SYMPTOMS,
        POSITIVE_MARKERS, INTENSIFIERS,
    )
except ImportError:
    HOPELESSNESS = SUICIDAL_IDEATION = DEPRESSION_EMOTIONAL = []
    COGNITIVE_DISTORTIONS = SOCIAL_ISOLATION = PHYSICAL_SYMPTOMS = []
    POSITIVE_MARKERS = INTENSIFIERS = []

# ---------------------------------------------------------------------------
# NLI-компонент: отрицание (улучшение #4)
# ---------------------------------------------------------------------------

# Стоп-слова для границы области действия отрицания
_SCOPE_STOPWORDS = {
    # RU
    'но', 'зато', 'хотя', 'потому', 'чтобы', 'если', 'когда', 'пока',
    'после', 'прежде', 'так', 'потому что', 'несмотря',
    # EN
    'but', 'although', 'though', 'because', 'if', 'when', 'while',
    'after', 'before', 'so', 'yet', 'however',
}

# Суицидальные глаголы/существительные (ядро)
_SUICIDAL_CORE_RU = [
    'убить', 'убиться', 'убивать', 'умереть', 'умирать', 'смерт',
    'суицид', 'вскрыть', 'повеситься', 'застрелиться', 'прыгнуть',
    'убью', 'умру', 'покончить', 'прекратить жизнь',
]
_SUICIDAL_CORE_EN = [
    'kill', 'die', 'suicide', 'hang', 'shoot', 'jump off', 'end it',
    'end my life', 'take my life',
]

# Паттерны явного отрицания суицидальных намерений
_EXPLICIT_NEGATION_PATTERNS_RU = [
    r'не\s+хочу\s+(?:себя\s+)?(?:убива|умира|убить|умере)',
    r'не\s+думаю\s+о\s+(?:суицид|смерт|убийств)',
    r'не\s+(?:планирую|собираюсь|буду|хочу)\s+(?:умира|убива|прыга|вешать)',
    r'(?:мысли|думы)\s+о\s+смерт\S*\s+меня\s+не',
    r'не\s+суицидальн',
    r'не\s+собираюсь\s+умира',
    r'жить\s+хочу',
    r'хочу\s+жить',
]
_EXPLICIT_NEGATION_PATTERNS_EN = [
    r"i(?:'m|\s+am)\s+not\s+suicidal",
    r"don'?t\s+want\s+to\s+(?:die|kill|hurt)",
    r"not\s+going\s+to\s+(?:die|kill|hurt|end)",
    r"i\s+will\s+not\s+(?:kill|hurt|harm)\s+(?:my)?self",
    r"no\s+thoughts\s+of\s+(?:suicide|killing|dying)",
    r"not\s+thinking\s+about\s+(?:suicide|death|killing)",
    r"i\s+want\s+to\s+live",
    r"i\s+choose\s+(?:to\s+)?live",
]

_ALL_NEG_PATTERNS = [re.compile(p, re.IGNORECASE)
                     for p in _EXPLICIT_NEGATION_PATTERNS_RU + _EXPLICIT_NEGATION_PATTERNS_EN]


def _compute_negation_scope_weight(text: str) -> float:
    """
    Scope-aware negation weight [0..1].

    Алгоритм:
    1. Токенизируем текст.
    2. Для каждого токена-отрицания (не/нет/никогда/no/not/never)
       открываем "окно" scope из SCOPE_WINDOW токенов.
    3. Если внутри окна есть суицидальный/депрессивный маркер,
       считаем это отрицанием суицидального намерения.
    4. Нормируем на кол-во суицидальных маркеров (0..1).
    """
    SCOPE_WINDOW = 5
    NEG_TOKENS_RU = {'не', 'нет', 'никогда', 'ни', 'никак', 'нисколько'}
    NEG_TOKENS_EN = {'no', 'not', "n't", 'never', 'neither', 'nor'}
    all_neg_tokens = NEG_TOKENS_RU | NEG_TOKENS_EN

    tokens = re.findall(r'\b\w+\b', text.lower())
    if not tokens:
        return 0.0

    # Все суицидальные маркеры для поиска в токенах
    suicidal_stems = [w[:5].lower() for w in (_SUICIDAL_CORE_RU + _SUICIDAL_CORE_EN)]
    depressive_stems = [w[:5].lower() for w in list(HOPELESSNESS) + list(SUICIDAL_IDEATION)]

    negated_suicidal = 0
    total_suicidal   = 0

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        # Проверяем: суицидальный маркер без отрицания?
        is_suicidal = any(tok.startswith(s) for s in suicidal_stems + depressive_stems)
        if is_suicidal:
            total_suicidal += 1
            # Смотрим назад — есть ли отрицание в окне SCOPE_WINDOW?
            window_start = max(0, i - SCOPE_WINDOW)
            window_tokens = tokens[window_start:i]
            # Граница scope: останавливаемся на стоп-словах
            in_scope = True
            for wt in reversed(window_tokens):
                if wt in _SCOPE_STOPWORDS:
                    in_scope = False
                    break
                if wt in all_neg_tokens:
                    if in_scope:
                        negated_suicidal += 1
                    break
        i += 1

    if total_suicidal == 0:
        return 0.0
    return min(negated_suicidal / total_suicidal, 1.0)


def _detect_explicit_negation(text: str) -> int:
    """Детектирует явные конструкции отрицания суицидальных намерений."""
    for pat in _ALL_NEG_PATTERNS:
        if pat.search(text):
            return 1
    return 0


def _compute_nli_negation_score(text: str) -> float:
    """
    Итоговый NLI-скор отрицания [0..1].
    Объединяет: явное отрицание + scope-aware.
    """
    explicit = _detect_explicit_negation(text)
    if explicit:
        return 1.0
    scope_w = _compute_negation_scope_weight(text)
    return scope_w


# ---------------------------------------------------------------------------
# Базовые функции извлечения признаков (оставлены из v1)
# ---------------------------------------------------------------------------

def _lex_freq(text_lower: str, lexicon: list) -> float:
    """Частота совпадений лексикона на 100 слов."""
    words = text_lower.split()
    n = len(words)
    if n == 0:
        return 0.0
    hits = sum(1 for w in words if any(w.startswith(m[:5].lower()) for m in lexicon))
    return hits / n * 100


def _sentence_count(text: str) -> int:
    return max(1, len(re.findall(r'[.!?…]+', text)) or 1)


def _lexical_diversity(words: list) -> float:
    if len(words) < 2:
        return 1.0
    return len(set(words)) / len(words)


def extract_features(text: str) -> Dict[str, float]:
    """
    Извлекает 32 лингвистических признака (v2: +nli_negation_score).
    Совместим с v1 FEATURE_NAMES (31 признак) + новый признак.
    """
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    n_words = max(len(words), 1)
    sentences = _sentence_count(text)

    # --- Базовые метрики ---
    text_length          = len(text)
    word_count           = len(words)
    sentence_count       = sentences
    avg_word_length      = sum(len(w) for w in words) / n_words
    avg_sentence_length  = n_words / sentences
    lexical_diversity    = _lexical_diversity(words)

    # --- Лексиконные признаки ---
    hopelessness_freq          = _lex_freq(text_lower, HOPELESSNESS)
    suicidal_freq              = _lex_freq(text_lower, SUICIDAL_IDEATION)
    depression_emotional_freq  = _lex_freq(text_lower, DEPRESSION_EMOTIONAL)
    cognitive_distortions_freq = _lex_freq(text_lower, COGNITIVE_DISTORTIONS)
    social_isolation_freq      = _lex_freq(text_lower, SOCIAL_ISOLATION)
    physical_symptoms_freq     = _lex_freq(text_lower, PHYSICAL_SYMPTOMS)
    positive_markers_freq      = _lex_freq(text_lower, POSITIVE_MARKERS)
    intensifiers_freq          = _lex_freq(text_lower, INTENSIFIERS)

    all_depressive_freq = (hopelessness_freq + suicidal_freq +
                           depression_emotional_freq + cognitive_distortions_freq +
                           social_isolation_freq + physical_symptoms_freq) / 6

    # --- Стилистические ---
    i_pronouns_ru = {'я', 'меня', 'мне', 'мной', 'мою', 'моя', 'моё', 'моего'}
    i_pronouns_en = {'i', 'me', 'my', 'mine', 'myself'}
    neg_tokens_ru = {'не', 'нет', 'никогда', 'ни', 'никак'}
    neg_tokens_en = {'no', 'not', "n't", 'never'}

    i_pronoun_freq = sum(1 for w in words
                         if w in i_pronouns_ru or w in i_pronouns_en) / n_words * 100
    negation_freq  = sum(1 for w in words
                         if w in neg_tokens_ru or w in neg_tokens_en) / n_words * 100

    question_ratio    = text.count('?') / n_words
    exclamation_ratio = text.count('!') / n_words
    ellipsis_ratio    = text.count('...') / n_words
    caps_freq         = sum(1 for c in text if c.isupper()) / max(len(text), 1)

    # --- Составные индексы ---
    negative_emotion_index  = (hopelessness_freq + depression_emotional_freq) / 2
    positive_emotion_index  = positive_markers_freq
    emotional_balance       = positive_emotion_index - negative_emotion_index
    intensification_index   = intensifiers_freq
    self_reference_index    = i_pronoun_freq
    depression_index        = (hopelessness_freq * 1.5 + depression_emotional_freq +
                               cognitive_distortions_freq * 1.2) / 3
    suicide_risk_index      = (suicidal_freq * 3 + hopelessness_freq * 2 +
                               social_isolation_freq) / 6
    risk_protective_ratio   = (suicide_risk_index + 1e-9) / (positive_emotion_index + 1e-9)
    text_complexity         = avg_word_length * avg_sentence_length / 10

    # --- УЛУЧШЕНИЕ #4: NLI-отрицание (scope-aware + явные паттерны) ---
    negation_of_suicidal_intent = _detect_explicit_negation(text)
    nli_negation_score          = _compute_nli_negation_score(text)

    return {
        # 31 базовых признака (совместимость с v1)
        'text_length':                  text_length,
        'word_count':                   word_count,
        'sentence_count':               sentence_count,
        'avg_word_length':              avg_word_length,
        'avg_sentence_length':          avg_sentence_length,
        'lexical_diversity':            lexical_diversity,
        'hopelessness_freq':            hopelessness_freq,
        'suicidal_freq':                suicidal_freq,
        'depression_emotional_freq':    depression_emotional_freq,
        'cognitive_distortions_freq':   cognitive_distortions_freq,
        'social_isolation_freq':        social_isolation_freq,
        'physical_symptoms_freq':       physical_symptoms_freq,
        'positive_markers_freq':        positive_markers_freq,
        'intensifiers_freq':            intensifiers_freq,
        'all_depressive_freq':          all_depressive_freq,
        'i_pronoun_freq':               i_pronoun_freq,
        'negation_freq':                negation_freq,
        'question_ratio':               question_ratio,
        'exclamation_ratio':            exclamation_ratio,
        'ellipsis_ratio':               ellipsis_ratio,
        'caps_freq':                    caps_freq,
        'negative_emotion_index':       negative_emotion_index,
        'positive_emotion_index':       positive_emotion_index,
        'emotional_balance':            emotional_balance,
        'intensification_index':        intensification_index,
        'self_reference_index':         self_reference_index,
        'depression_index':             depression_index,
        'suicide_risk_index':           suicide_risk_index,
        'risk_protective_ratio':        risk_protective_ratio,
        'text_complexity':              text_complexity,
        'negation_of_suicidal_intent':  negation_of_suicidal_intent,
        # Новый признак v2
        'nli_negation_score':           nli_negation_score,
    }


# Список признаков для модели (32 = 31 + 1 новый)
FEATURE_NAMES_V1 = [
    'text_length', 'word_count', 'sentence_count', 'avg_word_length',
    'avg_sentence_length', 'lexical_diversity', 'hopelessness_freq',
    'suicidal_freq', 'depression_emotional_freq', 'cognitive_distortions_freq',
    'social_isolation_freq', 'physical_symptoms_freq', 'positive_markers_freq',
    'intensifiers_freq', 'all_depressive_freq', 'i_pronoun_freq',
    'negation_freq', 'question_ratio', 'exclamation_ratio', 'ellipsis_ratio',
    'caps_freq', 'negative_emotion_index', 'positive_emotion_index',
    'emotional_balance', 'intensification_index', 'self_reference_index',
    'depression_index', 'suicide_risk_index', 'risk_protective_ratio',
    'text_complexity', 'negation_of_suicidal_intent',
]

FEATURE_NAMES_V2 = FEATURE_NAMES_V1 + ['nli_negation_score']

# Обратная совместимость
FEATURE_NAMES = FEATURE_NAMES_V1
