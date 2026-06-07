"""
Демо-скрипт для тестирования модели без Telegram.
Позволяет проверить работу экстрактора признаков и посмотреть на лингвистические метрики.
"""

from feature_extractor import extract_features, FEATURE_NAMES

# Тестовые тексты
TEST_CASES = [
    # Депрессивные
    "Мне кажется, всё потеряно. Я не вижу смысла продолжать. Жизнь бессмысленна.",
    "Каждый день одно и то же. Я устал бороться. Не хватает сил даже встать с кровати.",
    "Никому не нужен. Все отвернулись. Лучше бы я не родился.",
    "Зачем я живу? Обуза для семьи. Без меня всем было бы лучше.",
    "Прощайте. Не будет меня. Забудьте. Я обуза.",
    
    # Нормальные
    "Сегодня отличный день! Встретился с друзьями, погуляли в парке.",
    "Рад новому проекту на работе. Интересные задачи, хорошая команда.",
    "Учусь программировать. Пока сложно, но интересно. Есть прогресс.",
    "Встал рано, позанимался спортом. Чувствую прилив энергии.",
    "Планирую отпуск. Хочу посетить новые места, отдохнуть.",
]

print("=" * 70)
print("ДЕМО: ЛИНГВИСТИЧЕСКИЙ АНАЛИЗ ТЕКСТОВ")
print("=" * 70)

for i, text in enumerate(TEST_CASES, 1):
    features = extract_features(text)
    
    label = "ДЕПРЕССИЯ" if features["depression_index"] > 5 else "НОРМА"
    
    print(f"\n{'-' * 70}")
    print(f"Текст {i}: {text[:60]}...")
    print(f"{'-' * 70}")
    
    # Основные метрики
    print(f"  ?? Индекс депрессии:       {features['depression_index']:.2f}")
    print(f"  ?? Суицидальный риск:      {features['suicide_risk_index']:.2f}")
    print(f"  ?? Эмоц. баланс:           {features['emotional_balance']:+.2f}")
    print(f"  ?? Риск/защита ratio:      {features['risk_protective_ratio']:.2f}")
    
    # Детали
    print(f"  --------------------------")
    print(f"  ?? Слов:                   {int(features['word_count'])}")
    print(f"  ?? Лекс. разнообразие:     {features['lexical_diversity']:.2f}")
    print(f"  ?? Я/меня частота:         {features['i_pronoun_freq']:.1f}%")
    print(f"  ?? Маркеры депрессии:      {features['all_depressive_freq']:.1f}%")
    
    # Категории
    print(f"  --------------------------")
    if features["hopelessness_freq"] > 0:
        print(f"  ??  Безысходность:         {features['hopelessness_freq']:.1f}%")
    if features["suicidal_freq"] > 0:
        print(f"  ??  Суицидальные:          {features['suicidal_freq']:.1f}%")
    if features["social_isolation_freq"] > 0:
        print(f"  ??  Соц. изоляция:        {features['social_isolation_freq']:.1f}%")
    if features["cognitive_distortions_freq"] > 0:
        print(f"  ??  Когн. искажения:       {features['cognitive_distortions_freq']:.1f}%")
    if features["depression_emotional_freq"] > 0:
        print(f"  ??  Эмоц. депрессия:      {features['depression_emotional_freq']:.1f}%")
    if features["positive_markers_freq"] > 0:
        print(f"  ? Позитивные:            {features['positive_markers_freq']:.1f}%")
    
    # Интенсификаторы
    if features["intensification_index"] > 0:
        print(f"  ? Интенсификация:         {features['intensification_index']:.1f}")
    
    print(f"\n  ???  Вердикт: {label}")

print(f"\n{'=' * 70}")
print("ДЕМО ЗАВЕРШЕНО")
print(f"{'=' * 70}")
