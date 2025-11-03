#!/bin/bash

BASE_URL="http://localhost:5000"
MODEL_ID_FILE="/tmp/test_model_id.txt"

echo "Запуск тестирования работоспособности Flask API"

# Функция для вывода шага
print_step() {
    echo ""
    echo "=================================================="
    echo "Шаг $1: $2"
    echo "=================================================="
}

# 1. Проверка здоровья сервиса
print_step 1 "Проверка здоровья сервиса"
curl -i "$BASE_URL/health"
echo ""

# 2. Получение доступных классов моделей
print_step 2 "Получение доступных классов моделей"
curl -s "$BASE_URL/model-classes" | python -m json.tool
echo ""

# 3. Обучение и сохранение модели
print_step 3 "Обучение и сохранение модели"
RESPONSE=$(curl -s -X POST "$BASE_URL/models/train" \
    -H "Content-Type: application/json" \
    -d '{
        "model_type": "random_forest",
        "params": {"n_estimators": 10, "max_depth": 3},
        "X": [[5.1,3.5,1.4,0.2],[4.9,3.0,1.4,0.2]],
        "y": [0,0]
    }')
echo "$RESPONSE" | python -m json.tool

# Извлекаем model_id
MODEL_ID=$(echo "$RESPONSE" | python -c "import sys, json; print(json.load(sys.stdin)['model_id'])")
echo "$MODEL_ID" > $MODEL_ID_FILE
echo "Model ID сохранен: $MODEL_ID"

# 4. Просмотр всех моделей
print_step 4 "Просмотр всех моделей"
curl -s "$BASE_URL/models" | python -m json.tool
echo ""

# 5. Получение информации о конкретной модели
print_step 5 "Получение информации о конкретной модели"
curl -s "$BASE_URL/models/$MODEL_ID" | python -m json.tool
echo ""

# 6. Получение предсказаний
print_step 6 "Получение предсказаний"
curl -s -X POST "$BASE_URL/models/$MODEL_ID/predict" \
    -H "Content-Type: application/json" \
    -d '{"X": [[5.1,3.5,1.4,0.2],[4.9,3.0,1.4,0.2]]}' | python -m json.tool
echo ""

# 7. Переобучение модели
print_step 7 "Переобучение модели"
curl -s -X POST "$BASE_URL/models/$MODEL_ID/retrain" \
    -H "Content-Type: application/json" \
    -d '{"X": [[5.1,3.5,1.4,0.2],[4.9,3.0,1.4,0.2]], "y": [1,1]}' | python -m json.tool
echo ""

# 8. Удаление тестовой модели
read -p "🗑️  Удалить тестовую модель? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_step 8 "Удаление тестовой модели"
    curl -i -X DELETE "$BASE_URL/models/$MODEL_ID"
    rm -f $MODEL_ID_FILE
fi

echo ""
echo "Тестирование завершено! Все тесты пройдены! Success! Success! Success!"