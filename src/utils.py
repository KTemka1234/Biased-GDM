import json
from typing import Optional, Dict


def create_example_data():
    """
    Создание примера данных из статьи (Таблица 4)
    """
    # Нормализованные оценки из таблицы 4
    normalized_scores = [
        [  # DM1
            [0.48692, 0.41586, 0.54536],
            [0.45033, 0.52772, 0.34813],
            [0.48894, 0.32891, 0.48867],
            [0.37971, 0.37648, 0.42778],
            [0.33189, 0.43112, 0.33699],
            [0.54532, 0.52755, 0.40366],
        ],
        [  # DM2
            [0.57043, 0.75376, 0.30971],
            [0.67194, 0.59235, 0.83423],
            [0.61275, 0.72741, 0.71386],
            [0.43229, 0.79708, 0.53919],
            [0.27685, 0.56941, 0.77226],
            [0.42602, 0.77619, 0.85662],
        ],
        [  # DM3
            [0.67077, 0.57771, 0.49205],
            [0.72409, 0.69956, 0.54178],
            [0.75015, 0.55521, 0.71333],
            [0.62814, 0.63033, 0.76235],
            [0.70551, 0.56176, 0.72425],
            [0.64256, 0.50469, 0.41516],
        ],
        [  # DM4
            [0.59369, 0.51611, 0.60229],
            [0.60409, 0.58426, 0.62333],
            [0.68182, 0.56144, 0.49740],
            [0.49796, 0.63250, 0.68966],
            [0.51662, 0.57655, 0.50300],
            [0.59018, 0.63534, 0.58350],
        ],
        [  # DM5
            [0.36061, 0.34695, 0.21558],
            [0.35013, 0.28778, 0.17443],
            [0.16171, 0.42836, 0.30785],
            [0.39818, 0.26244, 0.34949],
            [0.20792, 0.30157, 0.46047],
            [0.24999, 0.40827, 0.46604],
        ],
    ]

    data = {
        "alternatives": ["A1", "A2", "A3", "A4", "A5", "A6"],
        "criteria": [
            {"name": "Criterion 1", "type": "positive"},
            {"name": "Criterion 2", "type": "positive"},
            {"name": "Criterion 3", "type": "positive"},
        ],
        "dms": [
            {"id": "DM1", "scores": normalized_scores[0]},
            {"id": "DM2", "scores": normalized_scores[1]},
            {"id": "DM3", "scores": normalized_scores[2]},
            {"id": "DM4", "scores": normalized_scores[3]},
            {"id": "DM5", "scores": normalized_scores[4]},
        ],
        "parameters": {"alpha": 0.95, "B": 2, "gamma": 0.5, "L": 0.25},
    }

    return data


def save_json(data: Dict, filename: str):
    """Сохранение данных в JSON файл"""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        raise Exception(f"❌ Ошибка при сохранении файла: {e}")


def load_json(filename: str) -> Optional[Dict]:
    """Загрузка данных из JSON файла"""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise Exception(f"❌ Ошибка при загрузке файла: {e}")


def validate_data(data: Dict) -> bool:
    """Проверка корректности загруженных данных"""
    
    # 1. Проверка наличия обязательных полей
    required_fields = ["alternatives", "criteria", "dms"]
    for field in required_fields:
        if field not in data:
            raise Exception(f"❌ Отсутствует обязательное поле: {field}")

    # 2. Валидация alternatives
    alternatives = data["alternatives"]
    if not isinstance(alternatives, list):
        raise ValueError("❌ Поле 'alternatives' должно быть списком")

    if len(alternatives) == 0:
        raise ValueError("❌ Список альтернатив не может быть пустым")

    for i, alt in enumerate(alternatives):
        if not isinstance(alt, str):
            raise ValueError(f"❌ Альтернатива #{i + 1} должна быть строкой")
        if not alt.strip():
            raise ValueError(f"❌ Альтернатива #{i + 1} не может быть пустой строкой")

    num_alternatives = len(alternatives)

    # 3. Валидация criteria
    criteria = data["criteria"]
    if not isinstance(criteria, list):
        raise ValueError("❌ Поле 'criteria' должно быть списком")

    if len(criteria) == 0:
        raise ValueError("❌ Список критериев не может быть пустым")

    for i, criterion in enumerate(criteria):
        if not isinstance(criterion, dict):
            raise ValueError(f"❌ Критерий #{i + 1} должен быть словарем")

        if "name" not in criterion:
            raise ValueError(f"❌ Критерий #{i + 1} должен иметь поле 'name'")
        if not isinstance(criterion["name"], str):
            raise ValueError(f"❌ Имя критерия #{i + 1} должно быть строкой")

        if "type" not in criterion:
            raise ValueError(f"❌ Критерий #{i + 1} должен иметь поле 'type'")
        if criterion["type"] not in ["positive", "negative"]:
            raise ValueError(
                f"❌ Тип критерия #{i + 1} должен быть 'positive' или 'negative'"
            )

    num_criteria = len(criteria)

    # 4. Валидация dms
    dms = data["dms"]
    if not isinstance(dms, list):
        raise ValueError("❌ Поле 'dms' должно быть списком")

    if len(dms) == 0:
        raise ValueError("❌ Список DM не может быть пустым")

    dm_ids = set()
    for i, dm in enumerate(dms):
        if not isinstance(dm, dict):
            raise ValueError(f"❌ DM #{i + 1} должен быть словарем")

        # Проверка id
        if "id" not in dm:
            raise ValueError(f"❌ DM #{i + 1} должен иметь поле 'id'")

        dm_id = dm["id"]
        if not isinstance(dm_id, str):
            raise ValueError(f"❌ ID DM #{i + 1} должен быть строкой")
        if not dm_id.strip():
            raise ValueError(f"❌ ID DM #{i + 1} не может быть пустой строкой")

        if dm_id in dm_ids:
            raise ValueError(f"❌ Дублирующийся ID DM: {dm_id}")
        dm_ids.add(dm_id)

        # Проверка scores
        if "scores" not in dm:
            raise ValueError(f"❌ DM #{i + 1} должен иметь поле 'scores'")

        scores = dm["scores"]
        if not isinstance(scores, list):
            raise ValueError(f"❌ Scores DM #{i + 1} должен быть списком")

        if len(scores) != num_alternatives:
            raise ValueError(
                f"❌ Количество строк в scores DM '{dm_id}' должно быть равно "
                f"количеству альтернатив ({num_alternatives}), получено {len(scores)}"
            )

        for alt_idx, alt_scores in enumerate(scores):
            if not isinstance(alt_scores, list):
                raise ValueError(
                    f"❌ Scores для альтернативы #{alt_idx + 1} у DM '{dm_id}' должен быть списком"
                )

            if len(alt_scores) != num_criteria:
                raise ValueError(
                    f"❌ Количество оценок для альтернативы '{alternatives[alt_idx]}' у DM '{dm_id}' "
                    f"должно быть равно количеству критериев ({num_criteria}), получено {len(alt_scores)}"
                )

            for crit_idx, score in enumerate(alt_scores):
                if not isinstance(score, (int, float)):
                    raise ValueError(
                        f"❌ Оценка для альтернативы '{alternatives[alt_idx]}' "
                        f"по критерию '{criteria[crit_idx]['name']}' у DM '{dm_id}' "
                        f"должна быть числом, получено {type(score)}"
                    )

    # 5. Валидация parameters (опционально)
    if "parameters" in data:
        parameters = data["parameters"]
        if not isinstance(parameters, dict):
            raise ValueError("❌ Поле 'parameters' должно быть словарем")

        # Проверка конкретных параметров
        if "alpha" in parameters:
            alpha = parameters["alpha"]
            if not isinstance(alpha, (int, float)):
                raise ValueError("❌ Параметр 'alpha' должен быть числом")
            if not (0 <= alpha <= 1):
                raise ValueError("❌ Параметр 'alpha' должен быть в диапазоне [0, 1]")

        if "B" in parameters:
            B = parameters["B"]
            if not isinstance(B, (int, float)):
                raise ValueError("❌ Параметр 'B' должен быть числом")
            if B <= 0:
                raise ValueError("❌ Параметр 'B' должен быть положительным числом")

        if "gamma" in parameters:
            gamma = parameters["gamma"]
            if not isinstance(gamma, (int, float)):
                raise ValueError("❌ Параметр 'gamma' должен быть числом")
            if not (0 <= gamma <= 1):
                raise ValueError("❌ Параметр 'gamma' должен быть в диапазоне [0, 1]")

    return True
