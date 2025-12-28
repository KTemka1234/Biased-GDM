import json
import os
from typing import Optional, Dict
from matplotlib.lines import Line2D
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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


def visualize_global_biases(json_file_path: str, B: Optional[int], show_plots: bool = False) -> None:
    """
    Визуализация глобальной предвзятости экспертов и их доверительных интервалов

    Args:
        json_file_path (str): Путь к JSON файлу с результатами и параметрами
        B (int): Пороговое значение для определения предвзятости
    """

    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    global_biasedness = results["global_biasedness_index"]
    biased_indices = results["biased_indices"]
    CIs = results["CIs"]

    if B is None:
        B = data["parameters"]["B"]

    # Создание DataFrame для удобства
    experts = list(global_biasedness.keys())
    is_biased = [expert in biased_indices for expert in experts]
    df = pd.DataFrame(
        {
            "Expert": experts,
            "Global_Bias": [global_biasedness[e] for e in experts],
            "Is_Biased": is_biased,
            "CI_Mean": [CIs[e]["mean"] for e in experts],
            "CI_Lower": [CIs[e]["lower_bound"] for e in experts],
            "CI_Upper": [CIs[e]["upper_bound"] for e in experts],
            "CI_Length": [CIs[e]["length"] for e in experts],
        }
    )

    df = df.sort_values("Global_Bias", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. График индексов предвзятости
    ax1 = axes[0]

    colors = ["#ff6b6b" if biased else "#51a351" for biased in df["Is_Biased"]]

    # Столбчатая диаграмма
    bars = ax1.bar(
        df["Expert"], df["Global_Bias"], color=colors, edgecolor="black", alpha=0.8
    )

    # Линия порога
    threshold_value = B
    ax1.axhline(
        y=threshold_value,
        color="green",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"Threshold (B={B})",
    )

    ax1.set_title(
        "Global bias indices of experts",
        fontsize=12,
        pad=15,
    )
    ax1.set_xlabel("Expert", fontsize=12)
    ax1.set_ylabel("Global bias index", fontsize=12)
    ax1.tick_params(axis="x", rotation=45, labelsize=10)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.legend(fontsize=8)

    # Добавляем значения на столбцы
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # 2. График доверительных интервалов
    ax2 = axes[1]

    # Отображаем доверительные интервалы
    y_positions = range(len(df))

    for idx, row in enumerate(df.itertuples()):
        # Цвет линии в зависимости от предвзятости
        line_color = "#ff6b6b" if row.Is_Biased else "#51a351"
        line_width = 2.5 if row.Is_Biased else 2.0

        # Линия доверительного интервала
        ax2.hlines(
            y=idx,
            xmin=row.CI_Lower,
            xmax=row.CI_Upper,
            colors=line_color,
            linewidth=line_width,
            alpha=0.8,
        )

        # Точка - среднее значение
        ax2.plot(
            row.CI_Mean,
            idx,
            "o",
            color=line_color,
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=1,
        )

        # Подпись эксперта с длиной интервала
        expert_name = row.Expert
        ax2.text(
            row.CI_Upper + 0.01,
            idx,
            f"{expert_name} ({row.CI_Length:.3f})",
            va="center",
            fontsize=8,
        )

    # Настройки графика
    ax2.set_title("Confidence intervals of experts", fontsize=12, pad=15)
    ax2.set_xlabel("Value", fontsize=12)
    ax2.set_ylabel("Expert", fontsize=12)
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels([row.Expert for row in df.itertuples()], fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.set_xlim(df["CI_Lower"].min() - 0.1, df["CI_Upper"].max() + 0.2)
    ax2.invert_yaxis()  # Чтобы первый эксперт был сверху

    # Легенда
    legend_elements = [
        Line2D(
            [0], [0], color="#ff6b6b", linewidth=2.5, label="Biased expert"
        ),
        Line2D(
            [0], [0], color="#51a351", linewidth=2, label="Unbiased expert"
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=8,
            label="Mean value",
        )
    ]

    ax2.legend(handles=legend_elements, fontsize=8)

    # Общий заголовок
    plt.suptitle(
        f"Global bias analysis (total experts: {len(experts)}, biased: {len(biased_indices)})",
        fontsize=14,
        fontweight="bold",
    )

    # Настраиваем расположение
    plt.tight_layout()

    plt.show() if show_plots else None

    os.makedirs("./images", exist_ok=True)
    file_name = os.path.basename(json_file_path)
    fig.savefig(f"./images/Global_{file_name.split('.')[0]}.png")


def visualize_local_biases(
    json_file_path: str,
    alternative_index: Optional[int] = None,
    criterion_index: Optional[int] = None,
    L_threshold: Optional[float] = None,
    show_plots: bool = False
):
    """
    Визуализация локальной предвзятости экспертов

    Args:
        json_file_path (str): Путь к JSON файлу с результатами
        alternative_index (int, optional): Индекс альтернативы для анализа. Если None, выбирается случайный
        criterion_index (int, optional): Индекс критерия для анализа (0-4). Если None, выбирается случайный
        L_threshold (float, optional): Уровень допустимого отклонения (L). Если None, берется из JSON-файла
    """

    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    parameters = data["parameters"]

    if L_threshold is None:
        L_threshold = parameters["L"]

    normalized_scores = results["normalized_scores"]
    consensus_scores = results["consensus_scores"]
    local_biasedness = results["local_biasedness_index"]
    biased_indices = results["biased_indices"]

    # Определение индексов для анализа, если не заданы
    if alternative_index is None:
        alternative_index = int(np.random.randint(0, len(consensus_scores)))
    if criterion_index is None:
        criterion_index = int(np.random.randint(0, len(normalized_scores.get(list(normalized_scores.keys())[0])[0])))

    # Названия альтернатив и критериев
    alternatives = list(consensus_scores.keys())
    criteria = [f"Criteria {i + 1}" for i in range(len(normalized_scores.get(list(normalized_scores.keys())[0])[0]))]
    alternative_name = alternatives[alternative_index]

    # Получение консенсусной оценки для выбранной альтернативы и критерия
    consensus_value = consensus_scores[alternative_name][criterion_index]

    # Получение оценок экспертов для выбранной альтернативы и критерия
    expert_scores = {}
    for expert, scores in normalized_scores.items():
        expert_scores[expert] = scores[alternative_index][criterion_index]

    # Создаем DataFrame для удобства
    experts = list(expert_scores.keys())
    df_local = pd.DataFrame(
        {
            "Expert": experts,
            "Score": [expert_scores[e] for e in experts],
            "Local_Bias": [local_biasedness[e] for e in experts],
            "Is_Biased": [e in biased_indices for e in experts],
            "Deviation": [expert_scores[e] - consensus_value for e in experts],
            "Abs_Deviation": [abs(expert_scores[e] - consensus_value) for e in experts],
        }
    )

    # Сортируем по отклонению от консенсуса
    df_local = df_local.sort_values("Abs_Deviation", ascending=False)

    # Создаем фигуру с двумя графиками
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. График отклонений экспертов от консенсуса
    ax1 = axes[0]

    # Консенсусная линия
    ax1.axhline(
        y=consensus_value,
        color="blue",
        linewidth=2,
        linestyle="--",
        alpha=0.7
    )

    for idx, row in enumerate(df_local.itertuples()):
        point_color = "#ff6b6b" if row.Is_Biased else "#51a351"
        point_size = 80

        # Отображение точки
        ax1.scatter(
            idx,
            row.Score,
            color=point_color,
            s=point_size,
            edgecolor="black",
            linewidth=1.5,
            zorder=5,
        )

        ax1.plot(
            [idx, idx],
            [consensus_value, row.Score],
            color=point_color,
            alpha=0.8,
            linewidth=1.5,
        )

        # Подпись эксперта
        deviation_str = f"{row.Deviation:+.2f}"
        ax1.text(
            idx,
            row.Score + (0.03 if row.Deviation >= 0 else -0.04),
            f"{row.Expert}\n{deviation_str}",
            ha="center",
            va="bottom" if row.Deviation >= 0 else "top",
            fontsize=8
        )

    # Настройки графика 1
    ax1.set_title(
        f"Local bias: {alternative_name}, {criteria[criterion_index]}",
        fontsize=12,
        pad=15,
    )
    ax1.set_xlabel("Experts", fontsize=12)
    ax1.set_ylabel("Normalized score", fontsize=12)
    ax1.set_xticks([])
    ax1.grid(True, alpha=0.5, linestyle="--")
    ax1.set_ylim(-0.2, 1.2)

    # Легенда для графика 1
    legend_elements = [
        # Консенсусная линия уже добавлена через label
        Line2D(
            [0],
            [0],
            color="blue",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Consensus score: {consensus_value:.3f}",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            markersize=8,
            label="Biased expert",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="green",
            markersize=8,
            label="Unbiased expert",
        ),
    ]

    ax1.legend(handles=legend_elements, fontsize=8)

    # 2. График индексов локальной предвзятости
    ax2 = axes[1]

    # Сортируем по локальной предвзятости
    df_local_bias = pd.DataFrame(
        {
            "Expert": experts,
            "Local_Bias": [local_biasedness[e] for e in experts],
            "Is_Biased": [e in biased_indices for e in experts],
        }
    ).sort_values("Local_Bias", ascending=False)

    # Цвета для столбцов
    colors = [
        "#ff6b6b" if biased else "#51a351" for biased in df_local_bias["Is_Biased"]
    ]

    # Столбчатая диаграмма
    bars = ax2.bar(
        df_local_bias["Expert"],
        df_local_bias["Local_Bias"],
        color=colors,
        edgecolor="black",
        alpha=0.8,
    )

    # Линия порога
    ax2.axhline(
        y=L_threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"Threshold (L): {L_threshold:.2f}",
    )

    # Настройки графика 2
    ax2.set_title(
        "Local bias indeces of experts",
        fontsize=12,
        pad=15
    )
    ax2.set_xlabel("Expert", fontsize=12)
    ax2.set_ylabel("Local bias index", fontsize=12)
    ax2.tick_params(axis="x", rotation=45, labelsize=10)
    ax2.grid(True, alpha=0.5, linestyle="--")
    ax2.set_ylim(0, max(df_local_bias["Local_Bias"]) + 0.1)
    ax2.legend(fontsize=8)

    # Добавляем значения на столбцы
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

    # Легенда для графика 1
    legend_elements = [
        # Консенсусная линия уже добавлена через label
        Line2D(
            [0],
            [0],
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Threshold (L): {L_threshold:.2f}",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            markersize=8,
            label="Biased expert",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="green",
            markersize=8,
            label="Unbiased expert",
        ),
    ]

    ax2.legend(handles=legend_elements, fontsize=8)

    # Общий заголовок
    total_experts = len(experts)
    biased_count = df_local["Is_Biased"].sum()
    plt.suptitle(
        f"Local bias analysis (total experts: {total_experts}, biased: {biased_count})",
        fontsize=14,
        fontweight="bold"
    )

    # Настраиваем расположение
    plt.tight_layout()

    plt.show() if show_plots else None
    
    os.makedirs("./images", exist_ok=True)
    file_name = os.path.basename(json_file_path)
    fig.savefig(f"./images/Local_{file_name.split('.')[0]}_A{alternative_index+1}_C{criterion_index+1}.png")
