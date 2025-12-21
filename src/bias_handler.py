from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from scipy.stats import t


class BiasDMHandlerContext:
    """
    Контекст для обработки предвзятости Decision Makers (DM).

    Содержит данные и параметры для алгоритмов определения предвзятости,
    а также делегирует выполнение конкретному обработчику.

    Attributes:
        _handler (BiasDMHandler): Текущий обработчик предвзятости
        data (dict): Входные данные с DM и критериями
        alpha (float): Уровень доверия для доверительных интервалов (по умолчанию 0.95)
        B_TH (Optional[int]): Пороговое значение индекса предвзятости для исключения DM
        gamma (Optional[float]): Коэффициент доли веса конкретного DM в общем весе в MABM/SABM методах
    """
    def __init__(
        self,
        handler: BiasDMHandler,
        data: dict,
        alpha: float = 0.95,
        B_TH: Optional[int] = None,
        gamma: Optional[float] = None,
    ) -> None:
        """
        Инициализация контекста обработки предвзятости.

        Args:
            handler: Обработчик предвзятости
            data: Входные данные
            alpha: Уровень доверия (0 < alpha < 1)
            B_TH: Пороговое значение индекса предвзятости
            gamma: Коэффициент доли веса конкретного DM в общем весе
        """
        self._handler = handler
        self.data = data
        self.alpha = alpha
        self.B_TH = B_TH
        self.gamma = gamma

    @property
    def handler(self) -> BiasDMHandler:
        return self._handler

    @handler.setter
    def handler(self, handler: BiasDMHandler) -> None:
        self._handler = handler

    def handle(self, normalized: bool = False) -> dict:
        """
        Метод делегирования обработки данных заданному обработчику.

        Args:
            normalized (bool): Флаг, указывающий нормализованы ли входные данные

        Returns:
            dict: Результаты обработки
        """
        return self._handler.handle(self, normalized)

    def normalize_scores(self, scores: list, criteria_types: list) -> np.ndarray:
        """
        Нормализация оценок по критериям.

        Для положительных критериев используется нормализация (x-min)/(max-min),
        для отрицательных - (max-x)/(max-min).

        Args:
            scores (list): Массив оценок размерности (I, J, K), где:
                   I - количество DM, J - альтернатив, K - критериев
            criteria_types (list): Список типов критериев ("positive" или "negative")

        Returns:
            np.ndarray: Нормализованный массив оценок той же размерности
        """
        normalized_scores = []
        min_vals = np.min(scores, axis=(0, 1))
        max_vals = np.max(scores, axis=(0, 1))

        for dm_scores in scores:
            dm_normalized = []
            for alt_scores in dm_scores:
                norm_alt = []
                for k, score in enumerate(alt_scores):
                    if max_vals[k] - min_vals[k] == 0:
                        norm_val = 0
                    elif criteria_types[k] == "positive":
                        norm_val = (score - min_vals[k]) / (max_vals[k] - min_vals[k])
                    else:
                        norm_val = (max_vals[k] - score) / (max_vals[k] - min_vals[k])
                    norm_alt.append(norm_val)
                dm_normalized.append(norm_alt)
            normalized_scores.append(dm_normalized)

        return np.array(normalized_scores)

    def calc_CIs(self, normalized_scores: list) -> list:
        """
        Расчет доверительных интервалов (confidence interval - CI) для каждого DM.

        Args:
            normalized_scores (list): Нормализованные оценки размерности (I, J, K)

        Returns:
            list: Список словарей с параметрами доверительных интервалов для каждого DM:
            - mean: Среднее значение
            - std: Стандартное отклонение
            - LB: Нижняя граница интервала
            - UB: Верхняя граница интервала
            - length: Длина интервала
        """
        I, J, K = normalized_scores.shape
        N = J * K

        # Преобразуем матрицу: каждый DM - строка из всех оценок
        flattened_scores = normalized_scores.reshape(I, -1)

        means = np.mean(flattened_scores, axis=1)
        stds = np.std(flattened_scores, axis=1, ddof=1)

        t_value = t.ppf(self.alpha, N - 1)

        CIs = []
        for i in range(I):
            margin = t_value * (stds[i] / np.sqrt(N))
            LB = means[i] - margin
            UB = means[i] + margin
            CI_length = UB - LB
            CIs.append(
                {
                    "mean": means[i],
                    "std": stds[i],
                    "LB": LB,
                    "UB": UB,
                    "length": CI_length,
                }
            )

        return CIs

    def calc_biasedness_index(self, CIs: list) -> np.ndarray:
        """
        Расчет индекса предвзятости для каждого DM.

        Индекс предвзятости - количество других DM, чьи доверительные интервалы
        пересекаются с интервалом текущего DM.

        Args:
            CIs (list): Список доверительных интервалов для всех DM

        Returns:
            np.ndarray: Массив индексов предвзятости B_i для каждого DM
        """
        I = len(CIs)
        B_i = np.zeros(I)

        for i in range(I):
            count = 0
            for j in range(I):
                if i != j:
                    if CIs[i]["UB"] >= CIs[j]["LB"] and CIs[i]["LB"] <= CIs[j]["UB"]:
                        count += 1
            B_i[i] = count

        return B_i

    def eliminate_biased_dms(self, normalized_scores: list, CIs: list, B_i: list) -> tuple:
        """
        Исключение предвзятых DM на основе порога B_TH.

        Args:
            normalized_scores (list): Нормализованные оценки всех DM
            CIs (list): Доверительные интервалы всех DM
            B_i (list): Индексы предвзятости всех DM

        Returns:
            Tuple: unbiased_scores, unbiased_CIs, unbiased_indices, biased_indices:
            - unbiased_scores: Оценки непредвзятых DM
            - unbiased_CIs: Доверительные интервалы непредвзятых DM
            - unbiased_indices: Индексы непредвзятых DM
            - biased_indices: Индексы предвзятых DM
        """
        if self.B_TH is None:
            self.B_TH = len(CIs) - 1  # максимальное значение по умолчанию

        unbiased_indices = [i for i, b in enumerate(B_i) if b >= self.B_TH]
        biased_indices = [i for i, b in enumerate(B_i) if b < self.B_TH]

        unbiased_scores = normalized_scores[unbiased_indices]
        unbiased_CIs = [CIs[i] for i in unbiased_indices]

        return unbiased_scores, unbiased_CIs, unbiased_indices, biased_indices

    def calc_overlap_ratio(self, CIs: list) -> tuple:
        """
        Расчет коэффициента перекрытия доверительных интервалов.

        Args:
            CIs (list): Список доверительных интервалов

        Returns:
            Tuple: overlap_matrix, total_overlap, O_i, O_tilde:
            - overlap_matrix: Матрица попарных перекрытий
            - total_overlap: Суммарные перекрытия для каждого DM
            - O_i: Перекрытия без учета самоперекрытий
            - O_tilde: Нормализованные коэффициенты перекрытия
        """
        I = len(CIs)
        overlap_matrix = np.zeros((I, I))

        # Расчет попарных перекрытий
        for i in range(I):
            for j in range(I):
                if i != j:
                    overlap = max(
                        0,
                        min(CIs[i]["UB"], CIs[j]["UB"])
                        - max(CIs[i]["LB"], CIs[j]["LB"]),
                    )
                    overlap_matrix[i, j] = overlap
                else:
                    overlap_matrix[i, j] = CIs[i]["length"]

        # Расчет общего перекрытия для каждого DM
        O_i = []
        M_i = []
        total_overlap = []
        for i in range(I):
            total_overlap.append(np.sum(overlap_matrix[i, :]))
            # Исключаем самоперекрытие (диагональный элемент)
            O_i.append(total_overlap[i] - overlap_matrix[i, i])

            # Максимально возможное перекрытие
            M_i_value = (I - 1) * CIs[i]["length"]
            M_i.append(M_i_value)

        # Коэффициент перекрытия
        O_tilde = [O_i[i] / M_i[i] if M_i[i] > 0 else 0 for i in range(I)]

        return overlap_matrix, total_overlap, O_i, O_tilde

    def calc_relative_CI(self, unbiased_CIs: list, unbiased_scores: np.ndarray) -> list:
        """
        Расчет относительных доверительных интервалов.

        Относительный CI - это отношение длины CI DM к длине общего CI всех DM.

        Args:
            unbiased_CIs (list): Доверительные интервалы непредвзятых DM
            unbiased_scores (np.ndarray): Оценки непредвзятых DM

        Returns:
            list: Список относительных доверительных интервалов CI_tilde
        """
        I, J, K = unbiased_scores.shape

        λ = I * J * K

        # Преобразуем в 1D массив всех оценок оставшихся DM
        flattened_all = unbiased_scores.reshape(-1)

        mean_total = np.mean(flattened_all)
        std_total = np.std(flattened_all, ddof=1)

        t_value_total = t.ppf(self.alpha, λ - 1)
        margin_total = t_value_total * (std_total / np.sqrt(λ))

        LB_total = mean_total - margin_total
        UB_total = mean_total + margin_total
        CI_total_length = UB_total - LB_total

        CI_tilde = []
        for ci in unbiased_CIs:
            relative_ci = ci["length"] / CI_total_length if CI_total_length > 0 else 0
            CI_tilde.append(relative_ci)

        return CI_tilde

    def calc_weights(self, O_tilde: list, CI_tilde: list) -> list:
        """
        Расчет весов для DM на основе коэффициентов перекрытия и относительных CI.

        Args:
            O_tilde (list): Нормализованные коэффициенты перекрытия
            CI_tilde (list): Относительные доверительные интервалы

        Returns:
            list: Список весов для каждого DM
        """
        products = [O_tilde[i] * CI_tilde[i] for i in range(len(O_tilde))]
        sum_products = np.sum(products)

        weights = [p / sum_products if sum_products != 0 else 0 for p in products]
        return weights


class BiasDMHandler(ABC):
    """
    Абстрактный базовый класс для обработчиков предвзятости DM.

    Определяет интерфейс для всех конкретных реализаций алгоритмов
    определения и обработки предвзятости (паттерн "Стратегия").
    """
    @abstractmethod
    def handle(self, context: BiasDMHandlerContext, normalized: bool = False) -> dict:
        """
        Основной метод обработки данных.

        Args:
            context (BiasDMHandlerContext): Контекст с данными и параметрами
            normalized (bool): Флаг нормализованности входных данных

        Returns:
            dict: результаты обработки
        """
        pass

    def _process_data(
        self,
        context: BiasDMHandlerContext,
        normalized: bool = False,
        eliminate: bool = False,
        apply_gamma: bool = False,
    ) -> dict:
        """
        Общая логика обработки для всех классов обработчиков предвзятости.

        Args:
            context (BiasDMHandlerContext): Контекст с данными и параметрами
            normalized (bool): Флаг нормализованности входных данных
            eliminate_biased (bool): Флаг исключения предвзятых DM
            apply_gamma (bool): Флаг применения gamma-коррекции весов

        Returns:
            dict: Словарь с результатами обработки, содержащий:
            - scores: Исходные оценки
            - normalized_scores: Нормализованные оценки
            - CIs: Доверительные интервалы
            - B_i: Индексы предвзятости
            - biased_indices: Индексы предвзятых DM
            - unbiased_indices: Индексы непредвзятых DM
            - overlap_matrix: Матрица перекрытий
            - total_overlap: Суммарные перекрытия
            - O_i: Абсолютные перекрытия
            - O_tilde: Нормализованные перекрытия
            - CI_tilde: Относительные CI
            - final_weights: Финальные веса DM
        """
        results = {
            "scores": [],
            "normalized_scores": [],
            "CIs": [],
            "B_i": [],
            "biased_indices": [],
            "unbiased_indices": [],
            "overlap_matrix": [],
            "total_overlap": [],
            "O_i": [],
            "O_tilde": [],
            "CI_tilde": [],
            "final_weights": [],
        }

        # Извлечение данных
        dms_data = context.data["dms"]
        criteria_types = [criterion["type"] for criterion in context.data["criteria"]]

        # Преобразование в numpy array
        scores = np.array([dm["scores"] for dm in dms_data])
        results["scores"] = scores

        # 1. Нормализация оценок
        if not normalized:
            normalized_scores = context.normalize_scores(scores, criteria_types)
        else:
            normalized_scores = scores
        results["normalized_scores"] = normalized_scores

        # 2. Расчет доверительных интервалов
        CIs = context.calc_CIs(normalized_scores)
        results["CIs"] = CIs

        # 3. Расчет индекса предвзятости
        B_i = context.calc_biasedness_index(CIs)
        results["B_i"] = B_i

        # 4. Исключение предвзятых DM (если требуется)
        if eliminate:
            unbiased_scores, unbiased_CIs, unbiased_indices, biased_indices = (
                context.eliminate_biased_dms(normalized_scores, CIs, B_i)
            )
            results["biased_indices"] = biased_indices
            results["unbiased_indices"] = unbiased_indices
        else:
            unbiased_scores = normalized_scores
            unbiased_CIs = CIs
            unbiased_indices = list(range(len(dms_data)))
            results["biased_indices"] = []
            results["unbiased_indices"] = unbiased_indices

        if len(unbiased_indices) == 0:
            results["final_weights"] = np.zeros(len(dms_data))
            return results

        # 5. Расчет коэффициента перекрытия для оставшихся DM
        overlap_matrix, total_overlap, O_i, O_tilde = context.calc_overlap_ratio(
            unbiased_CIs
        )
        results["overlap_matrix"] = overlap_matrix
        results["total_overlap"] = total_overlap
        results["O_i"] = O_i
        results["O_tilde"] = O_tilde

        # 6. Расчет относительных CI
        CI_tilde = context.calc_relative_CI(unbiased_CIs, unbiased_scores)
        results["CI_tilde"] = CI_tilde

        # 7. Расчет весов
        weights = context.calc_weights(O_tilde, CI_tilde)

        # Применение gamma коррекции (если требуется)
        if apply_gamma:
            if context.gamma is None or context.gamma < 0 or context.gamma > 1:
                raise ValueError("Gamma must be between 0 and 1")
            weights = self._apply_gamma_correction(
                weights, len(results["unbiased_indices"]), context.gamma
            )

        # Сопоставление весов с исходными индексами DM
        final_weights = np.zeros(len(dms_data))
        for i, idx in enumerate(unbiased_indices):
            final_weights[idx] = weights[i]
        results["final_weights"] = final_weights

        return results

    def _apply_gamma_correction(
        self, weights: list, num_unbiased: int, gamma: float
    ) -> list:
        """
        Применение gamma-коррекции к весам DM.

        Корректирует веса по формуле: gamma/n + (1-gamma)*original_weight

        Args:
            weights (list): Исходные веса DM
            num_unbiased (int): Количество непредвзятых DM
            gamma (float): Коэффициент коррекции (0 <= gamma <= 1)

        Returns:
            list: Список скорректированных весов
        """
        return [gamma / num_unbiased + (1 - gamma) * weight for weight in weights]


class EABMHandler(BiasDMHandler):
    """
    Extreme Anti-Biased Method (EABM) обработчик.

    Исключает предвзятых DM и рассчитывает веса только для оставшихся.
    Не применяет gamma-коррекцию.
    """
    def handle(self, context: BiasDMHandlerContext, normalized: bool = False) -> dict:
        """
        EABM метод.

        Args:
            context (BiasDMHandlerContext): Контекст с данными и параметрами
            normalized (bool): Флаг нормализованности входных данных

        Returns:
            dict: Словарь с результатами обработки
        """
        return self._process_data(context, normalized, True, False)


class MABMHandler(BiasDMHandler):
    """
    Moderate Anti-Biased Method (MABM) обработчик.

    Исключает предвзятых DM и применяет gamma-коррекцию к весам
    оставшихся непредвзятых DM.
    """
    def handle(self, context: BiasDMHandlerContext, normalized: bool = False) -> dict:
        """
        MABM метод.

        Args:
            context (BiasDMHandlerContext): Контекст с данными и параметрами
            normalized (bool): Флаг нормализованности входных данных

        Returns:
            dict: Словарь с результатами обработки
        """
        return self._process_data(context, normalized, True, True)


class SABMHandler(BiasDMHandler):
    """
    Soft Anti-Biased Method (SABM) обработчик.

    Не исключает предвзятых DM, но применяет gamma-коррекцию ко всем DM,
    смягчая влияние предвзятости.
    """
    def handle(self, context: BiasDMHandlerContext, normalized: bool = False) -> dict:
        """
        SABM метод.

        Args:
            context (BiasDMHandlerContext): Контекст с данными и параметрами
            normalized (bool): Флаг нормализованности входных данных

        Returns:
            dict: Словарь с результатами обработки
        """
        return self._process_data(context, normalized, False, True)
