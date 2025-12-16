from __future__ import annotations
from typing import Optional, Tuple
from bias_handler import BiasDMHandler, BiasDMHandlerContext
import numpy as np


class EnhancedBiasDMHandlerContext(BiasDMHandlerContext):
    """
    Расширенный контекст для обработки предвзятости Decision Makers (DM).

    Добавлена поддержка определения локальной предвзятости.
    """

    def __init__(
        self,
        handler: EnhancedBiasDMHandler,
        data: dict,
        alpha: float = 0.95,
        B_TH: Optional[int] = None,
        gamma: Optional[float] = None,
        L_TH: Optional[float] = None,
    ) -> None:
        """
        Инициализация расширенного контекста обработки предвзятости.

        Args:
            handler: Обработчик предвзятости
            data: Входные данные
            alpha: Уровень доверия (0 < alpha < 1)
            B_TH: Пороговое значение индекса глобальной предвзятости
            gamma: Коэффициент доли веса конкретного DM в общем весе
            L_TH: Пороговое значение индекса локальной предвзятости
        """
        super().__init__(handler, data, alpha, B_TH, gamma)
        self.L_TH = L_TH

    def calc_consensus_scores(self, normalized_scores: np.ndarray) -> np.ndarray:
        """
        Расчет консенсусных оценок для каждого DM.

        Args:
            normalized_scores (np.ndarray): Нормализованные оценки размерности (I, J, K)

        Returns:
            np.ndarray: Консенсусные оценки размерности (J, K)
        """
        return np.mean(normalized_scores, axis=0)

    def calc_local_bias_index(
        self, normalized_scores: np.ndarray, consensus_scores: np.ndarray
    ) -> np.ndarray:
        """
        Расчет индекса локальной предвзятости для каждого DM.

        Args:
            normalized_scores (np.ndarray): Нормализованные оценки размерности (I, J, K)
            consensus_scores (np.ndarray): Консенсусные оценки размерности (J, K)

        Returns:
            np.ndarray: Массив индексов целевой предвзятости для каждого DM
        """
        I, J, K = normalized_scores.shape
        L_i = np.zeros(I)

        for i in range(I):
            deviations = []

            # Рассчитываем отклонения для каждой оценки
            for j in range(J):
                for k in range(K):
                    deviation = normalized_scores[i, j, k] - consensus_scores[j, k]
                    deviations.append(deviation)

            deviations = np.array(deviations)

            # Стандартное отклонение отклонений
            if len(deviations) > 1:
                L_i[i] = np.std(deviations, ddof=1)
            else:
                L_i[i] = 0

        return L_i

    def eliminate_local_biased_dms(self, L_i: np.ndarray) -> Tuple[list, list]:
        """
        Исключение локальных предвзятых DM.

        Args:
            targeted_bias_indices (np.ndarray): Индексы целевой предвзятости

        Returns:
            Tuple[list, list]: (biased_indices, unbiased_indices)
        """
        biased_indices = []
        unbiased_indices = []

        for i, bias_index in enumerate(L_i):
            if bias_index > self.L_TH:
                biased_indices.append(i)
            else:
                unbiased_indices.append(i)

        return biased_indices, unbiased_indices


class EnhancedBiasDMHandler(BiasDMHandler):
    """
    Расширенный абстрактный базовый класс для обработчиков предвзятости DM.

    Добавлена поддержка детектирования локальной (целевой) предвзятости.
    """

    def _process_data_enhanced(
        self,
        context: EnhancedBiasDMHandlerContext,
        normalized: bool = False,
        eliminate: bool = False,
        apply_gamma: bool = False,
    ) -> dict:
        """
        Расширенная логика обработки с поддержкой целевой предвзятости.

        Args:
            context (EnhancedBiasDMHandlerContext): Контекст с данными и параметрами
            normalized (bool): Флаг нормализованности входных данных
            eliminate (bool): Флаг исключения предвзятых DM
            apply_gamma (bool): Флаг применения gamma-коррекции весов

        Returns:
            dict: Словарь с результатами обработки
        """
        results = {
            "scores": [],
            "normalized_scores": [],
            "consensus_scores": [],
            "CIs": [],
            "B_i": [],
            "L_i": [],
            "global_biased_indices": [],
            "local_biased_indices": [],
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

        # 2. Расчет консенсусных оценок
        consensus_scores = context.calc_consensus_scores(normalized_scores)
        results["consensus_scores"] = consensus_scores

        # 3. Расчет индекса локальной предвзятости
        L_i = context.calc_local_bias_index(normalized_scores, consensus_scores)
        results["L_i"] = L_i

        # 4. Расчет доверительных интервалов
        CIs = context.calc_CIs(normalized_scores)
        results["CIs"] = CIs

        # 5. Расчет индекса глобальной предвзятости
        B_i = context.calc_biasedness_index(CIs)
        results["B_i"] = B_i

        # 6. Исключение предвзятых DM (если требуется)
        if eliminate:
            unbiased_scores, unbiased_CIs, unbiased_indices, biased_indices = (
                context.eliminate_biased_dms(normalized_scores, CIs, B_i)
            )
            local_biased_indices, local_unbiased_indices = (
                context.eliminate_local_biased_dms(L_i)
            )

            results["global_biased_indices"] = biased_indices
            results["local_biased_indices"] = local_biased_indices

            biased_indices = list(set(biased_indices + local_biased_indices))
            unbiased_indices = [i for i in unbiased_indices if i not in biased_indices]
            unbiased_scores = np.array([unbiased_scores[i] for i in unbiased_indices])
            unbiased_CIs = [unbiased_CIs[i] for i in unbiased_indices]

            results["biased_indices"] = biased_indices
            results["unbiased_indices"] = unbiased_indices
        else:
            unbiased_scores = normalized_scores
            unbiased_CIs = CIs
            unbiased_indices = list(range(len(dms_data)))
            results["biased_indices"] = []
            results["unbiased_indices"] = unbiased_indices

        # 7. Если все DM предвзятые, то возвращаем нулевые веса
        if len(unbiased_indices) == 0:
            results["final_weights"] = np.zeros(len(dms_data))
            return results

        # 8. Расчет коэффициента перекрытия для оставшихся DM
        overlap_matrix, total_overlap, O_i, O_tilde = context.calc_overlap_ratio(
            unbiased_CIs
        )
        results["overlap_matrix"] = overlap_matrix
        results["total_overlap"] = total_overlap
        results["O_i"] = O_i
        results["O_tilde"] = O_tilde

        # 9. Расчет относительных CI
        CI_tilde = context.calc_relative_CI(unbiased_CIs, unbiased_scores)
        results["CI_tilde"] = CI_tilde

        # 10. Расчет весов
        weights = context.calc_weights(O_tilde, CI_tilde)

        # 11. Применение gamma коррекции (если требуется)
        if apply_gamma:
            if context.gamma is None or context.gamma < 0 or context.gamma > 1:
                raise ValueError("Gamma must be between 0 and 1")
            weights = self._apply_gamma_correction(
                weights, len(results["unbiased_indices"]), context.gamma
            )

        # 12. Сопоставление весов с исходными индексами DM
        final_weights = np.zeros(len(dms_data))
        for i, idx in enumerate(unbiased_indices):
            final_weights[idx] = weights[i]
        results["final_weights"] = final_weights

        return results


class EnhancedEABMHandler(EnhancedBiasDMHandler):
    """
    Enhanced Extreme Anti-Biased Method (EABM) обработчик.

    Исключает предвзятых DM и рассчитывает веса с учетом целевой предвзятости.
    """

    def handle(self, context: BiasDMHandlerContext, normalized: bool = False) -> dict:
        """
        Enhanced EABM метод.

        Args:
            context (EnhancedBiasDMHandlerContext): Контекст с данными и параметрами
            normalized (bool): Флаг нормализованности входных данных

        Returns:
            dict: Словарь с результатами обработки
        """
        return self._process_data_enhanced(context, normalized, True, False)


class EnhancedMABMHandler(EnhancedBiasDMHandler):
    """
    Enhanced Moderate Anti-Biased Method (MABM) обработчик.

    Исключает предвзятых DM и применяет gamma-коррекцию с учетом целевой предвзятости.
    """

    def handle(self, context: BiasDMHandlerContext, normalized: bool = False) -> dict:
        """
        Enhanced MABM метод.

        Args:
            context (EnhancedBiasDMHandlerContext): Контекст с данными и параметрами
            normalized (bool): Флаг нормализованности входных данных

        Returns:
            dict: Словарь с результатами обработки
        """
        return self._process_data_enhanced(context, normalized, True, True)


class EnhancedSABMHandler(EnhancedBiasDMHandler):
    """
    Enhanced Soft Anti-Biased Method (SABM) обработчик.

    Не исключает предвзятых DM, но применяет gamma-коррекцию с учетом целевой предвзятости.
    """

    def handle(self, context: BiasDMHandlerContext, normalized: bool = False) -> dict:
        """
        Enhanced SABM метод.

        Args:
            context (EnhancedBiasDMHandlerContext): Контекст с данными и параметрами
            normalized (bool): Флаг нормализованности входных данных

        Returns:
            dict: Словарь с результатами обработки
        """
        return self._process_data_enhanced(context, normalized, False, True)


# Пример использования
def example_usage():
    """
    Пример использования расширенного алгоритма детектирования предвзятости.
    """
    # Пример данных: 5 DM, 4 альтернативы, 3 критерия
    data = {
        "dms": [
            {
                "id": "Alex (DM1)",
                "scores": [[8, 7, 6], [6, 5, 7], [4, 3, 8]],
            },
            {
                "id": "Borya (DM2)",
                "scores": [[7, 8, 7], [5, 6, 6], [3, 4, 7]],
            },
            {
                "id": "Dasha (DM3)",
                "scores": [[7, 6, 8], [5, 4, 7], [3, 2, 9]],
            },
            {
                "id": "Egor (DM4)",
                "scores": [[1, 10, 10], [1, 10, 10], [1, 10, 10]],
            },
        ],
        "criteria": [
            {"name": "Camera", "type": "positive"},
            {"name": "Autonomy", "type": "positive"},
            {"name": "Cost", "type": "positive"},
        ],
    }

    # Создаем обработчики
    eabm_handler = EnhancedEABMHandler()

    # Создаем контекст с параметрами для целевой предвзятости
    context = EnhancedBiasDMHandlerContext(
        handler=eabm_handler, data=data, alpha=0.95, B_TH=2, gamma=0.5, L_TH=0.3
    )

    # Тестируем EABM метод
    print("=== Enhanced EABM Method ===")
    results_eabm = context.handle()

    print("Basic info:")
    print(f"Scores:\n{results_eabm['scores']}")
    print(f"\nNormalized scores:\n{results_eabm['normalized_scores']}")
    print(f"\nConsensus scores:\n{results_eabm['consensus_scores']}")

    print("\n\nBias detection:")
    print(f"Global bias indices: {results_eabm['B_i']}")
    print(f"Local bias indices: {results_eabm['L_i']}")
    print(f"Global bias DMs (indices): {results_eabm['global_biased_indices']}")
    print(f"Local bias DMs (indices): {results_eabm['local_biased_indices']}")
    print(f"Biased DMs (indices): {results_eabm['biased_indices']}")
    print(f"Unbiased DMs (indices): {results_eabm['unbiased_indices']}")
    print(f"Final weights: {results_eabm['final_weights']}")


if __name__ == "__main__":
    example_usage()
