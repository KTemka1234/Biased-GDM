from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import t


class BiasDMHandlerContext:
    def __init__(self, handler: BiasDMHandler, data, alpha=0.95, B_TH=None) -> None:
        self._handler = handler
        self.data = data
        self.alpha = alpha
        self.B_TH = B_TH

    @property
    def handler(self) -> BiasDMHandler:
        return self._handler

    @handler.setter
    def handler(self, handler: BiasDMHandler) -> None:
        self._handler = handler

    def handle(self, data, normalized=False):
        return self._handler.handle(data, normalized)

    def normalize_scores(self, scores, criteria_types):
        """
        Нормализация оценок по критериям
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
                    elif criteria_types[k] == "positive":  # positive criterion
                        norm_val = (score - min_vals[k]) / (max_vals[k] - min_vals[k])
                    else:  # negative criterion
                        norm_val = (max_vals[k] - score) / (max_vals[k] - min_vals[k])
                    norm_alt.append(norm_val)
                dm_normalized.append(norm_alt)
            normalized_scores.append(dm_normalized)

        return np.array(normalized_scores)

    def calc_CI(self, normalized_scores):
        """
        Расчет доверительных интервалов для каждого DM
        """
        I, J, K = normalized_scores.shape
        N = J * K  # общее количество оценок на DM

        # Преобразуем матрицу: каждый DM - строка из всех оценок
        flattened_scores = normalized_scores.reshape(I, -1)

        means = np.mean(flattened_scores, axis=1)
        stds = np.std(flattened_scores, axis=1, ddof=1)

        # t-значение для доверительного интервала
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

    def calc_biasedness_index(self, CIs):
        """
        Расчет индекса предвзятости для каждого DM
        """
        I = len(CIs)
        B_i = np.zeros(I)

        for i in range(I):
            count = 0
            for j in range(I):
                if i != j:
                    # Проверка пересечения доверительных интервалов
                    if CIs[i]["UB"] >= CIs[j]["LB"] and CIs[i]["LB"] <= CIs[j]["UB"]:
                        count += 1
            B_i[i] = count

        return B_i

    def eliminate_biased_dms(self, normalized_scores, CIs, B_i):
        """
        Исключение предвзятых DM
        """
        if self.B_TH is None:
            self.B_TH = len(CIs) - 1  # максимальное значение по умолчанию

        unbiased_indices = [i for i, b in enumerate(B_i) if b >= self.B_TH]
        biased_indices = [i for i, b in enumerate(B_i) if b < self.B_TH]

        unbiased_scores = normalized_scores[unbiased_indices]
        unbiased_CIs = [CIs[i] for i in unbiased_indices]

        return unbiased_scores, unbiased_CIs, unbiased_indices, biased_indices

    def calc_overlap_ratio(self, CIs):
        """
        Расчет коэффициента перекрытия
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

    def calc_relative_CI(self, unbiased_CIs, unbiased_scores):
        """
        Расчет относительного доверительного интервала
        """
        I, J, K = unbiased_scores.shape

        λ = I * J * K  # Общее количество оценок после исключения

        # Преобразуем в 1D массив всех оценок оставшихся DM
        flattened_all = unbiased_scores.reshape(-1)

        # Расчет общего доверительного интервала для всех оставшихся данных
        mean_total = np.mean(flattened_all)
        std_total = np.std(flattened_all, ddof=1)

        # t-значение для общего CI (степени свободы = λ - 1)
        t_value_total = t.ppf(self.alpha, λ - 1)
        margin_total = t_value_total * (std_total / np.sqrt(λ))

        # Длина общего CI
        LB_total = mean_total - margin_total
        UB_total = mean_total + margin_total
        CI_total_length = UB_total - LB_total

        # Относительные CI для каждого оставшегося DM
        CI_tilde = []
        for ci in unbiased_CIs:
            # Длина CI текущего DM делится на длину общего CI
            relative_ci = ci["length"] / CI_total_length if CI_total_length > 0 else 0
            CI_tilde.append(relative_ci)

        return CI_tilde

    def calc_weights(self, O_tilde, CI_tilde):
        """
        Расчет весов для DM
        """
        products = [O_tilde[i] * CI_tilde[i] for i in range(len(O_tilde))]
        sum_products = np.sum(products)

        weights = [
            p / sum_products if sum_products != 0 else 0 for p in products
        ]
        return weights


class BiasDMHandler(ABC):
    @abstractmethod
    def handle(self, context: BiasDMHandlerContext, normalized: bool = False):
        pass

class EABMHandler(BiasDMHandler):
    def handle(self, context: BiasDMHandlerContext, normalized: bool = False):
        """
        Основной метод EABM
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
        CIs = context.calc_CI(normalized_scores)
        results["CIs"] = CIs

        # 3. Расчет индекса предвзятости
        B_i = context.calc_biasedness_index(CIs)
        results["B_i"] = B_i

        # 4. Исключение предвзятых DM
        unbiased_scores, unbiased_CIs, unbiased_indices, biased_indices = (
            context.eliminate_biased_dms(normalized_scores, CIs, B_i)
        )
        results["biased_indices"] = biased_indices
        results["unbiased_indices"] = unbiased_indices

        if len(unbiased_indices) == 0:
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
        CI_tilde = context.calc_relative_CI(
            unbiased_CIs, unbiased_scores
        )
        results["CI_tilde"] = CI_tilde

        # 7. Расчет весов
        weights = context.calc_weights(O_tilde, CI_tilde)

        # Сопоставление весов с исходными индексами DM
        final_weights = [0] * len(dms_data)
        for i, idx in enumerate(unbiased_indices):
            final_weights[idx] = weights[i]
        results["final_weights"] = final_weights

        return results


class MABMHandler(BiasDMHandler):
    def handle(self, context: BiasDMHandlerContext, normalized: bool = False):
        print("MABM")


class SABMHandler(BiasDMHandler):
    def handle(self, context: BiasDMHandlerContext, normalized: bool = False):
        print("SABM")
