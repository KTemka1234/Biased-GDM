import enum
import click
import os
from typing import Optional
import numpy as np

from bias_handler import BiasDMHandlerContext, EABMHandler, MABMHandler, SABMHandler
from utils import create_example_data, load_json, save_json, validate_data


class BiasDMHandlerMethod(enum.Enum):
    EABM = (enum.auto(),)
    MABM = (enum.auto(),)
    SABM = (enum.auto(),)


def print_results(
    results: dict,
    data: dict,
    verbose: bool = False,
) -> None:
    """Красивый вывод результатов"""
    click.echo("\n" + "=" * 60)
    click.echo("🎯 РЕЗУЛЬТАТЫ")
    click.echo("=" * 60)

    if verbose:
        # 1. Входные данные
        click.echo(f"\n📊 МАТРИЦА ВХОДНЫХ ДАННЫХ:")
        scores_array = np.array(results["scores"])
        click.echo(f"Размерность: {scores_array.shape}")
        for i, dm in enumerate(data["dms"]):
            click.echo(f"   {dm['id']}:\n   {results['scores'][i]}")

        # 2. Нормализованные данные
        click.echo(f"\n🔄 НОРМАЛИЗОВАННАЯ МАТРИЦА:")
        for i, dm in enumerate(data["dms"]):
            click.echo(f"   {dm['id']}:\n   {results['normalized_scores'][i]}")

        # 3. Доверительные интервалы
        click.echo(f"\n📐 ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ:")
        for i, dm in enumerate(data["dms"]):
            ci = results["CIs"][i]
            status = "🚫" if i in results["biased_indices"] else "✅"
            click.echo(
                f"   {status} {dm['id']}: [{ci['LB']:.5f}, {ci['UB']:.5f}] | длина: {ci['length']:.5f}"
            )

        # 4. Индексы предвзятости
        click.echo(f"\n📊 ИНДЕКСЫ ПРЕДВЗЯТОСТИ (B_i):")
        for i, dm in enumerate(data["dms"]):
            status = "🚫 ПРЕДВЗЯТ" if i in results["biased_indices"] else "✅ НОРМА"
            click.echo(f"   {dm['id']}: {results['B_i'][i]:.5f} | {status}")

        # 5. Исключенные DM
        click.echo(f"\n🗑️  ИСКЛЮЧЕННЫЕ DM (ПРЕДВЗЯТЫЕ):")
        if results["biased_indices"]:
            for idx in results["biased_indices"]:
                click.echo(f"   ❌ {data['dms'][idx]['id']}")
        else:
            click.echo("   ✅ Нет исключенных DM")

        # 6. Матрица перекрытий (только для неуbiased DM)
        if "overlap_matrix" in results and len(results["overlap_matrix"]) > 0:
            click.echo(f"\n🔄 МАТРИЦА ПЕРЕКРЫТИЙ:")
            overlap_matrix = np.array(results["overlap_matrix"])
            click.echo(f"Размерность: {overlap_matrix.shape}")

            # Показываем только неуbiased DM в заголовках
            unbiased_dms = [
                dm
                for i, dm in enumerate(data["dms"])
                if i in results["unbiased_indices"]
            ]

            if unbiased_dms:
                # Заголовки
                header = "  " + "".join([f"{dm['id']:>10}" for dm in unbiased_dms])
                click.echo(header)
                click.echo("-" * (len(unbiased_dms) * 10 + 5))

                # Данные
                for i, dm in enumerate(unbiased_dms):
                    row = f"{dm['id']:>3} "
                    for j in range(len(unbiased_dms)):
                        row += f"{overlap_matrix[i][j]:>10.5f}"
                    click.echo(row)

        # 7. Индивидуальные перекрытия O_i
        if "O_i" in results and len(results["O_i"]) > 0:
            click.echo(f"\n🎯 ИНДИВИДУАЛЬНЫЕ ПЕРЕКРЫТИЯ (O_i):")
            unbiased_dms = [
                dm
                for i, dm in enumerate(data["dms"])
                if i not in results["biased_indices"]
            ]
            for i, dm in enumerate(unbiased_dms):
                click.echo(f"   {dm['id']}: {results['O_i'][i]:.5f}")

        # 8. Нормализованные перекрытия O_tilde
        if "O_tilde" in results and len(results["O_tilde"]) > 0:
            click.echo(f"\n📏 НОРМАЛИЗОВАННЫЕ ПЕРЕКРЫТИЯ (O_tilde):")
            unbiased_dms = [
                dm
                for i, dm in enumerate(data["dms"])
                if i not in results["biased_indices"]
            ]
            for i, dm in enumerate(unbiased_dms):
                click.echo(f"   {dm['id']}: {results['O_tilde'][i]:.5f}")

        # 9. Относительные доверительные интервалы
        if "CI_tilde" in results and len(results["CI_tilde"]) > 0:
            click.echo(f"\n📐 ОТНОСИТЕЛЬНЫЕ ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ (CI_tilde):")
            unbiased_dms = [
                dm
                for i, dm in enumerate(data["dms"])
                if i not in results["biased_indices"]
            ]
            for i, dm in enumerate(unbiased_dms):
                click.echo(f"   {dm['id']}: {results['CI_tilde'][i]:.5f}")

    click.echo(f"\n⚖️  ФИНАЛЬНЫЕ ВЕСА:")
    total_weight = 0
    weights = results["final_weights"]
    for i, dm in enumerate(data["dms"]):
        status = "🚫 ИСКЛЮЧЕН" if i in results["biased_indices"] else "✅ УЧАСТВУЕТ"
        click.echo(f"   {dm['id']}: {weights[i]:.5f} | {status}")
        total_weight += weights[i]

    click.echo(f"\n📈 Сумма весов: {total_weight:.5f}")


def save_json_results(
    results: dict,
    data: dict,
    alpha: float,
    b_threshold: int,
    gamma: float,
    input_file: str,
    output_file="results.json",
):
    """Сохранение результатов в JSON файл"""
    # Сохранение результатов
    out_results = {
        "parameters": {
            "alpha": alpha,
            "B_threshold": b_threshold,
            "gamma": gamma,
            "input_file": input_file,
            "output_file": output_file,
        },
        "results": {
            "scores": {
                data["dms"][i]["id"]: results["scores"].tolist()[i]
                for i in range(len(results["scores"]))
            },
            "normalized_scores": {
                data["dms"][i]["id"]: results["normalized_scores"].tolist()[i]
                for i in range(len(results["normalized_scores"]))
            },
            "CIs": {
                data["dms"][i]["id"]: {
                    "mean": float(ci["mean"]),
                    "std": float(ci["std"]),
                    "lower_bound": float(ci["LB"]),
                    "upper_bound": float(ci["UB"]),
                    "length": float(ci["length"]),
                }
                for i, ci in enumerate(results["CIs"])
            },
            "biasedness_index": {
                data["dms"][i]["id"]: float(b) for i, b in enumerate(results["B_i"])
            },
            "biased_indices": {
                data["dms"][i]["id"]: i for i in results["biased_indices"]
            },
            "unbiased_indices": {
                data["dms"][i]["id"]: i for i in results["unbiased_indices"]
            },
            "overlap_matrix": {
                data["dms"][idx]["id"]: [
                    float(val) for val in results["overlap_matrix"][i]
                ]
                for i, idx in enumerate(results["unbiased_indices"])
            },
            "total_overlap": {
                data["dms"][idx]["id"]: float(results["total_overlap"][i])
                for i, idx in enumerate(results["unbiased_indices"])
            },
            "O_i": {
                data["dms"][idx]["id"]: float(results["O_i"][i])
                for i, idx in enumerate(results["unbiased_indices"])
            },
            "O_tilde": {
                data["dms"][idx]["id"]: float(results["O_tilde"][i])
                for i, idx in enumerate(results["unbiased_indices"])
            },
            "CI_tilde": {
                data["dms"][idx]["id"]: float(results["CI_tilde"][i])
                for i, idx in enumerate(results["unbiased_indices"])
            },
            "final_weights": {
                data["dms"][i]["id"]: float(fw)
                for i, fw in enumerate(results["final_weights"])
            },
        },
    }

    if save_json(out_results, output_file):
        click.echo(f"\n💾 Результаты сохранены в: {output_file}")


@click.group()
def cli():
    """🎯 Anti-Biased Group Decision Making Framework

    Обнаружение и обработка предвзятости экспертов в групповом принятии решений
    """
    pass


@cli.command()
@click.option(
    "--method",
    "-m",
    type=click.Choice(BiasDMHandlerMethod, case_sensitive=False),
    default=BiasDMHandlerMethod.EABM,
    help="Метод обработки предвзятых DM",
    show_default=True,
)
@click.option(
    "--file",
    "-f",
    default="example_data.json",
    help="Путь к JSON файлу с входными данными",
    show_default=True,
)
@click.option(
    "--alpha",
    "-a",
    type=click.FloatRange(0.9, 0.99),
    help="Уровень доверия для доверительных интервалов",
    show_default=True,
)
@click.option(
    "-B",
    "--B_threshold",
    type=int,
    help="Порог предвзятости для исключения DM (по умолчанию: I-1)",
    show_default=True,
)
@click.option(
    "-g",
    "--gamma",
    type=click.FloatRange(0.0, 1.0),
    help="Процент доли веса DM в общем весе",
    show_default=True,
)
@click.option(
    "--output",
    "-o",
    help="Имя выходного файла для результатов (по умолчанию: results_<input_file>.json)",
)
@click.option("--verbose", "-v", is_flag=True, help="Подробный вывод")
def analyze(
    method: BiasDMHandlerMethod,
    file: str,
    alpha: float,
    b_threshold: Optional[int],
    gamma: Optional[float],
    output: str,
    verbose: bool,
):
    """Запустить анализ предвзятых экспертов"""
    normalized = False
    if file == "example_data.json":
        example_data = create_example_data()
        try:
            save_json(example_data, "example_data.json")
            verbose and click.echo("✅ Созданы демонстрационные данные")
            normalized = True
        except Exception as e:
            click.echo(e)
            return

    # Проверка существования файла
    if not os.path.exists(file):
        click.echo(f"❌ Файл {file} не существует!", err=True)
        return

    # Загрузка и валидация данных
    try:
        data = load_json(file)
        verbose and click.echo("✅ Данные загружены")
        validate_data(data)
        verbose and click.echo("✅ Данные прошли валидацию")
    except Exception as e:
        click.echo(e)
        return

    # Определение уровня доверия
    if alpha is None:
        alpha = data.get("parameters", {}).get("alpha", 0.95)

    # Определение порога предвзятости
    if b_threshold is None:
        b_threshold = data.get("parameters", {}).get("B", len(data["dms"]) - 1)

    # Определение процента доли DM в общем весе
    if gamma is None:
        gamma = data.get("parameters", {}).get("gamma", 0.5)

    # Определение имени выходного файла
    if output is None:
        base_name = os.path.splitext(os.path.basename(file))[0]
        output = f"{method.name}_results_{base_name}.json"

    if verbose:
        click.echo("\n⚙️ Параметры выполнения:")
        click.echo(f"* Метод обработки предвзятости: {method.name}")
        click.echo(f"* Файл исходных данных: {file}")
        click.echo(f"* Уровень доверия (a): {alpha}")
        click.echo(f"* Порог предвзятости (B): {b_threshold}")
        click.echo(f"* Процент доли DM в общем весе (gamma): {gamma}")
        click.echo(f"* Выходной файл: {output}")

    # Инициализация обработчика
    context = None
    match method:
        case BiasDMHandlerMethod.EABM:
            context = BiasDMHandlerContext(
                EABMHandler(), data, alpha, b_threshold, gamma
            )
        case BiasDMHandlerMethod.MABM:
            context = BiasDMHandlerContext(
                MABMHandler(), data, alpha, b_threshold, gamma
            )
        case BiasDMHandlerMethod.SABM:
            context = BiasDMHandlerContext(
                SABMHandler(), data, alpha, b_threshold, gamma
            )
        case _:
            click.echo("❌ Неизвестный метод обработки предвзятости DM", err=True)
            return

    # Применение метода из BiasDMHandlerMethod
    click.echo(f"\n🔄 Начало анализа предвзятости с помощью {method.name} метода...")
    results = context.handle(normalized)

    # Вывод результатов
    print_results(results, data, verbose)

    # Сохранение результатов
    try:
        save_json_results(results, data, alpha, b_threshold, gamma, file, output)
        verbose and click.echo(f"✅ Результаты сохранены в {output}")
    except Exception as e:
        click.echo(e)
        return

    click.echo("\n🎉 Анализ завершен успешно!")


@cli.command()
@click.option(
    "--file",
    "-f",
    default="example_data.json",
    help="Файл для проверки",
    show_default=True,
)
def validate(file: str):
    """Проверить корректность файла с данными"""
    if not os.path.exists(file):
        click.echo(f"❌ Файл {file} не существует!", err=True)
        return

    try:
        data = load_json(file)
        if data and validate_data(data):
            click.echo("🎉 Файл входных данны корректен и готов к использованию!")
        else:
            click.echo("❌ Файл входных данных некорректен!", err=True)
    except Exception as e:
        click.echo(e, err=True)


@cli.command()
def info():
    """Показать информацию о программе"""
    click.echo("🎯 Anti-Biased Group Decision Making Framework")
    click.echo("=" * 50)
    click.echo("📊 Обнаружение и обработка предвзятости экспертов")
    click.echo("   в процессах группового принятия решений")
    click.echo("\n🔧 Доступные команды:")
    click.echo("   analyze        - Запустить анализ предвзятости по выбранному методу")
    click.echo("   validate       - Проверить корректность файла входных данных")
    click.echo("   info           - Информация о программе")
    click.echo("   --help           - Помощь по командам")
    click.echo("\n📖 Примеры использования:")
    click.echo("   python ./src/main.py analyze")
    click.echo("   python ./src/main.py analyze -m eabm --verbose")
    click.echo(
        "   python ./src/main.py analyze -m eabm -f my_data.json --alpha 0.9 -B 2"
    )
    click.echo("   python ./src/main.py validate")
    click.echo("   python ./src/main.py validate -f my_data.json")
