import enum
import click
import os
from typing import Optional, Dict, List
import numpy as np

from bias_handler import BiasDMHandlerContext, EABMHandler, MABMHandler, SABMHandler
from utils import create_example_data, load_json, save_json, validate_data


class BiasDMHandlerMethod(enum.Enum):
    EABM = enum.auto(),
    MABM = enum.auto(),
    SABM = enum.auto(),
    

def print_results(
    weights: List[float],
    biased_indices: List[int],
    B_i: np.ndarray,
    CIs: List[Dict],
    data: Dict,
) -> None:
    """Красивый вывод результатов"""
    click.echo("\n" + "=" * 60)
    click.echo("🎯 РЕЗУЛЬТАТЫ")
    click.echo("=" * 60)

    click.echo(f"\n📊 Индексы предвзятости (B_i):")
    for i, dm in enumerate(data["dms"]):
        status = "🚫 ПРЕДВЗЯТ" if i in biased_indices else "✅ НОРМА"
        click.echo(f"   {dm['id']}: {B_i[i]:.2f} | {status}")

    click.echo(f"\n🗑️  Исключенные DM (предвзятые):")
    if biased_indices:
        for idx in biased_indices:
            click.echo(f"   ❌ {data['dms'][idx]['id']}")
    else:
        click.echo("   ✅ Нет исключенных DM")

    click.echo(f"\n⚖️  Финальные веса:")
    total_weight = 0
    for i, dm in enumerate(data["dms"]):
        status = "🚫 ИСКЛЮЧЕН" if i in biased_indices else "✅ УЧАСТВУЕТ"
        click.echo(f"   {dm['id']}: {weights[i]:.4f} | {status}")
        total_weight += weights[i]

    click.echo(f"\n📐 Доверительные интервалы:")
    for i, dm in enumerate(data["dms"]):
        ci = CIs[i]
        status = "🚫" if i in biased_indices else "✅"
        click.echo(
            f"   {status} {dm['id']}: [{ci['LB']:.4f}, {ci['UB']:.4f}] | длина: {ci['length']:.4f}"
        )

    click.echo(f"\n📈 Сумма весов: {total_weight:.6f}")


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
    type=click.Choice(
        BiasDMHandlerMethod, case_sensitive=False
    ),
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
    type=click.FloatRange(0.9, 0.99),
    help="Уровень доверия для доверительных интервалов",
    show_default=True,
)
@click.option(
    "-B",
    "--B_threshold",
    type=int,
    help="Порог предвзятости для исключения DM (по умолчанию: I-1)",
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
    output: str,
    verbose: bool,
):
    """Запустить анализ предвзятых экспертов"""
    if file == "example_data.json":
        example_data = create_example_data()
        try:
            save_json(example_data, "example_data.json")
            verbose and click.echo("✅ Созданы демонстрационные данные")
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

    # Определение имени выходного файла
    if output is None:
        base_name = os.path.splitext(os.path.basename(file))[0]
        output = f"results_{base_name}.json"

    if verbose:
        click.echo("\n⚙️ Параметры выполнения:")
        click.echo(f"* Метод обработки предвзятости: {method.name}")
        click.echo(f"* Файл исходных данных: {file}")
        click.echo(f"* Уровень доверия (a): {alpha}")
        click.echo(f"* Порог предвзятости (B): {b_threshold}")
        click.echo(f"* Выходной файл: {output}")

    # Инициализация обработчика
    context = None
    match method:
        case BiasDMHandlerMethod.EABM:
            context = BiasDMHandlerContext(EABMHandler(), data, alpha, b_threshold)
        case BiasDMHandlerMethod.MABM:
            context = BiasDMHandlerContext(MABMHandler(), data, alpha, b_threshold)
        case BiasDMHandlerMethod.SABM:
            context = BiasDMHandlerContext(SABMHandler(), data, alpha, b_threshold)
        case _:
            click.echo("❌ Неизвестный метод обработки предвзятости DM", err=True)
            return
            
    # Применение EABM метода
    click.echo(f"\n🔄 Начало анализа предвзятости с помощью {method.name} метода...")
    weights, biased_indices, B_i, CIs = context.handle(context)

    # Вывод результатов
    print_results(weights, biased_indices, B_i, CIs, data)

    # Сохранение результатов
    results = {
        "parameters": {"alpha": alpha, "B_threshold": b_threshold, "input_file": file},
        "weights": {
            data["dms"][i]["id"]: float(weights[i]) for i in range(len(weights))
        },
        "biased_dms": [data["dms"][i]["id"] for i in biased_indices],
        "biasedness_index": {
            data["dms"][i]["id"]: float(B_i[i]) for i in range(len(B_i))
        },
        "confidence_intervals": {
            data["dms"][i]["id"]: {
                "lower_bound": float(CIs[i]["LB"]),
                "upper_bound": float(CIs[i]["UB"]),
                "length": float(CIs[i]["length"]),
            }
            for i in range(len(CIs))
        },
    }

    if save_json(results, output):
        click.echo(f"\n💾 Результаты сохранены в: {output}")

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
    click.echo("   help           - Помощь по командам")
    click.echo("\n📖 Примеры использования:")
    click.echo("   python ./src/main.py analyze -m eabm --verbose")
    click.echo("   python ./src/main.py analyze -m eabm -f my_data.json --alpha 0.9 -B 2")
    click.echo("   python ./src/main.py validate")
    click.echo("   python ./src/main.py validate -f my_data.json")
