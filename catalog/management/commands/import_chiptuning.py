import re
from pathlib import Path
from contextlib import suppress

import pandas as pd
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from catalog.models import ChipCar, ChipStage, ChipMetric


def col_flex(df, *aliases):
    """Первое совпадение из списка псевдонимов колонки (без регистра)."""
    low = {str(c).strip().lower(): c for c in df.columns}
    for a in aliases:
        c = low.get(str(a).strip().lower())
        if c:
            return c
    return None


def as_str(v):
    return "" if pd.isna(v) else str(v).strip()


class Command(BaseCommand):
    help = (
        "Импорт чип-тюнинга из Excel. "
        "Ожидаемый формат колонок:\n"
        "Марка | Модель | Год | Мотор | Stage | Параметр | Заводские | После | Разница\n"
        "Каждая строка — одна характеристика (например 'Мощность')."
    )

    def add_arguments(self, parser):
        parser.add_argument("--xlsx", required=True, help="Путь к файлу Excel")
        parser.add_argument("--sheet", default=0, help="Лист (имя или индекс)")
        parser.add_argument(
            "--replace",
            action="store_true",
            help="Перед записью по (Марка,Модель,Год,Мотор,Stage) очистить старые строки ChipMetric",
        )

    @transaction.atomic
    def handle(self, *args, **opts):
        path = Path(opts["xlsx"])
        if not path.is_file():
            raise CommandError(f"Файл не найден: {path}")

        raw_sheet = opts.get("sheet", 0)
        if raw_sheet in (None, "", 0):
            sheet = 0
        else:
            try:
                sheet = int(raw_sheet)  # "0" -> 0
            except (TypeError, ValueError):
                sheet = raw_sheet       # оставляем имя листа, если это строка не-число

        df = pd.read_excel(path, sheet_name=sheet)

        # Ищем колонки (гибко по именам)
        COL_MAKE   = col_flex(df, "Марка", "Make")
        COL_MODEL  = col_flex(df, "Модель", "Model")
        COL_YEAR   = col_flex(df, "Год", "Year")
        COL_ENGINE = col_flex(df, "Мотор", "Двигатель", "Engine")
        COL_STAGE  = col_flex(df, "Stage", "Стадия", "Стейдж")

        COL_PARAM  = col_flex(df, "Параметр", "Показатель", "Metric")
        COL_STOCK  = col_flex(df, "Заводские", "Stock")
        COL_TUNED  = col_flex(df, "После тюнинга*", "После тюнинга", "Tuned", "После")
        COL_DELTA  = col_flex(df, "Разница", "Delta")

        required = [COL_MAKE, COL_MODEL, COL_YEAR, COL_ENGINE, COL_STAGE, COL_PARAM, COL_STOCK, COL_TUNED]
        if any(c is None for c in required):
            names = "\n".join(str(c) for c in df.columns)
            raise CommandError(
                "Не найдены нужные колонки. Нужны как минимум:\n"
                "Марка, Модель, Год, Мотор, Stage, Параметр, Заводские, После тюнинга*\n"
                f"Есть колонки:\n{names}"
            )

        # Группируем по «авто + стадия»
        group_fields = [COL_MAKE, COL_MODEL, COL_YEAR, COL_ENGINE, COL_STAGE]
        groups = df.groupby(group_fields, dropna=False)

        created_cars = updated_stages = written_rows = 0

        for (make, model, year, engine, stage_name), sub in groups:
            make   = as_str(make)
            model  = as_str(model)
            year   = as_str(year)
            engine = as_str(engine)
            stage_name = as_str(stage_name) or "Stage 1"

            if not (make and model and year and engine):
                # пропускаем неполные группы
                continue

            car, _car_created = ChipCar.objects.get_or_create(
                make=make, model=model, year=year, engine=engine
            )
            if _car_created:
                created_cars += 1

            stage, _ = ChipStage.objects.get_or_create(car=car, name=stage_name)

            if opts.get("replace"):
                stage.metrics.all().delete()

            # строки-показатели
            order = 0
            for _, row in sub.iterrows():
                label = as_str(row.get(COL_PARAM))
                stock = as_str(row.get(COL_STOCK))
                tuned = as_str(row.get(COL_TUNED))
                delta = as_str(row.get(COL_DELTA))

                # если дельты нет — попробуем аккуратно посчитать для чисел
                if not delta:
                    with suppress(Exception):
                        # извлекаем число (оставляем знак, цифры, точку и запятую)
                        def num(s):
                            s = re.sub(r"[^\d,.\-]", "", s.replace(",", "."))
                            return float(s) if s else None
                        a, b = num(stock), num(tuned)
                        if a is not None and b is not None:
                            val = b - a
                            # знак и «единицы» оставляем пользователю — просто число
                            delta = f"{val:+g}"

                if not label:
                    continue

                ChipMetric.objects.create(
                    stage=stage, order=order,
                    label=label, stock=stock, tuned=tuned, delta=delta
                )
                order += 1
                written_rows += 1

            updated_stages += 1

        self.stdout.write(self.style.SUCCESS(
            f"Импорт завершён: авто +{created_cars}, стадий {updated_stages}, строк метрик {written_rows}"
        ))
