# catalog/management/commands/dump_tuner.py
import os
import json
import time
from pathlib import Path

import requests
from django.core.management.base import BaseCommand, CommandError


API_BASE = "https://api.moysklad.ru/api/remap/1.2"


def _auth_header() -> str:
    """
    Читает токен из окружения TUNER_API_TOKEN.
    Принимает как 'Bearer xxx', так и просто 'xxx' — второй вариант обернём сами.
    """
    raw = ('5e7459df345678541a82637cee3add4a3197aecb').strip()
    if not raw:
        raise CommandError("Переменная окружения TUNER_API_TOKEN не задана.")
    return raw if raw.lower().startswith("bearer ") else f"Bearer {raw}"


def _session():
    s = requests.Session()
    s.headers.update({
        "Authorization": _auth_header(),
        "Accept-Encoding": "gzip",
        "Content-Type": "application/json",
        "User-Agent": "gonka-dump-tuner/1.0",
    })
    # requests сам распакует gzip по заголовку
    return s


def _get_json(sess: requests.Session, url: str, params: dict | None = None, retries: int = 3, sleep_s: float = 0.5):
    last_err = None
    for attempt in range(retries):
        try:
            r = sess.get(url, params=params or {}, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            if attempt + 1 < retries:
                time.sleep(sleep_s * (attempt + 1))
    raise last_err


def _product_description_and_attrs(sess: requests.Session, product_meta_href: str) -> tuple[str, list[dict], str | None, str | None]:
    """
    По meta.href продукта подтягиваем полную карточку:
    - description (строка или пусто)
    - attributes (массив атрибутов {name, value})
    - brand (если есть атрибут 'Производитель')
    - country (если есть атрибут 'Страна происхождения')
    Возвращает: (description, attributes, brand, country)
    """
    desc = ""
    attrs_out: list[dict] = []
    brand = None
    country = None

    if not product_meta_href:
        return desc, attrs_out, brand, country

    data = _get_json(sess, product_meta_href)

    # Описание
    desc = (data.get("description") or "").strip()

    # Атрибуты
    for a in (data.get("attributes") or []):
        name = (a.get("name") or "").strip()
        # value у МС бывает разных типов — приводим к строке
        val = a.get("value")
        if not name or val in (None, ""):
            continue
        val_str = str(val).strip()
        attrs_out.append({"name": name, "value": val_str})
        # Поймаем пару популярных штук, пригодится при импорте
        ln = name.lower()
        if ln in ("производитель", "бренд") and not brand:
            brand = val_str
        if ln in ("страна происхождения", "страна") and not country:
            country = val_str

    return desc, attrs_out, brand, country


class Command(BaseCommand):
    help = "Скачивает срез товаров «Тюнер»: stock + (для каждой позиции) карточку товара с description/attributes. Сохраняет в products.json и stock.json."

    def add_arguments(self, parser):
        parser.add_argument("--limit",  type=int, default=20, help="Сколько позиций взять (шаг пагинации отчёта)")
        parser.add_argument("--offset", type=int, default=0,  help="Смещение для начала (для повторного прогона)")
        parser.add_argument("--sleep",  type=float, default=0.2, help="Пауза между запросами (сек)")
        parser.add_argument("--out",    type=str, required=True, help="Папка, куда писать JSON")

    def handle(self, *args, **opts):
        out_dir = Path(opts["out"]).resolve()
        # NOTE: не пытаемся создавать родителя — чтобы права не ломать;
        # просим создать папку заранее с нужными владельцами/правами.
        if not out_dir.exists() or not out_dir.is_dir():
            raise CommandError(f"Папка '{out_dir}' не существует или не является директорией. Создай её и выдай права www-data.")

        limit  = max(1, int(opts["limit"]))
        offset = max(0, int(opts["offset"]))
        sleep_s = float(opts["sleep"])

        sess = _session()

        # 1) Тянем страницу отчёта stock/all
        # Доклад: этот отчёт отдаёт в удобном виде остатки/цены/пути категорий (pathName), но НЕ отдаёт description.
        stock_url = f"{API_BASE}/report/stock/all"
        stock_params = {
            "limit":  limit,
            "offset": offset,
        }
        stock = _get_json(sess, stock_url, stock_params)
        rows = stock.get("rows") or []

        # Сохраним stock «как есть» — поможет диагностировать цены/остатки
        (out_dir / "stock.json").write_text(
            json.dumps(stock, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        self.stdout.write(self.style.SUCCESS(f"✓ stock.json (rows={len(rows)})"))

        # 2) Соберём products.json — нормализованный список для импортера
        products_out: list[dict] = []

        for r in rows:
            # базовые поля из отчёта
            name = (r.get("name") or "").strip()
            code = (r.get("code") or "").strip()
            external_code = (r.get("externalCode") or "").strip()
            path_name = (r.get("pathName") or "").strip()  # "Категория/Подкатегория"
            images_meta = ((r.get("images") or {}).get("meta") or {})
            images_size = images_meta.get("size")  # это просто количество изображений у товара

            # мета продукта (для запроса карточки)
            meta = (r.get("meta") or {})
            product_href = meta.get("href") or ""  # https://api.../entity/product/<id>
            # аккуратно дотянем карточку, если доступен href
            try:
                desc, attrs, brand, country = _product_description_and_attrs(sess, product_href)
            except Exception as e:
                # Не валим весь дамп из-за одной записи — просто без описания/атрибутов
                desc, attrs, brand, country = "", [], None, None

            # Попробуем вытащить "article" (в МС может лежать как поле товара)
            article = None
            try:
                if product_href:
                    pdata = _get_json(sess, product_href)
                    article = (pdata.get("article") or "").strip() or None
            except Exception:
                pass

            products_out.append({
                "id":            (meta.get("uuidHref") or meta.get("href") or ""),  # для стабильности хранения id/href
                "href":          product_href,
                "name":          name,
                "code":          code,
                "externalCode":  external_code,
                "article":       article,
                "pathName":      path_name,
                "description":   desc,
                "attributes":    attrs,     # [{name, value}, ...]
                "brand":         brand,     # если вычислили из attrs
                "country":       country,   # если вычислили из attrs
                "images_size":   images_size,
            })

            time.sleep(sleep_s)

        (out_dir / "products.json").write_text(
            json.dumps({"products": products_out}, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        self.stdout.write(self.style.SUCCESS(f"✓ products.json (items={len(products_out)})"))

        self.stdout.write(self.style.SUCCESS("Готово."))
