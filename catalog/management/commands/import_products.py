import re
import json
from pathlib import Path
from decimal import Decimal
import pandas as pd

from django.core.files.images import ImageFile
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from catalog.models import (
    Category, Product, ProductImage, Attribute, ProductAttributeValue
)

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ---- Карта категорий/подкатегорий ----
CATEGORY_MAP = {
    "dvigatel": "Двигатель",
    "turbo": "Турбо",
    "vpusknaya_sistema": "Впускная система",
    "vykhlopnaya_sistema": "Выхлопная система",
    "okhlazhdenie": "Охлаждение",
    "podveska": "Подвеска",
    "tormoza": "Тормоза",
    "toplivnaya_sistema": "Топливная система",
    "transmissiya": "Трансмиссия",
    "wheels": "Шины и диски",
    "interer-i-eksterer": "Интерьер и Экстерьер",
    "masla_i_spetszhidkosti": "Масла и спецжидкости",
    "avtoelektronika": "Автоэлектроника",
    "clothing": "Одежда и атриБУБтика",
}

SUBCATEGORY_MAP = {
    # 1) Двигатель
    "svechi_zazhiganiya": ("Свечи зажигания", "dvigatel"),
    "katushki_zazhiganiya": ("Катушки зажигания", "dvigatel"),
    "komponenty_gbts": ("Компоненты ГБЦ", "dvigatel"),
    "maslyanye-poddony": ("Масляные поддоны", "dvigatel"),
    "porshnevaya_gruppa": ("Поршневая группа", "dvigatel"),
    "opory_i_vstavki": ("Опоры и вставки", "dvigatel"),
    # 2) Турбо
    "turbokity": ("Турбокиты", "turbo"),
    "aktuatory": ("Актуаторы и аксессуары", "turbo"),
    "komplekty_vodometanolnykh_sistem": ("Комплекты водометанольных систем", "turbo"),
    "roliki_i_shkivy": ("Ролики и шкивы", "turbo"),
    "perepusknye_klapany": ("Перепускные клапаны", "turbo"),
    "inlet_paypy_i_rekstriktory": ("Инлет-пайпы и рестрикторы", "turbo"),
    # 3) Впуск
    "vpusknye_sistemy": ("Впускные системы", "vpusknaya_sistema"),
    "vozdushnye_filtry_v_shtatnoe_mesto": ("Воздушные фильтры в штатное место", "vpusknaya_sistema"),
    "patrubki_i_paypy": ("Патрубки и пайпы", "vpusknaya_sistema"),
    "universalnye_vozdushnye_filtry": ("Универсальные воздушные фильтры", "vpusknaya_sistema"),
    "sredstva-dlya-obsluzhivaniya-vozdushnogo-filtra": ("Средства для обслуживания воздушного фильтра", "vpusknaya_sistema"),
    # 4) Выхлоп
    "daunpaypy_priemnye_truby": ("Даунпайпы/приемные трубы", "vykhlopnaya_sistema"),
    "srednyaya_i_zadnyaya_chast": ("Средняя и задняя часть", "vykhlopnaya_sistema"),
    "universalnye_katalizatory": ("Универсальные катализаторы", "vykhlopnaya_sistema"),
    "termoizolyatsiya": ("Термоизоляция", "vykhlopnaya_sistema"),
    # 5) Охлаждение
    "ventilyatory-spal": ("Вентиляторы SPAL", "okhlazhdenie"),
    "interkulery_i_radiatory_nadduva": ("Интеркулеры и радиаторы наддува", "okhlazhdenie"),
    "radiatory_dvigatelya": ("Радиаторы двигателя", "okhlazhdenie"),
    "radiatory_kpp": ("Радиаторы КПП", "okhlazhdenie"),
    "universalnye_maslyanye_radiatory": ("Универсальные масляные радиаторы", "okhlazhdenie"),
    # 6) Подвеска
    "amortizatory": ("Амортизаторы", "podveska"),
    "komplekty_podveski_v_sbore": ("Комплекты подвески в сборе", "podveska"),
    "pruzhiny": ("Пружины", "podveska"),
    "opory-amortizatorov": ("Опоры амортизаторов", "podveska"),
    "stabilizatory_i_aksessuary": ("Стабилизаторы и аксессуары", "podveska"),
    # 7) Тормоза
    "tormoznye_sistemy": ("Тормозные системы", "tormoza"),
    "tormoznye_diski": ("Тормозные диски", "tormoza"),
    "tormoznye_kolodki": ("Тормозные колодки", "tormoza"),
    "tormoznye_supporty": ("Тормозные суппорты", "tormoza"),
    "tormoznye_shlangi": ("Тормозные шланги", "tormoza"),
    # 8) Топливная
    "power_juice": ("POWER JUICE", "toplivnaya_sistema"),
    "toplivnye_nasosy": ("Топливные насосы", "toplivnaya_sistema"),
    # 9) Трансмиссия
    "differential": ("Дифференциалы", "transmissiya"),
    "komplekty_stsepleniya": ("Комплекты сцепления", "transmissiya"),
    "flywheels": ("Маховики", "transmissiya"),
    "korotkokhodnye_kulisy": ("Короткоходные кулисы", "transmissiya"),
    "drugie-tovary": ("Поддоны и другие товары", "transmissiya"),
    # 10) Шины и диски
    "bolts-nuts-etc": ("Болты/Гайки и прочее", "wheels"),
    "diski": ("Диски", "wheels"),
    "prostvaki": ("Проставки", "wheels"),  # оставляем, если так в данных
    "prostavi": ("Проставки", "wheels"),
    "tires": ("Шины", "wheels"),
    # 11) Интерьер/Экстерьер
    "lepestki": ("Лепестки", "interer-i-eksterer"),
    "nakladki-bampera-porogi": ("Накладки бампера, пороги, спойлеры", "interer-i-eksterer"),
    "zerkala-i-reshetki-radiatora": ("Зеркала и решетки радиатора", "interer-i-eksterer"),
    "krupnye-elementy-iz-karbona": ("Крупные элементы из карбона", "interer-i-eksterer"),
    "shift-knobs": ("Рукояти КПП", "interer-i-eksterer"),
    "nomernye-ramki": ("Номерные рамки", "interer-i-eksterer"),
    # 12) Масла и спецжидкости
    "masla": ("Масла и тормозные жидкости", "masla_i_spetszhidkosti"),
    "magnitnye-slivnye-probki": ("Магнитные сливные пробки", "masla_i_spetszhidkosti"),
    "maslyanye-filtry": ("Масляные фильтры", "masla_i_spetszhidkosti"),
    "korpusa-maslyanykh-filtrov": ("Корпуса масляных фильтров", "masla_i_spetszhidkosti"),
    "shchupy-urovnya-masla": ("Щупы уровня масла", "masla_i_spetszhidkosti"),
    # 13) Автоэлектроника
    "chip-tuning": ("MHD Tuning & xHP Flashtool", "avtoelektronika"),
    "bootmod3": ("BOOTMOD3", "avtoelektronika"),
    "burger-motorsports": ("Burger Motorsports", "avtoelektronika"),
    "dragy": ("Dragy", "avtoelektronika"),
    "dyno-spectrum": ("Dyno Spectrum", "avtoelektronika"),
    "ecumaster": ("ECUMASTER", "avtoelektronika"),
    "maxxecu": ("MaxxECU", "avtoelektronika"),
    "thor": ("MHD & THOR & xHP", "avtoelektronika"),
    "datchiki_i_aksessuary": ("Датчики и аксессуары", "avtoelektronika"),
    "bust_kontrollery": ("Буст-контроллеры", "avtoelektronika"),
    "vasya_diagnost": ("ВАСЯ диагност", "avtoelektronika"),
    # 14) Одежда и атриБУБтика
    "kruzhki-remuvki": ("Кружки, ремувки", "clothing"),
    "stickers": ("Наклейки", "clothing"),
    "clothes": ("Одежда", "clothing"),
}

# ---- Утилиты ----
def as_str(v):
    return "" if pd.isna(v) else str(v).strip()

def norm_decimal(v):
    if pd.isna(v):
        return Decimal("0")
    s = str(v).strip().replace("\u00A0", "").replace(" ", "").replace(",", ".")
    s = re.sub(r"[^0-9.\-]", "", s)
    try:
        return Decimal(s) if s else Decimal("0")
    except Exception:
        return Decimal("0")

def split_list(text):
    if not text:
        return []
    return [p.strip() for p in re.split(r"[;,|]", str(text)) if p.strip()]

def ensure_category_path(path_text):
    if not path_text:
        return None
    parts = [p.strip() for p in re.split(r"[>/\\]+", path_text) if p.strip()]
    parent = None
    for name in parts:
        slug = re.sub(r"[^a-z0-9\-]+", "-", name.lower()).strip("-") or "cat"
        cat, _ = Category.objects.get_or_create(slug=slug, defaults={"name": name, "parent": parent})
        if cat.parent_id != (parent.id if parent else None):
            cat.parent = parent
            cat.name = name
            cat.save(update_fields=["parent", "name"])
        parent = cat
    return parent

def parse_specs(text):
    s = as_str(text)
    if not s:
        return []
    # JSON?
    if s.startswith("{") or s.startswith("["):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return [(k.strip(), str(v).strip()) for k, v in obj.items()]
        except Exception:
            pass
    # Пары через двоеточие/тире/равно/перенос
    pairs = []
    for part in re.split(r"[;\n]+", s):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            k, v = part.split(":", 1)
        elif "—" in part:
            k, v = part.split("—", 1)
        elif "=" in part:
            k, v = part.split("=", 1)
        elif "-" in part and not re.match(r"^-?\d+(\.\d+)?$", part):
            k, v = part.split("-", 1)
        else:
            k, v = "Характеристика", part
        pairs.append((k.strip(), v.strip()))
    return [(k, v) for k, v in pairs if k and v]

def split_multi_values(key, val):
    """Для некоторых ключей разбиваем значения на отдельные элементы."""
    key_l = (key or "").strip().lower()
    if key_l in {"модель авто", "марка авто", "двигатель", "производитель"}:
        parts = re.split(r"[;,/|]+", val)
        return [p.strip() for p in parts if p.strip()]
    return [val]

def normalize_folder_under(root: Path, folder_text: str) -> Path | None:
    raw = as_str(folder_text)
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute():
        return p
    parts = [q for q in re.split(r"[\\\/]+", raw) if q]
    return root.joinpath(*parts)

def clean_sku(s) -> str:
    s = as_str(s)
    s = re.sub(r"^\s*артикул\s*[:\-]?\s*", "", s, flags=re.IGNORECASE)
    return s.strip()

def ensure_cat_by_subslug(sub_slug: str, CategoryModel):
    """Создать/получить подкатегорию из SUBCATEGORY_MAP. Вернёт child."""
    info = SUBCATEGORY_MAP.get(sub_slug)
    if not info:
        sub_name = sub_slug.replace("-", " ").replace("_", " ").title()
        parent_slug = "misc"
        parent_name = "Прочее"
    else:
        sub_name, parent_slug = info
        parent_name = CATEGORY_MAP.get(parent_slug, parent_slug)

    parent, _ = CategoryModel.objects.get_or_create(
        slug=parent_slug, defaults={"name": parent_name, "parent": None}
    )
    if parent.name != parent_name or parent.parent_id is not None:
        parent.name = parent_name
        parent.parent = None
        parent.save(update_fields=["name", "parent"])

    child, _ = CategoryModel.objects.get_or_create(
        slug=sub_slug, defaults={"name": sub_name, "parent": parent}
    )
    if child.name != sub_name or child.parent_id != parent.id:
        child.name = sub_name
        child.parent = parent
        child.save(update_fields=["name", "parent"])
    return child

def col_flex(df, *aliases):
    low = {str(c).lower(): c for c in df.columns}
    for a in aliases:
        c = low.get(str(a).lower())
        if c:
            return c
    return None

def _sheet_param(val):
    """Преобразует '0' -> 0, пустое -> 0, иначе возвращает как есть."""
    if val is None or val == "":
        return 0
    try:
        return int(val)
    except (TypeError, ValueError):
        return val  # строковое имя листа

@transaction.atomic
def import_brands(opts, stdout=None):
    """Отдельный режим: файл с колонками Артикул / Производитель."""
    path = Path(opts["brands_xlsx"])
    if not path.is_file():
        raise CommandError(f"Файл брендов не найден или это не файл: {path}")

    sheet = _sheet_param(opts.get("brands_sheet"))
    dfb = pd.read_excel(path, sheet_name=sheet)

    COL_SKU  = col_flex(dfb, "Артикул", "sku", "артикул", "article")
    COL_BRND = col_flex(dfb, "Производитель", "Бренд", "brand", "vendor")
    if not COL_SKU or not COL_BRND:
        raise CommandError("В файле брендов нет нужных колонок: 'Артикул' и 'Производитель'")

    attr_brand, _ = Attribute.objects.get_or_create(name="Производитель")
    if not attr_brand.slug:
        # если модель сама ставит slug в save()
        attr_brand.save(update_fields=["slug"])

    updated = 0
    missing = 0

    for _, row in dfb.iterrows():
        sku = clean_sku(row.get(COL_SKU))
        if not sku:
            continue
        br_val = as_str(row.get(COL_BRND))
        if not br_val:
            continue

        try:
            product = Product.objects.get(sku=sku)
        except Product.DoesNotExist:
            missing += 1
            continue

        if opts.get("replace_brands"):
            ProductAttributeValue.objects.filter(product=product, attribute=attr_brand).delete()

        for one in split_multi_values("Производитель", br_val):
            if not one:
                continue
            ProductAttributeValue.objects.get_or_create(
                product=product, attribute=attr_brand, value=one
            )

        updated += 1

    msg = f"Бренды: обновлено {updated}, пропущено (товар не найден) {missing}"
    if stdout:
        stdout.write(msg)
    else:
        print(msg)

class Command(BaseCommand):
    help = "Импорт из Excel (товары/фото/характеристики) + отдельный режим обновления брендов."

    def add_arguments(self, parser):
        # Основной файл каталога (делаем НЕобязательным, чтобы можно было запускать только режим брендов)
        parser.add_argument("--xlsx", help="Excel с каталогом")
        parser.add_argument("--images_root", default=".", help="Корневая папка для путей к фото")
        parser.add_argument("--sheet", help="Лист (имя/индекс) в основном Excel")
        parser.add_argument("--limit", type=int)
        parser.add_argument("--replace-images", action="store_true", help="Удалять старые фото перед загрузкой")
        parser.add_argument("--only-specs", dest="only_specs", action="store_true",
                            help="Импортировать ТОЛЬКО характеристики (товары/фото не трогать)")
        parser.add_argument("--only-categories", dest="only_categories", action="store_true",
                            help="Обновить категории/подкатегории и привязки товаров, остальное не трогать")

        # Отдельный файл брендов
        parser.add_argument("--brands-xlsx", dest="brands_xlsx",
                            help="Отдельный Excel с колонками Артикул/Производитель")
        parser.add_argument("--brands-sheet", dest="brands_sheet",
                            help="Лист в файле брендов (имя или индекс)")
        parser.add_argument("--replace-brands", dest="replace_brands",
                            action="store_true",
                            help="Стереть старые значения 'Производитель' перед записью новых")

    @transaction.atomic
    def handle(self, *args, **opts):
        # --- РЕЖИМ ОБНОВЛЕНИЯ БРЕНДОВ ---
        if opts.get("brands_xlsx"):
            return import_brands(opts, stdout=self.stdout)

        # --- Обычный импорт каталога ---
        if not opts.get("xlsx"):
            raise CommandError("Нужно указать --xlsx (или запустить с --brands-xlsx для режима брендов)")

        xlsx = Path(opts["xlsx"])
        if not xlsx.exists():
            raise CommandError(f"Excel не найден: {xlsx}")

        sheet_main = _sheet_param(opts.get("sheet"))
        df = pd.read_excel(xlsx, sheet_name=sheet_main)
        if opts.get("limit"):
            df = df.head(int(opts["limit"]))
        images_root = Path(opts["images_root"]).resolve()

        # быстрый доступ к колонкам по имени без кейса
        def col(name):
            low = {str(c).lower(): c for c in df.columns}
            return low.get(str(name).lower())

        COL_TITLE = col("Название")
        COL_CAT   = col("Категория")
        COL_SKU   = col("Артикул")
        COL_PRICE = col("Цена")
        COL_DESC  = col("Описание")
        COL_SPECS = col("Характеристики")
        COL_LINK  = col("Ссылка")
        COL_PHOTO = col("Фото")
        COL_DIR   = col("Папка")
        COL_SUB   = col("Подкатегория")

        for must, nm in [(COL_TITLE, "Название"), (COL_SKU, "Артикул")]:
            if not must:
                raise CommandError(f"Нет обязательной колонки: {nm}")

        created_p = 0
        updated_p = 0
        added_imgs = 0

        for _, row in df.iterrows():
            sku = clean_sku(row.get(COL_SKU))
            if not sku:
                continue

            sub_slug = as_str(row.get(COL_SUB))

            # --- Только категории ---
            if opts.get("only_categories"):
                if not sub_slug:
                    continue
                try:
                    product = Product.objects.get(sku=sku)
                except Product.DoesNotExist:
                    continue
                subcat = ensure_cat_by_subslug(sub_slug, Category)
                if product.category_id != subcat.id:
                    product.category = subcat
                    product.save(update_fields=["category"])
                continue

            # --- Только характеристики ---
            if opts.get("only_specs"):
                try:
                    product = Product.objects.get(sku=sku)
                except Product.DoesNotExist:
                    continue  # пропускаем отсутствующие товары
            else:
                # --- Полный импорт товара ---
                title = as_str(row.get(COL_TITLE)) or f"Товар {sku}"
                price = norm_decimal(row.get(COL_PRICE))
                description = as_str(row.get(COL_DESC))
                link = as_str(row.get(COL_LINK))
                cat_text = as_str(row.get(COL_CAT))

                # обычная категория по пути
                category = ensure_category_path(cat_text) if cat_text else None

                product, created = Product.objects.get_or_create(
                    sku=sku,
                    defaults={
                        "title": title,
                        "category": category,
                        "price": price,
                        "description": description,
                        "external_url": link,
                        "is_active": True,
                    },
                )
                if created:
                    created_p += 1
                else:
                    changed = False
                    if title and product.title != title:
                        product.title = title; changed = True
                    if category and product.category_id != (category.id if category else None):
                        product.category = category; changed = True
                    if description and not product.description:
                        product.description = description; changed = True
                    if link and not product.external_url:
                        product.external_url = link; changed = True
                    if price is not None and product.price != price:
                        product.price = price; changed = True
                    if changed:
                        product.save()
                        updated_p += 1

                # если задана подкатегория — перепривяжем
                if sub_slug:
                    subcat = ensure_cat_by_subslug(sub_slug, Category)
                    if product.category_id != subcat.id:
                        product.category = subcat
                        product.save(update_fields=["category"])

                # --- Фото (только в полном импорте) ---
                if opts.get("replace-images") and product.images.exists():
                    product.images.all().delete()

                files = set()

                folder_text = as_str(row.get(COL_DIR))
                if folder_text:
                    folder = normalize_folder_under(images_root, folder_text)
                    if folder and folder.exists() and folder.is_dir():
                        for f in folder.iterdir():
                            if f.is_file() and f.suffix.lower() in IMG_EXT:
                                files.add(f)

                photos_text = as_str(row.get(COL_PHOTO))
                for rel in split_list(photos_text):
                    p = Path(rel)
                    if not p.is_absolute():
                        p = images_root.joinpath(*re.split(r"[\\\/]+", rel))
                    if p.exists() and p.is_file() and p.suffix.lower() in IMG_EXT:
                        files.add(p)

                start_pos = product.images.count()
                for idx, path in enumerate(sorted(files)):
                    with path.open("rb") as fh:
                        ProductImage.objects.create(
                            product=product,
                            position=start_pos + idx,
                            image=ImageFile(fh, name=path.name),
                            alt=product.title
                        )
                        added_imgs += 1

            # --- Характеристики (в обоих режимах) ---
            specs = parse_specs(as_str(row.get(COL_SPECS)))
            for k, v in specs:
                key = (k or "").strip()
                val = (v or "").strip()
                if not key or not val:
                    continue
                attr, created_attr = Attribute.objects.get_or_create(name=key)
                if created_attr and not attr.slug:
                    attr.save(update_fields=["slug"])
                for one in split_multi_values(key, val):
                    ProductAttributeValue.objects.get_or_create(
                        product=product, attribute=attr, value=one
                    )

        self.stdout.write(self.style.SUCCESS(
            f"Импорт: products +{created_p}/~{updated_p}, images +{added_imgs}"
        ))
