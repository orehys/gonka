# catalog/views.py
from django.db.models import Value as V
from django.shortcuts import render
from django.core.paginator import Paginator
from django.db.models import Q, Exists, OuterRef, Prefetch, Max
from django.db.models.functions import ExtractYear
from django.utils.text import slugify
from .models import RobotsTxt, unique_slugify, Product, ProductAttributeValue, Category, Order, OrderItem, SavedCartItem, UserProfile, ProductImage, ChipCar, ChipStage, ChipMetric, ChipRequest, ServiceRequest, News, Brand, Attribute, CarMake, ContactRequest, CMSPage, SitePage, Banner, SEOEntry, RedirectRule, GoneURL, RobotsTxt, BannerPlacement, HomeLatestItem, fs_safe_segment
from django.db.models import Case, When, IntegerField
from django.shortcuts import render, get_object_or_404
from collections import defaultdict
from django.core.mail import EmailMultiAlternatives, send_mail, get_connection
import time, smtplib
import re
import requests
from django.http import HttpResponseNotAllowed
from urllib.parse import urlparse
from xml.sax.saxutils import escape
from datetime import datetime
from django.http import HttpResponse
from django.db import connection
from django.utils.timezone import now
from django.utils import timezone
from django.views.decorators.http import require_http_methods
from django.templatetags.static import static
from django.utils.html import strip_tags
from django.template.loader import render_to_string
from email.mime.image import MIMEImage
from pathlib import Path
from django.conf import settings
from .forms import SEOEntryForm, RedirectForm, GoneForm, RobotsForm
from django.core.mail import send_mail, EmailMessage
from django.views.decorators.http import require_POST
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.utils.http import urlencode
from django.db import transaction
from django.contrib.auth.decorators import login_required
from .cart import (
    add as cart_add,
    remove as cart_remove,
    set_qty as cart_set,
    items as cart_items,
    totals as cart_totals,
    clear as cart_clear,
)
from .forms import QtyForm, OrderForm, LoginForm, RegisterForm, ProfileForm, PasswordChangeForm, ProductImagesUploadForm, OrderStatusForm, ServiceRequestForm, AdminNewsForm, PasswordResetEmailForm, MailingPromoForm
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from django.contrib import messages
from django.contrib.auth import login, logout
from django.contrib.auth import update_session_auth_hash
from django.shortcuts import redirect
from django.urls import reverse
from django.http import JsonResponse
from .utils import push_recent, recent_qs
from django.contrib.auth import get_user_model
import logging
from .cart import CART_SESSION_ID
from django.utils.http import url_has_allowed_host_and_scheme
import secrets


logger = logging.getLogger(__name__)

def verify_recaptcha_v2(token: str, remote_ip: str | None = None):
    if not token:
        return False, {"error": "missing-token"}
    url = getattr(settings, "RECAPTCHA_VERIFY_URL",
                  "https://www.google.com/recaptcha/api/siteverify")
    data = {"secret": settings.RECAPTCHA_SECRET_KEY, "response": token}
    if remote_ip:
        data["remoteip"] = remote_ip
    try:
        r = requests.post(url, data=data, timeout=6)
        js = r.json()
        return bool(js.get("success")), js
    except Exception as e:
        logger.exception("reCAPTCHA verify exception")
        return False, {"error": "verify-exception", "detail": str(e)}

def verify_recaptcha(token: str, remote_ip: str | None = None) -> tuple[bool, dict]:
    """
    Проверка reCAPTCHA v2 на стороне сервера.
    Возвращает (ok, payload_json).
    """
    if not token:
        return False, {"error": "missing-token"}

    url = getattr(settings, "RECAPTCHA_VERIFY_URL",
                  "https://www.google.com/recaptcha/api/siteverify")
    data = {
        "secret": settings.RECAPTCHA_SECRET_KEY,
        "response": token,
    }
    if remote_ip:
        data["remoteip"] = remote_ip

    try:
        resp = requests.post(url, data=data, timeout=6)
        payload = resp.json()
        ok = bool(payload.get("success"))
        return ok, payload
    except Exception as e:
        logger.exception("reCAPTCHA verify failed: %s", e)
        return False, {"error": "verify-exception", "detail": str(e)}


def service_lead_submit(request):
    if request.method != "POST":
        ref = request.META.get("HTTP_REFERER")
        # если реферер отсутствует/чужой/указывает на этот же путь — на каталог
        if not ref or not url_has_allowed_host_and_scheme(
            ref, {request.get_host()}, require_https=request.is_secure()
        ) or urlparse(ref).path == request.path:
            return redirect(reverse("catalog:product_list"))
        return redirect(ref)

    form = ServiceRequestForm(request.POST)

    make   = (request.POST.get("make")   or "").strip()
    model  = (request.POST.get("model")  or "").strip()
    year   = (request.POST.get("year")   or "").strip()
    engine = (request.POST.get("engine") or "").strip()

    # Куда вернуться
    raw_next = (request.POST.get("next") or request.META.get("HTTP_REFERER")
                or reverse("catalog:product_list"))
    if not url_has_allowed_host_and_scheme(raw_next, {request.get_host()}):
        raw_next = reverse("catalog:product_list")

    # Не редиректим на сам submit (иначе можно получить петлю)
    submit_path = reverse("catalog:service_lead_submit")
    host = request.get_host()

    raw_next = request.POST.get("next")
    if raw_next and url_has_allowed_host_and_scheme(raw_next, {host}, require_https=request.is_secure()):
        next_url = raw_next
    else:
        referer = request.META.get("HTTP_REFERER")
        if referer and url_has_allowed_host_and_scheme(referer, {host}, require_https=request.is_secure()):
            next_url = referer
        else:
            next_url = reverse("catalog:product_list")

    # не редиректим на сам submit, чтобы не уйти в петлю
    if urlparse(next_url).path == submit_path:
        next_url = reverse("catalog:product_list")

    next_url     = raw_next
    referer_path = urlparse(next_url).path
    dyno_path    = reverse("catalog:page_dynostand")

    # reCAPTCHA
    token = request.POST.get("g-recaptcha-response")
    ok, payload = verify_recaptcha(token, request.META.get("REMOTE_ADDR"))
    if settings.DEBUG and not ok:
        ok = True

    def _home_context_for_error():
        brands  = _distinct_attr_values("Производитель")
        makes   = _distinct_attr_values("Марка авто")
        models_ = _distinct_attr_values("Модель авто")
        engines = _distinct_attr_values("Двигатель")

        curated = list(HomeLatestItem.objects.filter(is_active=True)
                       .select_related("product").order_by("position"))
        latest_products = [i.product for i in curated] if curated else \
                          list(Product.objects.filter(is_active=True).order_by("-created_at")[:8])

        recent_products = get_recent_products(request, limit=6)
        page = SitePage.objects.filter(slug="home").first()
        brands_qs = Brand.objects.order_by("name").only("id", "name", "slug", "logo")
        placements = (BannerPlacement.objects
                      .filter(page_key=BannerPlacement.PageKey.HOME, banner__is_active=True)
                      .select_related("banner").order_by("position"))
        banners_main = [p.banner for p in placements]

        return {
            "brands": brands_qs,
            "makes": makes,
            "models_": models_,
            "engines": engines,
            "latest_products": latest_products,
            "recent_products": recent_products,
            "page": page,
            "banners_main": banners_main,
            "RECAPTCHA_SITE_KEY": settings.RECAPTCHA_SITE_KEY,
            "form": form,   # важное: bound-форма с введёнными значениями
        }

    def _dyno_context_for_error():
        page = SitePage.objects.filter(slug="dyno").first()
        placements = (BannerPlacement.objects
                      .filter(page_key=BannerPlacement.PageKey.DYNO, banner__is_active=True)
                      .select_related("banner").order_by("position"))
        banners_main = [p.banner for p in placements]
        return {
            "page": page,
            "banners_main": banners_main,
            "RECAPTCHA_SITE_KEY": settings.RECAPTCHA_SITE_KEY,
            "form": form,
            "make":   (request.POST.get("make")   or ""),
            "model":  (request.POST.get("model")  or ""),
            "year":   (request.POST.get("year")   or ""),
            "engine": (request.POST.get("engine") or ""),
        }

    if not ok:
        messages.error(request, "Подтвердите, что вы не робот.")
        if referer_path == "/":
            return render(request, "home/index.html", _home_context_for_error())
        if referer_path == dyno_path:
            return render(request, "static_pages/dynostand.html", _dyno_context_for_error())
        return redirect(next_url)

    if form.is_valid():
        obj = form.save(commit=False)
        if request.user.is_authenticated:
            obj.user = request.user

        suffix = ""
        if any([make, model, year, engine]):
            suffix = f"\n\nВыбран авто: {make or '-'} {model or '-'} {year or '-'} {engine or '-'}"

        obj.comment    = (obj.comment or "") + suffix
        obj.source_url = request.META.get("HTTP_REFERER") or request.build_absolute_uri()
        obj.save()
        messages.success(request, "Заявка отправлена! Мы свяжемся с вами.")
        return redirect(next_url)
    else:
        messages.error(request, "Исправьте ошибки и отправьте снова.")
        if referer_path == "/":
            return render(request, "home/index.html", _home_context_for_error())
        if referer_path == dyno_path:
            return render(request, "static_pages/dynostand.html", _dyno_context_for_error())
        return redirect(next_url)

SUPPLIERS_NO_DEDUCT = {"tuner", "sttuning"}

def deduct_local_stock(order_items):
    """
    order_items: iterable из (product_id, qty) или объектов позиции заказа.
    Списывает остаток только у товаров НЕ от tuner/sttuning.
    """
    from catalog.models import Product  # чтобы без циклических импортов

    # соберём словарь {product_id: total_qty}
    qty_map = {}
    for it in order_items:
        # поддержим оба варианта: dict/объект
        pid = getattr(it, "product_id", None) or it["product_id"]
        q   = getattr(it, "qty", None) or it["qty"]
        qty_map[pid] = qty_map.get(pid, 0) + int(q)

    # блокируем строки товаров на время списания (защита от гонок)
    products = (
        Product.objects
        .select_for_update()
        .filter(id__in=qty_map.keys())
    )

    for p in products:
        # пропускаем поставщиков
        if (p.supplier or "").strip().lower() in SUPPLIERS_NO_DEDUCT:
            continue

        # если остаток не ведём — ничего не делаем
        if p.stock is None:
            continue

        new_stock = p.stock - qty_map[p.id]
        if new_stock < 0:
            new_stock = 0

        if new_stock != p.stock:
            p.stock = new_stock
            # опционально синхронизируем in_stock
            p.in_stock = (p.stock > 0)
            p.save(update_fields=["stock", "in_stock"])

def _exists_attr(attr_name_ru: str, value: str, partial: bool = True):
    """
    Вернёт выражение Exists(...) для проверки, что у товара есть характеристика
    attr_name_ru со значением value. partial=True — ищем по подстроке (icontains).
    """
    if not value:
        # вернём условие, которое всегда True (не фильтруем)
        return None
    base = ProductAttributeValue.objects.filter(
        product_id=OuterRef("pk"),
        attribute__name__iexact=attr_name_ru,
    )
    if partial:
        base = base.filter(value__icontains=value)
    else:
        base = base.filter(value__iexact=value)
    return Exists(base)



def _page_window(paginator, number, pad=1, max_len=9):
    """
    Возвращает список элементов для пагинации:
    [1, '…', n-1, n, n+1, '…', last].
    pad=1 → показываем по одной странице слева/справа от текущей.
    """
    total = paginator.num_pages
    if total <= max_len:
        return list(range(1, total + 1))

    start = max(2, number - pad)
    end = min(total - 1, number + pad)

    items = [1]
    if start > 2:
        items.append("…")
    items += list(range(start, end + 1))
    if end < total - 1:
        items.append("…")
    items.append(total)
    return items

def _apply_text_search(qs, query: str):
    """
    Ищем все слова из запроса:
      - в title, sku (включая вариант без дефисов/пробелов), description
      - в имени категории и родителя
      - в значениях характеристик (attributes__value)
    Слова комбинируются AND, поля внутри слова — OR.
    """
    if not query:
        return qs
    terms = [t for t in re.split(r"\s+", query.strip()) if t]
    for t in terms:
        t_flat = re.sub(r"[\s\-_/]+", "", t)
        block = (
            Q(title__icontains=t) |
            Q(sku__icontains=t) |
            Q(description__icontains=t) |
            Q(category__name__icontains=t) |
            Q(category__parent__name__icontains=t) |
            Q(attributes__value__icontains=t)
        )
        if t_flat and t_flat != t:
            block = block | Q(sku__icontains=t_flat)
        qs = qs.filter(block)
    return qs

def _facet_values_any(attr_aliases: list[str], base_qs):
    """
    Вернёт ОЧИЩЕННЫЙ уникальный список значений для любого из имён атрибутов (с алиасами),
    только по товарам из base_qs.
    """
    if not base_qs.exists():
        return []

    q_name = Q()
    for nm in attr_aliases:
        nm = (nm or "").strip()
        if nm:
            # icontains на всякий случай (если в БД 'Производитель ' с пробелом и т.п.)
            q_name |= Q(attribute__name__icontains=nm)

    rows = (ProductAttributeValue.objects
            .filter(q_name, product_id__in=base_qs.values_list("id", flat=True))
            .exclude(value__isnull=True).exclude(value__exact="")
            .values_list("value", flat=True)
            .distinct())

    # подчистим пробелы/мусор и уберём повторы «на всякий»
    cleaned = []
    seen = set()
    for v in rows:
        s = (v or "").strip()
        if not s:
            continue
        k = s.lower()
        if k not in seen:
            seen.add(k)
            cleaned.append(s)

    return sorted(cleaned, key=lambda x: x.lower())

def _cart_map(request):
    raw = request.session.get(CART_SESSION_ID)
    if isinstance(raw, dict):
        out = {}
        for pid, qty in raw.items():
            try:
                pid_i = int(pid)
                qty_i = int(qty)
                if qty_i > 0:
                    out[pid_i] = qty_i
            except Exception:
                continue
        return out
    return {}  # ← всегда словарь

def normalize_ru_phone(s: str):
    """
    Возвращает (+7XXXXXXXXXX, None) если ок,
    либо (None, 'текст ошибки') если не ок.
    """
    digits = re.sub(r"\D+", "", s or "")
    if not digits:
        return None, "Укажите телефон."
    # 8XXXXXXXXXX → 7XXXXXXXXXX
    if digits.startswith("8") and len(digits) == 11:
        digits = "7" + digits[1:]
    # +7… или 7… должны дать 11 цифр и начинаться с 7
    if len(digits) == 11 and digits[0] == "7":
        return "+7" + digits[1:], None  # храним в виде +7XXXXXXXXXX
    return None, "Телефон должен быть в формате +7XXXXXXXXXX (11 цифр)."

def product_list(request):
    qs = (Product.objects
          .filter(is_active=True)
          .prefetch_related("images")
          .select_related("category", "category__parent"))

    # поиск
    q = (request.GET.get("q") or "").strip()
    qs = _apply_text_search(qs, q)

    # параметры
    cat  = (request.GET.get("cat") or "").strip()
    sub  = (request.GET.get("sub") or "").strip()
    brand  = (request.GET.get("brand")  or "").strip()
    make   = (request.GET.get("make")   or "").strip()
    model  = (request.GET.get("model")  or "").strip()
    engine = (request.GET.get("engine") or "").strip()
    order  = (request.GET.get("order")  or "popular").strip()

    # NEW ↓↓↓
    stock = (request.GET.get("stock") or "").strip()   # "in" | "pre" | ""
    # NEW ↑↑↑

    # >>> ПЕРЕНЕСЕНО ВЫШЕ: сначала применим фильтр по категории/подкатегории,
    # чтобы на его основе посчитать фасеты (списки значений)
    if sub:
        qs = qs.filter(category__slug=sub)
    elif cat:
        qs = qs.filter(Q(category__parent__slug=cat) | Q(category__slug=cat))

    # >>> НОВОЕ: снимок "базовой" выборки для фасетов (без атрибутных фильтров)
    # >>> снимок базовой выборки

    # NEW: фильтр по наличию ДО snapshot'а base_qs
    if stock == "in":
        qs = qs.filter(in_stock=True)
    elif stock == "pre":
        qs = qs.filter(in_stock=False)

    base_qs = qs

    # 1) qs_* считаем с учётом остальных выбранных фильтров
    qs_brand  = base_qs
    qs_make   = base_qs
    qs_model  = base_qs
    qs_engine = base_qs

    if make:   qs_brand  = qs_brand.annotate(_m=_exists_attr("Марка авто", make)).filter(_m=True)
    if model:  qs_brand  = qs_brand.annotate(_mo=_exists_attr("Модель авто", model)).filter(_mo=True)
    if engine: qs_brand  = qs_brand.annotate(_e=_exists_attr("Двигатель", engine)).filter(_e=True)

    if brand:  qs_make   = qs_make.annotate(_b=_exists_attr("Производитель", brand)).filter(_b=True)
    if model:  qs_make   = qs_make.annotate(_mo=_exists_attr("Модель авто", model)).filter(_mo=True)
    if engine: qs_make   = qs_make.annotate(_e=_exists_attr("Двигатель", engine)).filter(_e=True)

    if brand:  qs_model  = qs_model.annotate(_b=_exists_attr("Производитель", brand)).filter(_b=True)
    if make:   qs_model  = qs_model.annotate(_m=_exists_attr("Марка авто", make)).filter(_m=True)
    if engine: qs_model  = qs_model.annotate(_e=_exists_attr("Двигатель", engine)).filter(_e=True)

    if brand:  qs_engine = qs_engine.annotate(_b=_exists_attr("Производитель", brand)).filter(_b=True)
    if make:   qs_engine = qs_engine.annotate(_m=_exists_attr("Марка авто", make)).filter(_m=True)
    if model:  qs_engine = qs_engine.annotate(_mo=_exists_attr("Модель авто", model)).filter(_mo=True)

    # 2) Опции селектов считаем из qs_*
    brand_options  = _facet_values_any(["Производитель", "Бренд"], qs_brand)
    make_options   = _facet_values_any(["Марка авто", "Марка"],    qs_make)
    model_options  = _facet_values_any(["Модель авто", "Модель"],  qs_model)
    engine_options = _facet_values_any(["Двигатель", "Engine"],    qs_engine)

    # 3) Если выбранное значение “устарело” — сбрасываем его
    if brand and brand not in brand_options:   brand  = ""
    if make and make not in make_options:      make   = ""
    if model and model not in model_options:   model  = ""
    if engine and engine not in engine_options:engine = ""

    # 4) ДАЛЬШЕ — твои атрибутные фильтры к основному qs (conds ...)


    # атрибутные фильтры (как было)
    conds = [
        ("_has_brand",  _exists_attr("Производитель", brand,  partial=True)),
        ("_has_make",   _exists_attr("Марка авто",    make,   partial=True)),
        ("_has_model",  _exists_attr("Модель авто",   model,  partial=True)),
        ("_has_engine", _exists_attr("Двигатель",     engine, partial=True)),
    ]
    for alias, expr in conds:
        if expr is not None:
            qs = qs.annotate(**{alias: expr}).filter(**{alias: True})

    # сортировка (оставил твою логику)
    if order == "price_asc":
        qs = qs.order_by("price", "-created_at")
    elif order == "price_desc":
        qs = qs.order_by("-price", "-created_at")
    elif order == "new":
        qs = qs.order_by("-created_at")
    else:
        qs = qs.order_by("-created_at")

    top_cats = Category.objects.filter(parent__isnull=True, is_hidden=False).order_by("name")
    sub_cats = Category.objects.filter(parent__slug=cat, is_hidden=False).order_by("name") if cat else []
    qs = qs.distinct()
    paginator = Paginator(qs, 24)

    raw_page = (request.GET.get("page") or "").strip()
    try:
        page_number = int(raw_page)
        if page_number < 1:
            page_number = 1
    except (TypeError, ValueError):
        page_number = 1
    page_obj = paginator.get_page(page_number)
    page_items = _page_window(paginator, page_obj.number, pad=1, max_len=9)

    found_count = paginator.count

    top_cats = Category.objects.filter(parent__isnull=True, is_hidden=False)\
           .prefetch_related("children")\
           .order_by("name")

    current_sub = Category.objects.select_related("parent").filter(slug=sub).first() if sub else None
    current_cat = (current_sub.parent if current_sub else
                Category.objects.filter(slug=cat, parent__isnull=True, is_hidden=False).first() if cat else None)

    params = request.GET.copy(); params.pop("page", None)

    recent_products = recent_qs(request, limit=3)

    cart_qty = _cart_map(request)
    cart_ids = set(cart_qty.keys())

    if request.user.is_authenticated:
        fav_ids = set(request.user.favorites.values_list("product_id", flat=True))
    else:
        fav_ids = set(map(int, request.session.get("fav_ids", [])))
    
    return render(request, "catalog/product_list.html", {
        "page_obj": page_obj, "q": q, "order": order,
        "brand": brand, "make": make, "model": model, "engine": engine,
        "params": params, "cat": cat, "sub": sub, "top_cats": top_cats, "sub_cats": sub_cats, "page_items": page_items, "current_cat": current_cat, "current_sub": current_sub, "recent_products": recent_products,"brand_options": brand_options,
        "make_options": make_options,
        "model_options": model_options,
        "engine_options": engine_options,
        "cart_qty": cart_qty,
        "cart_ids": cart_ids, "fav_ids": fav_ids, "found_count": found_count, "stock": stock,
    })


def _render_product_detail(request, product):
    # сгруппованные характеристики
    grouped = defaultdict(list)
    for row in product.attributes.select_related("attribute").order_by("attribute__name", "value"):
        name = (row.attribute.name or "").strip()
        val  = (row.value or "").strip()
        if val and (not grouped[name] or grouped[name][-1] != val):
            grouped[name].append(val)
    specs = [(name, values) for name, values in grouped.items()]

    top_cats = (Category.objects.filter(parent__isnull=True, is_hidden=False)
                .prefetch_related("children")
                .order_by("name"))

    is_faved = False
    if request.user.is_authenticated:
        is_faved = product.faved_by.filter(user=request.user).exists()

    push_recent(request, product.id)
    recent_products = recent_qs(request, limit=3, exclude_id=product.id)

    # ── ВАЖНО: определить current_cat/current_sub ─────────────────────────
    current_cat = None
    current_sub = None
    if product.category:
        if product.category.parent_id:
            current_cat = product.category.parent     # родитель
            current_sub = product.category            # подкатегория
        else:
            current_cat = product.category            # только родитель

    # ── хлебные крошки для JSON-LD ────────────────────────────────────────
    breadcrumb_items = []
    pos = 1
    if current_cat:
        url_cat = reverse('catalog:product_list') + f"?cat={current_cat.slug}"
        breadcrumb_items.append({
            "@type": "ListItem",
            "position": pos,
            "name": current_cat.name,
            "item": request.build_absolute_uri(url_cat),
        })
        pos += 1
    if current_cat and current_sub:
        url_sub = reverse('catalog:product_list') + f"?cat={current_cat.slug}&sub={current_sub.slug}"
        breadcrumb_items.append({
            "@type": "ListItem",
            "position": pos,
            "name": current_sub.name,
            "item": request.build_absolute_uri(url_sub),
        })
        pos += 1
    breadcrumb_items.append({
        "@type": "ListItem",
        "position": pos,
        "name": product.title,
        "item": request.build_absolute_uri(),  # текущая страница
    })

    # ── бренд и первая картинка для Product JSON-LD ───────────────────────
    brand_name = ""
    for label, values in specs:
        if label.lower() == "производитель" and values:
            brand_name = values[0]
            break
    im = product.images.first()
    first_image_url = im.image.url if im else ""

    return render(request, "catalog/product_detail.html", {
        "product": product,
        "specs": specs,
        "top_cats": top_cats,
        "is_faved": is_faved,
        "qty_in_cart": int(request.session.get("cart_v1", {}).get(str(product.id), 0)) if isinstance(request.session.get("cart_v1", {}), dict) else 0,
        "recent_products": recent_products,
        "current_cat": current_cat,
        "current_sub": current_sub,
        "breadcrumb_items": breadcrumb_items,
        "brand_name": brand_name,
        "first_image_url": first_image_url,
    })

# --- новый канонический путь: /catalog/<cat>/<sub>/<slug>/ ---
def product_detail_by_path(request, cat_slug, sub_slug, slug):
    product = get_object_or_404(
        Product.objects.select_related("category", "category__parent")
                       .prefetch_related("images", "attributes__attribute"),
        slug=slug, is_active=True
    )

    parent = product.category.parent if product.category else None
    if not product.category or not parent or parent.slug != cat_slug or product.category.slug != sub_slug:
        # если путь не совпал с фактической категорией товара — уходим на канонический
        return redirect(product.get_absolute_url(), permanent=True)

    return _render_product_detail(request, product)


# --- старый путь: /catalog/p/<slug>/ — делаем 301 на канонический ---
def product_detail(request, slug):
    product = get_object_or_404(
        Product.objects.select_related("category", "category__parent")
                       .prefetch_related("images", "attributes__attribute"),
        slug=slug, is_active=True
    )

    if product.category and product.category.parent:
        return redirect(product.get_absolute_url(), permanent=True)

    # фолбэк, если у товара нет нормальной иерархии
    return _render_product_detail(request, product)

def _rub(x: Decimal) -> Decimal:
    return Decimal(x).quantize(Decimal("1"), rounding=ROUND_HALF_UP)

def cart_detail(request):
    rows = list(cart_items(request))
    t = cart_totals(request)

    # "старая" цена = round(price*1.1) за единицу; суммируем по qty
    sum_gross = sum(_rub(r["price"] * Decimal("1.1")) * r["qty"] for r in rows)
    sum_net   = t["total"]
    sum_discount = (sum_gross - sum_net)
    if sum_discount < 0:
        sum_discount = Decimal("0")  # на всякий

    qty_forms = {row["product"].id: QtyForm(initial={"qty": row["qty"]}) for row in rows}
    form = OrderForm()

    initial = {}
    if request.user.is_authenticated:
        u = request.user
        p = getattr(u, "profile", None)
        initial = {
            "name":   (p.full_name or u.get_full_name() or u.username or ""),
            "email":  (u.email or ""),
            "phone":  (getattr(p, "phone", "") or ""),
            "address":getattr(p, "address", "") or "",
            "city":   getattr(p, "city", "") or "",
            # дефолты (если хочешь — можешь подставлять из последнего заказа)
            "delivery": "courier",
            "payment":  "card",
            "comment":  "",
        }

    recent_products = recent_qs(request, limit=3)

    return render(request, "catalog/cart.html", {
        "rows": rows, "totals": t, "qty_forms": qty_forms, "form": form,
        "sum_gross": sum_gross, "sum_discount": sum_discount, "sum_net": sum_net, "initial": initial,  "recent_products": recent_products, 
    })


def _cart_count(request):
    raw = request.session.get(CART_SESSION_ID, {})
    if not isinstance(raw, dict):
        return 0
    total = 0
    for _, qty in raw.items():
        try:
            total += int(qty)
        except Exception:
            continue
    return total

def _cart_qty(request, pid: int) -> int:
    raw = request.session.get(CART_SESSION_ID, {})
    try:
        return int(raw.get(str(int(pid)), 0))
    except Exception:
        return 0

@require_POST
def cart_add_ajax(request, pk):
    try:
        qty = int(request.POST.get("qty", 1) or 1)
        if qty < 1:
            qty = 1
    except Exception:
        qty = 1

    get_object_or_404(Product, pk=pk, is_active=True)

    try:
        cart_add(request, pk, qty=qty)
        return JsonResponse({
            "ok": True,
            "cart_count": _cart_count(request),
            "qty": _cart_qty(request, pk),   # ← вот это нужно фронту
        })
    except Exception as e:
        logger.exception("cart_add failed for pk=%s", pk)
        return JsonResponse({"ok": False, "error": str(e)}, status=400)

# опционально, чтобы инициализировать бейдж на любой странице
def cart_badge_count(request):
    return JsonResponse({"count": _cart_count(request)})

def _is_ajax(request) -> bool:
    return request.headers.get("x-requested-with") == "XMLHttpRequest" or \
           "application/json" in (request.headers.get("Accept") or "")

@require_POST
def cart_add_view(request, pk):
    qty = int(request.POST.get("qty", 1) or 1)
    cart_add(request, pk, qty=qty)
    if _is_ajax(request):
        return JsonResponse({"ok": True, "count": _cart_count(request)})
    return HttpResponseRedirect(request.META.get("HTTP_REFERER", reverse("catalog:cart_detail")))

@require_POST
def cart_set_view(request, pk):
    qty = int(request.POST.get("qty", 1) or 1)
    cart_set(request, pk, qty=qty)
    if _is_ajax(request):
        return JsonResponse({"ok": True, "count": _cart_count(request)})
    return HttpResponseRedirect(reverse("catalog:cart_detail"))

@require_POST
def cart_remove_view(request, pk):
    cart_remove(request, pk)
    if _is_ajax(request):
        return JsonResponse({"ok": True, "count": _cart_count(request)})
    return HttpResponseRedirect(reverse("catalog:cart_detail"))

@require_POST
@transaction.atomic
def checkout(request):
    rows = list(cart_items(request))
    if not rows:
        return HttpResponseRedirect(reverse("catalog:cart_detail"))
    for r in rows:
        p = r["product"]
        if (p.supplier or "").strip().lower() in SUPPLIERS_NO_DEDUCT:
            continue
        if p.stock is not None and r["qty"] > p.stock:
            messages.error(request, f"Недостаточно товара «{p.title}». Доступно: {p.stock}")
            return redirect("catalog:cart_detail")

    form = OrderForm(request.POST)
    if not form.is_valid():
        # вернёмся на корзину с ошибками
        return render(request, "catalog/cart.html", {"rows": rows, "totals": cart_totals(request), "form": form})

    order = Order.objects.create(
        user=request.user if request.user.is_authenticated else None,
        name=form.cleaned_data["name"],
        phone=form.cleaned_data["phone"],
        email=form.cleaned_data.get("email",""),
        city=form.cleaned_data.get("city",""),
        address=form.cleaned_data.get("address",""),
        comment=form.cleaned_data.get("comment",""),
        delivery=form.cleaned_data.get("delivery",""),
        payment=form.cleaned_data.get("payment",""),
        total=sum(r["subtotal"] for r in rows),
    )

    for r in rows:
        OrderItem.objects.create(
            order=order,
            product=r["product"],
            title=r["product"].title,
            sku=r["product"].sku,
            price=r["price"],
            qty=r["qty"],
            subtotal=r["subtotal"],
        )

    payload = [{"product_id": r["product"].id, "qty": r["qty"]} for r in rows]
    deduct_local_stock(payload)
    
    request.session["reach_goal"] = "checkout_success"
    cart_clear(request)
    return render(request, "catalog/checkout_success.html", {"order": order})


@login_required
def favorite_toggle(request, pk):
    p = get_object_or_404(Product, pk=pk, is_active=True)
    fav, created = request.user.favorites.get_or_create(product=p)
    if not created:
        fav.delete()
        state = "removed"
    else:
        state = "added"

    # Если AJAX — шлём JSON
    if request.headers.get("x-requested-with") == "XMLHttpRequest":
        return JsonResponse({"ok": True, "state": state})

    # Иначе ведём назад
    return HttpResponseRedirect(request.META.get("HTTP_REFERER", reverse("catalog:favorites")))



@login_required
def favorites_list(request):
    qs = Product.objects.filter(
        faved_by__user=request.user, is_active=True
    ).prefetch_related("images", "attributes__attribute") \
     .order_by("-created_at")

    # товары в корзине (для кнопки «В корзине»)
    raw_cart = request.session.get("cart_v1", {})
    cart_ids = {int(k) for k in raw_cart.keys()} if isinstance(raw_cart, dict) else set()

    paginator = Paginator(qs, 24)
    raw_page = (request.GET.get("page") or "").strip()
    try:
        page_number = int(raw_page)
        if page_number < 1:
            page_number = 1
    except (TypeError, ValueError):
        page_number = 1
    page_obj = paginator.get_page(page_number)
    page_items = _page_window(paginator, page_obj.number, pad=1, max_len=9)

    # соберём характеристики ТОЛЬКО для товаров текущей страницы
    for p in page_obj.object_list:
        grouped = defaultdict(list)
        for row in p.attributes.select_related("attribute").order_by("attribute__name", "value"):
            name = (row.attribute.name or "").strip()
            val  = (row.value or "").strip()
            if not val:
                continue
            if not grouped[name] or grouped[name][-1] != val:
                grouped[name].append(val)
        # положим прямо в объект, чтобы удобно использовать в шаблоне
        p.specs = [(name, values) for name, values in grouped.items()]

    return render(request, "catalog/favorites.html", {
        "page_obj": page_obj,
        "page_items": page_items,
        "cart_ids": cart_ids,
    })

def login_view(request):
    if request.method == "POST":
        form = LoginForm(request.POST)
        next_url = request.POST.get("next") or request.META.get("HTTP_REFERER") or reverse("catalog:product_list")
        if form.is_valid():
            user = form.user
            login(request, user)
            # remember me
            if form.cleaned_data.get("remember"):
                request.session.set_expiry(60*60*24*30)  # 30 дней
            else:
                request.session.set_expiry(0)           # до закрытия браузера
            messages.success(request, "Вы вошли в аккаунт")
            return redirect(next_url)
        else:
            messages.error(request, "Ошибка входа: проверьте логин и пароль")
            return redirect(request.META.get("HTTP_REFERER", reverse("catalog:product_list")))
    return redirect("catalog:product_list")

def logout_view(request):
    logout(request)
    messages.info(request, "Вы вышли из аккаунта")
    return redirect(request.META.get("HTTP_REFERER", reverse("catalog:product_list")))

def register_view(request):
    if request.user.is_authenticated:
        return redirect("catalog:account_profile")  # уже вошёл — в ЛК

    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Регистрация прошла успешно")
            return redirect("catalog:account_profile")
        else:
            messages.error(request, "Исправьте ошибки формы регистрации")
    else:
        form = RegisterForm()
    return render(request, "account/register.html", {"form": form})


@login_required
def account_orders(request):
    # базовый набор заказов для текущего пользователя (с учётом возможного отсутствия поля user)
    qs_base = Order.objects.all()
    if 'user' in {f.name for f in Order._meta.get_fields()}:
        qs_base = qs_base.filter(user=request.user)
    else:
        q = Q()
        if request.user.email:
            q |= Q(email__iexact=request.user.email)
        prof = getattr(request.user, "profile", None)
        if prof and prof.phone:
            q |= Q(phone__iexact=prof.phone)
        qs_base = qs_base.filter(q) if q else Order.objects.none()

    # доступные годы
    years_qs = qs_base.annotate(y=ExtractYear('created_at')) \
                      .values_list('y', flat=True).distinct()
    years = sorted([y for y in years_qs if y], reverse=True)

    # выбранный год из GET
    selected_year = (request.GET.get("year") or "").strip()
    qs = qs_base
    if selected_year.isdigit():
        qs = qs.filter(created_at__year=int(selected_year))

    # итоговый список + префетчи для картинок
    orders = qs.order_by("-created_at").prefetch_related(
        Prefetch(
            "items",
            queryset=OrderItem.objects.select_related("product")
                                      .prefetch_related("product__images")
        )
    )

    return render(request, "account/orders.html", {
        "orders": orders,
        "years": years,
        "selected_year": selected_year,
        "orders_total": qs_base.count(),   # общее число заказов пользователя
    })

@login_required
def account_profile(request):
    profile = request.user.profile
    if request.method == "POST":
        form = ProfileForm(request.POST, instance=profile)
        if form.is_valid():
            form.save()
            messages.success(request, "Данные обновлены")
            return redirect("catalog:account_profile")
        else:
            messages.error(request, "Проверьте поля")
    else:
        form = ProfileForm(instance=profile)
    return render(request, "account/profile.html", {"form": form, "user": request.user})

@login_required
def account_password(request):
    if request.method == "POST":
        form = PasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)
            messages.success(request, "Пароль изменён")
            return redirect("catalog:account_password")
        else:
            messages.error(request, "Проверьте поля")
    else:
        form = PasswordChangeForm(request.user)
    return render(request, "account/password.html", {"form": form})


def login_page(request):
    return render(request, "account/login_page.html", {
        "next": request.GET.get("next", "/")
    })

# Панель: дашборд
def m_dashboard(request):
    return render(request, "manage/base_dashboard.html", {})

# Товары: список
def _page_window1(paginator, current, pad=2):
    """Возвращает список страниц с '…' для компактной навигации."""
    n = paginator.num_pages
    rng = set([1, n] + list(range(max(1, current-pad), min(n, current+pad)+1)))
    # всегда покажем первые/последние 2
    rng |= set(range(1, min(n, 2)+1))
    rng |= set(range(max(1, n-1), n+1))
    out, last = [], 0
    for i in sorted(rng):
        if i - last > 1:
            out.append("…")
        out.append(i)
        last = i
    return out

@login_required
def m_products(request):
    q        = (request.GET.get("q") or "").strip()
    status   = (request.GET.get("status") or "").strip()    # '', 'site','hidden','in','pre'
    supplier = (request.GET.get("supplier") or "").strip()

    qs = Product.objects.all().order_by("-created_at")

    if q:
        qs = qs.filter(Q(title__icontains=q) | Q(sku__icontains=q))

        # Фильтр по поставщику
    if supplier == "__none__":
        qs = qs.filter(Q(supplier__isnull=True) | Q(supplier=""))
    elif supplier:
        qs = qs.filter(supplier__iexact=supplier)

    # Опции селекта (только непустые значения)
    supplier_options = (Product.objects
                        .exclude(Q(supplier__isnull=True) | Q(supplier=""))
                        .values_list("supplier", flat=True)
                        .distinct()
                        .order_by("supplier"))

    # фильтр по статусу
    if status == "site":
        qs = qs.filter(is_active=True)
    elif status == "hidden":
        qs = qs.filter(is_active=False)
    elif status == "in":
        qs = qs.filter(in_stock=True)
    elif status == "pre":
        qs = qs.filter(in_stock=False)

    paginator = Paginator(qs, 100)
    page_obj  = paginator.get_page(request.GET.get("page") or 1)
    page_items = _page_window1(paginator, page_obj.number, pad=2)

    # базовая строка для ссылок пагинации
    query_base = urlencode({"q": q, "status": status, "supplier": supplier})

    return render(request, "manage/products_list.html", {
        "page_obj": page_obj,
        "page_items": page_items,
        "q": q,
        "status": status,
        "supplier": supplier,
        "supplier_options": supplier_options,
        "query_base": query_base,   # напр. "q=xxx&status=in&supplier=sttuning"
    })

def _parse_specs_text(text: str):
    """
    Принимаем блок вида:
      'Марка авто: Audi\nМодель авто: A4, A5'
    Возвращаем список (name, [values...])
    """
    result = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or ":" not in line:
            continue
        name, vals = line.split(":", 1)
        name = name.strip()
        vals = [v.strip() for v in vals.split(",") if v.strip()]
        if name and vals:
            result.append((name, vals))
    return result

def _parse_decimal(s: str):
    """
    Поддерживает: 26 300,50; 26300.50; 26 300,50 (с неразрывными пробелами)
    Возвращает Decimal или None (если пусто).
    """
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    s = s.replace("\u00A0", " ").replace(" ", "").replace(",", ".")
    return Decimal(s)

# Товары: карточка/подробнее
def m_product_detail(request, pk):
    p = get_object_or_404(
        Product.objects.select_related("category__parent").prefetch_related("images", "attributes__attribute"),
        pk=pk
    )

    # для селектов категорий
    top_cats = Category.objects.filter(parent__isnull=True, is_hidden=False).order_by("name")
    parent = p.category.parent if p.category and p.category.parent_id else p.category  # если товар висит на родителе
    subs = Category.objects.filter(parent=parent, is_hidden=False).order_by("name") if parent else Category.objects.none()

    # собрать текущие характеристики в текст
    lines = []
    grouped = {}
    for row in p.attributes.all():
        grouped.setdefault(row.attribute.name, []).append(row.value)
    for name, values in grouped.items():
        lines.append(f"{name}: {', '.join(values)}")
    specs_text = "\n".join(lines)

    if request.method == "POST":
        action = request.POST.get("action")

        # 1) Сохранение основных полей/категорий/цен
        if action == "save_main":
            stock_raw = request.POST.get("stock", "").strip()
            p.stock = None if stock_raw == "" else int(stock_raw)
            p.title = (request.POST.get("title") or "").strip() or p.title
            p.description = (request.POST.get("description") or "").strip()
            sku_new = (request.POST.get("sku") or "").strip()
            if sku_new and sku_new != p.sku:
                # проверим уникальность, иначе получишь IntegrityError
                if Product.objects.filter(sku=sku_new).exclude(pk=p.pk).exists():
                    messages.error(request, "Такой SKU уже существует.")
                    return redirect("catalog:m_product_detail", pk=pk)
                p.sku = sku_new

            cost_price_txt = (request.POST.get("cost_price") or "").strip()
            price_str = request.POST.get("price")
            old_str   = request.POST.get("old_price")
            parent_id = (request.POST.get("parent_id") or "").strip()
            sub_id    = (request.POST.get("sub_id") or "").strip()

            try:
                val = _parse_decimal(price_str)
                if val is not None:
                    p.price = val
                    p.cost_price = new_cost_price = _parse_decimal(cost_price_txt)
            except InvalidOperation:
                messages.error(request, "Неверный формат цены.")  # ← эта ошибка и была

            try:
                val_old = _parse_decimal(old_str)
                p.old_price = val_old  # None если пусто
            except InvalidOperation:
                messages.error(request, "Неверный формат старой цены.")
            
            # выбор категории: если указана подкатегория — берём её; иначе, если указан родитель — привяжем к нему
            new_cat = None
            if sub_id:
                new_cat = Category.objects.filter(id=sub_id, is_hidden=False).first()
            elif parent_id:
                new_cat = Category.objects.filter(id=parent_id, is_hidden=False).first()
            if new_cat:
                p.category = new_cat

            p.save()
            messages.success(request, "Основные поля сохранены.")
            return redirect("catalog:m_product_detail", pk=pk)

        # 2) Полная замена характеристик текстовым блоком
        if action == "save_specs":
            parsed = _parse_specs_text(request.POST.get("specs_text", ""))
            # очистить и записать заново
            ProductAttributeValue.objects.filter(product=p).delete()
            for name, values in parsed:
                attr, _ = Attribute.objects.get_or_create(name=name)
                for v in values:
                    ProductAttributeValue.objects.create(product=p, attribute=attr, value=v)
            messages.success(request, "Характеристики обновлены.")
            return redirect("catalog:m_product_detail", pk=pk)

        # 3) Добавление фото (мультизагрузка)
        if action == "images_add":
            files = request.FILES.getlist("images")
            if not files:
                messages.error(request, "Выберите файлы для загрузки.")
            else:
                created = 0
                for f in files:
                    ProductImage.objects.create(product=p, image=f)
                    created += 1
                messages.success(request, f"Добавлено фото: {created}")
            return redirect("catalog:m_product_detail", pk=pk)


    brands = list(
        p.attributes
        .filter(attribute__name__iexact="Производитель")
        .values_list("value", flat=True)
    )

    upload_form = ProductImagesUploadForm()
    return render(request, "manage/product_detail.html", {
        "p": p,
        "upload_form": upload_form,
        "top_cats": top_cats,
        "parent": parent,
        "subs": subs,
        "specs_text": specs_text,
        "brands": brands,
    })

# Товары: скрыть/показать
def m_product_toggle(request, pk):
    if request.method != "POST":
        return redirect("catalog:m_product_detail", pk=pk)
    p = get_object_or_404(Product, pk=pk)
    p.is_active = not p.is_active
    p.save(update_fields=["is_active"])
    messages.success(request, f"Товар «{p.title}» теперь {'на сайте' if p.is_active else 'скрыт'}.")
    return redirect("catalog:m_product_detail", pk=pk)

# Товары: удалить
def m_product_delete(request, pk):
    if request.method != "POST":
        return redirect("catalog:m_product_detail", pk=pk)
    p = get_object_or_404(Product, pk=pk)
    title = p.title
    # При желании, можно удалить файлы картинок физически:
    for img in p.images.all():
        try:
            img.image.delete(save=False)
        except Exception:
            pass
    p.delete()
    messages.success(request, f"Товар «{title}» удалён.")
    return redirect("catalog:m_products")

# Товары: добавить фото
@require_POST
def m_product_images_add(request, pk):
    p = get_object_or_404(Product, pk=pk)

    # НЕ используем form.is_valid() — сразу забираем список файлов:
    files = request.FILES.getlist("images")

    if not files:
        messages.error(request, "Выберите файлы для загрузки.")
        return redirect("catalog:m_product_detail", pk=pk)

    created = 0
    for f in files:
        # Можно добавить элементарную фильтрацию по расширению/типу, если нужно
        ProductImage.objects.create(product=p, image=f)
        created += 1

    messages.success(request, f"Добавлено фото: {created}")
    return redirect("catalog:m_product_detail", pk=pk)

# Товары: удалить одно фото
def m_product_image_delete(request, pk, image_id):
    p = get_object_or_404(Product, pk=pk)
    img = get_object_or_404(ProductImage, pk=image_id, product=p)
    if request.method == "POST":
        try:
            img.image.delete(save=False)
        except Exception:
            pass
        img.delete()
        messages.success(request, "Фото удалено.")
    return redirect("catalog:m_product_detail", pk=pk)

# Пользователи: таблица
def m_users(request):
    User = get_user_model()
    q = (request.GET.get("q") or "").strip()
    qs = User.objects.all().select_related("profile").order_by("-date_joined")
    if q:
        qs = qs.filter(
            Q(username__icontains=q) |
            Q(email__icontains=q) |
            Q(profile__full_name__icontains=q) |
            Q(profile__phone__icontains=q) |
            Q(profile__city__icontains=q)
        )
    paginator = Paginator(qs, 50)
    page = paginator.get_page(request.GET.get("page") or 1)
    return render(request, "manage/users_list.html", {
        "page_obj": page,
        "q": q,
    })

# Заказы: таблица + смена статуса
def m_orders(request):
    q = (request.GET.get("q") or "").strip()
    status = (request.GET.get("status") or "").strip()

    qs = Order.objects.all().order_by("-created_at").prefetch_related(
        Prefetch("items", queryset=OrderItem.objects.select_related("product").prefetch_related("product__images"))
    )
    if q:
        qs = qs.filter(
            Q(id__icontains=q) |
            Q(email__icontains=q) |
            Q(name__icontains=q) |
            Q(phone__icontains=q)
        )
    if status:
        qs = qs.filter(status=status)

    status_choices = Order._meta.get_field("status").choices  # ← передадим в шаблон

    paginator = Paginator(qs, 30)
    page_obj = paginator.get_page(request.GET.get("page") or 1)

    # формы только для текущей страницы
    rows = [(o, OrderStatusForm(instance=o, prefix=f"o{o.id}")) for o in page_obj.object_list]

    return render(request, "manage/orders_list.html", {
        "q": q,
        "status": status,
        "status_choices": status_choices,  # ← используем в селекте
        "page_obj": page_obj,
        "rows": rows,                      # ← (order, form) на строку таблицы
    })

# POST endpoint смены статуса
def m_order_status(request, pk):
    o = get_object_or_404(Order, pk=pk)
    if request.method != "POST":
        return redirect("catalog:m_orders")
    form = OrderStatusForm(request.POST, instance=o, prefix=f"o{pk}")
    if form.is_valid():
        form.save()
        messages.success(request, f"Статус заказа №{pk} обновлён.")
    else:
        messages.error(request, f"Не удалось обновить статус заказа №{pk}.")
    return redirect("catalog:m_orders")

def _distinct_attr_values(attr_ru_name, limit=150):
    """
    Вернёт список уникальных значений характеристики (Производитель/Марка авто/…)
    для выпадашек на главной.
    """
    return (ProductAttributeValue.objects
            .filter(attribute__name__iexact=attr_ru_name)
            .values_list("value", flat=True)
            .order_by("value")
            .distinct()[:limit])

RECENT_SESSION_KEY = "recent_viewed"

def push_recent_viewed(request, product_id: int, limit: int = 20):
    """Добавляет товар в «ранее смотрели» (в начало), без дублей, с ограничением длины."""
    try:
        pid = int(product_id)
    except (TypeError, ValueError):
        return
    lst = request.session.get(RECENT_SESSION_KEY) or []
    # нормализуем к int
    norm = []
    for x in lst:
        try:
            xi = int(x)
        except (TypeError, ValueError):
            continue
        if xi != pid:
            norm.append(xi)
    request.session[RECENT_SESSION_KEY] = [pid] + norm[: max(0, limit - 1)]
    request.session.modified = True

def get_recent_products(request, limit: int = 6):
    """Возвращает список товаров из «ранее смотрели» в том же порядке, что в сессии."""
    lst = request.session.get(RECENT_SESSION_KEY) or []
    ids = []
    for x in lst:
        try:
            ids.append(int(x))
        except (TypeError, ValueError):
            continue
    if not ids:
        return []
    order = Case(*[When(pk=pk, then=pos) for pos, pk in enumerate(ids)], output_field=IntegerField())
    qs = (Product.objects
          .filter(pk__in=ids, is_active=True)
          .prefetch_related("images")
          .order_by(order)[:limit])
    return list(qs)

def home(request):
    # данные для фильтров
    brands  = _distinct_attr_values("Производитель")
    makes   = _distinct_attr_values("Марка авто")
    models_ = _distinct_attr_values("Модель авто")
    engines = _distinct_attr_values("Двигатель")

    # последние поступления (8)
    curated = list(HomeLatestItem.objects.filter(is_active=True)
                   .select_related("product").order_by("position"))
    if curated:
        latest_products = [i.product for i in curated]
    else:
        latest_products = Product.objects.filter(is_active=True).order_by("-created_at")[:8]

    # ранее вы смотрели — читаем из сессии (нужно чтобы в product_detail мы пушили id в recent_viewed)
    recent_ids = request.session.get("recent_viewed") or []
    recent_products = get_recent_products(request, limit=6)

    page = SitePage.objects.filter(slug="home").first()
    brands = Brand.objects.order_by("name").only("id", "name", "slug", "logo")
    placements = (BannerPlacement.objects
                  .filter(page_key=BannerPlacement.PageKey.HOME, banner__is_active=True)
                  .select_related("banner").order_by("position"))
    banners_main = [p.banner for p in placements]
    return render(request, "home/index.html", {
        "brands": brands,
        "makes": makes,
        "models_": models_,
        "engines": engines,
        "latest_products": latest_products,
        "recent_products": recent_products,
        "brands": brands, 
        "page": page,
        "banners_main": banners_main,
        "form": ServiceRequestForm(),    
        "RECAPTCHA_SITE_KEY": settings.RECAPTCHA_SITE_KEY,   
    })

def chip_tuning_view(request):
    # выбранные значения из GET
    make   = (request.GET.get("make")   or "").strip()
    model  = (request.GET.get("model")  or "").strip()
    year   = (request.GET.get("year")   or "").strip() 
    engine = (request.GET.get("engine") or "").strip()
    stage  = (request.GET.get("stage")  or "").strip()

    if request.method == "POST" and request.POST.get("chip_lead") == "1":
        name  = (request.POST.get("name")  or "").strip()
        phone = (request.POST.get("phone") or "").strip()
        car_id   = request.POST.get("car_id")
        stage_id = request.POST.get("stage_id")

        token = request.POST.get("g-recaptcha-response")
        ok, payload = verify_recaptcha(token, request.META.get("REMOTE_ADDR"))
        if settings.DEBUG and not ok:
            ok = True  # позволяем в DEV

        if not ok:
            messages.error(request, "Подтвердите, что вы не робот.")
            # просто отрисуем эту же страницу, чтобы значения не потерялись
            # (ниже рендер вернёт тот же контекст и шаблон)
            # Никаких редиректов здесь не делаем
            # ↓↓↓ выйдем к общему render() в конце вьюхи
        else:
            if not (name and phone and car_id):
                messages.error(request, "Заполните имя, телефон и выберите модификацию.")
                return redirect(request.get_full_path())  # вернемся с теми же GET

            car = get_object_or_404(ChipCar, id=car_id)
            stage_obj = ChipStage.objects.filter(id=stage_id).first() if stage_id else None

            ChipRequest.objects.create(
                user=request.user if request.user.is_authenticated else None,
                name=name,
                phone=phone,
                car=car,
                stage=stage_obj,
            )
            request.session["reach_goal"] = "chip_tuning_form_submit"
            messages.success(request, "Спасибо! Заявка отправлена, скоро свяжемся.")
            return redirect(request.get_full_path())

    # списки значений для селектов (каскадно)
    make_options = list(ChipCar.objects.order_by("make")
                        .values_list("make", flat=True).distinct())

    model_qs = ChipCar.objects.all()
    if make:
        model_qs = model_qs.filter(make=make)
    model_options = list(model_qs.order_by("model")
                         .values_list("model", flat=True).distinct())

    year_qs = model_qs
    if model:
        year_qs = year_qs.filter(model=model)
    year_options = list(year_qs.order_by("year")
                        .values_list("year", flat=True).distinct())

    engine_qs = year_qs
    if year:
        engine_qs = engine_qs.filter(year=year)
    engine_options = list(engine_qs.order_by("engine")
                          .values_list("engine", flat=True).distinct())

    # текущая машина
    car = None
    if make and model and year and engine:
        car = ChipCar.objects.filter(make=make, model=model, year=year, engine=engine).first()

    stage_raw = (request.GET.get("stage") or "").strip()

    stages = []
    current_stage = None
    metrics = []

    if car:
        stages = list(car.stages.order_by("id"))  # стабильный порядок

        # 1) если передали id
        if stage_raw.isdigit():
            sid = int(stage_raw)
            current_stage = next((s for s in stages if s.id == sid), None)

        # 2) иначе пробуем по имени (на случай старых ссылок)
        if current_stage is None and stage_raw:
            current_stage = next((s for s in stages if s.name.lower() == stage_raw.lower()), None)

        # 3) дефолт — первый stage
        if current_stage is None and stages:
            current_stage = stages[0]

        if current_stage:
            metrics = list(current_stage.metrics.order_by("order", "id"))

    # ---- «другие модификации» ----
    base_model = ""
    siblings_rows = []

    if car:
        base_model = (car.model or "").split()[0]

        siblings = (
            ChipCar.objects
            .filter(make=car.make, model__startswith=base_model)
            .exclude(id=car.id)
            .order_by("model", "engine", "year")
            .prefetch_related("stages__metrics")
        )


        def norm(s: str) -> str:
            return (s or "").lower().replace("ё", "е").strip()

        for sc in siblings:
            cfg_stages = list(sc.stages.all())  # <--- другое имя!
            if not cfg_stages:
                continue
            st = next((x for x in cfg_stages if norm(x.name) == "stage 1"), cfg_stages[0])

            mmap = {norm(m.label): m for m in st.metrics.all()}
            vol    = (mmap.get("объем двигателя") or mmap.get("обьем двигателя"))
            power  = mmap.get("мощность")
            torque = mmap.get("крутящий момент")

            qs = urlencode({"make": sc.make, "model": sc.model, "year": sc.year, "engine": sc.engine})
            siblings_rows.append({
                "title": f"{sc.make} {sc.model}",
                "engine_volume": getattr(vol, "stock", "") if vol else "",
                "power":          getattr(power, "stock", "") if power else "",
                "torque":         getattr(torque, "stock", "") if torque else "",
                "url":            f"{reverse('catalog:page_chip_tuning')}?{qs}",
            })

    return render(request, "static_pages/chip_tuning.html", {
        "make": make, "model": model, "year": year, "engine": engine, "stage": stage,
        "make_options": make_options, "model_options": model_options,
        "year_options": year_options, "engine_options": engine_options,
        "car": car, "stages": stages, "current_stage": current_stage, "metrics": metrics,
        "base_model": base_model, "siblings_rows": siblings_rows, "RECAPTCHA_SITE_KEY": settings.RECAPTCHA_SITE_KEY,
    })

def m_chip_list(request):
    q = (request.GET.get("q") or "").strip()
    qs = ChipCar.objects.all().order_by("make","model","year","engine")
    if q:
        from django.db.models import Q
        qs = qs.filter(
            Q(make__icontains=q) | Q(model__icontains=q) |
            Q(year__icontains=q) | Q(engine__icontains=q)
        )
    paginator = Paginator(qs, 1000)
    page_obj = paginator.get_page(request.GET.get("page") or 1)
    page_items = _page_window(paginator, page_obj.number)
    return render(request, "manage/chip_list.html", {
        "page_obj": page_obj, "page_items": page_items, "q": q,
    })

def _to_dec(s):
    if s is None:
        return None
    s = str(s).strip().replace(' ', '').replace(',', '.')
    try:
        return Decimal(s)
    except Exception:
        return None

def m_chip_detail(request, car_id):
    cfg = get_object_or_404(ChipCar, id=car_id)
    stages = list(cfg.stages.prefetch_related('metrics').all())

    if request.method == "POST":
        action = request.POST.get("action")

        if action == "save_stage":
            stage_id = int(request.POST.get("stage_id"))
            stage = next(s for s in stages if s.id == stage_id)

            # обновляем все существующие строки-метрики
            for m in stage.metrics.all():
                m.label = request.POST.get(f"label_{m.id}", m.label)
                m.stock = request.POST.get(f"stock_{m.id}", m.stock)
                m.tuned = request.POST.get(f"tuned_{m.id}", m.tuned)
                m.delta = request.POST.get(f"delta_{m.id}", m.delta)
                # позволяем хранить цену в любой строке; обычно будет только в одной
                price_key = f"price_{m.id}"
                if price_key in request.POST:
                    m.price = _to_dec(request.POST.get(price_key))
                m.save()

            # «этапная» цена одним полем — кладём её в первую строку (создадим при необходимости)
            if "price" in request.POST:
                p = _to_dec(request.POST.get("price"))
                first = stage.metrics.order_by('order', 'id').first()
                if not first:
                    first = ChipMetric.objects.create(stage=stage, order=0, label="Параметр")
                first.price = p
                first.save()

            messages.success(request, "Сохранено")
            return redirect("catalog:m_chip_detail", car_id=cfg.id)

        elif action == "add_metric":
            stage_id = int(request.POST.get("stage_id"))
            stage = next(s for s in stages if s.id == stage_id)
            label = (request.POST.get("new_label") or "Параметр").strip()
            order = (stage.metrics.aggregate(Max('order'))['order__max'] or 0) + 10
            ChipMetric.objects.create(stage=stage, order=order, label=label)
            return redirect("catalog:m_chip_detail", car_id=cfg.id)

        elif action == "del_metric":
            metric_id = int(request.POST.get("metric_id"))
            ChipMetric.objects.filter(id=metric_id, stage__car=cfg).delete()
            return redirect("catalog:m_chip_detail", car_id=cfg.id)

    return render(request, "manage/chip_detail.html", {
        "cfg": cfg,
        "stages": stages,
    })
    car = get_object_or_404(ChipCar, pk=car_id)
    stages = car.stages.prefetch_related("metrics").order_by("name")

    def find_metric(stage, needle: str):
        n = needle.lower()
        for m in stage.metrics.all():
            if n in m.label.lower():
                return m
        # «болванка», чтобы шаблон мог безопасно читать атрибуты
        return ChipMetric(stage=stage, label=needle, stock="", tuned="", delta="", price=None)

    # соберём данные для шаблона
    stage_rows = []
    for s in stages:
        power  = find_metric(s, "Мощность")
        torque = find_metric(s, "Крут")
        z100   = find_metric(s, "0-100")
        # где хранить цену? берём из строки «Мощность»
        price_metric = power
        stage_rows.append({
            "stage": s,
            "power": power,
            "torque": torque,
            "z100": z100,
            "price_metric": price_metric,
        })

    if request.method == "POST":
        # сохранение одной строки по кнопке этой стадии
        sid = int(request.POST.get("stage_id"))
        s   = stages.get(pk=sid)

        # читаем значения
        p_stock = (request.POST.get(f"power_stock_{sid}") or "").strip()
        p_tuned = (request.POST.get(f"power_tuned_{sid}") or "").strip()
        p_delta = (request.POST.get(f"power_delta_{sid}") or "").strip()

        t_stock = (request.POST.get(f"torque_stock_{sid}") or "").strip()
        t_tuned = (request.POST.get(f"torque_tuned_{sid}") or "").strip()
        t_delta = (request.POST.get(f"torque_delta_{sid}") or "").strip()

        a_stock = (request.POST.get(f"z100_stock_{sid}") or "").strip()
        a_tuned = (request.POST.get(f"z100_tuned_{sid}") or "").strip()
        a_delta = (request.POST.get(f"z100_delta_{sid}") or "").strip()

        price_raw = (request.POST.get(f"price_{sid}") or "").strip()
        price_val = None
        if price_raw:
            # нормализуем цену: «310000», «310 000», «310,000.00»
            clean = re.sub(r"[^\d.,]", "", price_raw).replace(",", ".")
            try:
                price_val = Decimal(clean)
            except Exception:
                messages.error(request, "Неверный формат цены.")
                return redirect(request.path)

        # upsert трёх строк метрик
        def upsert(label, stock, tuned, delta, price=None):
            obj = find_metric(s, label)
            creating = obj.pk is None
            obj.label = label
            obj.stock = stock
            obj.tuned = tuned
            obj.delta = delta
            if label == "Мощность":
                obj.price = price  # цену держим в строке «Мощность»
            obj.stage = s
            obj.save()
            if creating:
                s.metrics.add(obj)

        upsert("Мощность",         p_stock, p_tuned, p_delta, price_val)
        upsert("Крутящий момент",  t_stock, t_tuned, t_delta)
        upsert("0–100 км/ч",       a_stock, a_tuned, a_delta)

        messages.success(request, "Сохранено.")
        return redirect(request.path)

    return render(request, "manage/chip_detail.html", {
        "car": car,
        "rows": stage_rows,
    })


def autoservice_landing(request):
    # чтение выбранных фильтров (для селектов и редиректа после POST)
    make   = (request.GET.get("make") or "").strip()
    model  = (request.GET.get("model") or "").strip()
    year   = (request.GET.get("year") or "").strip()
    engine = (request.GET.get("engine") or "").strip()

    # ---- ОБРАБОТКА ФОРМЫ ЗАЯВКИ (POST) ----

    base_qs = ChipCar.objects.all()

    makes = list(base_qs.order_by("make")
                 .values_list("make", flat=True).distinct())

    models = (list(base_qs.filter(make=make).order_by("model")
                   .values_list("model", flat=True).distinct())
              if make else [])

    years = (list(base_qs.filter(make=make, model=model).order_by("year")
                  .values_list("year", flat=True).distinct())
             if (make and model) else [])

    engines = (list(base_qs.filter(make=make, model=model, year=year)
                    .order_by("engine")
                    .values_list("engine", flat=True).distinct())
               if (make and model and year) else [])

    news = News.objects.filter(is_published=True).order_by("-published_at")[:12]
    
    placements = (BannerPlacement.objects
                  .filter(page_key=BannerPlacement.PageKey.AUTOSERVICE, banner__is_active=True)
                  .select_related("banner").order_by("position"))
    banners_main = [p.banner for p in placements]

    if request.method == "POST":
        token = request.POST.get("g-recaptcha-response")
        ok, payload = verify_recaptcha_v2(token, request.META.get("REMOTE_ADDR"))
        logger.warning("reCAPTCHA payload: %s", payload)

        # Если хотите отключить в DEV:
        if settings.DEBUG and not ok:
            logger.warning("reCAPTCHA failed in DEBUG, payload=%s", payload)
            ok = True  # пропускаем в dev

        form = ServiceRequestForm(request.POST)  # bound-форма, сохраняет введённые поля

        logger.warning("g-recaptcha-response(first 16): %r", (token or "")[:16])

        if not ok:
            messages.error(request, "Подтвердите, что вы не робот.")

            # подготовка контекста (makes/models/years/engines/news/banners_main) — как в коде выше
            return render(request, "static_pages/avtoservis_landing.html", {
                "make": make, "model": model, "year": year, "engine": engine,
                "makes": makes, "models": models, "years": years, "engines": engines,
                "news": news, "banners_main": banners_main,
                "RECAPTCHA_SITE_KEY": settings.RECAPTCHA_SITE_KEY,
                "form": form,
            })

        if form.is_valid():
            obj = form.save(commit=False)
            if request.user.is_authenticated:
                obj.user = request.user
            # добавим выбранную конфигурацию в комментарий (чтобы менеджеру было проще)
            suffix = ""
            if any([make, model, year, engine]):
                suffix = f"\n\nВыбран авто: {make or '-'} {model or '-'} {year or '-'} {engine or '-'}"
            obj.comment = (obj.comment or "") + suffix
            obj.source_url = request.build_absolute_uri()
            obj.save()
            request.session["reach_goal"] = "autoservice_form_submit"
            messages.success(request, "Заявка отправлена! Мы свяжемся с вами.")
            qs = urlencode({"make": make, "model": model, "year": year, "engine": engine})
            return redirect(f"{request.path}?{qs}" if qs else f"{request.path}")
        else:
            messages.error(request, "Исправьте ошибки и отправьте снова.")
            return render(request, "static_pages/avtoservis_landing.html", {
                "make": make, "model": model, "year": year, "engine": engine,
                "makes": makes, "models": models, "years": years, "engines": engines,
                "news": news, "banners_main": banners_main,
                "RECAPTCHA_SITE_KEY": settings.RECAPTCHA_SITE_KEY,
                "form": form,  # ← важное: bound-форма из POST
            })

        # редирект на эту же страницу, сохраняя выбранные селекты
       

    # ---- ДАННЫЕ ДЛЯ СЕЛЕКТОВ ----


    return render(request, "static_pages/avtoservis_landing.html", {
        "make": make, "model": model, "year": year, "engine": engine,
        "makes": makes, "models": models, "years": years, "engines": engines, "news": news, "banners_main": banners_main, "RECAPTCHA_SITE_KEY": settings.RECAPTCHA_SITE_KEY,
    })


def m_chip_leads(request):
    qs = (ChipRequest.objects
          .select_related("car", "stage")
          .order_by("-created_at"))

    # простые фильтры в списке (по статусу/поиску)
    status = (request.GET.get("status") or "").strip()
    search = (request.GET.get("q") or "").strip()
    if status:
        qs = qs.filter(status=status)
    if search:
        qs = qs.filter(
            Q(name__icontains=search) |
            Q(phone__icontains=search) |
            Q(car__make__icontains=search) |
            Q(car__model__icontains=search) |
            Q(car__engine__icontains=search) |
            Q(car__year__icontains=search)
        )

    paginator = Paginator(qs, 30)
    page_obj = paginator.get_page(request.GET.get("page") or 1)

    return render(request, "manage/chip_leads_list.html", {
        "page_obj": page_obj,
        "status": status,
        "q": search,
        "status_choices": ChipRequest.STATUS_CHOICES,
    })


def m_chip_lead_detail(request, lead_id):
    lead = get_object_or_404(ChipRequest.objects.select_related("car", "stage", "user"), id=lead_id)

    if request.method == "POST":
        new_status = request.POST.get("status")
        if new_status in dict(ChipRequest.STATUS_CHOICES):
            lead.status = new_status
            lead.save(update_fields=["status"])
            messages.success(request, "Статус обновлён.")
            return redirect("catalog:m_chip_lead_detail", lead_id=lead.id)

    return render(request, "manage/chip_lead_detail.html", {"lead": lead, "status_choices": ChipRequest.STATUS_CHOICES,})

def m_service_leads(request):
    status = (request.GET.get("status") or "").strip()
    q = (request.GET.get("q") or "").strip()

    qs = ServiceRequest.objects.all()
    if status:
        qs = qs.filter(status=status)
    if q:
        qs = qs.filter(
            Q(name__icontains=q) |
            Q(phone__icontains=q) |
            Q(comment__icontains=q)
        )

    paginator = Paginator(qs, 10000)
    page_obj = paginator.get_page(request.GET.get("page") or 1)

    return render(request, "manage/service_leads_list.html", {
        "page_obj": page_obj,
        "status": status,
        "q": q,
        "status_choices": ServiceRequest.STATUS_CHOICES,
    })


def m_service_lead_detail(request, lead_id):
    lead = get_object_or_404(ServiceRequest, id=lead_id)

    if request.method == "POST":
        new_status = (request.POST.get("status") or "").strip()
        if new_status in dict(ServiceRequest.STATUS_CHOICES):
            lead.status = new_status
            lead.save(update_fields=["status"])
            messages.success(request, "Статус обновлён.")
        return redirect("catalog:m_service_lead_detail", lead_id=lead.id)

    return render(request, "manage/service_lead_detail.html", {
        "lead": lead,
        "status_choices": ServiceRequest.STATUS_CHOICES,
    })


def news_detail(request, slug):
    n = get_object_or_404(News, slug=slug, is_published=True)
    return render(request, "static_pages/news_detail.html", {"n": n})

def m_news_list(request):
    q = (request.GET.get("q") or "").strip()
    qs = News.objects.all()
    if q:
        qs = qs.filter(Q(title__icontains=q) | Q(excerpt__icontains=q) | Q(body__icontains=q))
    qs = qs.order_by("-published_at", "-id")

    paginator = Paginator(qs, 20)
    page_number = int(request.GET.get("page") or 1)
    page_obj = paginator.get_page(page_number)
    page_items = _page_window(paginator, page_obj.number, pad=1, max_len=9)

    return render(request, "manage/news_list.html", {
        "page_obj": page_obj,
        "page_items": page_items,
        "q": q,
    })

def m_news_new(request):
    if request.method == "POST":
        form = AdminNewsForm(request.POST, request.FILES)
        if form.is_valid():
            obj = form.save()
            messages.success(request, "Новость создана.")
            return redirect("catalog:m_news_edit", pk=obj.pk)
        messages.error(request, "Исправьте ошибки формы.")
    else:
        form = AdminNewsForm()
    return render(request, "manage/news_form.html", {"form": form, "obj": None})

def m_news_edit(request, pk):
    obj = get_object_or_404(News, pk=pk)
    if request.method == "POST":
        form = AdminNewsForm(request.POST, request.FILES, instance=obj)
        if form.is_valid():
            remove = form.cleaned_data.get("remove_image")
            obj = form.save(commit=False)
            if remove and obj.image:
                obj.image.delete(save=False)
                obj.image = None
            obj.save()
            messages.success(request, "Изменения сохранены.")
            return redirect("catalog:m_news_edit", pk=obj.pk)
        messages.error(request, "Исправьте ошибки формы.")
    else:
        form = AdminNewsForm(instance=obj)
    return render(request, "manage/news_form.html", {"form": form, "obj": obj})

def m_news_toggle(request, pk):
    if request.method != "POST":
        return redirect("catalog:m_news_list")
    obj = get_object_or_404(News, pk=pk)
    obj.is_published = not obj.is_published
    obj.save(update_fields=["is_published"])
    messages.success(request, ("Опубликовано" if obj.is_published else "Скрыто") + f": «{obj.title}».")
    # вернёмся туда, откуда пришли
    return redirect(request.META.get("HTTP_REFERER") or reverse("catalog:m_news_list"))

def m_news_delete(request, pk):
    if request.method != "POST":
        return redirect("catalog:m_news_list")
    obj = get_object_or_404(News, pk=pk)
    title = obj.title
    obj.delete()
    messages.success(request, f"Новость «{title}» удалена.")
    return redirect("catalog:m_news_list")

def news_list_public(request):
    qs = News.objects.filter(is_published=True).order_by("-published_at", "-id")
    paginator = Paginator(qs, 9)  # по 9 на страницу (3x3 сетка)
    try:
        page_number = int((request.GET.get("page") or "1"))
        if page_number < 1: page_number = 1
    except ValueError:
        page_number = 1
    page_obj = paginator.get_page(page_number)
    page_items = _page_window(paginator, page_obj.number, pad=1, max_len=9)
    return render(request, "static_pages/news.html", {
        "page_obj": page_obj,
        "page_items": page_items,
    })

def brands_public(request):
    qs = Brand.objects.order_by("name")
    paginator = Paginator(qs, 20)  # по 60 логотипов на страницу
    try:
        page_number = int((request.GET.get("page") or "1"))
        if page_number < 1: page_number = 1
    except ValueError:
        page_number = 1
    page_obj = paginator.get_page(page_number)
    page_items = _page_window(paginator, page_obj.number, pad=1, max_len=9)

    return render(request, "static_pages/brands.html", {
        "page_obj": page_obj,
        "page_items": page_items,
    })

def m_brands(request):
    qs = Brand.objects.order_by("name")
    paginator = Paginator(qs, 50)
    page_number = int((request.GET.get("page") or "1") or 1)
    page_obj = paginator.get_page(page_number)
    page_items = _page_window(paginator, page_obj.number, pad=1, max_len=9)
    return render(request, "manage/brands_list.html", {
        "page_obj": page_obj, "page_items": page_items
    })

@require_POST
def m_brand_add(request):
    name = (request.POST.get("name") or "").strip()
    if not name:
        messages.error(request, "Укажите название бренда.")
        return redirect("catalog:m_brands")
    Brand.objects.get_or_create(name=name)
    messages.success(request, f"Бренд «{name}» добавлен (или уже существовал).")
    return redirect("catalog:m_brands")

@require_POST
def m_brand_logo(request, brand_id):
    b = get_object_or_404(Brand, pk=brand_id)
    f = request.FILES.get("logo")
    if not f:
        messages.error(request, "Не выбран файл логотипа.")
        return redirect("catalog:m_brands")
    b.logo = f
    b.save(update_fields=["logo"])
    messages.success(request, f"Логотип для «{b.name}» обновлён.")
    return redirect("catalog:m_brands")

@require_POST
def m_brand_delete(request, brand_id):
    b = get_object_or_404(Brand, pk=brand_id)
    name = b.name
    b.delete()
    messages.success(request, f"Бренд «{name}» удалён.")
    return redirect("catalog:m_brands")

def m_brands_sync(request):
    # Подтянуть бренды из атрибута «Производитель»
    attr = Attribute.objects.filter(name="Производитель").first()
    created = 0
    if attr:
        values = (ProductAttributeValue.objects
                  .filter(attribute=attr)
                  .values_list("value", flat=True).distinct())
        for v in values:
            title = (v or "").strip()
            if title and not Brand.objects.filter(name=title).exists():
                Brand.objects.create(name=title)
                created += 1
    messages.success(request, f"Синхронизация завершена. Добавлено: {created}.")
    return redirect("catalog:m_brands")

def makes_public(request):
    makes = CarMake.objects.order_by("name")
    return render(request, "static_pages/car_makes.html", {"makes": makes})

# ── самописная админка ─────────────────────────────────────────────

def m_makes(request):
    page_obj = CarMake.objects.order_by("name")  # немного — пагинация не нужна
    return render(request, "manage/makes_list.html", {"items": page_obj})

@require_POST
def m_make_add(request):
    name = (request.POST.get("name") or "").strip()
    if not name:
        messages.error(request, "Укажите название марки.")
        return redirect("catalog:m_makes")
    CarMake.objects.get_or_create(name=name)
    messages.success(request, f"Марка «{name}» добавлена (или уже есть).")
    return redirect("catalog:m_makes")

@require_POST
def m_make_logo(request, make_id):
    m = get_object_or_404(CarMake, pk=make_id)
    f = request.FILES.get("logo")
    if not f:
        messages.error(request, "Выберите файл логотипа.")
        return redirect("catalog:m_makes")
    m.logo = f
    m.save(update_fields=["logo"])
    messages.success(request, f"Логотип «{m.name}» обновлён.")
    return redirect("catalog:m_makes")

@require_POST
def m_make_delete(request, make_id):
    m = get_object_or_404(CarMake, pk=make_id)
    name = m.name
    m.delete()
    messages.success(request, f"Марка «{name}» удалена.")
    return redirect("catalog:m_makes")

def m_makes_sync(request):
    """Собираем марки из:
       1) атрибута «Марка авто» у товаров
       2) таблицы ChipCar (если используешь чип-тюнинг)
    """
    created = 0

    # из атрибутов товаров
    attr = Attribute.objects.filter(name="Марка авто").first()
    if attr:
        vals = (ProductAttributeValue.objects
                .filter(attribute=attr)
                .values_list("value", flat=True).distinct())
        for v in vals:
            title = (v or "").strip()
            if title and not CarMake.objects.filter(name=title).exists():
                CarMake.objects.create(name=title); created += 1

    # из ChipCar (опционально — если модель есть)
    try:
        from .models import ChipCar
        vals = (ChipCar.objects.values_list("make", flat=True).distinct())
        for v in vals:
            title = (v or "").strip()
            if title and not CarMake.objects.filter(name=title).exists():
                CarMake.objects.create(name=title); created += 1
    except Exception:
        pass

    messages.success(request, f"Синхронизация завершена. Добавлено: {created}.")
    return redirect("catalog:m_makes")

def _parse_money(txt: str):
    if not txt:
        return None
    s = txt.strip().replace(" ", "").replace("\u00A0", "").replace(",", ".")
    s = re.sub(r"[^0-9.\-]", "", s)
    if not s:
        return None
    try:
        return Decimal(s)
    except InvalidOperation:
        return None

def _parse_specs1(text: str):
    """Простой разбор 'Ключ: значение1, значение2' построчно."""
    out = []
    if not text:
        return out
    for raw in re.split(r"[\r\n]+", text):
        row = raw.strip()
        if not row:
            continue
        if ":" in row:
            k, v = row.split(":", 1)
        elif "—" in row:
            k, v = row.split("—", 1)
        elif "=" in row:
            k, v = row.split("=", 1)
        else:
            # без разделителя — считаем это значением «Характеристика»
            k, v = "Характеристика", row
        k = k.strip()
        v = v.strip()
        if not k or not v:
            continue
        # множ. значения через , ; / |
        parts = [p.strip() for p in re.split(r"[;,/|]+", v) if p.strip()]
        out.append((k, parts))
    return out

@transaction.atomic
def m_product_add(request):
    top_cats = Category.objects.filter(parent__isnull=True, is_hidden=False).order_by("name")
    subs = Category.objects.select_related("parent").all().order_by("parent__name", "name")

    if request.method == "POST":
        title       = (request.POST.get("title") or "").strip()
        sku         = (request.POST.get("sku") or "").strip()
        description = (request.POST.get("description") or "").strip()
        ext_url     = (request.POST.get("external_url") or "").strip()

        parent_id = request.POST.get("parent_id") or ""
        sub_id    = request.POST.get("sub_id") or ""

        price_txt      = (request.POST.get("price") or "").strip()
        old_price_txt  = (request.POST.get("old_price") or "").strip()
        cost_price_txt = (request.POST.get("cost_price") or "").strip()

        is_active = bool(request.POST.get("is_active"))

        if not title or not sku:
            messages.error(request, "Заполните минимум название и артикул (SKU).")
            return redirect(request.path)

        price      = _parse_money(price_txt)
        old_price  = _parse_money(old_price_txt)
        cost_price = _parse_money(cost_price_txt)

        if price_txt and price is None:
            messages.error(request, "Неверный формат 'Цена'.")
            return redirect(request.path)
        if old_price_txt and old_price is None:
            messages.error(request, "Неверный формат 'Старая цена'.")
            return redirect(request.path)
        if cost_price_txt and cost_price is None:
            messages.error(request, "Неверный формат 'Себестоимость'.")
            return redirect(request.path)

        category = None
        if sub_id:
            category = Category.objects.filter(id=sub_id, is_hidden=False).first()
        elif parent_id:
            category = Category.objects.filter(id=parent_id, is_hidden=False).first()

        if Product.objects.filter(sku=sku).exists():
            messages.error(request, f"Товар с таким SKU уже существует: {sku}")
            return redirect(request.path)

        p = Product.objects.create(
            title=title,
            sku=sku,
            category=category,
            description=description,
            external_url=ext_url,
            price=price or Decimal("0"),
            old_price=old_price,
            cost_price=cost_price,
            is_active=is_active,
        )

        # характеристики (опционально)
        specs_text = (request.POST.get("specs_text") or "").strip()
        for key, values in _parse_specs1(specs_text):
            attr, _ = Attribute.objects.get_or_create(name=key)
            for val in values:
                ProductAttributeValue.objects.get_or_create(product=p, attribute=attr, value=val)

        # фото
        images = request.FILES.getlist("images")  # имя поля: images
        for idx, f in enumerate(images):
            ProductImage.objects.create(
                product=p, position=idx, image=f, alt=title
            )

        messages.success(request, f"Товар «{p.title}» добавлен.")
        return redirect("catalog:m_product_detail", p.id)

    return render(request, "manage/product_add.html", {
        "top_cats": top_cats,
        "subs": subs,
    })

def contact_request_view(request):
    """
    Публичная страница с формой «Обращение к нам»
    """
    # значения по умолчанию для возврата формы
    ctx = {
        "name_val":     "",
        "phone_val":    "",
        "comment_val":  "",
        "pd_consent_val": False,
        "RECAPTCHA_SITE_KEY": settings.RECAPTCHA_SITE_KEY,
    }

    if request.method == "POST":
        name    = (request.POST.get("name") or "").strip()
        phone   = (request.POST.get("phone") or "").strip()
        comment = (request.POST.get("comment") or "").strip()
        pd_consent = request.POST.get("pd_consent") in ("on", "true", "1")

        # заполним контекст, чтобы вернуть введённые значения при ошибке
        ctx.update({
            "name_val": name,
            "phone_val": phone,
            "comment_val": comment,
            "pd_consent_val": pd_consent,
        })

        # reCAPTCHA v2 проверка
        token = request.POST.get("g-recaptcha-response")
        ok, payload = verify_recaptcha(token, request.META.get("REMOTE_ADDR"))
        if settings.DEBUG and not ok:
            ok = True

        if not ok:
            messages.error(request, "Подтвердите, что вы не робот.")
            return render(request, "static_pages/contact_request.html", ctx, status=400)

        # ВАЛИДАЦИЯ ПОЛЕЙ
        if not name:
            messages.error(request, "Укажите имя.")
            return render(request, "static_pages/contact_request.html", ctx, status=400)

        # Проверка телефона
        normalized_phone, phone_err = normalize_ru_phone(phone)
        if phone_err:
            messages.error(request, phone_err)
            return render(request, "static_pages/contact_request.html", ctx, status=400)

        if not pd_consent:
            messages.error(request, "Подтвердите согласие на обработку персональных данных.")
            return render(request, "static_pages/contact_request.html", ctx, status=400)

        # Сохраняем уже нормализованный телефон
        ContactRequest.objects.create(
            name=name,
            phone=normalized_phone,
            comment=comment,
            page="contact_page",
        )
        request.session["reach_goal"] = "contact_form_submit"
        messages.success(request, "Спасибо! Мы получили ваше обращение и свяжемся с вами.")
        return redirect("catalog:page_contact_request")

    # GET
    return render(request, "static_pages/contact_request.html", ctx)



# ── админка: список обращений ────────────────────────────────────────────────
def m_contact_requests(request):
    qs = ContactRequest.objects.all()

    q = (request.GET.get("q") or "").strip()
    if q:
        qs = qs.filter(
            Q(name__icontains=q) |
            Q(phone__icontains=q) |
            Q(comment__icontains=q) |
            Q(page__icontains=q)
        )

    paginator = Paginator(qs, 30)
    raw_page = (request.GET.get("page") or "").strip()
    try:
        page_number = int(raw_page)
        if page_number < 1:
            page_number = 1
    except (TypeError, ValueError):
        page_number = 1

    page_obj   = paginator.get_page(page_number)
    page_items = _page_window(paginator, page_obj.number, pad=1, max_len=9)

    return render(request, "manage/contacts_list.html", {
        "page_obj": page_obj,
        "page_items": page_items,
        "q": q,
    })


def m_contact_delete(request, lead_id: int):
    lead = get_object_or_404(ContactRequest, id=lead_id)
    if request.method == "POST":
        lead.delete()
        messages.success(request, "Обращение удалено.")
        return redirect("catalog:m_contact_requests")
    # если кто-то зайдёт GET’ом — просто вернёмся
    return redirect("catalog:m_contact_requests")

def _ensure_sitepages():
    """Гарантируем наличие записей под все страницы."""
    existing = set(SitePage.objects.values_list("slug", flat=True))
    for slug, _name in SitePage.PAGES:
        if slug not in existing:
            SitePage.objects.create(slug=slug)


@transaction.atomic
def m_content_edit(request, slug: str):
    page = get_object_or_404(SitePage, slug=slug)

    if request.method == "POST":
        page.title      = request.POST.get("title", "").strip()
        page.subtitle   = request.POST.get("subtitle", "").strip()
        page.content_html = request.POST.get("content_html", "").strip()

        # картинка (можно удалить, если поставить чекбокс “удалить”)
        if "hero_image" in request.FILES:
            page.hero_image = request.FILES["hero_image"]
        if request.POST.get("remove_hero") == "1":
            page.hero_image = None

        # особый случай — “Контакты”
        if slug == "contacts":
            extra = page.extra or {}
            extra["address"] = request.POST.get("address", "").strip()
            extra["phone"]   = request.POST.get("phone", "").strip()
            extra["email"]   = request.POST.get("email", "").strip()
            # режим работы построчно, одна строка — один день
            hours_lines = request.POST.get("hours", "").splitlines()
            extra["hours"]  = [ln.strip() for ln in hours_lines if ln.strip()]
            page.extra = extra

        page.save()
        messages.success(request, "Страница сохранена.")
        return redirect("catalog:m_content_edit", slug=slug)

    # initial для формы “Контакты”
    ctx_extra = {}
    if slug == "contacts":
        ex = page.extra or {}
        ctx_extra = {
            "address": ex.get("address", ""),
            "phone":   ex.get("phone", ""),
            "email":   ex.get("email", ""),
            "hours":   "\n".join(ex.get("hours", [])),
        }

    return render(request, "manage/content_edit.html", {
        "page": page,
        **ctx_extra,
    })

def page_about(request):
    page = SitePage.objects.filter(slug="about").first()
    placements = (BannerPlacement.objects
                  .filter(page_key=BannerPlacement.PageKey.ABOUT, banner__is_active=True)
                  .select_related("banner").order_by("position"))
    banners_main = [p.banner for p in placements]
    return render(request, "static_pages/avtoservis_info.html", {"page": page, "banners_main": banners_main})

def page_help(request):
    page = SitePage.objects.filter(slug="help").first()
    return render(request, "static_pages/help.html", {"page": page})

def page_delivery(request):
    page = SitePage.objects.filter(slug="delivery").first()
    return render(request, "static_pages/delivery.html", {"page": page})

def page_payment(request):
    page = SitePage.objects.filter(slug="payment").first()
    return render(request, "static_pages/payment.html", {"page": page})

def page_dyno(request):
    page = SitePage.objects.filter(slug="dyno").first()
    placements = (BannerPlacement.objects
                  .filter(page_key=BannerPlacement.PageKey.DYNO, banner__is_active=True)
                  .select_related("banner").order_by("position"))
    banners_main = [p.banner for p in placements]
    return render(request, "static_pages/dynostand.html", {"page": page, "banners_main": banners_main, "RECAPTCHA_SITE_KEY": settings.RECAPTCHA_SITE_KEY,  # ключ для виджета
        "form": ServiceRequestForm(),          })

def page_contacts(request):
    page = SitePage.objects.filter(slug="contacts").first()
    return render(request, "static_pages/contacts.html", {"page": page})

def _next_banner_position(group: str) -> int:
    """
    Возвращает следующую позицию внутри группы баннеров.
    Шаг 10 — чтобы было удобно двигать вверх/вниз.
    """
    last = Banner.objects.filter(group=group).order_by("-position", "-id").first()
    return (last.position + 10) if last and last.position is not None else 10


@transaction.atomic
def m_banners(request):
    group = (request.GET.get("group") or request.POST.get("group") or "main-hero").strip()

    if request.method == "POST" and request.POST.get("action") == "add":
        files            = request.FILES.getlist("images")
        files_avif       = request.FILES.getlist("images_avif")
        files_mobile     = request.FILES.getlist("images_mobile")
        files_mobile_avif= request.FILES.getlist("images_mobile_avif")

        if not files:
            messages.error(request, "Выберите хотя бы одну картинку.")
            return redirect(request.get_full_path())

        created = 0
        pos = _next_banner_position(group)

        with transaction.atomic():
            for i, img in enumerate(files):
                b = Banner.objects.create(
                    group=group,
                    image=img,
                    position=pos,
                    is_active=True,  # если нужен дефолт «видим»
                )
                # привязка версий по индексу — если переданы
                if i < len(files_avif):
                    b.image_avif = files_avif[i]
                if i < len(files_mobile):
                    b.image_mobile = files_mobile[i]
                if i < len(files_mobile_avif):
                    b.image_mobile_avif = files_mobile_avif[i]
                b.save(update_fields=[
                    "image_avif", "image_mobile", "image_mobile_avif"
                ] if any([i < len(files_avif), i < len(files_mobile), i < len(files_mobile_avif)]) else [])

                created += 1
                pos += 10  # следующий слот

        messages.success(request, f"Добавлено слайдов: {created}")
        return redirect(request.get_full_path())

    banners = Banner.objects.filter(group=group).order_by("position", "id")
    return render(request, "manage/banners.html", {"group": group, "banners": banners})

def m_banner_toggle(request, bid: int):
    b = get_object_or_404(Banner, id=bid)
    b.is_active = not b.is_active
    b.save(update_fields=["is_active"])
    messages.success(request, "Статус изменён.")
    return redirect(request.META.get("HTTP_REFERER", reverse("catalog:m_banners")))

def m_banner_delete(request, bid: int):
    b = get_object_or_404(Banner, id=bid)
    grp = b.group
    b.delete()
    messages.success(request, "Слайд удалён.")
    return redirect(f"{reverse('catalog:m_banners')}?group={grp}")

@transaction.atomic
def m_banner_move(request, bid: int, dir: str):
    b = get_object_or_404(Banner, id=bid)
    qs = list(Banner.objects.filter(group=b.group).order_by("position", "id"))
    idx = qs.index(b)
    if dir == "up" and idx > 0:
        qs[idx-1].position, qs[idx].position = qs[idx].position, qs[idx-1].position
        qs[idx-1].save(update_fields=["position"]); qs[idx].save(update_fields=["position"])
    elif dir == "down" and idx < len(qs)-1:
        qs[idx+1].position, qs[idx].position = qs[idx].position, qs[idx+1].position
        qs[idx+1].save(update_fields=["position"]); qs[idx].save(update_fields=["position"])
    return redirect(request.META.get("HTTP_REFERER", reverse("catalog:m_banners")))

def m_content_index(request):
    _ensure_sitepages()
    pages = SitePage.objects.all()
    banners_main = Banner.objects.filter(group="main-hero").order_by("position","id")
    return render(request, "manage/content_index.html", {
        "pages": pages,
        "banners_main": banners_main
    })

def api_chip_options(request):
    """
    Каскад для страницы автосервиса (ChipCar):
    GET: make, model, year
    JSON: {models: [...], years: [...], engines: [...]}
    """
    make  = (request.GET.get("make")  or "").strip()
    model = (request.GET.get("model") or "").strip()
    year  = (request.GET.get("year")  or "").strip()

    qs = ChipCar.objects.all()

    resp = {"models": [], "years": [], "engines": []}

    if not make:
        return JsonResponse(resp)

    # МОДЕЛИ
    models_qs = (qs.filter(make=make)
                   .order_by("model")
                   .values_list("model", flat=True).distinct())
    resp["models"] = list(models_qs)

    if not model:
        return JsonResponse(resp)

    # ГОДА
    years_qs = (qs.filter(make=make, model=model)
                  .order_by("year")
                  .values_list("year", flat=True).distinct())
    resp["years"] = list(years_qs)

    if not year:
        return JsonResponse(resp)

    # МОТОРЫ
    engines_qs = (qs.filter(make=make, model=model, year=year)
                   .order_by("engine")
                   .values_list("engine", flat=True).distinct())
    resp["engines"] = list(engines_qs)

    return JsonResponse(resp)

def api_filter_options(request):
    make   = (request.GET.get("make")   or "").strip()
    model  = (request.GET.get("model")  or "").strip()
    engine = (request.GET.get("engine") or "").strip()

    def filtered_products(make=None, model=None, engine=None):
        qs = Product.objects.filter(is_active=True)
        if make:
            qs = qs.filter(
                attributes__attribute__name="Марка авто",
                attributes__value__icontains=make
            )
        if model:
            qs = qs.filter(
                attributes__attribute__name="Модель авто",
                attributes__value__icontains=model
            )
        if engine:
            qs = qs.filter(
                attributes__attribute__name="Двигатель",
                attributes__value__icontains=engine
            )
        return qs.distinct()

    # модели зависят от make
    base_for_models = filtered_products(make=make)
    model_values = (ProductAttributeValue.objects
                    .filter(product__in=base_for_models, attribute__name="Модель авто")
                    .values_list("value", flat=True).distinct().order_by("value"))

    # моторы зависят от make+model
    base_for_engines = filtered_products(make=make, model=model)
    engine_values = (ProductAttributeValue.objects
                     .filter(product__in=base_for_engines, attribute__name="Двигатель")
                     .values_list("value", flat=True).distinct().order_by("value"))

    # бренды зависят от make+model+engine (всё, что уже выбрано)
    base_for_brands = filtered_products(make=make, model=model, engine=engine)
    brand_values = (ProductAttributeValue.objects
                    .filter(product__in=base_for_brands, attribute__name="Производитель")
                    .values_list("value", flat=True).distinct().order_by("value"))

    return JsonResponse({
        "models":  list(model_values),
        "engines": list(engine_values),
        "brands":  list(brand_values),
    })

@login_required
def m_categories(request):
    # подтянем дерево (только два уровня: родитель + дети)
    top_cats = (Category.objects
                .filter(parent__isnull=True)
                .prefetch_related("children")
                .order_by("name"))
    return render(request, "manage/categories.html", {
        "top_cats": top_cats,
    })


@login_required
def m_category_rename(request, cat_id):
    if request.method != "POST":
        return redirect("catalog:m_categories")

    cat = get_object_or_404(Category, id=cat_id)

    new_name = (request.POST.get("name") or "").strip()
    new_slug_raw = (request.POST.get("slug") or "").strip()

    changed = False

    if new_name and new_name != cat.name:
        cat.name = new_name
        changed = True

    if new_slug_raw:
        # нормализуем пользовательский slug
        base = slugify(new_slug_raw, allow_unicode=True) or fs_safe_segment(new_slug_raw).lower()
        # уникальность в рамках (parent, slug) согласно unique_together
        new_slug = unique_slugify(cat, base, field_name="slug", max_len=220)
        if new_slug != cat.slug:
            cat.slug = new_slug
            changed = True
    else:
        # если поле slug оставили пустым — регенерируем из имени
        base = slugify(cat.name, allow_unicode=True) or fs_safe_segment(cat.name).lower()
        new_slug = unique_slugify(cat, base, field_name="slug", max_len=220)
        if new_slug != cat.slug:
            cat.slug = new_slug
            changed = True

    if changed:
        try:
            cat.save()
            messages.success(request, "Категория обновлена.")
        except Exception as e:
            messages.error(request, f"Не удалось сохранить: {e}")
    else:
        messages.info(request, "Изменений нет.")

    return redirect("catalog:m_categories")

def m_category_add(request):
    """Создать родительскую категорию."""
    if request.method != "POST":
        return redirect("catalog:m_categories")
    name = (request.POST.get("name") or "").strip()
    if not name:
        messages.error(request, "Введите название категории.")
        return redirect("catalog:m_categories")

    Category.objects.create(name=name, parent=None)
    messages.success(request, f"Категория «{name}» добавлена.")
    return redirect("catalog:m_categories")

def m_category_add_child(request, cat_id: int):
    """Создать подкатегорию у указанного родителя."""
    parent = get_object_or_404(Category, id=cat_id)
    if request.method != "POST":
        return redirect("catalog:m_categories")
    name = (request.POST.get("name") or "").strip()
    if not name:
        messages.error(request, "Введите название подкатегории.")
        return redirect("catalog:m_categories")

    Category.objects.create(name=name, parent=parent)
    messages.success(request, f"Подкатегория «{name}» добавлена в «{parent.name}».")
    return redirect("catalog:m_categories")

def _category_is_empty(cat: Category) -> bool:
    """Пуста, если нет товаров и нет подкатегорий (включая вложенные)."""
    if cat.products.exists():
        return False
    if cat.children.exists():
        return False
    return True

def m_category_delete(request, cat_id: int):
    """Удалить категорию ТОЛЬКО если она пустая (без товаров и без детей)."""
    cat = get_object_or_404(Category, id=cat_id)
    if request.method != "POST":
        return redirect("catalog:m_categories")

    if not _category_is_empty(cat):
        messages.error(request, "Нельзя удалить: у категории есть товары или подкатегории.")
        return redirect("catalog:m_categories")

    name = cat.name
    cat.delete()
    messages.success(request, f"Категория «{name}» удалена.")
    return redirect("catalog:m_categories")
# catalog/views.py

def robots_txt(request):
    """
    Надёжный robots.txt: никогда не падает.
    Если в БД нет таблицы/записей — отдаём дефолт.
    """
    default = (
        "User-agent: *\n"
        "Disallow:\n"
        "Sitemap: https://www.gonka.space/sitemap.xml\n"
    )
    try:
        # таблица уже мигрирована?
        tables = connection.introspection.table_names()
        if "catalog_robotstxt" not in tables:
            return HttpResponse(default, content_type="text/plain")

        from catalog.models import RobotsTxt  # импорт тут, чтобы не грузить при ошибках выше

        row = RobotsTxt.objects.order_by("-updated_at", "-id").first()
        content = (row.content or "").strip() if row else default
        if not content:
            content = default

        resp = HttpResponse(content, content_type="text/plain")
        resp["Cache-Control"] = "public, max-age=3600"
        resp["X-ROBOTS-TS"] = now().isoformat()
        return resp
    except Exception:
        # на всякий случай — никогда не отдаём 500
        return HttpResponse(default, content_type="text/plain")


def page_not_found(request, exception):
    # рендер кастомного шаблона 404.html
    from django.shortcuts import render
    return render(request, "404.html", status=404)

@login_required
def m_seo_dashboard(request):
    return render(request, "manage/seo/dashboard.html", {
        "cnt_entries": SEOEntry.objects.count(),
        "cnt_redirects": RedirectRule.objects.count(),
        "cnt_gone": GoneURL.objects.count(),
        "has_robots": RobotsTxt.objects.exists(),
    })


@login_required
def m_seo_entries(request):
    q = (request.GET.get("q") or "").strip()
    qs = SEOEntry.objects.all().order_by("-updated_at", "-id")
    if q:
        qs = qs.filter(
            Q(title__icontains=q) |
            Q(h1__icontains=q) |
            Q(meta_description__icontains=q) |
            Q(path_regex__icontains=q)
        )
    paginator = Paginator(qs, 100000)
    page = paginator.get_page(int(request.GET.get("page") or 1))
    return render(request, "manage/seo/entries_list.html", {"page_obj": page, "q": q})


@login_required
def m_seo_entry_edit(request, pk=None):
    obj = get_object_or_404(SEOEntry, pk=pk) if pk else None
    if request.method == "POST":
        form = SEOEntryForm(request.POST, instance=obj)
        if form.is_valid():
            form.save()
            return redirect("catalog:m_seo_entries")
    else:
        form = SEOEntryForm(instance=obj)
    return render(request, "manage/seo/entry_form.html", {"form": form, "obj": obj})


@login_required
def m_seo_entry_delete(request, pk):
    obj = get_object_or_404(SEOEntry, pk=pk)
    if request.method == "POST":
        obj.delete()
        return redirect("catalog:m_seo_entries")
    return render(request, "manage/seo/confirm_delete.html", {"obj": obj, "back": reverse("catalog:m_seo_entries")})


# ---- Redirects ----

@login_required
def m_redirects(request):
    if request.method == "POST":
        form = RedirectForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect("catalog:m_redirects")
    else:
        form = RedirectForm()

    q = (request.GET.get("q") or "").strip()
    qs = RedirectRule.objects.all().order_by("-priority", "-id")
    if q:
        qs = qs.filter(Q(from_path__icontains=q) | Q(to_url__icontains=q))
    paginator = Paginator(qs, 100000)
    page = paginator.get_page(int(request.GET.get("page") or 1))

    return render(request, "manage/seo/redirects.html", {"form": form, "page_obj": page, "q": q})


@login_required
def m_redirect_delete(request, pk):
    obj = get_object_or_404(RedirectRule, pk=pk)
    if request.method == "POST":
        obj.delete()
        return redirect("catalog:m_redirects")
    return render(request, "manage/seo/confirm_delete.html", {"obj": obj, "back": reverse("catalog:m_redirects")})


# ---- Gone (410) ----

@login_required
def m_gone(request):
    if request.method == "POST":
        form = GoneForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect("catalog:m_gone")
    else:
        form = GoneForm()

    q = (request.GET.get("q") or "").strip()
    qs = GoneURL.objects.all().order_by("-id")
    if q:
        qs = qs.filter(Q(path_or_regex__icontains=q) | Q(comment__icontains=q))
    paginator = Paginator(qs, 100000)
    page = paginator.get_page(int(request.GET.get("page") or 1))

    return render(request, "manage/seo/gone.html", {"form": form, "page_obj": page, "q": q})


@login_required
def m_gone_delete(request, pk):
    obj = get_object_or_404(GoneURL, pk=pk)
    if request.method == "POST":
        obj.delete()
        return redirect("catalog:m_gone")
    return render(request, "manage/seo/confirm_delete.html", {"obj": obj, "back": reverse("catalog:m_gone")})


# ---- robots.txt ----

@login_required
def m_robots(request):
    obj = RobotsTxt.objects.first()
    if not obj:
        obj = RobotsTxt.objects.create(content="User-agent: *\nDisallow:\n", is_active=True)
    if request.method == "POST":
        form = RobotsForm(request.POST, instance=obj)
        if form.is_valid():
            form.save()
            return redirect("catalog:m_robots")
    else:
        form = RobotsForm(instance=obj)
    return render(request, "manage/seo/robots.html", {"form": form})

def _abs(request, url: str) -> str:
    return request.build_absolute_uri(url)

@require_http_methods(["GET", "POST"])
def password_forgot(request):
    next_url = request.META.get("HTTP_REFERER") or reverse("catalog:product_list")
    if request.method == "GET":
        messages.info(request, "Введите e-mail в форме восстановления пароля.")
        return redirect(next_url)

    form = PasswordResetEmailForm(request.POST)
    if not form.is_valid():
        messages.error(request, "Укажите корректный e-mail.")
        return redirect(next_url)

    email = (form.cleaned_data["email"] or "").strip().lower()
    User = get_user_model()
    user = User.objects.filter(email__iexact=email).first()

    if user:
        new_password = secrets.token_urlsafe(9)
        user.set_password(new_password)
        user.save(update_fields=["password"])

        # ⚠️ используем JPG/PNG
        hero_url = _abs(request, static("img/mail_hero.png"))

        ctx = {
            "user": user,
            "new_password": new_password,
            "account_password_url": _abs(request, reverse("catalog:account_password")),
            "year": timezone.now().year,
            "hero_url": hero_url,
        }

        try:
            html = render_to_string("email/password_reset.html", ctx)
            text = strip_tags(html) or (
                f"Здравствуйте, {user.username}!\n"
                f"Ваш временный пароль: {new_password}\n"
                f"Сменить пароль: {ctx['account_password_url']}\n"
            )

            msg = EmailMultiAlternatives(
                subject="Восстановление доступа — Gonka shop",
                body=text,
                from_email=settings.DEFAULT_FROM_EMAIL,
                to=[email],
            )
            msg.attach_alternative(html, "text/html")
            msg.send(fail_silently=False)

        except Exception:
            logger.exception("Password reset HTML email failed for %s", email)
            send_mail(
                subject="Восстановление доступа — Gonka shop",
                message=(
                    f"Здравствуйте, {user.username}!\n\n"
                    f"Ваш временный пароль: {new_password}\n\n"
                    f"Сменить пароль: {ctx['account_password_url']}\n"
                    f"Если вы не запрашивали восстановление — проигнорируйте это письмо."
                ),
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[email],
                fail_silently=False,
            )

    messages.success(request, "Если такой e-mail существует, мы отправили письмо с паролем.")
    return redirect(next_url)

def m_users_mailing(request):
    if request.method == "POST":
        form = MailingPromoForm(request.POST)
        if not form.is_valid():
            messages.error(request, "Проверьте корректность полей.")
            return render(request, "manage/mailing_form.html", {"form": form})

        subject     = form.cleaned_data["subject"].strip()
        title       = form.cleaned_data["title"].strip()
        description = form.cleaned_data["description"].strip()
        button_url  = form.cleaned_data["button_url"].strip()
        button_text = (form.cleaned_data["button_text"] or "Открыть акцию").strip()
        hero_path   = (form.cleaned_data["hero_path"] or "img/mail_hero.png").strip()
        only_active = form.cleaned_data["only_active"]
        test_email  = (form.cleaned_data["test_email"] or "").strip()

        # Абсолютный URL на картинку из static
        hero_url = request.build_absolute_uri(static(hero_path))

        # Собираем HTML из шаблона письма
        ctx_mail = {
            "subject": subject,
            "title": title,
            "description": description,
            "button_url": button_url,
            "button_text": button_text,
            "hero_url": hero_url,
            "year": timezone.now().year,
        }
        html_body = render_to_string("email/mailing_promo.html", ctx_mail)
        text_body = strip_tags(f"{title}\n\n{description}\n\n{button_url}")

        # Тестовая отправка
        if test_email:
            try:
                m = EmailMultiAlternatives(subject=subject, body=text_body,
                                           from_email=settings.DEFAULT_FROM_EMAIL, to=[test_email])
                m.attach_alternative(html_body, "text/html")
                m.send(fail_silently=False)
                messages.success(request, f"Тестовое письмо отправлено на {test_email}.")
            except Exception as e:
                logger.exception("Mailing test send failed to %s", test_email)
                messages.error(request, f"Не удалось отправить тест: {e!r}")
            return render(request, "manage/mailing_form.html", {"form": form})

        # Основная рассылка — по одному, копим ошибки
        User = get_user_model()
        qs = User.objects.all()
        if only_active:
            qs = qs.filter(is_active=True)

        qs = qs.filter(profile__marketing_email=True)

        qs = qs.exclude(email__isnull=True).exclude(email="").order_by("id")
        
        recipients = list(qs.values_list("email", flat=True))
        total = len(recipients)
        if total == 0:
            messages.warning(request, "Нет получателей с e-mail.")
            return render(request, "manage/mailing_form.html", {"form": form})

        failed, sent = [], 0
        PAUSE = 0.4

        try:
            with get_connection(fail_silently=False) as conn:
                for addr in recipients:
                    msg = EmailMultiAlternatives(
                        subject=subject, body=text_body,
                        from_email=settings.DEFAULT_FROM_EMAIL, to=[addr],
                        connection=conn,
                    )
                    msg.attach_alternative(html_body, "text/html")
                    try:
                        msg.send(fail_silently=False)
                        sent += 1
                    except Exception as e:
                        logger.exception("Bulk mail send failed to %s", addr)
                        failed.append((addr, repr(e)))
                    time.sleep(PAUSE)
        except Exception as e:
            logger.exception("Mailing transport error")
            messages.error(request, f"Не удалось выполнить рассылку: {e!r}")
            return render(request, "manage/mailing_form.html", {
                "form": form,
                "result": {"total": total, "sent": sent, "failed_list": failed},
            })

        if failed:
            messages.warning(request, f"Готово: отправлено {sent} из {total}. Ошибок: {len(failed)}.")
        else:
            messages.success(request, f"Готово! Отправлено писем: {sent} из {total}.")

        return render(request, "manage/mailing_form.html", {
            "form": MailingPromoForm(),  # новая пустая форма
            "result": {"total": total, "sent": sent, "failed_list": failed},
        })

    # GET
    return render(request, "manage/mailing_form.html", {"form": MailingPromoForm()})

@login_required
@require_http_methods(["GET", "POST"])
def account_subscribe(request):
    profile = request.user.profile  # у тебя он уже используется в шаблонах
    if request.method == "POST":
        want_email = bool(request.POST.get("marketing_email"))
        if profile.marketing_email != want_email:
            profile.marketing_email = want_email
            profile.save(update_fields=["marketing_email"])
            messages.success(request, "Настройки уведомлений обновлены.")
        return redirect("catalog:account_subscribe")
    return render(request, "account/subscribe.html", {"profile": profile})

@require_POST
def m_banner_set_link(request, bid):
    from urllib.parse import urlparse
    from .models import Banner
    b = get_object_or_404(Banner, pk=bid)

    raw = (request.POST.get("link_url") or "").strip()

    # ничего не введено — просто остаёмся при старом значении
    if raw == "":
        messages.info(request, "Ссылка не изменена (поле было пустым).")
        return redirect(request.META.get("HTTP_REFERER") or reverse("catalog:m_banners"))

    # простая валидация URL
    p = urlparse(raw)
    if not p.scheme or not p.netloc:
        messages.error(request, "Некорректный URL. Укажите полный адрес, например https://example.com/page/")
        return redirect(request.META.get("HTTP_REFERER") or reverse("catalog:m_banners"))

    b.link_url = raw
    b.save()  # без update_fields — точно запишется
    messages.success(request, f"Ссылка обновлена: {raw}")
    return redirect(request.META.get("HTTP_REFERER") or reverse("catalog:m_banners"))

PAGE_CHOICES = BannerPlacement.PageKey.choices
PAGE_KEYS    = [k for k, _ in PAGE_CHOICES]

def _next_position_for_page(page_key: str) -> int:
    last = BannerPlacement.objects.filter(page_key=page_key).aggregate(m=Max("position"))["m"]
    return (last or 0) + 1

@require_http_methods(["GET"])
def m_banner_placements(request):
    """
    Список размещений по выбранной странице + форма добавления.
    """
    page_key = request.GET.get("page") or BannerPlacement.PageKey.HOME
    if page_key not in PAGE_KEYS:
        page_key = BannerPlacement.PageKey.HOME

    placements = (
        BannerPlacement.objects
        .filter(page_key=page_key)
        .select_related("banner")
        .order_by("position")
    )

    # для формы добавления — актуальные баннеры (можно сузить по группе, если нужно)
    banners_all = Banner.objects.filter(is_active=True).order_by("group", "position")

    ctx = {
        "page_key": page_key,
        "page_choices": PAGE_CHOICES,
        "placements": placements,
        "banners_all": banners_all,
    }
    return render(request, "manage/banner_placements.html", ctx)

@require_http_methods(["POST"])
def m_banner_place_add(request):
    """
    Добавить баннер в конкретную страницу (в конец списка).
    """
    page_key = request.POST.get("page_key") or BannerPlacement.PageKey.HOME
    banner_id = request.POST.get("banner_id")
    try:
        banner = Banner.objects.get(pk=banner_id)
    except Banner.DoesNotExist:
        messages.error(request, "Баннер не найден.")
        return redirect(f"{reverse('catalog:m_banner_placements')}?page={page_key}")

    # уникальность: один и тот же баннер можно добавить в каждую страницу максимум 1 раз
    obj, created = BannerPlacement.objects.get_or_create(
        page_key=page_key, banner=banner,
        defaults={"position": _next_position_for_page(page_key)}
    )
    if created:
        messages.success(request, "Баннер добавлен в размещения страницы.")
    else:
        messages.info(request, "Этот баннер уже есть на выбранной странице.")
    return redirect(f"{reverse('catalog:m_banner_placements')}?page={page_key}")

@require_http_methods(["POST"])
def m_banner_place_delete(request, pk: int):
    """
    Удалить размещение из страницы (сам баннер не трогаем).
    """
    plc = get_object_or_404(BannerPlacement, pk=pk)
    page_key = plc.page_key
    plc.delete()
    messages.success(request, "Размещение удалено.")
    return redirect(f"{reverse('catalog:m_banner_placements')}?page={page_key}")

@require_http_methods(["GET"])
def m_banner_place_move(request, pk: int, dir: str):
    """
    Меняем position в пределах одной страницы (up/down).
    """
    plc = get_object_or_404(BannerPlacement, pk=pk)
    page_key = plc.page_key

    if dir not in ("up", "down"):
        return redirect(f"{reverse('catalog:m_banner_placements')}?page={page_key}")

    if dir == "up":
        prev = (
            BannerPlacement.objects
            .filter(page_key=page_key, position__lt=plc.position)
            .order_by("-position")
            .first()
        )
        if prev:
            plc.position, prev.position = prev.position, plc.position
            plc.save(update_fields=["position"])
            prev.save(update_fields=["position"])
    else:
        nxt = (
            BannerPlacement.objects
            .filter(page_key=page_key, position__gt=plc.position)
            .order_by("position")
            .first()
        )
        if nxt:
            plc.position, nxt.position = nxt.position, plc.position
            plc.save(update_fields=["position"])
            nxt.save(update_fields=["position"])

    return redirect(f"{reverse('catalog:m_banner_placements')}?page={page_key}")

def _next_pos_home_latest():
    m = HomeLatestItem.objects.aggregate(m=Max("position"))["m"]
    return (m or 0) + 1

@require_http_methods(["GET", "POST"])
def m_home_latest(request):
    """
    Управление блоком 'Последние поступления' на главной.
    GET: список текущих позиций + поиск товара для добавления.
    POST: добавление выбранного товара.
    """
    q = request.GET.get("q", "").strip()
    items = HomeLatestItem.objects.select_related("product").order_by("position")

    # Поиск товаров для добавления
    products = Product.objects.none()
    if q:
        products = (Product.objects
                    .filter(Q(title__icontains=q) | Q(sku__icontains=q))
                    .order_by("-created_at")[:30])

    return render(request, "manage/home_latest.html", {
        "items": items,
        "q": q,
        "products": products,
    })

@require_http_methods(["POST"])
def m_home_latest_add(request):
    pid = request.POST.get("product_id")
    product = get_object_or_404(Product, pk=pid)
    obj, created = HomeLatestItem.objects.get_or_create(
        product=product,
        defaults={"position": _next_pos_home_latest(), "is_active": True},
    )
    if created:
        messages.success(request, "Товар добавлен в 'Последние поступления'.")
    else:
        messages.info(request, "Этот товар уже есть в списке.")
    return redirect(reverse("catalog:m_home_latest"))

@require_http_methods(["POST"])
def m_home_latest_toggle(request, pk):
    item = get_object_or_404(HomeLatestItem, pk=pk)
    item.is_active = not item.is_active
    item.save(update_fields=["is_active"])
    messages.success(request, "Статус изменён.")
    return redirect(reverse("catalog:m_home_latest"))

@require_http_methods(["POST"])
def m_home_latest_delete(request, pk):
    item = get_object_or_404(HomeLatestItem, pk=pk)
    item.delete()
    messages.success(request, "Позиция удалена.")
    return redirect(reverse("catalog:m_home_latest"))

@require_http_methods(["GET"])
def m_home_latest_move(request, pk, dir):
    a = get_object_or_404(HomeLatestItem, pk=pk)
    if dir not in ("up", "down"):
        return redirect(reverse("catalog:m_home_latest"))
    if dir == "up":
        b = HomeLatestItem.objects.filter(position__lt=a.position).order_by("-position").first()
    else:
        b = HomeLatestItem.objects.filter(position__gt=a.position).order_by("position").first()
    if b:
        a.position, b.position = b.position, a.position
        a.save(update_fields=["position"])
        b.save(update_fields=["position"])
    return redirect(reverse("catalog:m_home_latest"))

def _site_url():
    return (getattr(settings, "SITE_URL", "") or "").rstrip("/")

def _abs(url_path: str) -> str:
    base = _site_url() or ""
    return f"{base}{url_path}"

def _product_url(product) -> str:
    cat = getattr(product, "category", None)
    parent = getattr(cat, "parent", None)
    if parent and cat:
        path = f"/catalog/{parent.slug}/{cat.slug}/{product.slug}/"
    elif cat:
        path = f"/catalog/{cat.slug}/{product.slug}/"
    else:
        path = f"/catalog/p/{product.slug}/"
    return _abs(path)

def _first_image_url(product) -> str:
    im = product.images.first()
    return _abs(im.image.url) if im else ""

def _attr_value(product, names):
    """
    Вернёт первое подходящее значение среди атрибутов по списку имён.
    """
    qs = product.attributes.select_related("attribute").filter(attribute__name__in=names)
    val = qs.values_list("value", flat=True).first()
    return (val or "").strip()

def _num_from_text(s: str) -> str:
    s = (s or "").replace(",", ".")
    m = re.search(r"[\d.]+", s)
    return m.group(0) if m else ""

def _weight_kg(product) -> str:
    v = _attr_value(product, ["Вес (кг)", "Вес"])
    return _num_from_text(v)  # вернём "2.3" как строку

def feed_google(request):
    """
    Google Merchant feed: RSS 2.0 + g:
    /catalog/feed/google.xml
    """
    from .models import Product
    qs = (Product.objects
          .filter(is_active=True)
          .prefetch_related("images", "attributes__attribute", "category", "category__parent"))

    cur = "RUB"
    parts = []
    parts.append('<?xml version="1.0" encoding="UTF-8"?>')
    parts.append('<rss version="2.0" xmlns:g="http://base.google.com/ns/1.0">')
    parts.append("<channel>")
    parts.append(f"<title>{escape(getattr(settings, 'SITE_NAME', 'Gonka'))}</title>")
    parts.append(f"<link>{escape(_site_url() or '')}</link>")
    parts.append("<description>Product feed</description>")

    for p in qs:
        brand = _attr_value(p, ["Производитель", "Бренд"])
        gtin  = _attr_value(p, ["GTIN", "EAN", "Штрихкод"])
        mpn   = p.sku or ""
        identifier_exists = "TRUE" if (gtin or mpn) else "FALSE"
        availability = "in stock" if p.is_active else "out of stock"
        weight = _weight_kg(p)  # "2.1"
        url = _product_url(p)
        img = _first_image_url(p)
        title = (p.title or "")[:150]
        desc  = (p.description or "")[:4000]
        price = f"{p.price:.2f} {cur}"

        # Авто-детали
        make   = _attr_value(p, ["Марка авто", "Марка"])
        model  = _attr_value(p, ["Модель авто", "Модель"])
        engine = _attr_value(p, ["Двигатель", "Engine"])

        parts.append("<item>")
        parts.append(f"<g:id>{escape(str(p.id))}</g:id>")
        parts.append(f"<title>{escape(title)}</title>")
        parts.append(f"<link>{escape(url)}</link>")
        if img:
            parts.append(f"<g:image_link>{escape(img)}</g:image_link>")
        parts.append(f"<description>{escape(desc)}</description>")
        if brand:
            parts.append(f"<g:brand>{escape(brand)}</g:brand>")
        if mpn:
            parts.append(f"<g:mpn>{escape(mpn)}</g:mpn>")
        if gtin:
            parts.append(f"<g:gtin>{escape(gtin)}</g:gtin>")
        parts.append(f"<g:identifier_exists>{identifier_exists}</g:identifier_exists>")
        parts.append(f"<g:availability>{availability}</g:availability>")
        parts.append(f"<g:price>{price}</g:price>")
        if weight:
            parts.append(f"<g:shipping_weight>{escape(weight)} kg</g:shipping_weight>")
        parts.append("<g:condition>new</g:condition>")

        # Доп. характеристики (product_detail)
        for nm, val in [("Make", make), ("Model", model), ("Engine", engine)]:
            if val:
                parts.append("<g:product_detail>")
                parts.append(f"<g:attribute_name>{escape(nm)}</g:attribute_name>")
                parts.append(f"<g:attribute_value>{escape(val)}</g:attribute_value>")
                parts.append("</g:product_detail>")

        parts.append("</item>")

    parts.append("</channel></rss>")
    xml = "".join(parts)
    return HttpResponse(xml, content_type="application/xml; charset=utf-8")

def _abs_url(request, url_or_path: str) -> str:
    """Делает абсолютный URL из относительного пути/URL."""
    if not url_or_path:
        return ""
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        return url_or_path
    return request.build_absolute_uri(url_or_path)

def _product_url(request, p) -> str:
    try:
        return _abs_url(request, p.get_absolute_url())
    except Exception:
        return ""

def _first_image_url(request, p) -> str:
    img = p.images.first()
    if not img or not getattr(img, "image", None):
        return ""
    try:
        return _abs_url(request, img.image.url)
    except Exception:
        return ""

def _attr_value(p, names):
    """Вернёт первое совпавшее значение атрибута из списка имён."""
    al = getattr(p, "attributes", None)
    if not al:
        return ""
    want = {n.lower() for n in names}
    for pav in al.all():
        nm = (getattr(pav.attribute, "name", "") or "").lower()
        if nm in want:
            return pav.value or ""
    return ""

def _weight_kg(p):
    # если у тебя вес лежит в атрибутах – подправь сюда имена
    for nm in ["Вес", "Вес (кг)", "Weight"]:
        v = _attr_value(p, [nm])
        if v:
            return v
    return ""

def _site_url(request) -> str:
    # если в settings есть SITE_URL – используем его, иначе из request
    url = getattr(settings, "SITE_URL", "")
    return url or (request.build_absolute_uri("/")[:-1])

def feed_yml(request):
    """
    Yandex.Market YML feed:
    /catalog/feed/yml.xml
    """
    from .models import Product, Category

    products = (
        Product.objects
        .filter(is_active=True)
        .select_related("category", "category__parent")
        .prefetch_related("images", "attributes__attribute")
    )

    # Фильтруем офферы для Яндекса:
    # - положительная цена
    # - есть URL товара
    valid_products = []
    for p in products:
        try:
            price_ok = (p.price or 0) > 0
        except Exception:
            price_ok = False
        url_ok = bool(_product_url(request, p))
        if price_ok and url_ok:
            valid_products.append(p)

    # Категории, реально используемые
    cats = {}
    for p in valid_products:
        c = p.category
        if not c:
            continue
        cats[c.id] = c
        if c.parent:
            cats[c.parent.id] = c.parent

    # Отсортируем категории по id для стабильности
    cats_sorted = sorted(cats.values(), key=lambda x: x.id)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = []
    parts.append('<?xml version="1.0" encoding="utf-8"?>')
    parts.append('<!DOCTYPE yml_catalog SYSTEM "shops.dtd">')
    parts.append(f'<yml_catalog date="{now}">')
    parts.append("<shop>")
    parts.append(f"<name>{escape(getattr(settings, 'SITE_NAME', 'Gonka'))}</name>")
    parts.append(f"<company>{escape(getattr(settings, 'SITE_NAME', 'Gonka'))}</company>")
    parts.append(f"<url>{escape(_site_url(request) or '')}</url>")
    parts.append('<currencies><currency id="RUB" rate="1"/></currencies>')

    # Категории
    parts.append("<categories>")
    for c in cats_sorted:
        if c.parent:
            parts.append(
                f'<category id="{c.id}" parentId="{c.parent.id}">{escape(c.name)}</category>'
            )
        else:
            parts.append(f'<category id="{c.id}">{escape(c.name)}</category>')
    parts.append("</categories>")

    # Офферы
    parts.append("<offers>")
    for p in valid_products:
        c = p.category
        cat_id = getattr(c, "id", "")
        url = _product_url(request, p)
        img = _first_image_url(request, p)
        brand = _attr_value(p, ["Производитель", "Бренд"])
        gtin  = _attr_value(p, ["GTIN", "EAN", "Штрихкод"])
        weight = _weight_kg(p)

        # Яндекс любит доступность true/false, но при price>0 и наличии — true ок
        parts.append(f'<offer id="{p.id}" available="true">')

        parts.append(f"<url>{escape(url)}</url>")
        parts.append(f"<price>{p.price:.2f}</price>")
        parts.append("<currencyId>RUB</currencyId>")
        if cat_id:
            parts.append(f"<categoryId>{cat_id}</categoryId>")
        if img:
            parts.append(f"<picture>{escape(img)}</picture>")
        if brand:
            parts.append(f"<vendor>{escape(brand)}</vendor>")
        if p.sku:
            parts.append(f"<vendorCode>{escape(p.sku)}</vendorCode>")

        # Название
        parts.append(f"<name>{escape(p.title or '')}</name>")

        # Описание — правильно, тег + CDATA внутри
        desc = (p.description or "").strip()
        if desc:
            parts.append("<description><![CDATA[")
            parts.append(desc)
            parts.append("]]></description>")

        # Параметры (совместимость и вес)
        for nm in ["Марка авто", "Марка", "Модель авто", "Модель", "Двигатель", "Engine", "Год"]:
            val = _attr_value(p, [nm])
            if val:
                parts.append(f'<param name="{escape(nm)}">{escape(val)}</param>')

        if weight:
            parts.append(f'<param name="Вес (кг)">{escape(weight)}</param>')

        if gtin:
            parts.append(f"<barcode>{escape(gtin)}</barcode>")

        parts.append("</offer>")

    parts.append("</offers>")
    parts.append("</shop></yml_catalog>")
    xml = "".join(parts)
    return HttpResponse(xml, content_type="application/xml; charset=utf-8")


def _category_choices():
    # удобный список для <select>
    return Category.objects.order_by("parent__name", "name").select_related("parent")


@require_POST
def m_category_toggle_hidden(request, cat_id: int):
    cat = get_object_or_404(Category, id=cat_id)
    # чекбокс присылается если включён
    new_value = bool(request.POST.get("is_hidden"))
    if cat.is_hidden != new_value:
        cat.is_hidden = new_value
        cat.save(update_fields=["is_hidden"])
        state = "скрыта" if new_value else "показана"
        messages.success(request, f"Категория «{cat.name}» теперь {state}.")
    else:
        messages.info(request, "Изменений нет.")
    return redirect("catalog:m_categories")