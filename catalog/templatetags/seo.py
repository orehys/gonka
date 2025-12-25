# catalog/templatetags/seo.py
from django import template
from django.contrib.contenttypes.models import ContentType

import re

from django.conf import settings

from catalog.models import SEOEntry, SEOObject, SEOPath  # если app не catalog — поправь импорт

register = template.Library()


def _first(qs):
    """Безопасно получаем первый объект из queryset, либо None."""
    try:
        return qs.first()
    except Exception:
        return None


def _robots_value(obj) -> str:
    """Собираем значение meta robots из полей robots_index / robots_follow."""
    index = getattr(obj, "robots_index", True)
    follow = getattr(obj, "robots_follow", True)
    return f"{'index' if index else 'noindex'}, {'follow' if follow else 'nofollow'}"


def _seo_to_dict(obj):
    """Приводим любой SEO-объект к универсальному dict для шаблонов."""
    if not obj:
        return {}

    return {
        "title": getattr(obj, "title", "") or "",
        "h1": getattr(obj, "h1", "") or "",
        "description": getattr(obj, "meta_description", "") or "",
        "robots": _robots_value(obj),
        "canonical": getattr(obj, "canonical_url", "") or "",
        # json_ld пока не храним в моделях — оставим пустую строку для совместимости
        "json_ld": "",
    }


@register.simple_tag(takes_context=True)
def get_seo(context, obj=None):
    """
    Универсальный SEO-тег.

    obj может быть:
      - объектом модели (Product/Category/News и т.д.)
      - строковым путём ("/catalog/...")
      - request
      - None (тогда сначала попробуем объекты из контекста, потом request.path)

    Возвращает dict: {title, h1, description, robots, canonical, json_ld}

    ВАЖНО: если ничего не найдено, вернёт пустые строки — как и раньше.
    Фолбэки по умолчанию (Gonka.shop и т.п.) задаются уже в шаблоне.
    """
    request = context.get("request")

    # Базовый dict, чтобы не ломать существующие шаблоны
    data = {
        "title": "",
        "h1": "",
        "description": "",
        "robots": "",
        "canonical": "",
        "json_ld": "",
    }

    # Если передали request — ведём себя как будто obj не передали (берём request.path)
    if obj is not None and hasattr(obj, "build_absolute_uri"):
        obj = None

    path = None

    # Если obj - строка, считаем её путём
    if isinstance(obj, str):
        path = obj.strip()
    elif request is not None:
        # Если строку явно не задали – берём текущий путь
        path = request.path

    se_obj = None  # сюда положим найденный SEO-объект (SEOEntry/SEOObject/SEOPath)

    try:
        # 1) Сначала ищем SEO по объектам (явно переданный obj + объекты из контекста)
        candidates = []

        # 1.1. Явно переданный объект
        if obj is not None and not isinstance(obj, str):
            candidates.append(obj)

        # 1.2. Объекты из контекста (как в старой логике)
        for key in ("product", "current_sub", "current_cat", "category", "news", "object"):
            inst = context.get(key)
            if inst is not None:
                candidates.append(inst)

        for inst in candidates:
            try:
                ct = ContentType.objects.get_for_model(inst.__class__)
            except Exception:
                continue

            # Приоритет: SEOEntry (новая модель)
            entry = _first(
                SEOEntry.objects.filter(
                    is_active=True,
                    content_type=ct,
                    object_id=getattr(inst, "pk", None),
                ).order_by("-updated_at", "-id")
            )
            if entry:
                se_obj = entry
                break

            # Затем SEOObject (старая модель персональных настроек)
            old = _first(
                SEOObject.objects.filter(
                    content_type=ct,
                    object_id=getattr(inst, "pk", None),
                ).order_by("-updated_at")
            )
            if old:
                se_obj = old
                break

        # 2) Если по объектам ничего не нашли — ищем по URL
        if se_obj is None and path:
            # 2.1. SEOEntry с path_regex (регулярки)
            entries = SEOEntry.objects.filter(is_active=True).exclude(path_regex="").order_by(
                "-updated_at", "-id"
            )
            for entry in entries:
                try:
                    if re.search(entry.path_regex, path):
                        se_obj = entry
                        break
                except re.error:
                    # Некорректный regex не должен валить страницу
                    if getattr(settings, "DEBUG", False):
                        print(f"Bad SEOEntry.path_regex: {entry.path_regex!r}")
                    continue

            # 2.2. Если нет совпадения по regex – пробуем точный путь в SEOPath
            if se_obj is None:
                seopath = _first(
                    SEOPath.objects.filter(path=path).order_by("-id")
                )
                if seopath:
                    se_obj = seopath

    except Exception as e:
        # Не роняем сайт, просто логируем в DEBUG-режиме
        if getattr(settings, "DEBUG", False):
            print("SEO error in get_seo:", e)

    # Если что-то нашли — обновляем базовый dict полями из модели
    if se_obj is not None:
        data.update(_seo_to_dict(se_obj))

    return data
