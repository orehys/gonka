from .cart import totals, CART_SESSION_ID
from django.core.cache import cache
from .models import Category


def seo_defaults(request):
    return {
        "current_cat": None,
        "current_sub": None,
        "breadcrumb_items": [],
        "brand_name": "",
        "first_image_url": "",
    }
def cart_summary(request):
    t = totals(request)
    raw = request.session.get(CART_SESSION_ID, {})
    cart_ids = [int(k) for k in raw.keys()] if isinstance(raw, dict) else []
    cart_qty = {int(k): int(v) for k, v in raw.items()} if isinstance(raw, dict) else {}
    return {
        "cart_count": t["count"],
        "cart_total": t["total"],
        "cart_ids": cart_ids,
        "cart_qty": cart_qty,
    }

def header_categories(request):
    """
    Топ-категории с подкатегориями для селекта в шапке.
    Кэшируем на 5 минут, чтобы не дергать БД на каждый запрос.
    """
    cats = cache.get("header_cats_v1")
    if cats is None:
        qs = (Category.objects
              .filter(parent__isnull=True, is_hidden=False)
              .prefetch_related("children")
              .order_by("name"))
        cats = list(qs)  
        cache.set("header_cats_v1", cats, 300)
    return {"header_cats": cats}

def cart_context(request):
    raw = request.session.get(CART_SESSION_ID, {})
    try:
        count = sum(int(v) for v in raw.values()) if isinstance(raw, dict) else 0
    except Exception:
        count = 0
    return {"cart_count": count}