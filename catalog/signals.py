# catalog/signals.py
from django.contrib.auth import get_user_model
from django.contrib.auth.signals import user_logged_in 
from django.db.models.signals import pre_save, post_save
from django.dispatch import receiver
from django.conf import settings
from django.urls import reverse

import logging
from django.db import transaction
from .models import (
    Order,
    ChipRequest,
    ServiceRequest,
    ContactRequest,
    UserProfile,
    SavedCartItem,
    Favorite, Product, Category, RedirectRule,
)
from .cart import CART_SESSION_ID

User = get_user_model()
logger = logging.getLogger(__name__)
print("signals loaded")  

def _site():
    return getattr(settings, "SITE_URL", "http://127.0.0.1:8000/").rstrip("/")

def _safe(v):
    return (v or "").strip()




@receiver(post_save, sender=User)
def create_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.get_or_create(user=instance)


def _merge_cart_session_to_user(request, user):
    raw = request.session.get(CART_SESSION_ID, {})
    if isinstance(raw, dict):
        for pid, qty in raw.items():
            try:
                pid = int(pid)
                qty = int(qty)
                if qty < 1:
                    continue
                obj, created = SavedCartItem.objects.get_or_create(
                    user=user, product_id=pid, defaults={"qty": qty}
                )
                if not created:
                    obj.qty = max(obj.qty, qty)
                    obj.save(update_fields=["qty"])
            except Exception:
                continue


def _merge_cart_user_to_session(request, user):
    raw = request.session.get(CART_SESSION_ID, {})
    if not isinstance(raw, dict):
        raw = {}
    for it in SavedCartItem.objects.filter(user=user):
        raw[str(it.product_id)] = max(int(raw.get(str(it.product_id), 0)), it.qty)
    request.session[CART_SESSION_ID] = raw
    request.session.modified = True


def _merge_favorites_session_to_user(request, user):
    fav_ids = request.session.get("fav_ids", [])
    if isinstance(fav_ids, (list, tuple)):
        for pid in fav_ids:
            try:
                Favorite.objects.get_or_create(user=user, product_id=int(pid))
            except Exception:
                pass
        request.session["fav_ids"] = []
        request.session.modified = True


@receiver(user_logged_in)
def on_login(sender, user, request, **kwargs):
    _merge_cart_session_to_user(request, user)
    _merge_cart_user_to_session(request, user)
    _merge_favorites_session_to_user(request, user)

def _product_path(prod: Product) -> str:
    cat = prod.category
    if not cat or not cat.parent:
        return ""  # нет полного пути — не строим авто-редирект
    return f"/catalog/{cat.parent.slug}/{cat.slug}/{prod.slug}/"

@receiver(pre_save, sender=Product)
def auto_redirect_on_product_slug_change(sender, instance: Product, **kwargs):
    if not instance.pk:
        return
    try:
        old = Product.objects.select_related("category","category__parent").get(pk=instance.pk)
    except Product.DoesNotExist:
        return
    if old.slug != instance.slug:
        old_path = _product_path(old)
        new_path = _product_path(instance)
        if old_path and new_path and old_path != new_path:
            RedirectRule.objects.get_or_create(old_path=old_path, defaults={"new_url": new_path, "code": 301, "note": "auto product slug change"})
