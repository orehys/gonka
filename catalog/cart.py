from decimal import Decimal
from .models import Product

CART_SESSION_ID = "cart_v1"

def _cart(request) -> dict:
    cart = request.session.get(CART_SESSION_ID)
    if cart is None:
        cart = {}
        request.session[CART_SESSION_ID] = cart
    return cart

def add(request, product_id: int, qty: int = 1, replace: bool = False):
    cart = _cart(request)
    pid = str(product_id)
    cur = int(cart.get(pid, 0))
    cart[pid] = max(1, qty if replace else (cur + qty))
    request.session.modified = True

def set_qty(request, product_id: int, qty: int):
    add(request, product_id, qty=qty, replace=True)

def remove(request, product_id: int):
    cart = _cart(request)
    cart.pop(str(product_id), None)
    request.session.modified = True

def clear(request):
    request.session[CART_SESSION_ID] = {}
    request.session.modified = True

def items(request):
    """Итератор по позициям корзины с продуктами и суммами."""
    cart = _cart(request)
    ids = [int(pid) for pid in cart.keys()]
    products = {p.id: p for p in Product.objects.filter(id__in=ids).prefetch_related("images")}
    for pid, qty in cart.items():
        p = products.get(int(pid))
        if not p:  # продукт уже удалён
            continue
        qty = int(qty)
        yield {
            "product": p,
            "qty": qty,
            "price": Decimal(p.price or 0),
            "subtotal": Decimal(p.price or 0) * qty,
        }

def totals(request):
    total = sum(row["subtotal"] for row in items(request))
    count = sum(row["qty"] for row in items(request))
    return {"total": total, "count": count}
