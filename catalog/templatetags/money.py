from decimal import Decimal, ROUND_HALF_UP
from django import template

register = template.Library()
NBSP = "\u00A0"

def _dec(x) -> Decimal:
    try:
        return Decimal(str(x))
    except Exception:
        return Decimal("0")

@register.filter
def money(value, currency="₽"):
    d = _dec(value).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    s = f"{int(d):,}".replace(",", NBSP)
    return f"{s}{(' ' + currency) if currency else ''}"

@register.filter
def old_price(value, factor=1.1):
    d = _dec(value) * _dec(factor)
    return d.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
