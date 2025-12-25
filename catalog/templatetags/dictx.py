from django import template
register = template.Library()

@register.filter(name="dict_get")
def dict_get(d, key, default=0):
    """Безопасно получить d[key] в шаблоне."""
    try:
        return d.get(key, default)
    except Exception:
        return default
