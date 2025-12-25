from django import template

register = template.Library()

@register.filter
def dict_get(d, key, default=0):
    """
    Безопасно получить d[key] в шаблоне: {{ mydict|dict_get:some_id }}
    Возвращает 0 (или default) если ключа нет или d не словарь.
    """
    try:
        return d.get(key, default)
    except Exception:
        return default