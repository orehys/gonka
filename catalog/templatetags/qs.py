from django import template
register = template.Library()

@register.simple_tag(takes_context=True)
def qs(context, clear=False, **kwargs):
    """
    clear=True — удалить ВСЕ текущие параметры и собрать только из kwargs.
    Иначе сохраняем текущие и заменяем/удаляем указанные.
    """
    params = context['request'].GET.copy()
    if clear:
        params.clear()
    for k, v in kwargs.items():
        if v in ("", None, False):
            params.pop(k, None)
        else:
            params[k] = v
    q = params.urlencode()
    return ("?" + q) if q else ""
