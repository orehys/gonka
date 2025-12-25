# catalog/templatetags/htmlutils.py
import re
from django import template
from django.utils.html import escape
from django.utils.safestring import mark_safe
from django.template.defaultfilters import linebreaksbr

try:
    import bleach
except ImportError:
    bleach = None

register = template.Library()

_HTML_TAG_RE = re.compile(r"<[a-zA-Z][^>]*>")

# Разрешённые теги/атрибуты для очистки (подредактируй под себя)
BLEACH_TAGS = [
    "p", "br", "ul", "ol", "li",
    "strong", "b", "em", "i", "u",
    "h1", "h2", "h3", "h4",
    "a", "span",
    "table", "thead", "tbody", "tr", "th", "td",
]
BLEACH_ATTRS = {
    "*": ["class", "style"],
    "a": ["href", "title", "target", "rel"],
}
BLEACH_PROTOCOLS = ["http", "https", "mailto"]

@register.filter(name="smart_html")
def smart_html(value: str):
    """
    Если строка похожа на HTML — санитизирует (через bleach, если установлен)
    и возвращает mark_safe. Если это обычный текст — экранирует и ставит <br>.
    """
    if not value:
        return ""

    s = str(value)

    # Есть HTML-теги?
    if _HTML_TAG_RE.search(s):
        if bleach:
            cleaned = bleach.clean(
                s,
                tags=BLEACH_TAGS,
                attributes=BLEACH_ATTRS,
                protocols=BLEACH_PROTOCOLS,
                strip=True,
            )
            return mark_safe(cleaned)
        # fallback: просто доверимся (на свой страх и риск)
        return mark_safe(s)

    # Не HTML: экранируем и превращаем переводы строк в <br>
    return mark_safe(linebreaksbr(escape(s)))
