import re
from django import template
from django.utils.html import escape
from django.utils.safestring import mark_safe
from django.utils.html import strip_tags

register = template.Library()

TAG_RE = re.compile(r"<[a-zA-Z][^>]*>")

@register.filter
def smart_html(value: str):
    """
    HTML → отдать как есть.
    Plain text → разбить на абзацы и обернуть в <p>.
    """
    if not value:
        return ""
    s = str(value)

    # уже HTML? вернём как есть
    if TAG_RE.search(s):
        return mark_safe(s)

    # обычный текст → параграфы
    parts = [p.strip() for p in re.split(r"\n\s*\n", s) if p.strip()]
    html = "".join(f"<p>{p.replace('\n','<br>')}</p>" for p in parts) or f"<p>{s}</p>"
    return mark_safe(html)

@register.filter
def pretty_specs(text: str):
    """
    Превращает '... Спецификации: ...; ...; ...' в строки с <br>.
    Безопасно экранирует HTML внутри.
    """
    if not text:
        return ""
    s = str(text).replace("\r\n", "\n").replace("\r", "\n")

    # перенос сразу после "Спецификации:" / "Specifications:"
    s = re.sub(r"(?i)(спецификации|specifications)\s*:\s*", r"\1:\n", s)

    # делим по ; и переносам
    parts = [p.strip(" ;") for p in re.split(r"[;\n]+", s) if p.strip(" ;")]

    safe_lines = [escape(p) for p in parts]
    return mark_safe("<br>".join(safe_lines))

@register.filter
def pretty_specs_list(text: str):
    """
    Вариант как маркированный список <ul>.
    """
    if not text:
        return ""
    s = str(text).replace("\r\n","\n").replace("\r","\n")
    s = re.sub(r"(?i)(спецификации|specifications)\s*:\s*", r"\1:\n", s)
    parts = [p.strip(" ;") for p in re.split(r"[;\n]+", s) if p.strip(" ;")]
    if not parts:
        return ""
    items = "".join(f"<li>{escape(p)}</li>" for p in parts)
    return mark_safe(f"<ul class='desc-list'>{items}</ul>")

@register.filter
def trunc50_3dots(text):
    s = str(text or "")
    return s if len(s) <= 50 else s[:50].rstrip() + "..."