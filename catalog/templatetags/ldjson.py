# catalog/templatetags/ldjson.py
import json
from django import template
from django.utils.safestring import mark_safe
register = template.Library()

@register.simple_tag
def ldjson(data: dict):
    return mark_safe(f'<script type="application/ld+json">{json.dumps(data, ensure_ascii=False)}</script>')
