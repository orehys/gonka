# catalog/middleware.py
import re
from django.http import (
    HttpResponseGone,
    HttpResponsePermanentRedirect,
    HttpResponseRedirect,
)
from .models import RedirectRule, GoneURL


class SEORedirectMiddleware:
    """
    410 Gone + 301/302 редиректы из БД.
    Располагать после SecurityMiddleware и до CommonMiddleware.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        path = request.path

        # --- 410 Gone: точное совпадение
        if GoneURL.objects.filter(is_active=True, is_regex=False, path_or_regex=path).exists():
            return HttpResponseGone("Gone")

        # --- 410 Gone: regex
        for row in GoneURL.objects.filter(is_active=True, is_regex=True):
            try:
                if re.search(row.path_or_regex, path):
                    return HttpResponseGone("Gone")
            except re.error:
                # некорректная регулярка — просто пропускаем
                continue

        # --- Redirects ---
        # 1) Сначала точное совпадение (приоритет выше)
        rule = (RedirectRule.objects
                .filter(is_active=True, from_path=path)
                .order_by("-priority", "id")
                .first())
        if rule and rule.to_url and rule.to_url != path:
            is_301 = (rule.code or 301) == 301
            Resp = HttpResponsePermanentRedirect if is_301 else HttpResponseRedirect
            return Resp(rule.to_url)

        # 2) Затем regex-правила (в этой модели нет флага is_regex — пробуем как regex)
        for row in RedirectRule.objects.filter(is_active=True).exclude(from_path=path).order_by("-priority", "id"):
            src = row.from_path or ""
            dst = row.to_url or ""
            if not src or not dst:
                continue
            try:
                if re.search(src, path):
                    # поддерживаем подстановки по группам через re.sub
                    new_url = re.sub(src, dst, path)
                    if new_url and new_url != path:
                        is_301 = (row.code or 301) == 301
                        Resp = HttpResponsePermanentRedirect if is_301 else HttpResponseRedirect
                        return Resp(new_url)
            except re.error:
                # невалидная регулярка — пропускаем
                continue

        return self.get_response(request)
