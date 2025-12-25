# catalog/management/commands/enforce_price_visibility.py
from django.core.management.base import BaseCommand
from django.db import transaction
from django.db.models import Q
from decimal import Decimal
import os
import requests
from django.conf import settings
from catalog.models import Product

def send_tg(text_html: str):
    """Отправка уведомления в Telegram, если заданы TG_BOT_TOKEN/TG_CHAT_ID."""
    token = "8445903639:AAHcAncSxyKMYVJPV95C7ET8tqqtK9WpXk0"
    chat_id  = "-4918686740"
    if not token or not chat_id:
        print("TG: пропущено (нет TG_BOT_TOKEN/TG_CHAT_ID в окружении)")
        return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text_html, "parse_mode": "HTML"},
            timeout=15,
        )
        if r.status_code != 200:
            print(f'TG: ошибка {r.status_code}: {r.text}')
    except Exception as e:
        print(f"TG: исключение: {e}")

class Command(BaseCommand):
    help = (
        "Автоскрытие товаров с ценой 0 и автоматическое включение тех, "
        "которые ранее были скрыты этим скриптом, если цена снова > 0."
    )

    def add_arguments(self, parser):
        parser.add_argument("--dry-run", action="store_true", help="Только показать, что будет сделано")
        parser.add_argument("--batch", type=int, default=1000, help="Размер батча для апдейтов")

    @transaction.atomic
    def handle(self, *args, **opts):
        dry = opts["dry_run"]
        batch = opts["batch"]

        to_hide = (Product.objects
                   .filter(is_active=True)
                   .filter(Q(price__isnull=True) | Q(price__lte=Decimal("0"))))

        to_show = (Product.objects
                   .filter(is_active=False, auto_hidden_zero_price=True)
                   .filter(price__gt=Decimal("0")))

        hide_count = to_hide.count()
        show_count = to_show.count()

        self.stdout.write(f"Скрыть (price<=0): {hide_count}")
        self.stdout.write(f"Включить (price>0, auto_hidden_zero_price=True): {show_count}")

        if dry:
            self.stdout.write("DRY-RUN: изменений не вносилось.")
            return

        # Скрываем батчами
        if hide_count:
            pk_list = list(to_hide.values_list("pk", flat=True))
            for i in range(0, len(pk_list), batch):
                chunk = pk_list[i:i+batch]
                (Product.objects
                 .filter(pk__in=chunk)
                 .update(is_active=False, auto_hidden_zero_price=True))

        # Включаем батчами
        if show_count:
            pk_list = list(to_show.values_list("pk", flat=True))
            for i in range(0, len(pk_list), batch):
                chunk = pk_list[i:i+batch]
                (Product.objects
                 .filter(pk__in=chunk)
                 .update(is_active=True, auto_hidden_zero_price=False))

        msg = (
            "<b>Ежедневный прогон цен (автоскрытие)</b>\n"
            f"Скрыто (price≤0): <b>{hide_count}</b>\n"
            f"Включено обратно (price>0): <b>{show_count}</b>"
        )
        send_tg(msg)

        self.stdout.write(self.style.SUCCESS(
            f"Готово. Скрыто: {hide_count}, включено: {show_count}"
        ))
