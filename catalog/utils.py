# catalog/utils.py
from django.db.models import Case, When, IntegerField
from .models import Product

SESSION_KEY = "recent_ids"

def push_recent(request, product_id: int, maxlen: int = 20):
    """Поместить товар в начало списка недавно просмотренных."""
    try:
        pid = int(product_id)
    except Exception:
        return
    data = request.session.get(SESSION_KEY, [])
    if not isinstance(data, list):
        data = []
    # убрать дубликаты + в начало
    data = [x for x in data if x != pid]
    data.insert(0, pid)
    # ограничить длину
    data = data[:maxlen]
    request.session[SESSION_KEY] = data
    request.session.modified = True

def recent_qs(request, limit: int = 3, exclude_id: int | None = None):
    """Вернуть QuerySet товаров в порядке просмотра."""
    ids = request.session.get(SESSION_KEY, [])
    if not isinstance(ids, list) or not ids:
        return Product.objects.none()
    # исключить текущий товар при необходимости
    if exclude_id is not None:
        ids = [i for i in ids if i != exclude_id]
    if not ids:
        return Product.objects.none()

    ids_cut = ids[:limit]
    # сохранить порядок ids с помощью CASE
    when_list = [When(id=pk, then=pos) for pos, pk in enumerate(ids_cut)]
    order = Case(*when_list, default=len(ids_cut), output_field=IntegerField())

    return (Product.objects.filter(id__in=ids_cut, is_active=True)
            .prefetch_related("images")
            .order_by(order))
