# catalog/models.py
import os, re
from django.db import models
from django.utils.text import slugify
from django.conf import settings
from decimal import Decimal, ROUND_HALF_UP
from django.utils import timezone
from django.urls import reverse
from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType
from django.core.validators import MinValueValidator

def fs_safe_segment(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r'[<>:"/\\|?*#]+', "_", s)  
    s = re.sub(r"\s+", "-", s)
    return s[:60] or "item"

def unique_slugify(instance, base: str, field_name: str = "slug", max_len: int = 120):
    Model = instance.__class__
    slug = base[:max_len].strip("-")
    n = 2
    while Model.objects.filter(**{field_name: slug}).exclude(pk=instance.pk).exists():
        suffix = f"-{n}"
        slug = (base[: max_len - len(suffix)] + suffix).strip("-")
        n += 1
    return slug

def product_image_upload_path(instance, filename):
    sku = getattr(instance.product, "sku", "") or "unknown"
    folder = fs_safe_segment(sku)
    base, ext = os.path.splitext(os.path.basename(filename))
    safe_name = f"{fs_safe_segment(base)}{(ext or '.jpg').lower()}"
    return f"products/{folder}/{safe_name}"
    sku = getattr(instance.product, "sku", "") or "unknown"
    folder = fs_safe_segment(sku)
    base, ext = os.path.splitext(os.path.basename(filename))
    safe_name = f"{fs_safe_segment(base)}{(ext or '.jpg').lower()}"
    return f"products/{folder}/{safe_name}"

class Category(models.Model):
    name = models.CharField(max_length=200)
    slug = models.SlugField(max_length=220, unique=True, blank=True)
    parent = models.ForeignKey("self", null=True, blank=True, on_delete=models.CASCADE, related_name="children")
    is_hidden = models.BooleanField(default=False, db_index=True, verbose_name="Скрыта")
    class Meta:
        unique_together = [("parent","slug")]
        ordering = ["name"]
    def save(self, *a, **kw):
        if not self.slug:
            base = slugify(self.name, allow_unicode=True) or fs_safe_segment(self.name).lower()
            self.slug = unique_slugify(self, base, "slug", 220)
        super().save(*a, **kw)

class Product(models.Model):
    title = models.CharField(max_length=255)
    slug = models.SlugField(max_length=280, unique=True, blank=True)
    category = models.ForeignKey(Category, on_delete=models.SET_NULL, null=True, related_name="products")
    sku = models.CharField("Артикул", max_length=100, unique=True)
    price = models.DecimalField(max_digits=12, decimal_places=2)
    supplier     = models.CharField(max_length=50, blank=True, db_index=True) 
    supplier_ref = models.CharField(max_length=120, blank=True)    
    old_price = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)  
    cost_price = models.DecimalField(
        "Себестоимость (тех.)", max_digits=12, decimal_places=2, null=True, blank=True
    )

    def get_old_price(self):
        """Если задана old_price — вернуть её, иначе 10% надбавка от price."""
        if self.old_price is not None:
            return self.old_price
        return (self.price * Decimal("1.10")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        
    description = models.TextField(blank=True)
    external_url = models.URLField("Ссылка", blank=True)
    is_active = models.BooleanField(default=True)
    in_stock  = models.BooleanField(default=True) 
    created_at = models.DateTimeField(auto_now_add=True)
    auto_hidden_zero_price = models.BooleanField(default=False)
    stock = models.PositiveIntegerField(
        "Остаток",
        null=True, blank=True,
        validators=[MinValueValidator(0)]
    )

    def get_absolute_url(self):
        if self.category and self.category.parent:
            return reverse(
                "catalog:product_detail_by_path",
                kwargs={
                    "cat_slug": self.category.parent.slug,
                    "sub_slug": self.category.slug,
                    "slug": self.slug,
                },
            )
        # фолбэк, если категория не задана/неполная
        return reverse("catalog:product_detail", kwargs={"slug": self.slug})

    class Meta:
        indexes = [models.Index(fields=["sku"]), models.Index(fields=["slug"])]

    def save(self,*a,**kw):
        if not self.slug:
            # используем slugify(title) +  SKU 
            base = f"{slugify(self.title) or 'item'}-{fs_safe_segment(self.sku).lower()}"
            self.slug = unique_slugify(self, base, "slug", 280)
        super().save(*a,**kw)

    def __str__(self): return f"{self.title} ({self.sku})"

class Attribute(models.Model):
    name = models.CharField(max_length=100)                     
    slug = models.SlugField(max_length=120, unique=True, blank=True)

    def save(self, *a, **kw):
        if not self.slug:
            base = slugify(self.name, allow_unicode=True) or "attr"
            self.slug = unique_slugify(self, base, "slug", 120)
        super().save(*a, **kw)

    def __str__(self): return self.name

class ProductAttributeValue(models.Model):
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name="attributes")
    attribute = models.ForeignKey(Attribute, on_delete=models.CASCADE)
    value = models.CharField(max_length=255)
    class Meta:
        unique_together = [("product","attribute","value")]

class ProductImage(models.Model):
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name="images")
    image = models.ImageField(upload_to=product_image_upload_path)
    alt = models.CharField(max_length=255, blank=True)
    position = models.PositiveIntegerField(default=0)
    class Meta:
        ordering = ["position","id"]

class Order(models.Model):
    NEW, PROCESS, DONE, CANCELED = "new", "process", "done", "canceled"
    STATUSES = [(NEW,"Новый"),(PROCESS,"В обработке"),(DONE,"Выполнен"),(CANCELED,"Отменён")]

    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=16, choices=STATUSES, default=NEW)

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True, blank=True,
        on_delete=models.SET_NULL,
        related_name="orders"
    )

    name = models.CharField(max_length=100)
    phone = models.CharField(max_length=50)
    email = models.EmailField(blank=True)
    city = models.CharField(max_length=100, blank=True)
    address = models.CharField(max_length=200, blank=True)
    comment = models.TextField(blank=True)

    total = models.DecimalField(max_digits=12, decimal_places=2, default=0)

    delivery = models.CharField(max_length=20, blank=True)
    payment  = models.CharField(max_length=20, blank=True)


    def __str__(self):
        return f"Заказ #{self.id} ({self.get_status_display()})"

class OrderItem(models.Model):
    order = models.ForeignKey(Order, related_name="items", on_delete=models.CASCADE)
    product = models.ForeignKey("catalog.Product", null=True, blank=True, on_delete=models.SET_NULL)
    title = models.CharField(max_length=255)
    sku = models.CharField(max_length=100, blank=True)
    price = models.DecimalField(max_digits=12, decimal_places=2)
    qty = models.PositiveIntegerField(default=1)
    subtotal = models.DecimalField(max_digits=12, decimal_places=2)

    def __str__(self):
        return f"{self.title} x{self.qty}"

class Favorite(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="favorites")
    product = models.ForeignKey("catalog.Product", on_delete=models.CASCADE, related_name="faved_by")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("user", "product")

class UserProfile(models.Model):
    user     = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="profile")
    full_name = models.CharField(max_length=200, blank=True)
    phone     = models.CharField(max_length=50, blank=True)
    city      = models.CharField(max_length=100, blank=True)
    address   = models.CharField(max_length=255, blank=True)
    marketing_email = models.BooleanField(default=True, verbose_name="Согласие на e-mail рассылку")

    def __str__(self):
        return f"Профиль {self.user}"

class SavedCartItem(models.Model):
    user    = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="saved_cart")
    product = models.ForeignKey("catalog.Product", on_delete=models.CASCADE)
    qty     = models.PositiveIntegerField(default=1)

    class Meta:
        unique_together = ("user", "product")

# ── ЧИП-ТЮНИНГ ────────────────────────────────────────────────────────────────

class ChipCar(models.Model):
    make   = models.CharField(max_length=100)   
    model  = models.CharField(max_length=150)   
    year   = models.CharField(max_length=50)    
    engine = models.CharField(max_length=150)   

    class Meta:
        unique_together = ("make", "model", "year", "engine")
        ordering = ("make", "model", "year", "engine")

    def __str__(self):
        return f"{self.make} {self.model} {self.year} {self.engine}"


class ChipStage(models.Model):
    car  = models.ForeignKey(ChipCar, related_name="stages", on_delete=models.CASCADE)
    name = models.CharField(max_length=50)    

    def _price_metric(self):
        """Первая строка с непустой ценой (если есть)."""
        return self.metrics.exclude(price__isnull=True).order_by('order', 'id').first()

    @property
    def price(self):
        m = self._price_metric()
        return m.price if m else None

    @property
    def old_price(self):
        p = self.price
        if p is None:
            return None
        return (p * Decimal('1.10')).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    class Meta:
        unique_together = ("car", "name")
        ordering = ("name",)

    def __str__(self):
        return f"{self.car} · {self.name}"


class ChipMetric(models.Model):
    """
    Гибкая таблица показателей (строки таблицы):
    label  — "Мощность", "Крутящий момент", "0–100 км/ч", "Объём двигателя" и т.д.
    значения храним строками, чтобы не терять единицы измерения (лс, Нм, сек, см³).
    """
    stage = models.ForeignKey(ChipStage, related_name="metrics", on_delete=models.CASCADE)
    order = models.PositiveIntegerField(default=0)
    label = models.CharField(max_length=120)

    stock = models.CharField(max_length=50, blank=True, default="")
    tuned = models.CharField(max_length=50, blank=True, default="")
    delta = models.CharField(max_length=50, blank=True, default="")

    price = models.DecimalField(
        "Цена, ₽", max_digits=12, decimal_places=2, null=True, blank=True
    )

    @property
    def old_price(self):
        if self.price is None:
            return None
        return (self.price * Decimal("1.10")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    class Meta:
        ordering = ("order", "id")

    def __str__(self):
        return f"{self.stage} · {self.label}"

class ChipRequest(models.Model):
    STATUS_NEW = "new"
    STATUS_INPROG = "in_progress"
    STATUS_DONE = "done"
    STATUS_CHOICES = [
        (STATUS_NEW, "Новая"),
        (STATUS_INPROG, "В работе"),
        (STATUS_DONE, "Завершена"),
    ]

    created_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True,
        on_delete=models.SET_NULL, related_name="chip_requests"
    )

    name = models.CharField("Имя", max_length=150)
    phone = models.CharField("Телефон", max_length=100)

    car = models.ForeignKey(ChipCar, on_delete=models.CASCADE, related_name="leads", verbose_name="Модификация")
    stage = models.ForeignKey(ChipStage, null=True, blank=True, on_delete=models.SET_NULL, related_name="leads")

    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_NEW)

    class Meta:
        ordering = ("-created_at",)

    def __str__(self):
        return f"[{self.get_status_display()}] {self.name} {self.phone} · {self.car} · {self.stage or '—'}"

class ServiceRequest(models.Model):
    STATUS_CHOICES = (
        ("new", "Новая"),
        ("in_work", "В работе"),
        ("done", "Завершена"),
        ("cancelled", "Отменена"),
    )

    name       = models.CharField(max_length=150)
    phone      = models.CharField(max_length=50)
    comment    = models.TextField()
    status     = models.CharField(max_length=20, choices=STATUS_CHOICES, default="new")
    created_at = models.DateTimeField(auto_now_add=True)


    user       = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL)
    source_url = models.CharField(max_length=500, blank=True, default="")

    class Meta:
        ordering = ("-created_at",)

    def __str__(self):
        return f"{self.name} · {self.phone} · {self.get_status_display()}"

class News(models.Model):
    title = models.CharField("Заголовок", max_length=200)
    slug = models.SlugField("Слаг", max_length=220, unique=True, blank=True)
    author = models.CharField("Автор", max_length=120, blank=True)
    excerpt = models.TextField("Короткий текст", blank=True)
    body = models.TextField("Полный текст", blank=True)
    image = models.ImageField("Обложка", upload_to="news/", blank=True, null=True)
    published_at = models.DateTimeField("Дата публикации", default=timezone.now)
    is_published = models.BooleanField("Опубликовано", default=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("-published_at", "-id")
        verbose_name = "Новость"
        verbose_name_plural = "Новости"

    def __str__(self):
        return self.title

    def save(self, *args, **kwargs):
        if not self.slug:
            base = slugify(self.title) or "news"
            slug = base
            i = 2
            while News.objects.filter(slug=slug).exclude(pk=self.pk).exists():
                slug = f"{base}-{i}"
                i += 1
            self.slug = slug
        super().save(*args, **kwargs)

class Brand(models.Model):
    name = models.CharField(max_length=120, unique=True)
    slug = models.SlugField(max_length=140, unique=True, blank=True)
    logo = models.ImageField(upload_to="brands/", null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("name",)

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        if not self.slug:
            base = slugify(self.name) or "brand"
            slug = base
            i = 2
            from .models import Brand 
            while Brand.objects.filter(slug=slug).exclude(pk=self.pk).exists():
                slug = f"{base}-{i}"
                i += 1
            self.slug = slug
        super().save(*args, **kwargs)

class CarMake(models.Model):
    name = models.CharField(max_length=120, unique=True)
    slug = models.SlugField(max_length=140, unique=True, blank=True)
    logo = models.ImageField(upload_to="car_makes/", null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("name",)

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        if not self.slug:
            base = slugify(self.name) or "make"
            slug = base
            i = 2
            while CarMake.objects.filter(slug=slug).exclude(pk=self.pk).exists():
                slug = f"{base}-{i}"
                i += 1
            self.slug = slug
        super().save(*args, **kwargs)

class ContactRequest(models.Model):
    name       = models.CharField("Имя", max_length=120)
    phone      = models.CharField("Телефон", max_length=64)
    comment    = models.TextField("Комментарий", blank=True, default="")
    page       = models.CharField("Источник (страница)", max_length=120, blank=True, default="")
    created_at = models.DateTimeField("Создано", auto_now_add=True)

    class Meta:
        ordering = ("-created_at",)
        verbose_name = "обращение"
        verbose_name_plural = "обращения"

    def __str__(self):
        return f"{self.name} · {self.phone} · {self.created_at:%Y-%m-%d %H:%M}"


class CMSPage(models.Model):
    SLUGS = [
        ("about",    "О сервисе"),
        ("help",     "Помощь"),
        ("delivery", "Условия доставки"),
        ("payment",  "Условия оплаты"),
        ("dyno",     "Диностенд"),
        ("home",     "Главная"),
    ]
    slug       = models.SlugField("Системное имя", max_length=50, unique=True, choices=SLUGS)
    title      = models.CharField("Заголовок", max_length=200)
    content    = models.TextField("Текст (HTML или обычный)", blank=True, default="")
    updated_at = models.DateTimeField("Обновлено", auto_now=True)

    class Meta:
        ordering = ("slug",)
        verbose_name = "страница наполнения"
        verbose_name_plural = "страницы наполнения"

    def __str__(self):
        return dict(self.SLUGS).get(self.slug, self.slug)

class SitePage(models.Model):
    PAGES = [
        ("home",      "Главная"),
        ("about",     "О сервисе"),
        ("help",      "Помощь"),
        ("delivery",  "Условия доставки"),
        ("payment",   "Условия оплаты"),
        ("dyno",      "Диностенд"),
        ("contacts",  "Контакты"),
    ]
    slug        = models.CharField(max_length=32, choices=PAGES, unique=True)
    title       = models.CharField(max_length=255, blank=True)          
    subtitle    = models.CharField(max_length=255, blank=True)         
    hero_image  = models.ImageField(upload_to="pages/", null=True, blank=True)
    content_html = models.TextField(blank=True)                         
    extra       = models.JSONField(default=dict, blank=True)            
    updated_at  = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["id"]

    def __str__(self):
        return dict(self.PAGES).get(self.slug, self.slug)

class Banner(models.Model):
    GROUPS = [
        ("main-hero", "Главный слайдер"),   
    ]
    group      = models.CharField(max_length=50, choices=GROUPS, default="main-hero")
    title      = models.CharField(max_length=255, blank=True)
    alt        = models.CharField(max_length=255, blank=True)
    link_url   = models.CharField(max_length=400, blank=True)       # клик по баннеру (необязательно)
    image      = models.ImageField(upload_to="banners/")            # JPG/PNG/WebP — базовая картинка
    image_avif = models.ImageField(upload_to="banners/", null=True, blank=True)  # AVIF (опционально)
    image_mobile = models.ImageField(
        upload_to="banners/", blank=True, null=True, verbose_name="Мобильная картинка"
    )
    image_mobile_avif = models.ImageField(
        upload_to="banners/", blank=True, null=True, verbose_name="Мобильная картинка (AVIF)"
    )
    position   = models.PositiveIntegerField(default=0)              # для сортировки
    is_active  = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("position", "id")

    def __str__(self):
        return f"[{self.group}] {self.title or self.alt or self.id}"

class SEOObject(models.Model):
    """Персональные SEO-настройки для конкретной записи (Product, Category, News, CMSPage...)."""
    content_type = models.ForeignKey(ContentType, on_delete=models.CASCADE)
    object_id    = models.PositiveIntegerField()
    content_object = GenericForeignKey("content_type", "object_id")

    title            = models.CharField(max_length=255, blank=True)
    meta_description = models.CharField(max_length=300, blank=True)
    robots_index     = models.BooleanField(default=True)
    robots_follow    = models.BooleanField(default=True)
    canonical_url    = models.URLField(blank=True)
    h1               = models.CharField(max_length=255, blank=True)

    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = (("content_type", "object_id"),)

    def robots_value(self) -> str:
        return f"{'index' if self.robots_index else 'noindex'}, {'follow' if self.robots_follow else 'nofollow'}"


class SEOPath(models.Model):
    """SEO-настройки для произвольного URL-пути (например, /catalog/help/)."""
    path             = models.CharField(max_length=400, unique=True, help_text="Абсолютный путь, напр. /catalog/help/")
    title            = models.CharField(max_length=255, blank=True)
    meta_description = models.CharField(max_length=300, blank=True)
    robots_index     = models.BooleanField(default=True)
    robots_follow    = models.BooleanField(default=True)
    canonical_url    = models.URLField(blank=True)
    h1               = models.CharField(max_length=255, blank=True)
    updated_at       = models.DateTimeField(auto_now=True)

    def robots_value(self) -> str:
        return f"{'index' if self.robots_index else 'noindex'}, {'follow' if self.robots_follow else 'nofollow'}"

class SEOEntry(models.Model):
    """
    Либо привязка к объекту (content_type + object_id), либо path_regex (регулярка по URL).
    """
    is_active = models.BooleanField(default=True)

    # Привязка к объекту (опционально)
    content_type = models.ForeignKey(ContentType, null=True, blank=True, on_delete=models.CASCADE)
    object_id    = models.PositiveIntegerField(null=True, blank=True)
    content_object = GenericForeignKey("content_type", "object_id")

    # Либо матчинг по пути
    path_regex = models.CharField("Регулярка пути", max_length=400, blank=True, default="")

    # Метаданные
    title            = models.CharField("Title", max_length=255, blank=True, default="")
    h1               = models.CharField("H1", max_length=255, blank=True, default="")
    meta_description = models.TextField("Meta description", blank=True, default="")
    canonical_url    = models.CharField("Canonical URL", max_length=400, blank=True, default="")
    robots_index     = models.BooleanField("index", default=True)
    robots_follow    = models.BooleanField("follow", default=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("-updated_at", "-id")
        verbose_name = "SEO-запись"
        verbose_name_plural = "SEO-записи"

    def __str__(self):
        target = self.path_regex or (self.content_type and f"{self.content_type.app_label}.{self.content_type.model}#{self.object_id}") or "—"
        return f"[{'on' if self.is_active else 'off'}] {target}"


class RedirectRule(models.Model):
    """Простые редиректы. from_path может быть regex (если начинается с ^)."""
    is_active = models.BooleanField(default=True)
    from_path = models.CharField("Откуда (путь или regex)", max_length=400)
    to_url    = models.CharField("Куда (абс/отн URL)", max_length=500)
    code      = models.PositiveSmallIntegerField("Код", choices=[(301, "301"), (302, "302")], default=301)
    priority  = models.IntegerField("Приоритет", default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-priority", "-id")
        verbose_name = "редирект"
        verbose_name_plural = "редиректы"

    def __str__(self):
        return f"{self.from_path} → {self.to_url} ({self.code})"


class GoneURL(models.Model):
    path_or_regex = models.CharField("Путь или regex", max_length=400)
    is_regex   = models.BooleanField(default=False)
    is_active  = models.BooleanField(default=True)
    comment    = models.CharField("Комментарий", max_length=200, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)

    class Meta:
        ordering = ("-created_at",)

    def __str__(self):
        return self.path_or_regex


class RobotsTxt(models.Model):
    is_active = models.BooleanField("Активен", default=True)
    content   = models.TextField("robots.txt", default="User-agent: *\nDisallow:\n")
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "robots.txt"
        verbose_name_plural = "robots.txt"

    def __str__(self):
        return f"robots.txt ({'on' if self.is_active else 'off'})"

    is_active = models.BooleanField("Активен", default=True)
    content   = models.TextField("robots.txt", default="User-agent: *\nDisallow:\n")
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "robots.txt"
        verbose_name_plural = "robots.txt"

    def __str__(self):
        return f"robots.txt ({'on' if self.is_active else 'off'})"

# --- Баннеры: размещения по страницам ---

class BannerPlacement(models.Model):
    class PageKey(models.TextChoices):
        HOME        = "home",        "Главная"
        AUTOSERVICE = "autoservice", "Автосервис"
        ABOUT       = "about",       "О сервисе"
        DYNO        = "dyno",        "Диностенд"

    banner   = models.ForeignKey("Banner", on_delete=models.CASCADE, related_name="placements")
    page_key = models.CharField(max_length=32, choices=PageKey.choices)
    position = models.PositiveIntegerField(default=0)

    class Meta:
        unique_together = ("banner", "page_key")
        ordering = ["page_key", "position"]

    def __str__(self):
        return f"{self.get_page_key_display()} — #{self.position} — {self.banner_id}"

class HomeLatestItem(models.Model):
    product   = models.ForeignKey("Product", on_delete=models.CASCADE, related_name="home_latest_items")
    position  = models.PositiveIntegerField(default=0)
    is_active = models.BooleanField(default=True)
    created   = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["position"]
        unique_together = (("product",),)

    def __str__(self):
        return f"#{self.position} — {self.product_id}"
