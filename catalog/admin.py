# catalog/admin.py
from django.contrib import admin
from .models import (
    Category, Product, ProductImage,
    Attribute, ProductAttributeValue,   # ← именно Attribute
    Order, OrderItem, Favorite, ChipCar, ChipStage, ChipMetric, News
)
from .models import SEOObject, SEOPath

class ProductImageInline(admin.TabularInline):
    model = ProductImage
    extra = 0

class ProductAttrValueInline(admin.TabularInline):
    model = ProductAttributeValue
    extra = 0

@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = ("title", "sku", "price", "category", "is_active", "created_at")
    list_filter  = ("is_active", "category")
    search_fields = ("title", "sku")
    prepopulated_fields = {"slug": ("title",)}
    inlines = [ProductImageInline, ProductAttrValueInline]

@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ("name", "slug", "parent")
    search_fields = ("name", "slug")
    prepopulated_fields = {"slug": ("name",)}

@admin.register(Attribute)  # ← было ProductAttribute
class AttributeAdmin(admin.ModelAdmin):
    list_display = ("name",)
    search_fields = ("name",)

@admin.register(Order)
class OrderAdmin(admin.ModelAdmin):
    list_display = ("id", "created_at", "status", "name", "phone", "total")
    list_filter  = ("status", "created_at")
    search_fields = ("id", "name", "phone", "email")

@admin.register(OrderItem)
class OrderItemAdmin(admin.ModelAdmin):
    list_display = ("order", "title", "sku", "price", "qty", "subtotal")

# опционально
admin.site.register(ProductImage)
admin.site.register(ProductAttributeValue)
admin.site.register(Favorite)

@admin.register(ChipCar)
class ChipCarAdmin(admin.ModelAdmin):
    list_display = ("make", "model", "year", "engine")
    search_fields = ("make", "model", "year", "engine")
    list_filter = ("make", "year", "engine")

@admin.register(ChipStage)
class ChipStageAdmin(admin.ModelAdmin):
    list_display = ("car", "name")
    list_filter = ("name", "car__make")

@admin.register(ChipMetric)
class ChipMetricAdmin(admin.ModelAdmin):
    list_display = ("stage", "order", "label", "stock", "tuned", "delta")
    list_filter = ("stage__name", "stage__car__make")

@admin.register(News)
class NewsAdmin(admin.ModelAdmin):
    list_display = ("title", "published_at", "is_published")
    list_filter  = ("is_published", "published_at")
    search_fields = ("title", "excerpt", "body")
    prepopulated_fields = {"slug": ("title",)}
    date_hierarchy = "published_at"