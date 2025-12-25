"""
URL configuration for config project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.urls import include, path, re_path
from django.conf import settings
from django.conf.urls.static import static
from catalog import views as cat_views
from django.views.generic import TemplateView


urlpatterns = [
    path("", cat_views.home, name="home"),
    path("catalog/", include("catalog.urls")),
    path('admin/', admin.site.urls),
    re_path(r"^robots\.txt$", cat_views.robots_txt, name="robots_txt"),
    
    path("avtoservis/", cat_views.autoservice_landing, name="page_avtoservis_short"),
    path("avtoservis/chip-tyuning/",       cat_views.chip_tuning_view,    name="page_chip_tuning_short"),
    path("avtoservis/dinostend/",          cat_views.page_dyno,           name="page_dynostand_short"),
    path("soglasie/", TemplateView.as_view(template_name="static_pages/soglasie.html"), name="page_agreement_short"),
    path("cookie/", TemplateView.as_view(template_name="static_pages/cookie.html"), name="page_cookie_short"),
    path("dostavka-i-oplata/", cat_views.page_help, name="page_help_short"),
    path("kontakty/", cat_views.page_contacts, name="page_contacts_short"),
    path("contact-us/", cat_views.contact_request_view, name="page_contact_request_short"),
    path("company-info/", cat_views.page_about, name="page_avtoservis_info_short"),
    path("blog/", cat_views.news_list_public, name="page_news_short"),
    path("blog/article<slug:slug>/", cat_views.news_detail, name="news_detail_short"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
handler404 = "catalog.views.page_not_found"


