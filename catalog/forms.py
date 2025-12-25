# catalog/forms.py
from django import forms
from django.contrib.auth import authenticate, get_user_model
from django.contrib.auth.forms import PasswordChangeForm as DjangoPasswordChangeForm
from .models import UserProfile, Order, ServiceRequest, News, SEOEntry, RedirectRule, GoneURL, RobotsTxt
from django.db import transaction
from django.utils import timezone
# from captcha.fields import ReCaptchaField
# from captcha.widgets import ReCaptchaV2Checkbox, ReCaptchaV2Invisible, ReCaptchaV3


DELIVERY_CHOICES = [("courier", "Курьер"), ("pickup", "Самовывоз")]
PAYMENT_CHOICES  = [("card", "Карта"), ("cash", "Наличные")]

class QtyForm(forms.Form):
    qty = forms.IntegerField(
        min_value=1,
        initial=1,
        widget=forms.NumberInput(attrs={"class": "form-control", "style": "width:90px"})
    )

class OrderForm(forms.Form):
    name     = forms.CharField(label="Имя", max_length=100)
    phone    = forms.CharField(label="Телефон", max_length=50)
    email    = forms.EmailField(label="Email", required=False)
    city     = forms.CharField(label="Город", max_length=100, required=False)
    address  = forms.CharField(label="Адрес", max_length=200, required=False)
    delivery = forms.ChoiceField(label="Способ доставки", choices=DELIVERY_CHOICES)
    payment  = forms.ChoiceField(label="Способ оплаты", choices=PAYMENT_CHOICES)
    comment  = forms.CharField(label="Комментарий", widget=forms.Textarea(attrs={"rows": 3}), required=False)

User = get_user_model()

class LoginForm(forms.Form):
    username = forms.CharField(label="E-mail или логин")
    password = forms.CharField(label="Пароль", widget=forms.PasswordInput)
    remember = forms.BooleanField(label="Запомнить", required=False)

    def clean(self):
        cd = super().clean()
        u = cd.get("username")
        p = cd.get("password")
        user = authenticate(username=u, password=p)
        if not user:
            # если ввели e-mail вместо логина
            try:
                obj = User.objects.get(email=u)
                user = authenticate(username=obj.username, password=p)
            except User.DoesNotExist:
                user = None
        if not user:
            raise forms.ValidationError("Неверный логин или пароль")
        self.user = user
        return cd

class RegisterForm(forms.Form):
    username  = forms.CharField(label="Логин", max_length=150)
    full_name = forms.CharField(label="Фамилия Имя Отчество", max_length=200)
    email     = forms.EmailField(label="E-mail")
    phone     = forms.CharField(label="Телефон", max_length=50)
    password  = forms.CharField(label="Пароль", widget=forms.PasswordInput)

    def clean_username(self):
        u = self.cleaned_data["username"]
        if User.objects.filter(username=u).exists():
            raise forms.ValidationError("Логин уже занят")
        return u

    def clean_email(self):
        e = self.cleaned_data["email"].lower()
        if User.objects.filter(email=e).exists():
            raise forms.ValidationError("E-mail уже используется")
        return e

    @transaction.atomic
    def save(self):
        data = self.cleaned_data
        user = User.objects.create_user(
            username=data["username"],
            email=data["email"].lower(),
            password=data["password"],
        )
        # профиль мог быть создан сигналом post_save — берём/создаём безопасно
        profile, _ = UserProfile.objects.get_or_create(user=user)
        profile.full_name = data["full_name"]
        profile.phone = data["phone"]
        profile.save()
        return user

class PasswordResetEmailForm(forms.Form):
    email = forms.EmailField(label="E-mail")

    def clean_email(self):
        return (self.cleaned_data["email"] or "").strip().lower()

class ProfileForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ["full_name", "phone", "city", "address"]
        widgets = {f: forms.TextInput(attrs={"class":"form-control auth-input"}) for f in fields}

class PasswordChangeForm(DjangoPasswordChangeForm):
    # чтобы стили совпадали
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        for f in self.fields.values():
            f.widget.attrs.update({"class":"form-control auth-input"})

# --- кастомный виджет ДОЛЖЕН стоять выше формы ---
# catalog/forms.py
class MultipleFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True

class ProductImagesUploadForm(forms.Form):
    images = forms.FileField(
        label="Добавить фото",
        widget=MultipleFileInput(attrs={"multiple": True, "class": "form-control"}),
        required=False,   # ← ВАЖНО
    )



class OrderStatusForm(forms.ModelForm):
    class Meta:
        model = Order
        fields = ["status"]
        widgets = {
            "status": forms.Select(attrs={"class": "form-select form-select-sm"})
        }

class ServiceRequestForm(forms.ModelForm):
    class Meta:
        model = ServiceRequest
        fields = ["name", "phone", "comment"]

    def clean_phone(self):
        import re
        phone = (self.cleaned_data.get("phone") or "").strip()
        # очень мягкая валидация
        if len(re.sub(r"\D+", "", phone)) < 7:
            raise forms.ValidationError("Укажите корректный телефон.")
        return phone

class AdminNewsForm(forms.ModelForm):
    remove_image = forms.BooleanField(
        required=False, initial=False, label="Удалить текущую обложку"
    )

    class Meta:
        model = News
        fields = ["title", "author", "excerpt", "body", "image", "published_at", "is_published"]
        widgets = {
            # удобный ввод даты/времени
            "published_at": forms.DateTimeInput(
                attrs={"type": "datetime-local"}, format="%Y-%m-%dT%H:%M"
            ),
            "title":   forms.TextInput(attrs={"class": "form-control"}),
            "author": forms.TextInput(attrs={"class": "form-control"}),
            "excerpt": forms.Textarea(attrs={"class": "form-control", "rows": 3}),
            "body":    forms.Textarea(attrs={"class": "form-control", "rows": 10}),
            "image":   forms.ClearableFileInput(attrs={"class": "form-control"}),
            "is_published": forms.CheckboxInput(attrs={"class": "form-check-input"}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # корректно проставим initial для datetime-local
        if self.instance and self.instance.pk and self.instance.published_at:
            tz = timezone.get_current_timezone()
            self.initial["published_at"] = self.instance.published_at.astimezone(tz) \
                .strftime("%Y-%m-%dT%H:%M")
        elif "published_at" not in self.initial:
            self.initial["published_at"] = timezone.now().strftime("%Y-%m-%dT%H:%M")

class SEOEntryForm(forms.ModelForm):
    class Meta:
        model = SEOEntry
        fields = [
            "is_active",
            "content_type", "object_id", "path_regex",
            "title", "h1", "meta_description",
            "robots_index", "robots_follow",
            "canonical_url",
        ]
        widgets = {
            "meta_description": forms.Textarea(attrs={"rows": 3}),
            "path_regex": forms.TextInput(attrs={"placeholder": r"^/catalog/.*$ (опционально)"}),
            "canonical_url": forms.TextInput(attrs={"placeholder": "https://example.com/... (опционально)"}),
        }

    def clean(self):
        data = super().clean()
        ct = data.get("content_type")
        oid = data.get("object_id")
        rgx = data.get("path_regex", "").strip()
        if not ((ct and oid) or rgx):
            raise forms.ValidationError(
                "Нужно заполнить либо связку «Объект (content_type+object_id)», либо «path_regex»."
            )
        return data


class RedirectForm(forms.ModelForm):
    class Meta:
        model = RedirectRule
        fields = ["is_active", "from_path", "to_url", "code", "priority"]
        widgets = {
            "from_path": forms.TextInput(attrs={"placeholder": "/staryi-url/ или ^/blog/"}),
            "to_url": forms.TextInput(attrs={"placeholder": "/novyi-url/ или https://..."}),
        }


class GoneForm(forms.ModelForm):
    class Meta:
        model = GoneURL
        fields = ["path_or_regex", "is_regex", "is_active"]  # без 'comment'


class RobotsForm(forms.ModelForm):
    class Meta:
        model = RobotsTxt
        fields = ["is_active", "content"]
        widgets = {"content": forms.Textarea(attrs={"rows": 16, "spellcheck": "false", "class": "font-monospace"})}

class MailingPromoForm(forms.Form):
    subject      = forms.CharField(label="Тема письма", max_length=200)
    title        = forms.CharField(label="Заголовок акции", max_length=200)
    description  = forms.CharField(label="Описание", widget=forms.Textarea(attrs={"rows": 4}))
    button_url   = forms.URLField(label="Ссылка кнопки")
    button_text  = forms.CharField(label="Текст кнопки", max_length=60, initial="Открыть акцию", required=False)
    hero_path    = forms.CharField(
        label="Картинка hero (static path)", required=False, initial="img/mail_hero.png",
        help_text="Путь внутри static/, например img/mail_hero.png"
    )
    only_active  = forms.BooleanField(label="Только активные пользователи", required=False, initial=True)
    test_email   = forms.EmailField(label="Отправить тест на e-mail", required=False)