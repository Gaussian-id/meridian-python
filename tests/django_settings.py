"""Settings for the Django adapter tests (`tests/test_django.py` configures Django with them), and the URLconf: each
test mounts the view it drives in `urlpatterns`. A module of its own, so Django and the tests share one list."""

from typing import Any

SECRET_KEY = 'test-only'
ROOT_URLCONF = 'tests.django_settings'
ALLOWED_HOSTS = ['testserver']
INSTALLED_APPS: list[str] = []
MIDDLEWARE = ['django.middleware.csrf.CsrfViewMiddleware']
MERIDIAN: dict[str, str] = {}

urlpatterns: list[Any] = []
