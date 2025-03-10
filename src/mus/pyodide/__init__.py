from .anthropic import patch_anthropic

def patch_all():
    patch_anthropic()