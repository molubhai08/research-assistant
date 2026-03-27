from django import forms

from .models import Projects


class Project(forms.ModelForm):
    class Meta:
        model = Projects
        fields = ["name"]

