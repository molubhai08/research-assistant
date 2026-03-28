from django.db import models
from django.contrib.auth.models import User

class Projects(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    name = models.CharField(max_length=255)
    proj_notes = models.TextField(blank=True)
    papers = models.TextField(blank=True)

    def __str__(self):
        return self.name