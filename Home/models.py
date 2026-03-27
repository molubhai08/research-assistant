from django.db import models

# Create your models here.
class Projects(models.Model):
    name = models.CharField(max_length=1000)
    proj_notes = models.TextField()
    papers = models.TextField()

    def __str__(self):
        return f"{self.name}"

