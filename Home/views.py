from django.shortcuts import render , redirect

from .models import Projects

from .forms import Project

# Create your views here.

def Home(request):
    return render(request , 'index.html')

def Proj(request):
    if request.method == 'POST':
        form = Project(request.POST)

        if form.is_valid():
            name = form.cleaned_data['name']

            try:
                obj = Projects.objects.get(name=name)
            except Projects.DoesNotExist:
                obj = form.save()

            return redirect('workplace', name=obj.name)
    else:
        form = Project()
    return render(request , "proj.html" , {'form' : form})

