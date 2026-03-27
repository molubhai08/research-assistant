from django.shortcuts import render , redirect

from .models import Projects

from .forms import Project

# Create your views here.

def Home(request):
    return render(request , 'index.html')

def Proj(request):
    if request.method == 'POST':
        mode = request.POST.get('mode', 'new')

        if mode == 'existing':
            name = request.POST.get('existing_name', '').strip()
            try:
                obj = Projects.objects.get(name=name)
                return redirect('workplace', name=obj.name)
            except Projects.DoesNotExist:
                pass  # fall through to re-render with error

        else:
            form = Project(request.POST)
            if form.is_valid():
                name = form.cleaned_data['name']
                try:
                    obj = Projects.objects.get(name=name)
                except Projects.DoesNotExist:
                    obj = form.save()
                return redirect('workplace', name=obj.name)

    existing_projects = Projects.objects.all().values_list('name', flat=True)
    form = Project()
    return render(request, 'proj.html', {'form': form, 'existing_projects': existing_projects})

