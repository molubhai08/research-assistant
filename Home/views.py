from django.shortcuts import render, redirect
from .models import Projects
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, authenticate, logout


def Home(request):
    return render(request, 'index.html')


@login_required(login_url='/sign_in/')
def Proj(request):
    error = None

    if request.method == 'POST':
        mode = request.POST.get('mode', 'new').strip()

        if mode == 'existing':
            name = request.POST.get('existing_name', '').strip()

            if not name:
                error = 'Please select a project from the list.'
            else:
                try:
                    obj = Projects.objects.get(name=name, user=request.user)
                    return redirect('workplace', name=obj.name)
                except Projects.DoesNotExist:
                    error = f'Project "{name}" not found. It may have been deleted.'
                except Exception as e:
                    error = f'Something went wrong: {e}'

        else:  # mode == 'new'
            name = request.POST.get('name', '').strip()
            if not name:
                error = 'Please enter a project name.'
            else:
                obj, created = Projects.objects.get_or_create(
                    name=name,
                    user=request.user
                )
                return redirect('workplace', name=obj.name)

    # GET request or POST with errors — render the form fresh (or with errors)
    existing_projects = Projects.objects.filter(user=request.user).values_list('name', flat=True)

    return render(request, 'proj.html', {
        'existing_projects': existing_projects,
        'error': error,
    })


def signout(request):
    logout(request)
    return redirect('signin')


def signin(request):
    error = None

    if request.method == 'POST':
        if 'signup' in request.POST:
            username = request.POST.get('username', '').strip()
            email    = request.POST.get('email', '').strip()
            password = request.POST.get('password', '')

            if not username or not password:
                error = 'Username and password are required.'
            elif User.objects.filter(username=username).exists():
                error = 'Username already taken.'
            else:
                user = User.objects.create(username=username, email=email)
                user.set_password(password)
                user.save()
                login(request, user)
                return redirect('proj')

        elif 'login' in request.POST:
            username = request.POST.get('username', '').strip()
            password = request.POST.get('password', '')
            user = authenticate(request, username=username, password=password)

            if user is not None:
                login(request, user)
                return redirect('proj')
            else:
                error = 'Invalid username or password.'

    return render(request, 'sign_in.html', {'error': error})