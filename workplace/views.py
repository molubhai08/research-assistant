import sys
import os

# make the agent importable from the workspace root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from Home.models import Projects
from agent.app import ask, _store
from langchain_core.messages import HumanMessage, AIMessage


# per-project conversation history stored in memory
_histories: dict[str, list] = {}


from django.contrib.auth.decorators import login_required


@login_required(login_url='/sign_in/')
def WorkPlace(request, name):
    project = Projects.objects.get(name=name , user = request.user)
    return render(request, 'workplace.html', {'project': project})


@require_POST
def SaveNotes(request, name):
    try:
        project = Projects.objects.get(name=name , user = request.user)
        project.proj_notes = request.POST.get('notes', '')
        project.save()
        return JsonResponse({'status': 'ok'})
    except Projects.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Project not found'}, status=404)


@require_POST
def ChatView(request, name):
    try:
        project = Projects.objects.get(name=name , user = request.user)
    except Projects.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Project not found'}, status=404)

    message = request.POST.get('message', '').strip()
    if not message:
        return JsonResponse({'status': 'error', 'message': 'Empty message'}, status=400)

    history = _histories.get(name, [])
    try:
        result = ask(message, history)
    except Exception as exc:
        return JsonResponse({'status': 'error', 'message': str(exc)}, status=500)

    # update in-memory history
    history.append(HumanMessage(content=message))
    history.append(AIMessage(content=result['answer']))
    _histories[name] = history

    # persist any newly found papers to the project
    if result['papers']:
        existing_entries = [p.strip() for p in project.papers.split('\n') if p.strip()]
        existing_titles = set(e.split('||')[0].strip() for e in existing_entries)
        new_entries = [
            f"{p['title']}||{p['link']}"
            for p in result['papers']
            if p['title'] not in existing_titles
        ]
        if new_entries:
            existing_entries.extend(new_entries)
            project.papers = '\n'.join(existing_entries)
            project.save()

    return JsonResponse({
        'status': 'ok',
        'answer': result['answer'],
        'steps': result['steps'],
        'papers': result['papers'],
    })


@require_POST
def AddPaper(request, name):
    try:
        project = Projects.objects.get(name=name , user = request.user)
    except Projects.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Project not found'}, status=404)

    title = request.POST.get('title', '').strip()
    if not title:
        return JsonResponse({'status': 'error', 'message': 'No title'}, status=400)

    existing = [p.strip() for p in project.papers.split('\n') if p.strip()]
    if title not in existing:
        existing.append(title)
        project.papers = '\n'.join(existing)
        project.save()

    return JsonResponse({'status': 'ok', 'papers': existing})
