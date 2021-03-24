from django.shortcuts import render,redirect
from django.contrib import messages
from django.http import HttpResponse,HttpResponseRedirect
from Bank.models import Customers,Accounts
from django.contrib.sessions.models import Session
# Create your views here.

def login(request):
    return render(request,'login.html')


def loginto(request):
    username = request.POST['username']
    password = request.POST['password']
    users = Customers.objects.filter(name=username)
    if users:
        user = users[0]
        pas = user.password
        if password ==pas:
            print("suceess")
            request.session['userid'] = user.user_id
            request.session['username'] =user.name
            request.session['is_logged'] = "Yes"
            print(request.session['userid'])
            return redirect('/')
        else:  
            print("wrong pass")
            messages.info(request,'Invalid password')
            post = request.POST.copy()
            post.update({'username':None,'password':None})
            request.POST = post
            return redirect('login')
    else:
        print("wrong user anme")
        messages.info(request,'Invalid User Name')
        post = request.POST.copy()
        post.update({'username':None,'password':None})
        request.POST = post
        return redirect('login')


         
