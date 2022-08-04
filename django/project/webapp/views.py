import pandas as pd ### For reading the csv file
import numpy as np ##### For joins the different array
import os
from tensorflow.compat.v1 import Session
from tensorflow import Graph
from keras.preprocessing import image
#from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
model_graph = Graph()

from django.shortcuts import render,redirect
from django.shortcuts import get_object_or_404
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from django.http import HttpResponse
from django.forms import inlineformset_factory
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .models import *
from .forms import CreateUserForm

# Create your views here.
def registerPage(request):
    if request.user.is_authenticated:
        return redirect('home')
    else:
        form = CreateUserForm()
        if request.method == 'POST':
            form = CreateUserForm(request.POST)
            if form.is_valid():
                form.save()
                user = form.cleaned_data.get('username')
                messages.success(request, 'Account was created for ' + user)
                return redirect('login')

        context = {'form': form}
        return render(request, 'accounts/register.html', context)

def loginPage(request):
    if request.user.is_authenticated:
        return redirect('home')
    else:
        if request.method == 'POST':
            username = request.POST.get('username')
            password = request.POST.get('password')

            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('home')
            else:
                messages.info(request, 'Username OR password is incorrect')

        context = {}
        return render(request, 'accounts/login.html', context)

def logoutUser(request):
    logout(request)
    return redirect('login')

@login_required(login_url='login')
def home(request):
    return render(request, 'accounts/index.html')

with model_graph.as_default():
	tf_session = Session()
	with tf_session.as_default():
		pass
		model = load_model("weights.h5")

IMG_WIDTH = 224
IMG_HEIGHT = 224


@login_required(login_url='login')
def predict(request):
    print(request.POST.dict())
    fileObj = request.FILES["document"]
    fs=FileSystemStorage()
    filePathName = fs.save(fileObj.name,fileObj)
    filePathName = fs.url(filePathName)
    test_image = "."+filePathName
    img = image.load_img(test_image,target_size=(IMG_WIDTH,IMG_HEIGHT,3))
    img = img_to_array(img)
    img = img/255
    x = img.reshape(1,IMG_WIDTH,IMG_HEIGHT,3)

    with model_graph.as_default():
        with tf_session.as_default():
            result = np.argmax(model.predict(x))

            if result== 0:
                print("lung present")
                return render(request, "accounts/Stage_1.html")
            elif result== 1:
                print("normal lung")
                return render(request, "accounts/Normal lung.html")



