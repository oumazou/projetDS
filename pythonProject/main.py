from fastapi import FastAPI,File,UploadFile
import shutil
import os
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, Lambda, Activation, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
import random
import numpy as np
import pandas as pd
import cv2
from fastapi.middleware.cors import CORSMiddleware
from pydantic.main import BaseModel


#*********************************** email

from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form
from starlette.responses import JSONResponse
from starlette.requests import Request
from fastapi_mail import FastMail, MessageSchema,ConnectionConfig
from pydantic import BaseModel, EmailStr
from typing import List

from dotenv import load_dotenv
from dotenv import dotenv_values


creddentials=dotenv_values(".editorconfig")



#*********************************** email





origins = [
    "*"
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

alphabets = u"!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "
max_str_len = 100 # max length of input labels
num_of_characters = len(alphabets) + 1 # +1 for ctc pseudo blank(epsilon)
num_of_timestamps = 64 # max length of predicted labels


def label_to_num(label):
    label_num = []
    for ch in label:
        label_num.append(alphabets.find(ch))
        # find() method returns the lowest index of the substring if it is found in given string otherwise -1

    return np.array(label_num)


def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret += alphabets[ch]
    return ret



def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)





from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights("model.h5")
print("Loaded model from disk")
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


def preprocess(img):
    (h, w) = img.shape

    final_img = np.ones([100, 800]) * 255  # black white image

    # crop
    if w > 800:
        img = img[:, 200:2280]

    if h > 100:
        img = img[300:400, 400:1200]

    final_img[:h, :w] = img
    thresh, image_black = cv2.threshold(final_img, 170, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.erode(image_black, kernel, iterations=1)
    resize_img = cv2.resize(img, (256, 64))

    return resize_img  # cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)



file_path_best = "model.h5"

loaded_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(lr = 0.0001))

checkpoint = ModelCheckpoint(filepath=file_path_best, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [checkpoint]

one = ["", "one ", "two ", "three ", "four ",
       "five ", "six ", "seven ", "eight ",
       "nine ", "ten ", "eleven ", "twelve ",
       "thirteen ", "fourteen ", "fifteen ",
       "sixteen ", "seventeen ", "eighteen ",
       "nineteen "];

# strings at index 0 and 1 are not used,
# they is to make array indexing simple
ten = ["", "", "twenty ", "thirty ", "forty ",
       "fifty ", "sixty ", "seventy ", "eighty ",
       "ninety "];


def numToWords(n, s):
    str = "";

    # if n is more than 19, divide it
    if (n > 19):
        str += ten[n // 10] + one[n % 10];
    else:
        str += one[n];
        # if n is non-zero
    if (n):
        str += s;

    return str;



def convertToWords(n):
    # stores word representation of given
    # number n
    out = "";

    # handles digits at ten millions and
    # hundred millions places (if any)
    out += numToWords((n // 10000000),
                      "crore ");

    # handles digits at hundred thousands
    # and one millions places (if any)
    out += numToWords(((n // 100000) % 100),
                      "lakh ");

    # handles digits at thousands and tens
    # thousands places (if any)
    out += numToWords(((n // 1000) % 100),
                      "thousand ");

    # handles digit at hundreds places (if any)
    out += numToWords(((n // 100) % 10),
                      "hundred ");

    if (n > 100 and n % 100):
        out += "and ";

    # handles digits at ones and tens
    # places (if any)
    out += numToWords((n % 100), "");

    return out;




@app.get("/")
async def root():
    return {"message": "Hello World"}


def test():
    ch = "aa"
    return(ch)


@app.post('/test')
async def testt(img: UploadFile=File(...)):
    wiss = test()
    # api_host = 'http://http://127.0.0.1:8000/upload/'
    # headers = {'Content-Type': 'image/jpeg'}
    # image_url = 'http://image.url.com/sample.jpeg'

    # img_file = urllib2.urlopen(image_url)

    # response = requests.post(api_host, data=img_file.read(),
    #   
    #                    headers=headers, verify=False)
    with open('imgs/'+img.filename,"wb") as buffer:
        shutil.copyfileobj(img.file,buffer)

    img_dir = "imgs/"+img.filename
    print("aaaaaaaaaaaaaaaaa")
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    thresh, image_black = cv2.threshold(image, 170, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 1), np.uint8)
    c2 = cv2.erode(image_black, kernel, iterations=1)

    image = preprocess(c2)
    image = image / 255
    pred = loaded_model.predict(image.reshape(1, 256, 64, 1))
    decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0]) * pred.shape[1], greedy=True)[0][0])

    output = num_to_label(decoded[0]) + ' '

    var1 = output.split(',')
    number = str(var1[1])
    number=int(number)

    letter = convertToWords(number)

    print(letter)
    return ({"letter":letter,"number":number})







#*************************************emails


class EmailSchema(BaseModel):
    email: List[EmailStr]

class EmailContent(BaseModel):
    message:str
    subject:str



conf = ConnectionConfig(
    MAIL_USERNAME = creddentials['EMAIL'],
    MAIL_PASSWORD = creddentials['PASS'],
    MAIL_FROM = creddentials['EMAIL'],
    MAIL_PORT = 587,
    MAIL_SERVER = "smtp.gmail.com",
    MAIL_TLS = True,
    MAIL_SSL = False,
    USE_CREDENTIALS = True,
    VALIDATE_CERTS = True
)







class emailsender(BaseModel):
    adress:str
@app.get('/email/{adress}/{subject}/{message}/{name}')
async def simple_send(adress:str,subject:str,message:str,name:str):
    html = "HELLO Sir\Mrs "+name+" \n"+message+"\nBest Regards"
    message = MessageSchema(
        subject=subject,
        recipients=[adress],  # List of recipients, as many as you can pass
        body=html,
        #subtype="html"
        )

    fm = FastMail(conf)
    await fm.send_message(message)
    return JSONResponse(status_code=200, content={"message": "email has been sent"})
