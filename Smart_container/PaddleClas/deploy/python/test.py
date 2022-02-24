#图片处理
import base64
import binascii
import hashlib
import json
import os
from typing import Container

import memcache
import pymysql
import requests
from django.http import JsonResponse
from django.shortcuts import HttpResponse, render
#检索
from fuzzywuzzy import fuzz, process
#登陆用
from pyDes import CBC, PAD_PKCS5, des
from xpinyin import Pinyin

# 数据库相关操作
from app01 import models

# Create your views here.

KEY='mHAxsLYz'      #秘钥
PICTURE_ROOT = '/root/Smart_container/PaddleClas/dataset/retail'

def des_encrypt(s):
    """
    DES 加密
    :param s: 原始字符串
    :return: 加密后字符串，16进制
    """
    secret_key = KEY
    iv = secret_key
    k = des(secret_key, CBC, iv, pad=None, padmode=PAD_PKCS5)
    en = k.encrypt(s, padmode=PAD_PKCS5)
    return binascii.b2a_hex(en)


def des_descrypt(s):
    """
    DES 解密
    :param s: 加密后的字符串，16进制
    :return:  解密后的字符串
    """
    secret_key = KEY
    iv = secret_key
    k = des(secret_key, CBC, iv, pad=None, padmode=PAD_PKCS5)
    de = k.decrypt(binascii.a2b_hex(s), padmode=PAD_PKCS5)
    sessionID = de.split('_')
    openid = sessionID[0]
    return openid


def SKexpired(old_sessionID, code):
    
    s_openid = des_descrypt(old_sessionID)

    appid = "wx433732b2940b7d4c"
    secret = "b4e95c5b998cd13ba9d09e077343f2e7"
    code2SessionUrl = "https://api.weixin.qq.com/sns/jscode2session?appid={appid}&secret={secret}&js_code={code}&grant_type=authorization_code".format(
        appid=appid, secret=secret, code=code)
    resp = requests.get(code2SessionUrl)
    respDict = resp.json()
    s_session_key = respDict.get("session_key")

    s = str(s_openid) + '_' +str(s_session_key)
    sessionID = des_encrypt(s)

    models.TUser.objects.filter(openid=s_openid).update(session_key=s_session_key) 

    return sessionID



def information():
    container = models.TContainer.objects.all()

    container_all = []
    for i in container:
        temp = []
        temp.append(i.number)  
        temp.append(i.container_name)
        temp.append(i.container_price)
        temp.append(i.picture_address)
        container_all.append(temp)
    
    return container_all


def update():
    container_all = information()

    TXT_PATH='/root/Smart_container/PaddleClas/dataset/retail/data_update.txt'
             
    with open(os.path.abspath(TXT_PATH),'w+',encoding='utf-8') as fh:

        for container_single in container_all:
            container_name = container_single[1]
            container_address = container_single[3]
        
            fh.write(container_address + '\t' + container_name + '\n')
        fh.close()
   #有问题要修改
    os.system('python3 python/build_gallery.py -c configs/build_product.yaml -o IndexProcess.data_file="/root/Smart_container/PaddleClas/dataset/retail/data_update.txt" -o IndexProcess.index_dir="/root/Smart_container/PaddleClas/dataset/retail/index_update"')


# 识别模块
def reference(request):
    if request.method == "POST":
        sessionID = request.POST.get('sessionID')
        isSKexpried = request.POST.get('isSKexpried')
        code = request.POST.get('code')
        value = request.POST.get('picture')

        res_all = models.TContainer.objects.all()

        if isSKexpried:
            sessionID = SKexpired(sessionID, code)

        image_name = base64.b64decode(value)
        
        print(image_name)

        image_file = '/root/Smart_container/PaddleClas/dataset/retail/test1.jpg'
        with open(image_file, "wb") as fh:
             fh.write(image_name)
             fh.close()

###      商品识别

        rec_docs_list = []

        rec_docs_price_all = []

        price_all = 0.0

        # self.picture_file = '/home/thomas/Smart_container/PaddleClas/dataset/retail/test.jpg'
        #
        # cv2.imwrite(self.picture_file, self.image)

        os.system(
            'python /root/Smart_container/PaddleClas/deploy/python/predict_system.py -c /root/Smart_container/PaddleClas/deploy/configs/inference_product.yaml -o Global.use_gpu=False')
        print('3')
        log_path = '/root/Smart_container/PaddleClas/dataset/log.txt'
        

        rec_docs_str = ''
        rec_deplay_str = ''

        with open(log_path, 'r', encoding='utf8') as F:

            str_result_list = F.readlines()
            print(str_result_list)

            if str_result_list[0] == "Please connect root to upload container's name and it's price!":

                rec_deplay_str_all = str_result_list[0]

            else:

                for str_result in str_result_list:

                    price_all = 0

                    rec_docs_price = []

                    dict_result = eval(str_result)

                    rec_docs = dict_result['rec_docs']  # 结果
                    rec_docs_list.append(rec_docs)
                    print('2')
                    print(rec_docs_list)
                    for res in res_all:
                        for rec_docs_sig in rec_docs_list:
                            if rec_docs_sig == res.container_name:
                                rec_price = res.container_price
                                price_all += float(rec_price)
                                rec_docs_price.append(rec_docs)
                                rec_docs_price.append(rec_price)
                                rec_docs_price_all.append(rec_docs_price)

            
            # print("1")
            # print(rec_docs_price_all)
            os.remove(log_path)
            return JsonResponse({"state": 'true',"container": rec_docs_price_all,"price_all": price_all})
    else:
        return JsonResponse({"state": 'false'})



#登录

def login_in(request):
    if request.method == "POST":
        code = request.POST.get('code')
        userinfo = request.POST.get('userinfo')
    userinfo = json.loads(userinfo)
    s_nickname = userinfo['nickName']

    appid = "wx433732b2940b7d4c"
    secret = "b4e95c5b998cd13ba9d09e077343f2e7"
    code2SessionUrl = "https://api.weixin.qq.com/sns/jscode2session?appid={appid}&secret={secret}&js_code={code}&grant_type=authorization_code".format(
        appid=appid, secret=secret, code=code)
    resp = requests.get(code2SessionUrl)
    respDict = resp.json()
    s_openid = respDict.get("openid")    #需要存入的openid
    s_session_key = respDict.get("session_key")    #需要存入的session_key

    s = str(s_openid) + '_' +str(s_session_key)
    sessionID = des_encrypt(s)
    sessionID = str(sessionID)

    old_openid = models.TUser.objects.filter(openid=s_openid)   #old_openid是查询数据库中是否有s_openid，无为空
    old_openid = old_openid.values()
    if not bool(old_openid):                                           #判断表中是否还有对应openid
        s_user = models.TUser(openid = s_openid, nickname = s_nickname, session_key = s_session_key)  
        s_user.save()
        update()
    else:
        models.TUser.objects.filter(openid=s_openid).update(session_key=s_session_key)  #替换session_key


    return JsonResponse({"sessionID": sessionID})



def record(request):             #增加模块
    if request.method == "POST":
        sessionID = request.POST.get('sessionID')
        isSKexpried = request.POST.get('isSKexpried')
        code = request.POST.get('code')
        s_container_name = request.POST.get('container_name')         #商品名称 str
        s_container_price = request.POST.get('container_price')       #商品单价 float

        picture = request.FILES['productimage']   #照片

        if isSKexpried:
            sessionID = SKexpired(sessionID, code)

        value_name = s_container_name


        p = Pinyin()                 
        name = p.get_pinyin(value_name).replace('-','')
        
        s_picture_address = 'gallery/'+ name + '.jpg'

        with open(os.path.join(PICTURE_ROOT,s_picture_address), 'wb') as fh:
            for chunk in picture.chunks():
                fh.write(chunk)
            fh.close()

        last_data = models.TContainer.objects.last()           #查询t_container表中最后一条数据，以便于商品录入排序
        if not bool(last_data.number):
            s_number = 1                                         #序号
        else:
            s_number = last_data.number + 1
        
        old_container = models.TContainer.objects.filter(container_name=s_container_name)     
        old_container = old_container.values() 

        if not bool(old_container): 

            s_container = models.TContainer(number = s_number, container_name = s_container_name, container_price = s_container_price,picture_address = s_picture_address)
            s_container.save()

            update()
            
            return JsonResponse({"state": 'true', "sessionID": sessionID})
        else:
            return JsonResponse({"state": 'true', "sessionID": sessionID})
    else:
        return JsonResponse({"state": 'false'})



def delete(request):                #删除模块
    if request.method == "POST":
        sessionID = request.POST.get('sessionID')
        isSKexpried = request.POST.get('isSKexpried')
        code = request.POST.get('code')
        d_number = request.POST.get('number')
        d_container_name = request.POST.get('container_name')

        if isSKexpried:
             sessionID = SKexpired(sessionID, code)

        d_number = int(d_number)
        old_container = models.TContainer.objects.filter(number = d_number)     #查询t_container表中所有数据，判断表中是否已经包含目标商品
        old_container = old_container.values()

        if not bool(old_container):                                         #表内不含待删除商品
            return JsonResponse({"state": 'false', "sessionID": sessionID})
        else:
            models.TContainer.objects.filter(number = d_number).delete()
            
            update()

            return JsonResponse({"state": 'true', "sessionID": sessionID})
    else:
        return JsonResponse({"state": 'false'})


def replace(request):               #修改模块
    if request.method == "POST":
        sessionID = request.POST.get('sessionID')
        isSKexpried = request.POST.get('isSKexpried')
        code = request.POST.get('code')
        number = request.POST.get('number')
        r_container_name = request.POST.get('container_name')
        r_container_price = request.POST.get('container_price')
        r_picture = request.FILES['productimage']
        # print(r_container_name)


        if isSKexpried:
            sessionID = SKexpired(sessionID, code)

        models.TContainer.objects.filter(number = number).update(container_name = r_container_name)
        models.TContainer.objects.filter(number = number).update(container_price = r_container_price)
        
        g = models.TContainer.objects.filter(number = number)

        result = models.TContainer.objects.filter(number = number)

        with open(os.path.join(PICTURE_ROOT,result[0].picture_address), 'wb') as fh:
            for chunk in r_picture.chunks():
                fh.write(chunk)
            fh.close()
        
        update()

        return JsonResponse({"state": 'true', "sessionID": sessionID})
    else:
        return JsonResponse({"state": 'false'})



def search(request):             #查询模块
    if request.method == "POST":
        sessionID = request.POST.get('sessionID')
        isSKexpried = request.POST.get('isSKexpried')
        code = request.POST.get('code')

        if isSKexpried:
            sessionID = SKexpired(sessionID, code)

        container_all = information()
  
        return JsonResponse({"state": 'true', "sessionID": sessionID, 'container_all': container_all})
    else:
        return JsonResponse({"state": 'false'})


def find(request):    #检索模块
    if request.method== "POST":
        sessionID = request.POST.get('sessionID')
        isSKexpried = request.POST.get('isSKexpried')
        code = request.POST.get('code')
        searchtarget = request.POST.get('searchtarget')

        container = models.TContainer.objects.all()

    
        find_result = []
        for i in container:
            
            value = fuzz.partial_ratio("%s"%searchtarget,i.container_name)
            
            if value>=80:
                temp = []
                temp.append(i.number)  
                temp.append(i.container_name)
                temp.append(i.container_price)
                temp.append(i.picture_address)
                find_result.append(temp)

        return JsonResponse({"state": 'true', "sessionID": sessionID,"container_all":find_result})
    else:
        return JsonResponse({"state": 'false'})
