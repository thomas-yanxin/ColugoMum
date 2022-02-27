#-*-coding:GBK -*- 
import json
import os
import random
from turtle import pos

#-*-coding:GBK -*- 
import pymysql

###核心代码

db = pymysql.connect(host="localhost", user="SM_C", passwd="105316", db="container")
print('数据库连接成功！')
cur = db.cursor()
sql = "select * from t_container"
cur.execute(sql)
result = cur.fetchall()
result = list(result)
container_list = []
for row in result:
    container_list.append(list(row))
print(container_list)
db.commit()
cur.close()
db.close()


# 获得商品名称和销量的降序列表



import random

from pyecharts import options as opts
from pyecharts.charts import Bar, Funnel, Gauge, Geo, Page, Pie, Scatter3D
from pyecharts.globals import ThemeType


def stock_warining():
    ## 获得商品名称和库存的升序排列表

    name_stock = []

    for i in container_list:
        temp = []
        # print(i)
        name = i[2]
        # print(name)
        stock = i[5]
        # print(int(sale))
        temp.append(name)
        temp.append(int(stock))
        name_stock.append(temp)

    name_stock_sort = name_stock
    name_stock_sort.sort(key=lambda x :(x[1]))    # 依据库存降序排列
    # print(name_stock_sort)

    c = (
        Funnel()
        .add(
            "商品",
            name_stock_sort[:5],
            sort_="ascending",
            label_opts=opts.LabelOpts(position="center"),
            
        )
        .set_global_opts(title_opts=opts.TitleOpts(title="库存告急",
                                                       title_textstyle_opts=opts.TextStyleOpts(color="#F39C12",font_family="黑体",font_size=20),
                                                       pos_left="0%"),
                                                       legend_opts=opts.LegendOpts(pos_left='20%', type_='scroll',pos_top='5%',inactive_color='#D2E9FF'),)
        # .render("funnel_stock_ascending.html")
    )
    return c

# for m in container_list:

def sale_bar():
    name_sale = []

    for i in container_list:
        temp = []
        name = i[2]
        # print(name)
        sale = i[6]
        # print(int(sale))
        temp.append(name)
        temp.append(int(sale))
        name_sale.append(temp)

    name_sale_sort = name_sale
    name_sale_sort.sort(key=lambda x :(x[1]),reverse = True)    # 依据销量降序排列
    # print(name_sale_sort)


    name_sale_list = []
    sale_sale_list = []

    n = 0
    for i in name_sale_sort:
        n += 1
        name = i[0]
        sale = i[1]
        name_sale_list.append(name)
        sale_sale_list.append(sale)
        if n == 30:
            break
    c = (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))
        .add_xaxis(name_sale_list)
        .add_yaxis("销量", sale_sale_list)
        .set_series_opts(
                             label_opts=opts.LabelOpts(is_show=True,color="#ABEBC6",font_size=10))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="销量排行榜",
                                                       title_textstyle_opts=opts.TextStyleOpts(color="#F39C12",font_family="黑体",font_size=20),
                                                       pos_left="10%",pos_top='5%'),
                                                       legend_opts=opts.LegendOpts(textstyle_opts=opts.TextStyleOpts(color="#58D68D")),
                             xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(color="#58D68D")),
                             yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(color="#58D68D")),
            datazoom_opts=opts.DataZoomOpts(),
        )
        .set_colors(colors='#0080FF')
        # .render("bar_datazoom_slider.html")
    )
    return c


def stock_sale_bar(): #柱状图
    # 获得商品名称、销量和库存的降序列表

    name_sale_stock = []

    for i in container_list:
        temp = []
        name = i[2]
        sale = i[6]
        stock = i[5]
        # print(int(sale))
        temp.append(name)
        temp.append(int(stock))
        temp.append(int(sale))
        name_sale_stock.append(temp)

    name_sale_stock_sort = name_sale_stock
    name_sale_stock_sort.sort(key=lambda x :(x[1]),reverse = True)    # 依据库存降序排列

    name_sale_stock_name_list = []
    name_sale_stock_stock_list = []
    name_sale_stock_sale_list = []

    n = 0
    for i in name_sale_stock_sort:
        n += 1
        name = i[0]
        stock = i[1]
        sale = i[2]
        name_sale_stock_name_list.append(name)
        name_sale_stock_stock_list.append(stock)
        name_sale_stock_sale_list.append(sale)
        if n == 20:
            break
    cate = name_sale_stock_name_list
    c = (      
         Bar(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))
            .add_xaxis(cate)
            .add_yaxis("销售量", name_sale_stock_stock_list)
            .add_yaxis("库存数", name_sale_stock_sale_list)
            .set_series_opts(
                             label_opts=opts.LabelOpts(is_show=True,color="#ABEBC6",font_size=12)
            )
            .set_global_opts(title_opts=opts.TitleOpts(title="销售库存对比图",
                                                       title_textstyle_opts=opts.TextStyleOpts(color="#F39C12",font_family="黑体",font_size=20),
                                                       pos_left="5%"),
                             legend_opts=opts.LegendOpts(textstyle_opts=opts.TextStyleOpts(color="#ABEBC6")),
                             xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(color="#58D68D")),
                             yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(color="#58D68D")),
                             datazoom_opts=opts.DataZoomOpts(),
                                                     
            )
            .set_colors(["blue", "green"])
            #.render("bar_stack0.html")
    )
    return c


def tab0(name,color): #标题
    c = (Pie().
        set_global_opts(
        title_opts=opts.TitleOpts(title=name,pos_left='center',pos_top='center',
                                title_textstyle_opts=opts.TextStyleOpts(color=color,font_size=40,font_family="方正舒体"))))
    return c
 

 
def gau():#库存利用率
    c = (
        Gauge(init_opts=opts.InitOpts(width="400px", height="400px"))
            .add(series_name="库位利用率", data_pair=[["", 90]])
            .set_global_opts(title_opts=opts.TitleOpts(title="库存利用率",
                                                       title_textstyle_opts=opts.TextStyleOpts(color="#F39C12",font_family="黑体",font_size=20),
                                                       pos_left="20%"),
            legend_opts=opts.LegendOpts(is_show=False),
            tooltip_opts=opts.TooltipOpts(is_show=True, formatter="{a} <br/>{b} : {c}%"),
            
        )
            #.render("gauge.html")
    )
    return c
 
 
# def radius():
#     cate = ['客户A', '客户B', '客户C', '客户D', '客户E', '其他客户']
#     data = [153, 124, 107, 99, 89, 46]
#     c=Pie()
#     c.add('', [list(z) for z in zip(cate, data)],
#             radius=["30%", "75%"],
#             rosetype="radius")
#     c.set_global_opts(title_opts=opts.TitleOpts(title="客户销售额占比", padding=[1,250],title_textstyle_opts=opts.TextStyleOpts(color="#FFFFFF")),
#                       legend_opts=opts.LegendOpts(textstyle_opts=opts.TextStyleOpts(color="#FFFFFF"),type_="scroll",orient="vertical",pos_right="5%",pos_top="middle")
#                       )
#     c.set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {d}%"))
#     c.set_colors(['red',"orange", "yellow", "green", "Cyan", "purple"])
    
#     return c
 
 
# def funnel():
#     cate = ['访问', '注册', '加入购物车', '提交订单', '付款成功']
#     data = [30398, 15230, 10045, 8109, 5698]
#     c = Funnel()
#     c.add("用户数", [list(z) for z in zip(cate, data)], 
#                sort_='ascending',
#                label_opts=opts.LabelOpts(position="inside"))
#     c.set_global_opts(title_opts=opts.TitleOpts(title=""))
 
 
#     return c
 
 
def geo():
    city_num = [('武汉',105),('成都',70),('北京',99),
            ('西安',80),('杭州',60),('贵阳',34),
            ('上海',65),('深圳',54),('乌鲁木齐',76),
            ('哈尔滨',47),('兰州',56),('信阳',85)]
    start_end = [('宁波','成都'),('武汉','北京'),('武汉','西安'),
             ('长沙','杭州'),('武汉','贵阳'),('武汉','上海'),
             ('甘肃','深圳'),('北京','乌鲁木齐'),('上海','哈尔滨'),
             ('武汉','兰州'),('西藏','信阳')]
    c = Geo()
    c.add_schema(maptype='china', 
                itemstyle_opts=opts.ItemStyleOpts(color='#0080FF', border_color='white'))
    # 4.添加数据
    c.add('', data_pair=city_num, color='white')
    c.add('', data_pair=start_end, type_="lines",label_opts=opts.LabelOpts(is_show=False),
         effect_opts=opts.EffectOpts(symbol="arrow", 
                                     color='gold', 
                                     symbol_size=1))
    c.set_global_opts(
        title_opts = opts.TitleOpts(title="各门店销售情况",
                                                       title_textstyle_opts=opts.TextStyleOpts(color="#F39C12",font_family="黑体",font_size=20),
                                                       pos_left="20%"),)
    
    return c
 
 
def scatter3D():
    data = [(random.randint(0, 10), random.randint(0, 10), random.randint(0, 10)) for _ in range(80)]
    c = (Scatter3D()
            .add("", data)
            .set_global_opts(
              title_opts=opts.TitleOpts(title="购买力分析",
                                                       title_textstyle_opts=opts.TextStyleOpts(color="#F39C12",font_family="黑体",font_size=20),
                                                       pos_left="20%"),
            )
        )
    return c
        

from pyecharts.charts import Page

page = Page() 
page.add(
         tab0("袋鼯麻麻——智能零售数据可视化","#FDFEFE"),
        #  tab0("OFFICETOUCH","#2CB34A"), 
         geo(),
         sale_bar(),
         stock_warining(),
         stock_sale_bar(),
         gau(),
        #  radius(),
        #  funnel(),
         
         scatter3D(),
         )
page.render("../datacenter.html")

from bs4 import BeautifulSoup

with open("datacenter.html", "r+", encoding='utf-8') as html:
    html_bf = BeautifulSoup(html, 'lxml')
    divs = html_bf.select('.chart-container')
    divs[0]["style"] = "width:40%;height:10%;position:absolute;top:0%;left:30%;"   
    divs[1]["style"] = "width:30%;height:50%;position:absolute;top:10%;left:2%;"     
    divs[2]["style"] = "width:40%;height:45%;position:absolute;top:56%;left:2%;"   
    divs[3]["style"] = "width:32%;height:45%;position:absolute;top:12%;left:67%;"  
    divs[4]["style"] = "width:34%;height:40%;position:absolute;top:12%;left:34%;"   
    divs[5]["style"] = "width:45%;height:40%;position:absolute;top:58%;left:32%;"    
    divs[6]["style"] = "width:35%;height:40%;position:absolute;top:55%;left:62%;"   
    # divs[7]["style"] = "width:35%;height:40%;position:absolute;top:50%;left:60%;"
    body = html_bf.find("body")
    body["style"] = "background-image: url(https://img.zcool.cn/community/017d6a5c513b9ca801213f26c6f65d.png@1280w_1l_2o_100sh.png)"  # 背景颜色
    html_new = str(html_bf)
    html.seek(0, 0)
    html.truncate()
    html.write(html_new)
print('Start visualing!')
