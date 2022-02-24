import pymysql

db = pymysql.connect(host="localhost", user="root", passwd="105316", db="goods_identification")
cur = db.cursor()
rec_docs_list='达利园'
containers = []
price = []
sqlQuery = " select * from t_goods"
cur.execute(sqlQuery)
result = cur.fetchall()
print(result)
print(len(result))
print(result[257])
for s in result:

    if  rec_docs_list == s[1]:
        print(s[1])
        print(s[2])
        containers.append(s[1])
        price.append(s[2])
print(price)                  
db.commit()
cur.close()
db.close()