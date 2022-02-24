import pymysql
import random

db = pymysql.connect(host="localhost", user="root", passwd="105316", db="container")
print('数据库连接成功！')
cur = db.cursor()
sql = "select * from t_container"
cur.execute(sql)
result=cur.fetchall()
for row in result:
    sqlQuery = "UPDATE t_container SET stock = %s WHERE number = %s"
    value = (random.randint(1000, 5000), row[0])
    cur.execute(sqlQuery, value)


db.commit()
cur.close()
db.close()

print("============")
print("Done! ")