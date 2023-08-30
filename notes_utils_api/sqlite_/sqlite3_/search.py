import sqlite3

# 如果数据库文件不存在，会自动创建一个
conn = sqlite3.connect('test.db')

# 创建一个游标对象
cursor = conn.cursor()


# 执行查询语句
cursor.execute('SELECT * FROM user')

# 获取查询结果
results = cursor.fetchall()

# 遍历查询结果并打印
for row in results:
    print(row)


# 关闭游标和连接
cursor.close()
conn.close()
