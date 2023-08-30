import sqlite3

# 连接到 SQLite 数据库
# 如果数据库文件不存在，会自动创建一个
conn = sqlite3.connect('test.db')

# 创建一个游标对象
cursor = conn.cursor()

# 执行一条 SQL 语句，创建一张表
cursor.execute('CREATE TABLE user (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, age INTEGER)')

# 插入一条数据
cursor.execute("INSERT INTO user (name, age) VALUES ('Alice', 20)")

# 提交事务
conn.commit()

# 关闭游标和连接
cursor.close()
conn.close()
