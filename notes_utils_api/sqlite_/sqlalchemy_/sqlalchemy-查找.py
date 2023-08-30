from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# 创建对象的基类
Base = declarative_base()

# 定义 User 对象
class User(Base):
    # 表的名字
    __tablename__ = 'user'

    # 表的结构
    id = Column(Integer, primary_key=True)
    name = Column(String(20))
    age = Column(Integer)

# 初始化数据库连接
engine = create_engine('sqlite:///test.db')
# 创建 DBSession 类型
DBSession = sessionmaker(bind=engine)

# 创建 session 对象
session = DBSession()
# 查询 User 表中所有用户的姓名和年龄
results = session.query(User.name, User.age).all()
# 遍历查询结果并打印
for row in results:
    print(row)
# 关闭 session
session.close()
