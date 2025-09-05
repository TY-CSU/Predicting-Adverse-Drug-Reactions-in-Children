import sqlite3

# 连接到SQLite数据库
conn = sqlite3.connect('faers-data_Child.sqlite')

# 使用 cursor() 方法创建 cursor 对象
cursor = conn.cursor()

# 检索数据库中所有表的列表
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

# 获取所有结果
tables = cursor.fetchall()

# 打印表列表
for table in tables:
    print(table[0])

# 关闭连接
conn.close()