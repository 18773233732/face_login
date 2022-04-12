import datetime
import pymysql
from dbutils.persistent_db import PersistentDB

POOL = PersistentDB(
    creator=pymysql,  # 使用链接数据库的模块
    maxusage=None,  # 一个链接最多被重复使用的次数，None表示无限制
    setsession=[],  # 开始会话前执行的命令列表。如：["set datestyle to ...", "set time zone ..."]
    ping=0,
    # ping MySQL服务端，检查是否服务可用。# 如：0 = None = never, 1 = default = whenever it is requested, 2 = when a cursor is created, 4 = when a query is executed, 7 = always
    closeable=False,
    # 如果为False时， conn.close() 实际上被忽略，供下次使用，再线程关闭时，才会自动关闭链接。如果为True时， conn.close()则关闭链接，那么再次调用pool.connection时就会报错，因为已经真的关闭了连接（pool.steady_connection()可以获取一个新的链接）
    threadlocal=None,  # 本线程独享值得对象，用于保存链接对象，如果链接对象被重置
    host='127.0.0.1',
    port=3306,
    user='root',
    password='root',
    database='face_sys',
    charset='utf8'
)


def insert_facejson(user, pwd, phone, vector, provinces, city, area):
    conn = POOL.connection()
    cursor = conn.cursor()
    dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sql = "insert into face_users(user_name, password, vector, phone, date, state,provinces, city, area) values('%s' ,'%s','%s','%s', '%s', '%d','%d','%d','%d') ;" % (
        user, pwd, vector, phone, dt, 0, provinces, city, area)
    # print(sql)
    try:
        # 执行sql语句
        cursor.execute(sql)
        # 提交到数据库执行
        conn.commit()
    except:
        # Rollback in case there is any error
        print('Error:insert into error')
        conn.rollback()
    finally:
        conn.close()


def findall_facejson():
    conn = POOL.connection()
    cursor = conn.cursor()
    sql = "select id, vector from face_users;"
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        return results
    except:
        print("Error:unable to fecth data")
    finally:
        conn.close()


def get_userID(uid, pwd):
    conn = POOL.connection()
    cursor = conn.cursor()
    # print(cursor, 111111111)
    if uid is None or pwd is None:
        return 0
    sql = "SELECT `id` FROM `face_users` WHERE `user_name`=%s AND `password`=%s;"
    try:
        cursor.execute(sql, (uid, pwd))
        result = cursor.fetchone()
        # print(result)
        if result is None:
            return 0
        return result[0]
    except:
        print("Error:unable to fetch data")
    finally:
        conn.close()


def get_users_length():
    conn = POOL.connection()
    cursor = conn.cursor()
    sql = "SELECT COUNT(*) FROM `face_users`"
    try:
        cursor.execute(sql)
        result = cursor.fetchone()
        if result is None:
            return 0
        return result[0]
    except:
        print('Error:unable to fetch data')
    finally:
        conn.close()

# def get_user_info(user_id):
#     conn = POOL.connection()
#     cursor = conn.cursor()
#     sql = "SELECT id, user_name, phone, date, state, provinces, city, area FROM `face_users` WHERE id=%d" % user_id
#     try:
#         cursor.execute(sql)
#         result = cursor.fetchone()
#         if result is None:
#             return {}
#         print(result)
#         print(result[0])
#         return result[0]
#     except:
#         print('Error:unable to fetch data')
#     finally:
#         conn.close()


def get_user_info(id):
    conn = POOL.connection()
    cursor = conn.cursor()
    # sql = "SELECT id, user_name, phone FROM face_users WHERE id=%d;" % id
    sql = "SELECT id, user_name, phone, date, state, provinces, city, area FROM `face_users` WHERE id=%d" % id
    # print(sql)
    try:
        cursor.execute(sql)
        result = cursor.fetchone()
        # print(result)
        if result is not None:
            return {
                'id': result[0],
                'name': result[1],
                'phone': result[2],
                'date': result[3].strftime("%Y-%m-%d %H:%M:%S") if result[3] else '',
                'geographic': {
                    'provinces': result[4],
                    'city': result[5],
                    'area': result[6]
                }
            }
        return {}
    except:
        print('Error:unable to fetch data')
    finally:
        conn.close()
