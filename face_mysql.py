import mysql.connector
import datetime


class face_mysql:
    def __init__(self):
        pass
    #设置数据库和密码
    def conn_mysql(self):
        db = mysql.connector.connect(user='root', password='root', host='127.0.0.1', database='face', auth_plugin='mysql_native_password')
        return db

    def insert_facejson(self, pic_name, pic_json, uid):
        db = self.conn_mysql()
        cursor = db.cursor()
        dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sql = "insert into face_json(json,pic_name,date,state,uid) values('%s' ,'%s','%s','%d','%s') ;" % (
            pic_json, pic_name, dt, 1, uid)
        #print("sql=",sql)
        try:
            # 执行sql语句
            cursor.execute(sql)
            # 提交到数据库执行
            lastid = int(cursor.lastrowid)
            db.commit()
        except:
            # Rollback in case there is any error
            db.rollback()
        db.close()
        return lastid

    def findall_facejson(self):
        db = self.conn_mysql()
        cursor = db.cursor()

        sql = "select * from face_json where state=1;"
        try:
            cursor.execute(sql)
            results = cursor.fetchall()
            return results
        except:
            print("Error:unable to fecth data")
        db.close()

if __name__ == "__main__":
    data = face_mysql()
    db = data.conn_mysql()
    print(db)
    db.close()
    