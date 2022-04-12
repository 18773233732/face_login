database_config = {
    'user': 'root',
    'password': 'root',
    'host': '127.0.0.1',
    'database': 'face',
    'auth_plugin': 'mysql_native_password',
}

app_config = {
    # 密钥，可随意修改
    'SECRET_KEY': 'hnusttsunh',
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,
    # 最大的相似距离，1.22是facenet基于lfw计算得到的
    'MAX_DISTINCT': 1.22,
    'UPLOAD_FOLDER': 'pic_tmp/',
    # 上传的图片路径和格式
    'ALLOWED_EXTENSIONS': set(['png', 'jpg', 'jpeg']),
    # 训练模型路径
    'MODEL_PATH': "models",
}
