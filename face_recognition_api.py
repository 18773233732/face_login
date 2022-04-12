# -*- coding:utf-8 -*-
import cv2
from flask_cors import CORS
from flask_httpauth import HTTPTokenAuth
from flask import Flask, request
import os
import base64
import face_mysql
import matrix_fun
import numpy as np
from config.config import app_config
import src.facenet
import tensorflow as tf
import src.align.detect_face
import utils.utils

app = Flask(__name__)
auth = HTTPTokenAuth(scheme='Bearer')


# 配置跨域
CORS(app, supports_credentials=True)

# 图片最大为16M
app.config['MAX_CONTENT_LENGTH'] = app_config['MAX_CONTENT_LENGTH']
app.config['UPLOAD_FOLDER'] = app_config['UPLOAD_FOLDER']

# 设置post请求中获取的图片保存的路径
if not os.path.exists(app_config['UPLOAD_FOLDER']):
    os.makedirs(app_config['UPLOAD_FOLDER'])
else:
    # print('文件上传目录已存在')
    pass


with tf.Graph().as_default():
    gpu_memory_fraction = 1.0
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(
        gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = src.align.detect_face.create_mtcnn(sess, None)


with tf.Graph().as_default():
    sess = tf.Session()
    # src.facenet.load_model(modelpath)
    # 加载模型
    meta_file, ckpt_file = src.facenet.get_model_filenames(
        app_config['MODEL_PATH'])
    saver = tf.train.import_meta_graph(
        os.path.join(app_config['MODEL_PATH'], meta_file))
    saver.restore(sess, os.path.join(app_config['MODEL_PATH'], ckpt_file))
    # Get input and output tensors
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    # 进行人脸识别，加载
    # print('Creating networks and loading parameters')

    # 获取post中的图片并执行插入到库 返回数据库中保存的id
    @app.route('/api/register', methods=['POST'])
    @auth.login_required
    def face_insert():
        # 获取post请求中的username作为图片信息
        uid = face_mysql.get_users_length() + 1
        user = request.json.get('username')
        password = request.json.get('password')
        phone = request.json.get('phone')
        provinces = request.json.get('provinces')
        city = request.json.get('city')
        area = request.json.get('area')
        pic_base64 = request.json.get('picBase64')
        # print(pic_base64)
        if not utils.utils.params_check(user, password, phone, provinces, city, area):
            return utils.utils.response_json({'success': True, 'data': {}, 'errorMessage': 'register_failed'}, 200)
        imgData = base64.b64decode(pic_base64)
        nparr = np.fromstring(imgData, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # cv2.imshow("test", img_np)

        # 从post请求图片保存到本地路径中
        # file = upload_files
        # if file and utils.utils.allowed_file(file.filename):
        # filename = secure_filename(file.filename)
        # cv2.imwrite(os.path.join(
        #     app.config['UPLOAD_FOLDER'], '%d.jpg' % uid), img_np)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], '%d.jpg' % uid)
        # print(image_path)

        # opencv读取图片，开始进行人脸识别
        # img = cv2.imread(image_path)
        # img = misc.imread(os.path.expanduser(image_path), mode='RGB')
        # img = cv2.imread(image_path)
        # print(img, 2222222222)
        # 设置默认插入时 detect_multiple_faces =Flase只检测图中的一张人脸，True则检测人脸中的多张
        # 一般入库时只检测一张人脸，查询时检测多张人脸
        images = utils.utils.image_array_align_data(
            img_np, image_path, pnet, rnet, onet, detect_multiple_faces=False)

        feed_dict = {images_placeholder: images,
                     phase_train_placeholder: False}
        # emb_array保存的是经过facenet转换的128维的向量
        emb_array = sess.run(embeddings, feed_dict=feed_dict)
        filename_base = os.path.splitext(image_path)[0]
        id_list = []
        # 存入数据库
        # print(len(emb_array), '人脸数量')
        for j in range(0, len(emb_array)):
            face_mysql.insert_facejson(user, password, phone, ",".join(
                str(li) for li in emb_array[j].tolist()), provinces, city, area)

        # 设置返回类型
        request_result = {'success': True, 'data': {}, 'errorMessage': ''}
        request_result['data']['id'] = uid
        # print(request_result)
        return utils.utils.response_json(request_result, 200)

    @app.route('/api/queryface', methods=['POST'])
    @auth.login_required
    def face_query():
        # 获取查询条件  在ugroup中查找相似的人脸
        # ugroup = request.form['ugroup']
        # upload_files = request.files['imagefile']
        pic_base64 = request.json.get('picBase64')
        # print(pic_base64)
        if not utils.utils.params_check(pic_base64):
            return utils.utils.response_json({'success': True, 'data': {}, 'errorMessage': 'register_failed'}, 200)
        imgData = base64.b64decode(pic_base64)
        nparr = np.fromstring(imgData, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # print(image_path)

        # 读取本地的图片
        # img = misc.imread(os.path.expanduser(image_path), mode='RGB')
        images = utils.utils.image_array_align_data(
            img_np, None, pnet, rnet, onet)

        # 判断如果如图没有检测到人脸则直接返回
        if len(images.shape) < 4:
            # return json.dumps({'error': "not found face"})
            return utils.utils.response_json({'success': True, 'data': {}, 'errorMessage': "not_found_face"}, 200)

        feed_dict = {images_placeholder: images,
                     phase_train_placeholder: False}
        emb_array = sess.run(embeddings, feed_dict=feed_dict)
        face_query = matrix_fun.matrix()
        # 分别获取距离该图片中人脸最相近的人脸信息
        # pic_min_scores 是数据库中人脸距离（facenet计算人脸相似度根据人脸距离进行的）
        # pic_min_names 是当时入库时保存的文件名
        # pic_min_uid  是对应的用户id
        pic_min_scores, pic_min_uid = face_query.get_socres(
            emb_array)

        # 如果提交的query没有group 则返回
        if len(pic_min_scores) == 0:
            # return json.dumps({'error': "not found user group"})
            return utils.utils.response_json({'success': True, 'data': {}, 'errorMessage': "not_found_face"}, 200)

        # 设置返回结果
        result = []
        for i in range(0, len(pic_min_scores)):
            if pic_min_scores[i] < app_config['MAX_DISTINCT']:
                rdict = {'id': pic_min_uid[i], 'distance': pic_min_scores[i]}
                result.append(rdict)
        # print(result)
        if len(result) == 0:
            # return json.dumps({"state": "success, but not match face"})
            return utils.utils.response_json({'success': True, 'data': {}, 'errorMessage': "not_match_face"}, 200)
        else:
            return utils.utils.response_json({"success": True, "data": result, "errorMessage": ""}, 200)


@app.route('/api/uploadpic', methods=['POST'])
@auth.login_required
def get_pic():
    file = request.files['image']
    # print(file.filename)
    return utils.utils.response_json({'data': {'filename': file.filename}, 'errorMessage': '', 'success': True}, 200)


@app.route('/api/login', methods=['POST'])
# @auth.login_required
def login():
    user_name = request.json.get('username')
    password = request.json.get('password')
    user_id = face_mysql.get_userID(user_name, password)
    if user_id:
        token = utils.utils.generate_auth_token(user_id)
        user_data = face_mysql.get_user_info(user_id)
        return utils.utils.response_json({'success': True, 'data': {'userData': user_data, 'token': token}}, 200)
    return utils.utils.response_json({'success': True, 'data': {}, 'errorMessage': 'login_failed'}, 200)


@app.route('/api/userslength', methods=['GET'])
@auth.login_required
def get_users_len():
    user_length = face_mysql.get_users_length()
    return utils.utils.response_json({'success': True, 'data': {'length': user_length}, 'errorMessage': ''}, 200)


@app.route('/api/getUserInfo', methods=['POST'])
@auth.login_required
def get_user_info():
    token = request.headers.get('Authorization').split(' ')[1]
    user_id = utils.utils.get_auth_token_to_id(token)
    if not user_id:
        return utils.utils.response_json({'success': True, 'data': {}, 'errorMessage': 'params_error'}, 200)
    user_info = face_mysql.get_user_info(user_id)
    # print(user_info)
    return utils.utils.response_json({"success": True, "data": {"userInfo": user_info}, 'errorMessage': ''}, 200)


@ auth.verify_token
def verify_token(token):
    # 先验证token
    if token:
        user_id = utils.utils.verify_auth_token(token)
        # print(user_id)
        if user_id:
            return True
        else:
            return False
    return False


@ app.errorhandler(utils.utils.TokenIncorrectError)
def token_incorrect(error):
    return utils.utils.response_json({'errorMessage': error.msg, 'data': {}, 'success': True}, 200)


@ app.errorhandler(utils.utils.TokenExpiredError)
def token_expired(error):
    return utils.utils.response_json({'errorMessage': error.msg, 'data': {}, 'success': True}, 200)


@ auth.error_handler
def unauthorized():
    return utils.utils.response_json({'errorMessage': 'nnauthorized_access', 'success': True, 'data': {}}, 401)


@ app.errorhandler(400)
def not_found(error):
    return utils.utils.response_json({'errorMessage': 'invalid_data', 'success': True, 'data': {}}, 400)


@ app.errorhandler(404)
def page_not_found(error):
    return utils.utils.response_json({'errorMessage': 'data_not_found', 'success': True, 'data': {}}, 404)


@ app.errorhandler(500)
def special_exception_handler(error):
    return utils.utils.response_json({'errorMessage': 'server_error', 'success': True, 'data': {}}, 500)


@ app.route('/api/getTestData', methods=['GET'])
@ auth.login_required
def getTestData():
    return utils.utils.response_json({'data': 'getTestData', 'success': True, 'errorMessage': ''}, 200)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
