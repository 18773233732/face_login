from flask import make_response
import json
from html5lib import serialize
import numpy as np
from scipy import misc
from config.config import app_config
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer
from itsdangerous import BadSignature, SignatureExpired
import src.facenet
import os


# Token Expired Error
class TokenExpiredError(Exception):
    def __init__(self, msg):
        Exception.__init__(self)
        self.msg = msg

    def to_dict(self):
        return {"success": True, "data": {}, "errorMessage": self.msg}


# Token incorrect Error
class TokenIncorrectError(Exception):
    def __init__(self, msg):
        Exception.__init__(self)
        self.msg = msg

    def to_dict(self):
        return {"success": True, 'data': {}, 'errorMessage': self.msg}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app_config['ALLOWED_EXTENSIONS']


def value_is_not_empty(value):
    return value not in ['', None, {}, []]


def empty_json_data(data):
    if isinstance(data, dict):
        temp_data = dict()
        for key, value in data.items():
            if value_is_not_empty(value):
                new_value = empty_json_data(value)
                if value_is_not_empty(new_value):
                    temp_data[key] = new_value
        return None if not temp_data else temp_data

    elif isinstance(data, list):
        temp_data = list()
        for value in data:
            if value_is_not_empty(value):
                new_value = empty_json_data(value)
                if value_is_not_empty(new_value):
                    temp_data.append(new_value)
        return None if not temp_data else temp_data

    elif value_is_not_empty(data):
        return data


# 生成token, 有效时间为一小时
def generate_auth_token(user_id, expiration=3600):
    s = Serializer(app_config['SECRET_KEY'], expires_in=expiration)
    return s.dumps({'user_id': user_id}).decode()


# 解析token
def verify_auth_token(token):
    s = Serializer(app_config['SECRET_KEY'])
    # token正确
    try:
        data = s.loads(token)
        user_id = data.get('user_id')
        # print('正确')
        return user_id
    # token过期
    except SignatureExpired:
        # print('过期')
        # return None
        # return {'data': 'token_expired', 'success': False}
        raise TokenExpiredError('token_expired')
    # token错误
    except BadSignature:
        # print('错误')
        # return {'data': 'token_', 'success': False}
        raise TokenIncorrectError('token_incorrect')


def get_auth_token_to_id(token):
    s = Serializer(app_config['SECRET_KEY'])
    try:
        data = s.loads(token)
        user_id = data.get('user_id')
        return user_id
    except Exception:
        return 0


def value_is_not_empty(value):
    return value not in ['', None, {}, []]


def empty_json_data(data):
    if isinstance(data, dict):
        temp_data = dict()
        for key, value in data.items():
            if value_is_not_empty(value):
                new_value = empty_json_data(value)
                if value_is_not_empty(new_value):
                    temp_data[key] = new_value
        return None if not temp_data else temp_data

    elif isinstance(data, list):
        temp_data = list()
        for value in data:
            if value_is_not_empty(value):
                new_value = empty_json_data(value)
                if value_is_not_empty(new_value):
                    temp_data.append(new_value)
        return None if not temp_data else temp_data

    elif value_is_not_empty(data):
        return data


def response_json(json_data, status):
    # 过滤空数据字段
    # data = empty_json_data(json_data)
    return make_response(json.dumps(json_data, ensure_ascii=False, sort_keys=True), status)


def params_check(*params):
    result = True
    for item in params:
        if value_is_not_empty(item):
            continue
        else:
            result = False
            break
    return result

# 检测图片中的人脸  image_arr是opencv读取图片后的3维矩阵  返回图片中人脸的位置信息


def image_array_align_data(image_arr, image_path, pnet, rnet, onet, image_size=160, margin=32, detect_multiple_faces=True):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    img = image_arr
    bounding_boxes, _ = src.align.detect_face.detect_face(
        img, minsize, pnet, rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]

    nrof_successfully_aligned = 0
    if nrof_faces > 0:
        det = bounding_boxes[:, 0:4]
        det_arr = []
        img_size = np.asarray(img.shape)[0:2]
        if nrof_faces > 1:
            if detect_multiple_faces:
                for i in range(nrof_faces):
                    det_arr.append(np.squeeze(det[i]))
            else:
                bounding_box_size = (
                    det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                img_center = img_size / 2
                offsets = np.vstack(
                    [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                index = np.argmax(
                    bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                det_arr.append(det[index, :])
        else:
            det_arr.append(np.squeeze(det))

        images = np.zeros((len(det_arr), image_size, image_size, 3))
        for i, det in enumerate(det_arr):
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
            # 进行图片缩放 cv2.resize(img,(w,h))
            scaled = misc.imresize(
                cropped, (image_size, image_size), interp='bilinear')
            nrof_successfully_aligned += 1

            if image_path:
                # 保存检测的头像
                filename_base = 'pic_tmp'
                filename = os.path.basename(image_path)
                filename_name, file_extension = os.path.splitext(filename)
                # 多个人脸时，在picname后加_0 _1 _2 依次累加。
                output_filename_n = "{}/{}{}".format(
                    filename_base, filename_name, file_extension)
                misc.imsave(output_filename_n, scaled)

            scaled = src.facenet.prewhiten(scaled)
            scaled = src.facenet.crop(scaled, False, 160)
            scaled = src.facenet.flip(scaled, False)

            images[i] = scaled
            # 只保存第一张人脸
            break
    if nrof_faces > 0:
        return images
    else:
        # 如果没有检测到人脸  直接返回一个1*3的0矩阵  多少维度都行  只要能和是不是一个图片辨别出来就行
        return np.zeros((1, 3))
