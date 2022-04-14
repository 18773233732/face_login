from flask import make_response
import json
import numpy as np
import wmi
import socket
from scipy import misc
from config.config import app_config
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer
from itsdangerous import BadSignature, SignatureExpired
import src.facenet
import os

w = wmi.WMI()

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


def get_device_info():
    result = {}
    # 获取电脑使用者信息
    for CS in w.Win32_ComputerSystem():
        # print(CS)
        # print("电脑名称: %s" % CS.Caption)
        result['ComputerName'] = "%s" % CS.Caption
        # print("使用者: %s" % CS.UserName)
        result['User'] = "%s" % CS.UserName
        # print("制造商: %s" % CS.Manufacturer)
        result['Manufacturer'] = "%s" % CS.Manufacturer
        # print("系统信息: %s" % CS.SystemFamily)
        result['SystemInformation'] = "%s" % CS.SystemFamily
        # print("工作组: %s" % CS.Workgroup)
        result['WorkingGroup'] = "%s" % CS.Workgroup
        # print("机器型号: %s" % CS.model)
        result['MachineModel'] = "%s" % CS.model
        # print("")
    # 获取操作系统信息
    for OS in w.Win32_OperatingSystem():
        # print(OS)
        # print("操作系统: %s" % OS.Caption)
        result['OperatingSystem'] = "%s" % OS.Caption
        # print("语言版本: %s" % OS.MUILanguages)
        result['LanguageVersion'] = "%s" % OS.MUILanguages
        # print("系统位数: %s" % OS.OSArchitecture)
        result['SystemBits'] = "%s" % OS.OSArchitecture
        # print("注册人: %s" % OS.RegisteredUser)
        result['Registrant'] = "%s" % OS.RegisteredUser
        # print("系统驱动: %s" % OS.SystemDevice)
        result['SystemDrive'] = "%s" % OS.SystemDevice
        # print("系统目录: %s" % OS.SystemDirectory)
        result['SystemDirectory'] = "%s" % OS.SystemDirectory
        # print("")
    # 获取电脑IP和MAC信息
    for address in w.Win32_NetworkAdapterConfiguration(ServiceName="e1dexpress"):
        # print(address)
        # print("IP地址: %s" % address.IPAddress)
        result['IPAddress'] = "%s" % address.IPAddress
        # print("MAC地址: %s" % address.MACAddress)
        result['MACAddress'] = "%s" % address.MACAddress
        # print("网络描述: %s" % address.Description)
        result['NetworkDescription'] = "%s" % address.Description
        # print("")
    # 获取电脑CPU信息
    for processor in w.Win32_Processor():
        # print(processor)
        # print("CPU型号: %s" % processor.Name.strip())
        result['CPUModel'] = "%s" % processor.Name.strip()
        # print("CPU核数: %s" % processor.NumberOfCores)
        result['CPUCoresNumber'] = "%s" % processor.NumberOfCores
        # print("")
    # 获取BIOS信息
    for BIOS in w.Win32_BIOS():
        # print(BIOS)
        # print("使用日期: %s" % BIOS.Description)
        result['UseDate'] = "%s" % BIOS.Description
        # print("主板型号: %s" % BIOS.SerialNumber)
        result['MotherboardModel'] = "%s" % BIOS.SerialNumber
        # print("当前语言: %s" % BIOS.CurrentLanguage)
        result['CurrentLanguage'] = "%s" % BIOS.CurrentLanguage
        # print("")
    # 获取内存信息
    for memModule in w.Win32_PhysicalMemory():
        totalMemSize = int(memModule.Capacity)
        # print("内存厂商: %s" % memModule.Manufacturer)
        result['MemoryManufacturer'] = "%s" % memModule.Manufacturer
        # print("内存型号: %s" % memModule.PartNumber)
        result['MemoryModel'] = "%s" % memModule.PartNumber
        # print("内存大小: %.2fGB" % (totalMemSize/1024**3))
        result['MemorySize'] = "%.2fGB" % (totalMemSize/1024**3)
        print("")
    # 获取磁盘信息
    for disk in w.Win32_DiskDrive():
        diskSize = int(disk.size)
        # print("磁盘名称: %s" % disk.Caption)
        result['DiskName'] = "%s" % disk.Caption
        # print("硬盘型号: %s" % disk.Model)
        result['HardDiskModel'] = "%s" % disk.Model
        # print("磁盘大小: %.2fGB" % (diskSize/1024**3))
        result['DiskSize'] = "%.2fGB" % (diskSize/1024**3)
    # 获取显卡信息
    for xk in w.Win32_VideoController():
        # print("显卡名称: %s" % xk.name)
        result['GraphicsCardName'] = "%s" % xk.name
        # print("")
    # 获取计算机名称和IP
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    # print("计算机名称: %s" % hostname)
    # print("IP地址: %s" % ip)
    # print(result)
    return {'DeviceData': result}

result = get_device_info()

def get_system_info():
    return result