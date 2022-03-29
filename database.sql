CREATE DATABASE IF NOT EXISTS face;
use face;
CREATE TABLE IF NOT EXISTS `face_json` (
  `id` int(32) NOT NULL AUTO_INCREMENT COMMENT 'id自增',
  `uid` varchar(64) DEFAULT NULL COMMENT '图片用户id',
  `json` text COMMENT '人脸的向量',
  `pic_name` varchar(255) DEFAULT NULL COMMENT '图片名称',
  `date` datetime DEFAULT NULL COMMENT '插入时间',
  `state` tinyint(1) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8;