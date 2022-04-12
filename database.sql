/*用户表*/
-- DROP DATABASE if EXISTS face_sys;

CREATE DATABASE face_sys;

use face_sys;

CREATE TABLE IF NOT EXISTS `face_users` (
  `id` int(32) NOT NULL AUTO_INCREMENT COMMENT 'id自增',
  `user_name` varchar(50) NULL COMMENT '用户名',
  `password` varchar(150) NULL COMMENT '密码',
  `vector` text COMMENT '人脸的向量',
  `pic_name` varchar(255) DEFAULT NULL COMMENT '图片名称',
  `phone` varchar(20) NULL COMMENT '手机',
  `date` datetime DEFAULT NULL COMMENT '插入时间',
  `state` tinyint(1) DEFAULT NULL COMMENT '状态',
  `provinces` int(10) NULL DEFAULT '0' COMMENT '省',
  `city` int(10) NULL DEFAULT '0' COMMENT '城市',
  `area` int(10) NULL DEFAULT '0' COMMENT '地区',
  PRIMARY KEY (`id`)
) ENGINE = InnoDB AUTO_INCREMENT = 1 DEFAULT CHARSET = utf8;


CREATE TABLE IF NOT EXISTS `face_device_info` (
  `id` int(32) NOT NULL AUTO_INCREMENT COMMENT 'id自增',
  `ip` varchar(20) NULL COMMENT 'ip地址',
  `cpu_use` numeric NULL COMMENT 'cpu占用率',
  `mem_use` numeric NULL COMMENT '内存占用率',
  `disk_use` numeric NULL COMMENT '硬盘使用率',
  `temperature` numeric NULL COMMENT '温度',
  `humidity` numeric NULL COMMENT '湿度',
  `date` datetime DEFAULT NULL COMMENT '插入时间',
  PRIMARY KEY (`id`)
) ENGINE = InnoDB AUTO_INCREMENT = 1 DEFAULT CHARSET = utf8;
