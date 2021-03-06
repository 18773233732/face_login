import wmi
import json
import socket
w = wmi.WMI()  # 获取电脑使用者信息
result = {}
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
    #print(processor)
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
print(json.dumps(result))
