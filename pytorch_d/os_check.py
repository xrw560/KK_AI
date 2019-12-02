import platform


def print_platform_info():
    print("----operation system----------")
    print(platform.python_version())  # python版本
    print(platform.architecture())  # 操作系统可执行程序的结构
    print(platform.node())  # 计算机的网路名称
    print(platform.platform())  # 操作系统的名称及版本号
    print(platform.processor())  # 计算机处理器
    print(platform.python_build())  # 操作系统python的构建日期
    print(platform.python_compiler())  # 系统中的python解释器的信息
    if platform.python_branch() == "":
        print(platform.python_implementation())
        print(platform.python_revision())
    print(platform.release())
    print(platform.version())  # 操作系统的版本
    print(platform.uname())  # 包含上面所有信息的汇总
    print(platform.system())


def check_os():
    return platform.system()


if __name__ == "__main__":
    print_platform_info()
