import platform


def check_os():
    return platform.system()


def get_fashion_mnist_path():
    sys_name = check_os()
    if sys_name == "Windows":
        return "F:/data/pytorch_data/FashionMNIST"
    elif sys_name == "Linux":
        return "/root/private/data/FashionMNIST"
    else:
        print("未检索到当前系统名称")
        exit(0)

if __name__=="__main__":
    print(check_os())