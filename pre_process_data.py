from PIL import Image
import os
import math
from io import BytesIO
import uuid
import pandas as pd

target_width = 256
target_height = target_width

img_num_pre_folder = 1000

data_info = {
    "path": [],
    "type": []
}


def crop_img_to_fit(img_path):
    global target_width
    global target_height

    img = Image.open(img_path)

    try:
        img = img.convert("RGB")
    except OSError as e:
        # 解决 "OSError: image file is truncated (0 bytes not processed)"
        print(e)

        with open(img_path, 'rb') as f:
            f = f.read()
        f = f + B'\xff' + B'\xd9'

        img = Image.open(BytesIO(f))
        img = img.convert("RGB")

    width, height = img.width, img.height

    if width < target_width:
        while width < target_width:
            width *= 1.2

        width = math.ceil(width)

    if height < target_height:
        while height < target_height:
            height *= 1.2

        height = math.ceil(height)

    img = img.resize((width, height))

    left = width - target_width
    left = math.floor(left / 2)

    top = height - target_height
    top = math.floor(top / 2)

    right = left + target_width
    bottom = top + target_height

    img = img.crop((left, top, right, bottom))

    return img


def shrink_img_to_fit(img_path):
    global target_width
    global target_height

    img = Image.open(img_path)

    try:
        img = img.convert("RGB")
    except OSError as e:
        # 解决 "OSError: image file is truncated (0 bytes not processed)"
        print(e)

        with open(img_path, 'rb') as f:
            f = f.read()
        f = f + B'\xff' + B'\xd9'

        img = Image.open(BytesIO(f))
        img = img.convert("RGB")

    img = img.resize((target_width, target_height))

    return img


def gen_img(img_data):
    # 接收一个 PIL 图像对象 返回一个四元组 四元组中元素均为 PIL.Image
    # 分别是:
    # 逆时针旋转 90 度,
    # 逆时针旋转 180 度,
    # 逆时针旋转 270 度,
    # 镜像反转后的图像数据

    rotate_90 = img_data.rotate(90, expand=True)
    rotate_180 = img_data.rotate(180, expand=True)
    rotate_270 = img_data.rotate(270, expand=True)
    transpose = img_data.transpose(Image.FLIP_LEFT_RIGHT)

    return (
        rotate_90,
        rotate_180,
        rotate_270,
        transpose
    )


for folder_name in os.listdir("./Data/original"):
    img_list = os.listdir(f"./Data/original/{folder_name}")

    files_num_in_folder = len(img_list)

    target_folder = f"./Data/available/{folder_name}"

    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    for img_name in img_list:
        print(f"processing ./Data/original/{folder_name}/{img_name} ...")

        img = shrink_img_to_fit(f"./Data/original/{folder_name}/{img_name}")

        new_img_name = img_name.split(".")[0] + ".jpg"
        target_path = f"{target_folder}/{new_img_name}"

        img.save(target_path, "jpeg")

        data_info["path"].append(target_path)
        data_info["type"].append(folder_name)

    print(f"fixing {folder_name} ...")

    # 数据不够 需要增补
    if files_num_in_folder < img_num_pre_folder:
        # 需要使用多少张图像
        num_of_img = math.ceil((img_num_pre_folder - files_num_in_folder) / 4)

        img_paths = os.listdir(f"./Data/available/{folder_name}")[:num_of_img]

        for im_path in img_paths:
            img = Image.open(f"./Data/available/{folder_name}/{im_path}")

            for im in gen_img(img):
                target_path = f"./Data/available/{folder_name}/{uuid.uuid4()}.jpg"

                im.save(target_path, "jpeg")

                data_info["path"].append(target_path)
                data_info["type"].append(folder_name)
        else:
            # 现在文件夹中的图像数量有可能多于 1000
            # 删除掉多余部分

            img_paths = os.listdir(f"./Data/available/{folder_name}")

            num = len(img_paths)
            num = num - img_num_pre_folder

            for i in range(num):
                os.remove(data_info["path"][-1])

                del data_info["path"][-1]
                del data_info["type"][-1]

print("checking ...")

# 检查
for path in data_info["path"]:
    if not os.path.exists(path):
        print(f"err: {path} not exit!")

# 生成 csv
df = pd.DataFrame(
    data=data_info,
    index=range(len(data_info["path"]))
)

df.to_csv("./Data/available/info.csv")

print("done.")


