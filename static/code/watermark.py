import os
from PIL import Image

watermark_path = "/Users/a1-6/Workspaces/AIInfer/static/watermark.png" # 水印图片的路径
source_folder = "/Users/a1-6/Workspaces/AIInfer/01AICluster/01Roadmap/images_src" # 原始图片的文件夹
output_folder = "/Users/a1-6/Workspaces/AIInfer/01AICluster/01Roadmap/watermark" # 输出图片的文件夹


def check_image(img_path):
    if(img_path.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))):
        return True
    else:
        return False


def resize_image(input_image, limit_w=1200, limit_h=680):
    """
    调整图片尺寸，限制最大宽度为 1000px，最大高度为 600px
    """
    original_width, original_height = input_image.size

    # 计算新的尺寸，保持宽高比
    ratio = min(limit_w/original_width, limit_h/original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    # 调整图片尺寸
    resized_img = input_image.resize((new_width, new_height), Image.LANCZOS)
    return resized_img

def png_to_jpg(input_image, output_path=None, quality=100):
    """
    将 PNG 图片转换为 JPG 格式，透明背景替换为白色
    """
    # 创建白色背景画布
    background = Image.new('RGB', input_image.size, (255, 255, 255))
    # 将原图合并到白色背景上
    background.paste(input_image, mask=input_image.split()[-1])  # 使用 alpha 通道作为 mask
    return background


def pastic_watermask(input_image, output_path=None):
    """
    将输入图片添加水印
    """
    watermark = Image.open(watermark_path).convert("RGBA") # 打开并转换水印图片
    watermark_width, watermark_height = watermark.size # 获取水印图片的尺寸
    w_ration = watermark_width/watermark_height

    margin = 10 # 边距
    image_width, image_height = input_image.size # 获取图片文件的尺寸
    new_watermark_hight = int(image_height/10)
    new_watermark_width = int(image_height/10 * w_ration)
    watermark_x = image_width - new_watermark_width - margin # 水印图片在 x 轴上的位置
    watermark_y = image_height - new_watermark_hight - margin # 水印图片在 y 轴上的位置

    new_watermark = watermark.resize((new_watermark_width, new_watermark_hight))
    input_image.paste(new_watermark, (watermark_x, watermark_y), new_watermark) # 将水印图片合成到原始图片上
    return input_image


for filename in os.listdir(source_folder): # 遍历原始图片的文件夹
    if check_image(filename): # 判断是否是图片文件
        
        print("dealing with image:" + filename)
        image_path = os.path.join(source_folder, filename) # 拼接图片文件的路径
        output_path = os.path.join(output_folder, filename) # 拼接输出文件的路径

        try:
            image = Image.open(image_path).convert("RGBA") # 打开并转换图片文件
            image_resize = resize_image(image)
            image_watermask = pastic_watermask(image_resize)
            image_out = png_to_jpg(image_watermask)

            image_out.save(output_path, quality=100) # 保存输出文件
            print("\t Finished image:" + filename)
        except Exception as e:
            print(f"\t >>>>>>>>>>> Cannot deling with image: {filename} for reasion: {e}")