import os
import cv2
import numpy as np
from tqdm import tqdm


def detect_contours(image, draw_on_original=True):
    """
    检测并绘制轮廓
    :param image: 输入图像
    :param draw_on_original: 是否在原图上绘制轮廓
    :return: 带轮廓的图像和轮廓数据
    """
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 二值化处理
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 轮廓检测
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建结果图像
    if draw_on_original:
        result = image.copy()
    else:
        result = np.zeros_like(image)
    
    # 绘制轮廓（红色，线宽2）
    cv2.drawContours(result, contours, -1, (0, 0, 255), 2)
    
    return result, contours

def process_image(input_path, output_path):
    """
    处理单张图像
    :param input_path: 输入图像路径
    :param output_path: 输出图像路径
    """
    img = cv2.imread(input_path)
    if img is None:
        print(f"无法读取图像: {input_path}")
        return
    
    # 仅进行轮廓检测
    img, contours = detect_contours(img)
    print(f"检测到 {len(contours)} 个轮廓")  # 调试信息
    
    # 保存结果
    cv2.imwrite(output_path, img)

def batch_process_images(input_folder, output_folder):
    """
    批量处理文件夹中的所有图像
    :param input_folder: 输入文件夹路径
    :param output_folder: 输出文件夹路径
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有图像文件
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    image_files = [f for f in os.listdir(input_folder) 
                  if f.lower().endswith(supported_formats)]
    
    if not image_files:
        print(f"在文件夹 {input_folder} 中没有找到支持的图像文件")
        return
    
    print(f"开始处理 {len(image_files)} 张图像...")
    
    # 使用进度条显示处理进度
    for filename in tqdm(image_files, desc="处理进度"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        process_image(input_path, output_path)
    
    print(f"所有图像处理完成! 输出目录: {output_folder}")

if __name__ == "__main__":
    # 配置参数
    BASE_DIR = r"D:\app\pythonMODEDEMO"
    
    # 设置路径
    input_dir = os.path.join(BASE_DIR, "input_images")
    output_dir = os.path.join(BASE_DIR, "output_images")
    
    # 检查输入目录
    if not os.path.exists(input_dir):
        print(f"错误：输入目录不存在 {input_dir}")
        print("请创建 input_images 文件夹并放入待处理图片")
        input("按回车键退出...")
        exit()
    
    # 批量处理
    batch_process_images(input_dir, output_dir)
    
    input("处理完成，按回车键退出...")
