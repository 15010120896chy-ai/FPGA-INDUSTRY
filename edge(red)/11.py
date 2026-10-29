import os
import cv2
import numpy as np
from tqdm import tqdm



#降噪111111111111111
def denoise_image(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

#对比度222222222222222
def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def process_image(input_path, output_path):
   
    img = cv2.imread(input_path)
    if img is None:
        print(f"无法读取图像: {input_path}")
        return



# 进行处理
    img = denoise_image(img)
    img = enhance_contrast(img)
    
# 保存
    cv2.imwrite(output_path, img)





"""批量处理文件夹中的所有图像"""
def batch_process_images(input_folder, output_folder):
    
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
    # 路径
    BASE_DIR = r"D:\app\pythonMODEDEMO"
    
    # 设置输入和输出文件夹路径
    input_dir = os.path.join(BASE_DIR, "input_images")
    output_dir = os.path.join(BASE_DIR, "output_images")
    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"错误：输入目录不存在 {input_dir}")
        print("请创建 input_images 文件夹并放入待处理图片")
        input("按回车键退出...")
        exit()
    
    # 批量
    batch_process_images(input_dir, output_dir)
    
    input("处理完成，按回车键退出...")
