import os
import cv2
import numpy as np
from tqdm import tqdm

def detect_contours(image, draw_on_original=True):
    # 灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 使用自适应阈值代替全局阈值，减少黑斑影响
    blockSize = 19  # 增加块大小
    C =  2         # 减少常数值
    binary = cv2.adaptiveThreshold(gray, 255, 
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, blockSize, C)
    
    # 形态学操作去除小噪点
    kernel_size_open = 3  # 开运算核大小
    kernel_open = np.ones((kernel_size_open, kernel_size_open), np.uint8)
    iterations_open = 2   # 开运算迭代次数
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=iterations_open)
    
    # 进一步形态学操作：闭运算
    kernel_size_close = 3 # 闭运算核大小
    kernel_close = np.ones((kernel_size_close, kernel_size_close), np.uint8)
    iterations_close = 2   # 闭运算迭代次数
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=iterations_close)
    
    # 轮廓检测
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # 创建结果图像
    if draw_on_original:
        result = image.copy()
    else:
        result = np.zeros_like(image)
    
    # 过滤掉两侧区域
    filtered_contours = []
    height, width = image.shape[:2]
    left_boundary_percentage = 0.2  # 忽略左侧20%区域
    right_boundary_percentage = 0.8 # 忽略右侧20%区域
    
    left_boundary = width * left_boundary_percentage
    right_boundary = width * right_boundary_percentage
    
    for cnt in contours:
        # 计算轮廓的边界矩形
        x, y, w, h = cv2.boundingRect(cnt)
        
        # 过滤条件：不在两侧
        if (x > left_boundary and x + w < right_boundary):
            filtered_contours.append(cnt)
    
    # 绘制过滤后的轮廓（红色，线宽5）
    cv2.drawContours(result, filtered_contours, -1, (0, 0, 255), 5)
    
    return result, filtered_contours

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
    
    # 进行轮廓检测
    img, contours = detect_contours(img)
    print(f"检测到 {len(contours)} 个有效轮廓")  # 调试信息
    
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



