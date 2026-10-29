import os
import cv2
import numpy as np
from tqdm import tqdm

def optimize_for_metal_defects(img):
    """专为金属表面缺陷检测优化的预处理流程"""
    # 转换为灰度图（确保是单通道）
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # 1. 强对比度拉伸
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(32, 32))
    enhanced = clahe.apply(gray)
    
    # 2. 缺陷特异性增强
    laplacian = cv2.Laplacian(enhanced, cv2.CV_64F)
    sharpened = enhanced - 0.5*laplacian
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    # 3. 双阈值缺陷提取
    _, black_defects = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, white_defects = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    defects = cv2.bitwise_or(black_defects, white_defects)
    
    # 4. 形态学处理
    kernel = np.ones((3,3), np.uint8)
    enhanced_defects = cv2.morphologyEx(defects, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 5. 结果合成
    result = cv2.addWeighted(sharpened, 0.7, enhanced_defects, 0.3, 0)
    return result

def process_image(input_path, output_path, show_comparison=False):
    """处理单个图像并可选显示对比"""
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法读取图像: {input_path}")
        return
    
    processed = optimize_for_metal_defects(img)
    
    if show_comparison:
        comparison = np.hstack((img, processed))
        cv2.imshow('Original vs Processed', comparison)
        cv2.waitKey(300)  # 显示0.3秒
        cv2.destroyAllWindows()
    
    cv2.imwrite(output_path, processed)

def batch_process_images(input_folder, output_folder, show_comparison=False):
    """批量处理文件夹中的所有图像
    
    参数:
        input_folder: 输入图片文件夹路径
        output_folder: 输出图片文件夹路径
        show_comparison: 是否显示处理前后对比(默认为False)
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有图像文件
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(input_folder) 
                  if f.lower().endswith(supported_formats)]
    
    if not image_files:
        print(f"在文件夹 {input_folder} 中没有找到支持的图像文件")
        return
    
    print(f"开始处理 {len(image_files)} 张铝片图像...")
    print("检测类型: 划痕/黑点/褶皱等表面缺陷")
    
    # 使用进度条显示处理进度
    for filename in tqdm(image_files, desc="处理进度"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        process_image(input_path, output_path, show_comparison)
    
    print(f"所有图像处理完成! 结果已保存到: {output_folder}")

if __name__ == "__main__":
    # 配置路径
    base_dir = r"D:\app\pythonMODEDEMO"  # 修改为你的实际路径
    input_dir = os.path.join(base_dir, "input_images")
    output_dir = os.path.join(base_dir, "output_images")
    
    # 检查输入目录
    if not os.path.exists(input_dir):
        print(f"错误：输入目录不存在 {input_dir}")
        print("请创建 input_images 文件夹并放入待检测的铝片图片")
        input("按回车键退出...")
        exit()
    
    # 执行批量处理(show_comparison=True可显示处理前后对比)
    batch_process_images(input_dir, output_dir, show_comparison=False)
    
    # 防止窗口闪退
    input("按回车键退出程序...")
