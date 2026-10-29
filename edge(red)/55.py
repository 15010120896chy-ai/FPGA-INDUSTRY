import os
import cv2
import numpy as np
from tqdm import tqdm

def optimize_for_metal_defects(img):
    """专为金属表面缺陷检测优化的预处理流程 - 高对比度低亮度版"""
    # 转换为灰度图（确保是单通道）
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # 1. 初始亮度降低 (降低整体亮度)
    gray = cv2.convertScaleAbs(gray, alpha=0.7, beta=-30)  # 降低亮度
    
    # 2. 自适应直方图均衡化 - 增强对比度
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(12, 12))  # 更高clipLimit，更小网格
    enhanced = clahe.apply(gray)
    
    # 3. 缺陷特异性增强 - 改进的锐化
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)  # 轻微模糊
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    sharpened = cv2.addWeighted(enhanced, 1.8, blurred, -0.8, 0)  # 更强锐化
    
    # 4. 多阈值缺陷提取
    # 针对暗色缺陷 (降低阈值灵敏度)
    _, black_defects = cv2.threshold(sharpened, 30, 255, cv2.THRESH_BINARY_INV)  # 固定阈值
    
    # 针对亮色缺陷 (提高阈值灵敏度)
    _, white_defects = cv2.threshold(sharpened, 220, 255, cv2.THRESH_BINARY)  # 固定高阈值
    
    # 自适应阈值增强 (提高对比度)
    adaptive_thresh = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 15, 5)  # 增大block size和C值
    
    # 组合多种阈值结果
    defects = cv2.bitwise_or(black_defects, white_defects)
    defects = cv2.bitwise_or(defects, adaptive_thresh)
    
    # 5. 改进的形态学处理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    defects = cv2.morphologyEx(defects, cv2.MORPH_OPEN, kernel, iterations=1)
    defects = cv2.morphologyEx(defects, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 6. 边缘增强 (降低阈值)
    edges = cv2.Canny(sharpened, 30, 100)  # 更低阈值检测更多边缘
    defects = cv2.bitwise_or(defects, edges)
    
    # 7. 结果合成 - 高对比度混合
    # 先增强原始图像的对比度
    sharpened_high_contrast = cv2.convertScaleAbs(sharpened, alpha=1.5, beta=0)
    # 与缺陷图混合
    result = cv2.addWeighted(sharpened_high_contrast, 0.7, defects, 0.5, -20)  # 降低整体亮度
    
    # 8. 最终对比度调整
    result = cv2.convertScaleAbs(result, alpha=1.3, beta=-30)  # 进一步提高对比度，降低亮度
    
    return result

def process_image(input_path, output_path, show_comparison=False):
    """处理单个图像并可选显示对比"""
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法读取图像: {input_path}")
        return
    
    processed = optimize_for_metal_defects(img)
    
    if show_comparison:
        # 调整显示大小以适应屏幕
        scale_percent = 60  # 调整为原图的60%
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        resized_original = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        resized_processed = cv2.resize(processed, dim, interpolation=cv2.INTER_AREA)
        
        comparison = np.hstack((resized_original, resized_processed))
        cv2.imshow('Original (左) vs Processed (右)', comparison)
        cv2.waitKey(800)  # 显示0.8秒
        cv2.destroyAllWindows()
    
    cv2.imwrite(output_path, processed)

def batch_process_images(input_folder, output_folder, show_comparison=False):
    """批量处理文件夹中的所有图像"""
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
    print("处理模式: 高对比度/低亮度")
    
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
    batch_process_images(input_dir, output_dir, show_comparison=True)  # 默认显示对比
    
    # 防止窗口闪退
    input("按回车键退出程序...")
