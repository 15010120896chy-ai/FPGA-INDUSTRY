import os
import cv2
import numpy as np
from tqdm import tqdm

def enhanced_metal_defect_detection(img):
    """增强版金属缺陷检测算法"""
    # 转换为灰度图（兼容彩色/灰度输入）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) > 2 else img.copy()
    
    # 1. 多尺度光照校正
    blurred = cv2.GaussianBlur(gray, (0, 0), 10)
    illumination_corrected = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    
    # 2. 频域增强（突出微观缺陷）
    rows, cols = illumination_corrected.shape
    crow, ccol = rows//2, cols//2
    
    # 傅里叶变换
    dft = np.fft.fft2(illumination_corrected)
    dft_shift = np.fft.fftshift(dft)
    
    # 创建高通滤波器
    mask = np.ones((rows, cols), np.uint8)
    r = 30
    cv2.circle(mask, (ccol, crow), r, 0, -1)
    
    # 应用滤波
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    highpass = np.fft.ifft2(f_ishift)
    highpass = np.abs(highpass)
    highpass = cv2.normalize(highpass, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 3. 多特征融合
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    local_contrast = clahe.apply(highpass)
    
    # 4. 动态阈值分割
    thresh = cv2.adaptiveThreshold(local_contrast, 255, 
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 31, 5)
    
    # 5. 形态学优化
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 6. 缺陷标注
    result = img.copy() if len(img.shape) > 2 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    result[cleaned > 0] = [0, 0, 255]  # 红色标记
    
    return result

def process_image(input_path, output_path, show_comparison=False):
    """处理单个图像"""
    img = cv2.imread(input_path)
    if img is None:
        print(f"无法读取图像: {input_path}")
        return
    
    processed = enhanced_metal_defect_detection(img)
    
    if show_comparison:
        h, w = img.shape[:2]
        
        # 确保zoom区域尺寸有效
        zoom_h, zoom_w = h//3, w//3
        if zoom_h == 0 or zoom_w == 0:
            zoom_h, zoom_w = h, w
            
        # 计算zoom区域坐标（确保不越界）
        y1, y2 = max(0, h//3), min(h, 2*h//3)
        x1, x2 = max(0, w//3), min(w, 2*w//3)
        
        zoom_region = processed[y1:y2, x1:x2]
        zoom_region = cv2.resize(zoom_region, (zoom_w, zoom_h))
        
        # 创建对比图
        comparison = np.hstack((img, processed))
        
        # 确保zoom区域能放入对比图右下角
        if zoom_h <= comparison.shape[0] and zoom_w <= comparison.shape[1]:
            comparison[-zoom_h:, -zoom_w:] = zoom_region
        
        cv2.imshow('Original vs Processed', cv2.resize(comparison, (1200, 600)))
        cv2.waitKey(500)
        cv2.destroyAllWindows()
    
    cv2.imwrite(output_path, processed)

def batch_process_images(input_folder, output_folder, show_comparison=False):
    """批量处理"""
    os.makedirs(output_folder, exist_ok=True)
    
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(input_folder) 
                  if f.lower().endswith(supported_formats)]
    
    if not image_files:
        print(f"在文件夹 {input_folder} 中没有找到支持的图像文件")
        return
    
    print(f"开始处理 {len(image_files)} 张金属表面图像...")
    
    for filename in tqdm(image_files, desc="处理进度"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        process_image(input_path, output_path, show_comparison)
    
    print(f"处理完成! 结果保存在: {output_folder}")

if __name__ == "__main__":
    # 配置路径
    base_dir = r"D:\app\pythonMODEDEMO"
    input_dir = os.path.join(base_dir, "input_images")
    output_dir = os.path.join(base_dir, "output_images")
    
    if not os.path.exists(input_dir):
        print(f"错误：输入目录不存在 {input_dir}")
        print("请创建 input_images 文件夹并放入待检测图片")
        input("按回车键退出...")
        exit()
    
    batch_process_images(input_dir, output_dir, show_comparison=True)
    input("按回车键退出程序...")
