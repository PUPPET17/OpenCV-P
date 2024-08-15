import cv2
import numpy as np
import os

# 设置图片路径
image_path = "C:/Users/10023/Desktop/fun/OpenCV/pic/side.JPEG"

# 检查图像是否存在
if not os.path.exists(image_path):
    print(f"Error: Image not found at {image_path}")
    exit()

# 读取图像
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not read image.")
    exit()

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Canny 边缘检测
canny = cv2.Canny(gray, 70, 150)
canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
cv2.putText(canny, 'Canny', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

# Sobel 边缘检测
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobelx, sobely)
sobel = cv2.convertScaleAbs(sobel)
sobel = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
cv2.putText(sobel, 'Sobel', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

# Laplacian 边缘检测
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)
laplacian = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
cv2.putText(laplacian, 'Laplacian', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

# Scharr 边缘检测
scharrx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
scharry = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
scharr = cv2.magnitude(scharrx, scharry)
scharr = cv2.convertScaleAbs(scharr)
scharr = cv2.cvtColor(scharr, cv2.COLOR_GRAY2BGR)
cv2.putText(scharr, 'Scharr', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

# Prewitt 边缘检测
kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=int)
kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
prewittx = cv2.filter2D(gray, cv2.CV_32F, kernelx)  # 转换为 CV_32F
prewitty = cv2.filter2D(gray, cv2.CV_32F, kernely)  # 转换为 CV_32F
prewitt = cv2.magnitude(prewittx, prewitty)
prewitt = cv2.convertScaleAbs(prewitt)
prewitt = cv2.cvtColor(prewitt, cv2.COLOR_GRAY2BGR)
cv2.putText(prewitt, 'Prewitt', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

# 将所有结果合并为一个窗口进行显示
combined = np.hstack((canny, sobel, laplacian, scharr, prewitt))

# 调整窗口大小并显示结果
cv2.namedWindow('Edge Detection Comparison', cv2.WINDOW_NORMAL)
cv2.imshow('Edge Detection Comparison', combined)
cv2.imwrite('Edge_detection.jpg', combined)

# 按下 'q' 键退出
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 关闭所有窗口
cv2.destroyAllWindows()
