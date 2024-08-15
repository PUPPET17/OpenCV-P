import cv2
import numpy as np

def canny_edge_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    canny_edges = cv2.Canny(blurred, 70, 150)
    canny_edges_colored = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR)
    cv2.putText(canny_edges_colored, 'Canny', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return canny_edges_colored

def sobel_edge_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = cv2.convertScaleAbs(sobel)
    sobel_colored = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
    cv2.putText(sobel_colored, 'Sobel', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return sobel_colored

def laplacian_edge_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    laplacian_colored = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
    cv2.putText(laplacian_colored, 'Laplacian', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return laplacian_colored

def scharr_edge_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    scharrx = cv2.Scharr(blurred, cv2.CV_64F, 1, 0)
    scharry = cv2.Scharr(blurred, cv2.CV_64F, 0, 1)
    scharr = cv2.magnitude(scharrx, scharry)
    scharr = cv2.convertScaleAbs(scharr)
    scharr_colored = cv2.cvtColor(scharr, cv2.COLOR_GRAY2BGR)
    cv2.putText(scharr_colored, 'Scharr', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return scharr_colored

def prewitt_edge_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=int)
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    prewittx = cv2.filter2D(blurred, cv2.CV_32F, kernelx)
    prewitty = cv2.filter2D(blurred, cv2.CV_32F, kernely)
    prewitt = cv2.magnitude(prewittx, prewitty)
    prewitt = cv2.convertScaleAbs(prewitt)
    prewitt_colored = cv2.cvtColor(prewitt, cv2.COLOR_GRAY2BGR)
    cv2.putText(prewitt_colored, 'Prewitt', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return prewitt_colored

def main():
    # 打开默认摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    while True:
        # 从摄像头读取帧
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # 调整图像大小
        resized_frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
        
        # 调用不同的边缘检测函数
        canny = canny_edge_detection(resized_frame)
        sobel = sobel_edge_detection(resized_frame)
        laplacian = laplacian_edge_detection(resized_frame)
        scharr = scharr_edge_detection(resized_frame)
        prewitt = prewitt_edge_detection(resized_frame)
        
        # 合并所有边缘检测结果
        combined = np.hstack((canny, sobel, laplacian, scharr, prewitt))
        
        cv2.namedWindow('Edge Detection Comparison', cv2.WINDOW_NORMAL)
        # 显示合并后的图像
        cv2.imshow('Edge Detection Comparison', combined)
        
        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放摄像头并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
